// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Software kernels against hardware arithmetic over identical operands.
//!
//! BF compares the integer implementation with its specialized double route,
//! including zero-sign and NaN handling. Other formats compare software with
//! an eligible host round trip. The public operator is deliberately not the
//! software baseline: BF operators already select the hardware implementation.
//!
//! The unary table covers conversions and signs. Use --bf for every BF width,
//! --json for machine-readable rows, and docs/benchmarking.md for the protocol.

#include "minifloat.hpp"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace skymizer::minifloat; // NOLINT(google-build-using-namespace)

namespace {

//! Operand pairs one measurement runs over, small enough to stay in L1
constexpr std::size_t PAIRS = 1 << 10;
//! Passes to take the minimum over
constexpr int PASSES = 30;
//! Sweeps of the operand array inside one pass
constexpr int REPEATS = 200;

//! Keep a computed value from being optimized away
//!
//! The value has to reach a register and the compiler has to forget what it
//! is; nothing else needs to be spilled, so there is no memory clobber here.
template <typename T> void black_box(T x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : "+r"(x)); // NOLINT(hicpp-no-assembler)
#else
  volatile T sink = x;
  static_cast<void>(sink);
#endif
}

//! Make completed array stores observable without converting back to packed bits.
template <typename T> void observe_buffer(const std::vector<T> &values) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : : "r"(values.data()) : "memory"); // NOLINT(hicpp-no-assembler)
#else
  const auto *bytes = reinterpret_cast<const volatile unsigned char *>(values.data());
  for (std::size_t i = 0; i < values.size() * sizeof(T); ++i)
    black_box(bytes[i]);
#endif
}

//! Deterministic pseudo-random source using Knuth's MMIX constants
class Lcg {
  std::uint64_t state_;

public:
  explicit constexpr Lcg(std::uint64_t seed) : state_(seed) {}

  std::uint32_t next() {
    state_ = state_ * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return static_cast<std::uint32_t>(state_ >> 32);
  }
};

//! Draw `PAIRS` operand pairs, one bit pattern at a time
//!
//! Drawing raw codes gives NaNs, infinities and subnormals the density they
//! have in the format, which is the mix an operator has to survive.  Both
//! routes then run over the very same pairs.
template <typename T> std::vector<std::pair<T, T>> draw_pairs() {
  Lcg random{UINT64_C(0x0FEDCBA987654321)};
  constexpr auto MASK = UINT32_MAX >> (31 - T::EXPONENT_BITS - T::MANTISSA_BITS);

  std::vector<std::pair<T, T>> pairs;
  pairs.reserve(PAIRS);
  for (std::size_t i = 0; i < PAIRS; ++i) {
    const auto draw = [&random] {
      return T::from_bits(static_cast<typename T::Storage>(random.next() & MASK));
    };
    const T x = draw();
    pairs.emplace_back(x, draw());
  }
  return pairs;
}

//! The host float a shape may be compared against
enum struct Route { None, Float, Double };

//! The narrowest host float that rounds every operator like the shape does
//!
//! Two things have to hold.  The operands must be exact, or the round trip
//! starts from a different number.  And the intermediate must carry at least
//! 2p + 2 digits, where p is the shape's own precision: below that, rounding to
//! the intermediate and then to the shape can differ from rounding to the shape
//! once (Figueroa 1995).  Exactness alone is not enough — `IEEE<2, 13>` is
//! exact in `float`, yet a product of two of its significands is 28 digits.
//!
//! BF uses its actual double implementation, including BF32. The wider
//! intermediate is what makes its final rounding independent of the host mode.
template <typename T> constexpr Route route() {
  constexpr int DIGITS = 2 * T::MANTISSA_DIGITS + 2;

  if (T::IS_BFLOAT)
    return Route::Double;
  if (T::HAS_EXACT_F32_CONVERSION && DIGITS <= FLT_MANT_DIG)
    return Route::Float;
  if (T::HAS_EXACT_F64_CONVERSION && DIGITS <= DBL_MANT_DIG)
    return Route::Double;
  return Route::None;
}

//! Nanoseconds per element, minimum across `PASSES` passes of `Body`
template <typename Body> double measure(Body body) {
  double best = HUGE_VAL;

  for (int pass = 0; pass < PASSES; ++pass) {
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < REPEATS; ++repeat)
      body();
    const std::chrono::duration<double, std::nano> elapsed =
        std::chrono::steady_clock::now() - start;
    best = std::min(best, elapsed.count() / (PAIRS * REPEATS));
  }
  return best;
}

//! Emit `github-action-benchmark` rows instead of the two tables
//!
//! `--json` is what `.github/workflows/bench.yml` runs; the tables are what a
//! person reads, and no header, footer or skip notice belongs in a dataset.
bool emit_json = false;
bool json_started = false;
bool bf_only = false;

//! One `customSmallerIsBetter` row
//!
//! `{shape}/{op}/{soft|f32|f64}` from the ratio table, which has two routes to
//! tell apart, and `{shape}/{op}` from the unary one, which has none.  The
//! layout is the minifloat-rs sibling's -- a criterion group per shape, one
//! benchmark per operator -- but the ids are not interchangeable with it: the
//! two repositories spell most shapes differently, `E5M10` and `E8M7` here
//! against `F16` and `BF16` there, and only `E11M4`, `E2M13` and `E12M3` are
//! spelt alike.
void emit_row(const char *shape, const char *op, const char *route, double ns) {
  if (!emit_json)
    return;
  std::printf(
      "%s\n  {\"name\": \"%s/%s%s%s\", \"unit\": \"ns/element\", \"value\": %.6f}",
      json_started ? "," : "[", shape, op, route ? "/" : "", route ? route : "", ns
  );
  json_started = true;
}

double total_log_ratio = 0.0;
int comparisons = 0;
int wins = 0;

template <typename Op, typename T> T software(Op, T x, T y) {
  if constexpr (std::is_same_v<Op, std::plus<>>)
    return detail::add_impl(x, y, false);
  if constexpr (std::is_same_v<Op, std::minus<>>)
    return detail::add_impl(x, y, true);
  if constexpr (std::is_same_v<Op, std::multiplies<>>)
    return detail::mul_impl(x, y);
  if constexpr (std::is_same_v<Op, std::divides<>>)
    return detail::div_impl(x, y);
}

template <typename Op, typename T> T hardware_bf(Op, T x, T y) {
  if constexpr (std::is_same_v<Op, std::plus<>>)
    return detail::bf_arithmetic<detail::BfOp::Add>(x, y);
  if constexpr (std::is_same_v<Op, std::minus<>>)
    return detail::bf_arithmetic<detail::BfOp::Sub>(x, y);
  if constexpr (std::is_same_v<Op, std::multiplies<>>)
    return detail::bf_arithmetic<detail::BfOp::Mul>(x, y);
  if constexpr (std::is_same_v<Op, std::divides<>>)
    return detail::bf_arithmetic<detail::BfOp::Div>(x, y);
}

//! Time one operator both ways and report the ratio in the integer route's
//! favour
template <typename Op, Route R, typename T>
void bench_op(const char *shape, const char *name, const std::vector<std::pair<T, T>> &pairs) {
  const Op op;

  const double soft = measure([&pairs, op] {
    for (const auto &pair : pairs)
      black_box(software(op, pair.first, pair.second).to_bits());
  });

  const double hard = measure([&pairs, op] {
    for (const auto &pair : pairs) {
      if constexpr (T::IS_BFLOAT)
        black_box(hardware_bf(op, pair.first, pair.second).to_bits());
      else if constexpr (R == Route::Float)
        black_box(T{op(pair.first.to_float(), pair.second.to_float())}.to_bits());
      else
        black_box(T{op(pair.first.to_double(), pair.second.to_double())}.to_bits());
    }
  });

  const double ratio = hard / soft;
  total_log_ratio += std::log(ratio);
  ++comparisons;
  wins += ratio > 1.0;

  emit_row(shape, name, "soft", soft);
  emit_row(shape, name, R == Route::Float ? "f32" : "f64", hard);

  if (!emit_json)
    std::printf(
        "%-14s %-3s %8.3f %8.3f  %6.3fx %s\n", shape, name, soft, hard, ratio,
        R == Route::Float ? "float" : "double"
    );
}

//! Time one body and report nanoseconds per element
//!
//! Unlike an operator, none of these has a second route to be timed against:
//! `to_float` and the constructor *are* the host route, and negation has no
//! host analogue worth faking.  The figure is therefore absolute, and it means
//! something only against the same line from another build — which is what the
//! interleaved protocol in `docs/benchmarking.md` does.
template <typename Body> void bench_unary(const char *shape, const char *name, Body body) {
  const double ns = measure(body);

  emit_row(shape, name, nullptr, ns);

  if (!emit_json)
    std::printf("%-14s %-4s %8.3f\n", shape, name, ns);
}

//! One line per unary body for a shape
//!
//! Every shape is timed here, including the ones `route` skips: a conversion
//! path is exactly what a shape no host float can round for still has.
template <typename T> void bench_unary_shape(const char *shape) {
  const auto pairs = draw_pairs<T>();

  // A shape's own `to_float` never leaves its contract: a format without a NaN
  // has none to hand back, so `T{...}` below is always a value the format can
  // represent.
  std::vector<float> floats;
  std::vector<double> doubles;
  floats.reserve(pairs.size());
  doubles.reserve(pairs.size());
  for (const auto &pair : pairs) {
    floats.push_back(pair.first.to_float());
    doubles.push_back(pair.first.to_double());
  }

  bench_unary(shape, "neg", [&pairs] {
    for (const auto &pair : pairs)
      black_box((-pair.first).to_bits());
  });
  bench_unary(shape, "abs", [&pairs] {
    for (const auto &pair : pairs)
      black_box(pair.first.abs().to_bits());
  });
  bench_unary(shape, "f32", [&pairs] {
    for (const auto &pair : pairs)
      black_box(pair.first.to_float());
  });
  bench_unary(shape, "f64", [&pairs] {
    for (const auto &pair : pairs)
      black_box(pair.first.to_double());
  });
  bench_unary(shape, "from", [&floats] {
    for (const float x : floats)
      black_box(T{x}.to_bits());
  });
  bench_unary(shape, "from_f64", [&doubles] {
    for (const double x : doubles)
      black_box(T{x}.to_bits());
  });

  // The packed sinks above can cancel BF's final alignment. These rows keep
  // the actual object representation and allow the compiler to vectorize the
  // array conversion. Allocate outside timing and observe every completed pass.
  std::vector<T> output(pairs.size());
  bench_unary(shape, "store_f32", [&floats, &output] {
    for (std::size_t i = 0; i < output.size(); ++i)
      output[i] = T{floats[i]};
    observe_buffer(output);
  });
  bench_unary(shape, "store_f64", [&doubles, &output] {
    for (std::size_t i = 0; i < output.size(); ++i)
      output[i] = T{doubles[i]};
    observe_buffer(output);
  });
}

//! One line per operator for a shape, or one line saying why it has none
//!
//! IEEE<12, 3> exceeds every host float's range and has no hardware route.
//! BF retains a meaningful comparison by calling the integer kernels directly.
template <typename T> void bench_shape(const char *shape) {
  constexpr Route R = route<T>();

  if constexpr (R == Route::None) {
    if (!emit_json)
      std::printf("%-14s skipped: no host float rounds like it\n", shape);
  } else {
    const auto pairs = draw_pairs<T>();
    bench_op<std::plus<>, R>(shape, "add", pairs);
    bench_op<std::minus<>, R>(shape, "sub", pairs);
    bench_op<std::multiplies<>, R>(shape, "mul", pairs);
    bench_op<std::divides<>, R>(shape, "div", pairs);
  }
}

//! Every shape the tables run over, so one list serves both
//!
//! The visitor takes a value rather than an explicit template argument, which
//! a C++17 lambda cannot: `tests/support.hpp` spells its type lists the same
//! way.
template <typename Visit, std::size_t... I>
void for_each_bf(Visit visit, std::index_sequence<I...>) {
  (visit(BF<static_cast<int>(I) + 10>{}, ("BF" + std::to_string(I + 10)).c_str()), ...);
}

template <typename Visit> void for_each_shape(Visit visit) {
  if (bf_only) {
    for_each_bf(visit, std::make_index_sequence<23>{});
    return;
  }
  visit(E2M1FN{}, "E2M1FN");
  visit(E2M3FN{}, "E2M3FN");
  visit(E3M2FN{}, "E3M2FN");
  visit(E3M4{}, "E3M4");
  visit(E4M3{}, "E4M3");
  visit(E4M3FN{}, "E4M3FN");
  visit(E4M3FNUZ{}, "E4M3FNUZ");
  visit(E4M3B11FNUZ{}, "E4M3B11FNUZ");
  visit(E5M2{}, "E5M2");
  visit(E5M2FNUZ{}, "E5M2FNUZ");
  visit(E5M10{}, "E5M10");
  visit(E8M7{}, "E8M7");
  visit(BF<20>{}, "BF20");
  visit(BF<24>{}, "BF24");
  visit(BF<32>{}, "BF32");
  visit(IEEE<11, 4>{}, "E11M4");
  visit(IEEE<2, 13>{}, "E2M13");
  visit(IEEE<12, 3>{}, "E12M3");
}

} // namespace

int main(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    emit_json |= std::strcmp(argv[i], "--json") == 0;
    bf_only |= std::strcmp(argv[i], "--bf") == 0;
  }

  if (!emit_json)
    std::printf("%-14s %-3s %8s %8s  %7s %s\n", "shape", "op", "soft", "host", "ratio", "route");
  for_each_shape([](auto sample, const char *shape) { bench_shape<decltype(sample)>(shape); });

  if (!emit_json)
    std::printf(
        "\ninteger route wins %d of %d, geomean %.3fx in its favour\n", wins, comparisons,
        std::exp(total_log_ratio / comparisons)
    );

  if (!emit_json)
    std::printf("\n%-14s %-4s %8s\n", "shape", "op", "ns");
  for_each_shape([](auto sample, const char *shape) {
    bench_unary_shape<decltype(sample)>(shape);
  });

  if (emit_json)
    std::puts(json_started ? "\n]" : "[]");
}
