// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Integer arithmetic against a round trip through a host float
//!
//! Each operator is timed twice on the same operands: once as the library
//! computes it, on integer significands, and once the way a caller would fake
//! it — widen both operands, let the FPU work, round the result back.  `route`
//! decides which host float a shape is entitled to; a shape no host float can
//! round for is skipped rather than compared against a different answer.
//!
//! The two routes do not agree on every input, which is the reason the integer
//! one exists: a host float cannot referee a shape it cannot hold, and its NaN
//! carries a sign that means nothing.  Speed is the bonus this file measures.
//!
//! Both routes are timed in one binary, alternating within every pass, and the
//! reported figure is the minimum across passes.  Noise on a benchmark is
//! one-sided: nothing makes a loop run faster than it can.  Pin the run to one
//! core (`taskset -c 2`) on an idle box, or the numbers mean nothing.

#include "minifloat.hpp"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
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
  constexpr auto MASK = (1U << (T::EXPONENT_BITS + T::MANTISSA_BITS + 1)) - 1U;

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
template <typename T> constexpr Route route() {
  constexpr int DIGITS = 2 * T::MANTISSA_DIGITS + 2;

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

double total_log_ratio = 0.0;
int comparisons = 0;
int wins = 0;

//! Time one operator both ways and report the ratio in the integer route's
//! favour
template <typename Op, Route R, typename T>
void bench_op(const char *shape, const char *name, const std::vector<std::pair<T, T>> &pairs) {
  const Op op;

  const double soft = measure([&pairs, op] {
    for (const auto &pair : pairs)
      black_box(op(pair.first, pair.second).to_bits());
  });

  const double hard = measure([&pairs, op] {
    for (const auto &pair : pairs) {
      if constexpr (R == Route::Float)
        black_box(T{op(pair.first.to_float(), pair.second.to_float())}.to_bits());
      else
        black_box(T{op(pair.first.to_double(), pair.second.to_double())}.to_bits());
    }
  });

  const double ratio = hard / soft;
  total_log_ratio += std::log(ratio);
  ++comparisons;
  wins += ratio > 1.0;

  std::printf(
      "%-14s %-3s %8.3f %8.3f  %6.3fx %s\n", shape, name, soft, hard, ratio,
      R == Route::Float ? "float" : "double"
  );
}

//! One line per operator for a shape, or one line saying why it has none
template <typename T> void bench_shape(const char *shape) {
  constexpr Route R = route<T>();

  if constexpr (R == Route::None) {
    std::printf("%-14s skipped: no host float rounds like it\n", shape);
  } else {
    const auto pairs = draw_pairs<T>();
    bench_op<std::plus<>, R>(shape, "add", pairs);
    bench_op<std::minus<>, R>(shape, "sub", pairs);
    bench_op<std::multiplies<>, R>(shape, "mul", pairs);
    bench_op<std::divides<>, R>(shape, "div", pairs);
  }
}

} // namespace

int main() {
  std::printf("%-14s %-3s %8s %8s  %7s %s\n", "shape", "op", "soft", "host", "ratio", "route");

  bench_shape<E2M1FN>("E2M1FN");
  bench_shape<E2M3FN>("E2M3FN");
  bench_shape<E3M2FN>("E3M2FN");
  bench_shape<E3M4>("E3M4");
  bench_shape<E4M3>("E4M3");
  bench_shape<E4M3FN>("E4M3FN");
  bench_shape<E4M3FNUZ>("E4M3FNUZ");
  bench_shape<E4M3B11FNUZ>("E4M3B11FNUZ");
  bench_shape<E5M2>("E5M2");
  bench_shape<E5M2FNUZ>("E5M2FNUZ");
  bench_shape<E5M10>("E5M10");
  bench_shape<E8M7>("E8M7");
  bench_shape<IEEE<11, 4>>("E11M4");
  bench_shape<IEEE<2, 13>>("E2M13");
  bench_shape<IEEE<12, 3>>("E12M3");

  std::printf(
      "\ninteger route wins %d of %d, geomean %.3fx in its favour\n", wins, comparisons,
      std::exp(total_log_ratio / comparisons)
  );
}
