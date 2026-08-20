// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "minifloat.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace minifloat_test {
using namespace skymizer::minifloat; // NOLINT(google-build-using-namespace)

//! Test floating-point identity like Object.is in JavaScript
inline bool same_float(float x, float y) {
  return bit_cast<std::uint32_t>(x) == bit_cast<std::uint32_t>(y) ||
         ((std::isnan)(x) && (std::isnan)(y));
}

//! Test floating-point identity like Object.is in JavaScript
inline bool same_double(double x, double y) {
  return bit_cast<std::uint64_t>(x) == bit_cast<std::uint64_t>(y) ||
         ((std::isnan)(x) && (std::isnan)(y));
}

//! Test minifloat identity, treating every NaN representation alike
template <class Format> bool same_mini(Minifloat<Format> x, Minifloat<Format> y) {
  return x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
}

//! Comparison result similar to `x <=> y` in C++20
template <typename T> int compare(T x, T y) { return (x > y) - (x < y); }

//! Name a type in a failure message, since checks run over type lists
template <typename T> std::string describe() {
  return "E" + std::to_string(T::EXPONENT_BITS) + "M" + std::to_string(T::MANTISSA_BITS) +
         " B=" + std::to_string(T::BIAS) + (T::HAS_INF ? " inf" : "") + (T::HAS_NAN ? " nan" : "") +
         (T::HAS_NEG_ZERO ? " -0" : "");
}

//! Iterate over every representation of a minifloat type
template <typename T, typename Predicate> bool for_all(Predicate pred) {
  constexpr unsigned END = 1U << (T::EXPONENT_BITS + T::MANTISSA_BITS + 1);

  for (unsigned bits = 0; bits < END; ++bits)
    if (!pred(T::from_bits(static_cast<typename T::Storage>(bits))))
      return false;
  return true;
}

//! Run `Checker::check<T>()` for every `T` in the pack
template <typename Checker, typename... Ts> void check_each() {
  const auto run = [](auto sample) {
    using T = decltype(sample);
    EXPECT_TRUE(Checker::template check<T>()) << describe<T>();
  };
  (run(Ts{}), ...);
}

//! Every format layer over representative shapes through 8 bits
template <typename Checker> void test_small_types() {
  check_each<
      Checker, //
      Finite<2, 1>, Finite<2, 3>, Finite<3, 2>, Finite<2, 5>, Finite<3, 4>, Finite<4, 3>,
      Finite<4, 3, 11>, Finite<5, 2>, Finite<6, 1>, Finite<7, 0>, //
      IEEE<2, 1>, IEEE<2, 3>, IEEE<3, 2>, IEEE<2, 5>, IEEE<3, 4>, IEEE<4, 3>, IEEE<4, 3, 11>,
      IEEE<5, 2>, IEEE<6, 1>, //
      FN<2, 1>, FN<2, 3>, FN<3, 2>, FN<2, 5>, FN<3, 4>, FN<4, 3>, FN<4, 3, 11>, FN<5, 2>, FN<6, 1>,
      FN<7, 0>, //
      FNUZ<2, 1>, FNUZ<2, 3>, FNUZ<3, 2>, FNUZ<2, 5>, FNUZ<3, 4>, FNUZ<4, 3>, FNUZ<4, 3, 11>,
      FNUZ<5, 2>, FNUZ<6, 1>, FNUZ<7, 0>>();
}

//! Shapes reaching conversion paths that no 8-bit type can reach
template <typename Checker> void test_wide_types() {
  check_each<
      Checker, IEEE<5, 10>, IEEE<8, 7>, IEEE<11, 4>, IEEE<12, 3>, IEEE<12, 3, 1000>, FN<12, 3>,
      FNUZ<12, 3>, Finite<12, 3>, IEEE<2, 13>>();
}

template <typename Checker> void test_all_types() {
  test_small_types<Checker>();
  test_wide_types<Checker>();
}

//! Quadratic checks retain the old 11-bit stress shapes
template <typename Checker> void test_paired_types() {
  test_small_types<Checker>();
  check_each<Checker, IEEE<5, 5>, FN<5, 5>, FNUZ<5, 5>>();
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

constexpr std::uint64_t bit_mask(unsigned width) { return width ? UINT64_MAX >> (64 - width) : 0; }

struct Scaled {
  std::uint64_t significand;
  int exponent;
};

inline int bit_width(std::uint64_t x) {
  int width = 0;
  for (; x; x >>= 1)
    ++width;
  return width;
}

//! Exact comparison of non-negative significand-times-power-of-two pairs
inline int compare_scaled(Scaled x, Scaled y) {
  if (!x.significand || !y.significand)
    return compare(x.significand, y.significand);

  const int x_lead = x.exponent + bit_width(x.significand);
  const int y_lead = y.exponent + bit_width(y.significand);
  if (x_lead != y_lead)
    return compare(x_lead, y_lead);

  if (x.exponent >= y.exponent)
    return compare(x.significand << (x.exponent - y.exponent), y.significand);
  return compare(x.significand, y.significand << (y.exponent - x.exponent));
}

//! Decompose a finite double magnitude exactly
inline Scaled decompose(double x) {
  const auto bits = bit_cast<std::uint64_t>(x);
  const auto field = static_cast<int>(bits >> 52);
  const std::uint64_t fraction = bits & bit_mask(52);

  if (!field)
    return {fraction, DBL_MIN_EXP - DBL_MANT_DIG};
  return {fraction | UINT64_C(1) << 52, field + DBL_MIN_EXP - 1 - DBL_MANT_DIG};
}

//! Exact value of a magnitude code, extended one code beyond the maximum
template <typename T> Scaled code_value(std::uint64_t code) {
  const auto field = code >> T::MANTISSA_BITS;
  const auto fraction = code & bit_mask(T::MANTISSA_BITS);

  if (!field)
    return {fraction, 1 - T::BIAS - T::MANTISSA_BITS};
  return {
      fraction | UINT64_C(1) << T::MANTISSA_BITS,
      static_cast<int>(field) - T::BIAS - T::MANTISSA_BITS,
  };
}

//! Magnitudes of the overflow result and maximum finite value
template <typename T> constexpr std::pair<std::uint64_t, std::uint64_t> huge_and_max() {
  constexpr auto ABS_MASK = bit_mask(T::EXPONENT_BITS + T::MANTISSA_BITS);
  constexpr auto HUGE = T::HAS_INF ? bit_mask(T::EXPONENT_BITS) << T::MANTISSA_BITS
                        : T::HAS_NAN && T::HAS_NEG_ZERO ? ABS_MASK - 1
                                                        : ABS_MASK;
  return {HUGE, HUGE - T::HAS_INF};
}

//! Correctly round an exact value supplied through comparisons
template <typename T, typename Compare> T reference_round(bool negative, Compare cmp) {
  const auto [huge, max] = huge_and_max<T>();
  std::uint64_t lo = 0;
  std::uint64_t hi = max + 1;

  while (lo < hi) {
    const auto mid = (lo + hi + 1) / 2;
    if (cmp(code_value<T>(mid)) < 0)
      hi = mid - 1;
    else
      lo = mid;
  }

  std::uint64_t code = huge;
  if (lo <= max) {
    const auto lower = code_value<T>(lo);
    const int midpoint = cmp({2 * lower.significand + 1, lower.exponent - 1});
    code = std::min(lo + (midpoint > 0 || (midpoint == 0 && (lo & 1))), huge);
  }

  const auto sign = std::uint64_t{negative} << (T::EXPONENT_BITS + T::MANTISSA_BITS);
  const bool signed_result = T::HAS_NEG_ZERO || code;
  return T::from_bits(static_cast<typename T::Storage>(code | signed_result * sign));
}

//! Correctly rounded encoding derived independently from the library
template <typename T> T reference_encode(double x) {
  if ((std::isnan)(x)) {
    const auto [huge, max] = huge_and_max<T>();
    static_cast<void>(huge);
    std::uint64_t magnitude;
    if constexpr (!T::HAS_NAN)
      magnitude = max;
    else if constexpr (T::HAS_INF)
      magnitude = bit_mask(T::EXPONENT_BITS + 1) << (T::MANTISSA_BITS - 1);
    else if constexpr (T::HAS_NEG_ZERO)
      magnitude = bit_mask(T::EXPONENT_BITS + T::MANTISSA_BITS);
    else
      magnitude = UINT64_C(1) << (T::EXPONENT_BITS + T::MANTISSA_BITS);
    const auto sign = std::uint64_t{std::signbit(x)} << (T::EXPONENT_BITS + T::MANTISSA_BITS);
    return T::from_bits(static_cast<typename T::Storage>(magnitude | sign));
  }

  if ((std::isinf)(x)) {
    const auto [huge, max] = huge_and_max<T>();
    static_cast<void>(max);
    const auto sign = std::uint64_t{std::signbit(x)} << (T::EXPONENT_BITS + T::MANTISSA_BITS);
    return T::from_bits(static_cast<typename T::Storage>(huge | sign));
  }

  const auto value = decompose(std::abs(x));
  return reference_round<T>(std::signbit(x), [value](Scaled candidate) {
    return compare_scaled(value, candidate);
  });
}

//! Inputs at and around every rounding decision reachable through double
template <typename T> std::vector<double> rounding_inputs() {
  const auto [huge, max] = huge_and_max<T>();
  static_cast<void>(huge);
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> inputs{
      0.0,
      -0.0,
      inf,
      -inf,
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::min(),
      -std::numeric_limits<double>::min(),
      std::numeric_limits<double>::denorm_min(),
      -std::numeric_limits<double>::denorm_min(),
  };

  if constexpr (T::HAS_NAN) {
    inputs.push_back(std::numeric_limits<double>::quiet_NaN());
    inputs.push_back(-std::numeric_limits<double>::quiet_NaN());
    inputs.push_back(bit_cast<double>(UINT64_C(0x7FF4000000000BAD)));
    inputs.push_back(bit_cast<double>(UINT64_C(0xFFF80000DEADBEEF)));
  }

  inputs.reserve(inputs.size() + static_cast<std::size_t>(max + 2) * 8);
  for (std::uint64_t code = 0; code <= max + 1; ++code) {
    const auto value = code_value<T>(code);
    const double exact = std::ldexp(static_cast<double>(value.significand), value.exponent);
    const double midpoint =
        std::ldexp(static_cast<double>(2 * value.significand + 1), value.exponent - 1);

    for (double x :
         {exact, midpoint, std::nextafter(midpoint, inf), std::nextafter(midpoint, -inf)}) {
      inputs.push_back(x);
      inputs.push_back(-x);
    }
  }
  return inputs;
}

} // namespace minifloat_test
