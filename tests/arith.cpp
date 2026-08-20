// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "support.hpp"

#include <functional>

using namespace minifloat_test;      // NOLINT(google-build-using-namespace)
using namespace skymizer::minifloat; // NOLINT(google-build-using-namespace)

namespace {
template <typename Operation, typename T> bool matches_host(T x, T y) {
  const Operation op;
  const double reference = op(x.to_double(), y.to_double());
  if (!T::HAS_NAN && (std::isnan)(reference))
    return true;
  return same_mini(op(x, y), T{reference});
}

struct CheckHostArithmetic {
  template <typename T> static bool check() {
    return for_all<T>([](T x) {
      return for_all<T>([x](T y) {
        return matches_host<std::plus<>>(x, y) && matches_host<std::minus<>>(x, y) &&
               matches_host<std::multiplies<>>(x, y) && matches_host<std::divides<>>(x, y);
      });
    });
  }
};

struct CheckWideHostArithmetic {
  template <typename T> static bool check() {
    if (!T::HAS_EXACT_F64_CONVERSION)
      return true;

    Lcg random{UINT64_C(0x0FEDCBA987654321)};
    constexpr auto MASK = bit_mask(T::EXPONENT_BITS + T::MANTISSA_BITS + 1);
    for (unsigned i = 0; i < 1U << 13; ++i) {
      const auto draw = [&random] {
        return T::from_bits(static_cast<typename T::Storage>(random.next() & MASK));
      };
      const T x = draw();
      const T y = draw();
      if (!matches_host<std::plus<>>(x, y) || !matches_host<std::minus<>>(x, y) ||
          !matches_host<std::multiplies<>>(x, y) || !matches_host<std::divides<>>(x, y)) {
        ADD_FAILURE() << describe<T>() << " host arithmetic mismatch for bits " << +x.to_bits()
                      << ", " << +y.to_bits();
        return false;
      }
    }
    return true;
  }
};

template <int E, int M> bool test_snowball_sum() {
  using T = FN<E, M>;
  using Bits = typename T::Storage;

  constexpr Bits STEP = 1U << M;
  constexpr Bits SIGNIFICAND = STEP - 1U;
  constexpr Bits EXPONENT = 1U << (E + M - 1);
  constexpr Bits GREATER = EXPONENT | SIGNIFICAND;

  for (Bits lesser = SIGNIFICAND; lesser <= GREATER; lesser += STEP) {
    const T x = T::from_bits(GREATER);
    const T y = T::from_bits(lesser);
    if (!same_mini(x + y, T{x.to_double() + y.to_double()}))
      return false;
  }
  return true;
}

struct Exact {
  bool negative;
  Scaled numerator;
  std::uint64_t denominator;
};

int compare_exact(const Exact &exact, Scaled candidate) {
  candidate.significand *= exact.denominator;
  return compare_scaled(exact.numerator, candidate);
}

//! Exact sum with one sticky bit for an addend beyond the aligned range
Exact exact_sum(bool xn, Scaled x, bool yn, Scaled y) {
  constexpr int ALIGN_CAP = 54;
  const int top = std::max(x.exponent, y.exponent);
  const int bottom = std::min(x.exponent, y.exponent);
  const int base = std::max(bottom, top - ALIGN_CAP);
  const auto align = [base](bool negative, Scaled value) {
    std::int64_t magnitude =
        !value.significand ? 0
        : value.exponent >= base
            ? static_cast<std::int64_t>(value.significand << (value.exponent - base))
            : 1;
    return negative ? -magnitude : magnitude;
  };
  const std::int64_t sum = align(xn, x) + align(yn, y);
  const auto magnitude = static_cast<std::uint64_t>(sum < 0 ? -sum : sum);
  return {sum == 0 ? xn && yn : sum < 0, {magnitude, base}, 1};
}

template <typename T> T signed_huge(bool negative) {
  const auto [huge, max] = huge_and_max<T>();
  static_cast<void>(max);
  const auto sign = std::uint64_t{negative} << (T::EXPONENT_BITS + T::MANTISSA_BITS);
  return T::from_bits(static_cast<typename T::Storage>(huge | sign));
}

template <typename T> bool check_exact_pair(T x, T y) {
  if (!x.is_finite() || !y.is_finite())
    return true;

  const bool xn = x.signbit();
  const bool yn = y.signbit();
  const Scaled xp = code_value<T>(x.to_bits() & T::ABS_MASK);
  const Scaled yp = code_value<T>(y.to_bits() & T::ABS_MASK);

  const auto check = [x, y](char op, T actual, const Exact &exact) {
    const T expected = reference_round<T>(exact.negative, [&exact](Scaled candidate) {
      return compare_exact(exact, candidate);
    });
    if (same_mini(actual, expected))
      return true;
    ADD_FAILURE() << describe<T>() << " bits " << +x.to_bits() << ' ' << op << ' ' << +y.to_bits()
                  << ": got " << +actual.to_bits() << ", expected " << +expected.to_bits();
    return false;
  };

  if (!check('+', x + y, exact_sum(xn, xp, yn, yp)) ||
      !check('-', x - y, exact_sum(xn, xp, !yn, yp)) ||
      !check(
          '*', x * y, {xn != yn, {xp.significand * yp.significand, xp.exponent + yp.exponent}, 1}
      ))
    return false;

  if (!yp.significand) {
    if (!xp.significand)
      return !T::HAS_NAN || (x / y).is_nan();
    return same_mini(x / y, signed_huge<T>(xn != yn));
  }
  return check('/', x / y, {xn != yn, {xp.significand, xp.exponent - yp.exponent}, yp.significand});
}

struct CheckExactSmallArithmetic {
  template <typename T> static bool check() {
    return for_all<T>([](T x) { return for_all<T>([x](T y) { return check_exact_pair(x, y); }); });
  }
};
} // namespace

TEST(Arith, MatchesHostRoundTrip) { test_paired_types<CheckHostArithmetic>(); }

TEST(Arith, WideFormatsMatchHostRoundTrip) { test_wide_types<CheckWideHostArithmetic>(); }

TEST(Arith, CorrectlyRoundedSmallFormats) { test_small_types<CheckExactSmallArithmetic>(); }

TEST(Arith, CompoundAssignment) {
  using T = E5M2;
  T x{2.0F};
  x += T{1.0F};
  EXPECT_EQ(x, T{3.0F});
  x -= T{0.5F};
  EXPECT_EQ(x, T{2.5F});
  x *= T{2.0F};
  EXPECT_EQ(x, T{5.0F});
  x /= T{2.0F};
  EXPECT_EQ(x, T{2.5F});
}

TEST(Arith, SnowballSum) {
  EXPECT_TRUE((test_snowball_sum<2, 11>()));
  EXPECT_TRUE((test_snowball_sum<3, 11>()));
  EXPECT_TRUE((test_snowball_sum<4, 11>()));
  EXPECT_TRUE((test_snowball_sum<2, 12>()));
  EXPECT_TRUE((test_snowball_sum<3, 12>()));
}
