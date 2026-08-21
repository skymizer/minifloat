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

//! `matches_host`'s twin through `float`
//!
//! `float` deliberately, not `double`: this is the referee for the route
//! `route` in `benches/arith.cpp` grants `E5M10` and `E8M7`, and nothing else
//! in the suite covers it.  Below 2p + 2 digits in the intermediate, rounding
//! twice can differ from rounding once, so a shape gets this treatment only
//! where the bench would grant it a `float`.
template <typename Operation, typename T> bool matches_float(T x, T y) {
  const Operation op;
  const float reference = op(x.to_float(), y.to_float());
  if (!T::HAS_NAN && (std::isnan)(reference))
    return true;
  return same_mini(op(x, y), T{reference});
}

//! All four operators of one ordered pair against the float route
template <typename T> bool float_arithmetic_matches(T x, T y) {
  return matches_float<std::plus<>>(x, y) && matches_float<std::minus<>>(x, y) &&
         matches_float<std::multiplies<>>(x, y) && matches_float<std::divides<>>(x, y);
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
  // A 14-bit significand shifted by more overflows `int64_t` once the 16-bit
  // shapes run through here; 48 keeps both aligned addends under 2**62.  The
  // sticky substitute below is deliberately unlike the engine's drop to zero,
  // and both round alike.
  constexpr int ALIGN_CAP = 48;
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

//! The same exact oracle, sampled where every ordered pair is out of reach
//!
//! 2**16 pairs, not 2**13: the Rust crate found that 2**13 misses an `E2M13`
//! double rounding.  The seed differs from the host sweep's so the two draw
//! different pairs.
struct CheckExactWideArithmetic {
  template <typename T> static bool check() {
    Lcg random{UINT64_C(0x0123456789ABCDEF)};
    constexpr auto MASK = bit_mask(T::EXPONENT_BITS + T::MANTISSA_BITS + 1);

    for (unsigned i = 0; i < 1U << 16; ++i) {
      const auto draw = [&random] {
        return T::from_bits(static_cast<typename T::Storage>(random.next() & MASK));
      };
      const T x = draw();
      const T y = draw();
      if (!check_exact_pair(x, y))
        return false;
    }
    return true;
  }
};

//! Every arm of the special-value ladder, pinned to an exact bit pattern
//!
//! `same_mini` would let a signed NaN pass; arithmetic never manufactures one,
//! so these compare codes.  The invalid result is the format's NaN, or its
//! positive maximum where it has none, derived here from the test's own
//! encoder rather than from the library.
struct CheckSpecialLadder {
  template <typename T> static bool check() {
    constexpr auto SIGN = std::uint64_t{1} << (T::EXPONENT_BITS + T::MANTISSA_BITS);
    const T invalid = reference_encode<T>(std::numeric_limits<double>::quiet_NaN());
    const T zero = T::from_bits(0);
    const T one{1.0};

    // `SIGN` is captured although it is a constant expression: MSVC rejects
    // the implicit use outright (C3493) where GCC and Clang allow it.
    const auto signed_zero = [SIGN](bool negative) {
      const bool keep = T::HAS_NEG_ZERO && negative;
      return T::from_bits(static_cast<typename T::Storage>(keep ? SIGN : 0));
    };

    bool ok = true;
    const auto same = [&ok](const char *what, T actual, T expected) {
      if (actual.to_bits() == expected.to_bits())
        return;
      ADD_FAILURE() << describe<T>() << ' ' << what << ": got " << +actual.to_bits()
                    << ", expected " << +expected.to_bits();
      ok = false;
    };

    same("1 * 1", one * one, one);
    same("(-1) * 1", -one * one, -one);
    same("(-1) * (-1)", -one * -one, one);
    same("1 / (-1)", one / -one, -one);
    same("(-1) / (-1)", -one / -one, one);

    same("0 - 0", zero - zero, signed_zero(false));
    same("1 - 1", one - one, signed_zero(false));
    same("1 - 0", one - zero, one);
    same("0 - 1", zero - one, -one);
    same("0 * (-1)", zero * -one, signed_zero(true));

    same("0 / 0", zero / zero, invalid);
    same("1 / 0", one / zero, signed_huge<T>(false));
    same("(-1) / 0", -one / zero, signed_huge<T>(true));

    if constexpr (T::HAS_NEG_ZERO) {
      const T neg_zero = T::from_bits(static_cast<typename T::Storage>(SIGN));
      same("(-0) - (-0)", neg_zero - neg_zero, signed_zero(false));
      same("(-0) - 0", neg_zero - zero, signed_zero(true));
      same("(-0) + (-0)", neg_zero + neg_zero, signed_zero(true));
      same("0 + (-0)", zero + neg_zero, signed_zero(false));
      same("(-0) / 0", neg_zero / zero, invalid);
      same("1 / (-0)", one / neg_zero, signed_huge<T>(true));
    }

    if constexpr (T::HAS_NAN) {
      const T nan = T::quiet_NaN();
      const T signed_nan = T::from_bits(static_cast<typename T::Storage>(nan.to_bits() | SIGN));
      same("nan + 1", nan + one, invalid);
      same("1 + nan", one + nan, invalid);
      same("nan - 1", nan - one, invalid);
      same("1 - nan", one - nan, invalid);
      same("nan * 1", nan * one, invalid);
      same("1 * nan", one * nan, invalid);
      same("nan / 1", nan / one, invalid);
      same("1 / nan", one / nan, invalid);
      same("(-nan) + 1", signed_nan + one, invalid);
      same("1 * (-nan)", one * signed_nan, invalid);
    }

    if constexpr (T::HAS_INF) {
      const T inf = T::infinity();
      same("inf + inf", inf + inf, inf);
      same("(-inf) + (-inf)", -inf + -inf, -inf);
      same("inf + (-inf)", inf + -inf, invalid);
      same("(-inf) + inf", -inf + inf, invalid);
      same("inf - inf", inf - inf, invalid);
      same("inf - (-inf)", inf - -inf, inf);
      same("inf + 1", inf + one, inf);
      same("1 + inf", one + inf, inf);
      same("inf - 1", inf - one, inf);
      same("1 - inf", one - inf, -inf);
      same("inf * inf", inf * inf, inf);
      same("inf * (-1)", inf * -one, -inf);
      same("inf * 0", inf * zero, invalid);
      same("0 * inf", zero * inf, invalid);
      same("inf / inf", inf / inf, invalid);
      same("inf / (-1)", inf / -one, -inf);
      same("inf / 0", inf / zero, inf);
      same("1 / inf", one / inf, signed_zero(false));
      same("(-1) / inf", -one / inf, signed_zero(true));
      same("1 / (-inf)", one / -inf, signed_zero(true));
    }
    return ok;
  }
};
} // namespace

TEST(Arith, MatchesHostRoundTrip) { test_paired_types<CheckHostArithmetic>(); }

//! Every ordered pair of `E5M10` and `E8M7`, against the float route
//!
//! `CheckHostArithmetic` stops at 11 bits because the check is quadratic; this
//! carries the same idea to the two shapes `route` puts on the float route,
//! all 2**32 pairs of each.
//!
//! No other 16-bit shape belongs here, because none of them is on that route.
//! `IEEE<2, 13>` is exact in a `float` and still not entitled to one -- a
//! product of two of its significands is 28 digits -- so `route` times it
//! against a `double`, as it does `IEEE<11, 4>`; `IEEE<12, 3>` is the one
//! shape `route` skips outright.  The double route is already covered by
//! `CheckWideHostArithmetic` and refereed by the exact oracle.
TEST(Arith, EveryPairMatchesFloatRoundTrip) {
  expect_all_pairs<E5M10>(float_arithmetic_matches<E5M10>);
  expect_all_pairs<E8M7>(float_arithmetic_matches<E8M7>);
}

TEST(Arith, WideFormatsMatchHostRoundTrip) { test_wide_types<CheckWideHostArithmetic>(); }

TEST(Arith, CorrectlyRoundedSmallFormats) { test_small_types<CheckExactSmallArithmetic>(); }

TEST(Arith, CorrectlyRoundedWideFormats) { test_wide_types<CheckExactWideArithmetic>(); }

TEST(Arith, SpecialValueLadder) {
  check_each<CheckSpecialLadder, Finite<4, 3>, IEEE<4, 3>, FN<4, 3>, FNUZ<4, 3>>();
  test_wide_types<CheckSpecialLadder>();
}

//! Results beyond `double` that the destination shape still holds exactly
TEST(Arith, PastDoubleRange) {
  using T = IEEE<12, 3>;
  // Every code here is 1 * 2**k, whose exponent field is `k` + `BIAS`.
  const auto power = [](int k) {
    return T::from_bits(static_cast<T::Storage>((k + T::BIAS) << T::MANTISSA_BITS));
  };

  EXPECT_TRUE(same_mini(power(-1000) * power(-1000), power(-2000))); // double flushes to zero
  EXPECT_TRUE(same_mini(power(1000) * power(1000), power(2000)));    // and this to infinity
  EXPECT_TRUE(same_mini(power(-2000) / power(-1000), power(-1000)));
  EXPECT_TRUE(same_mini(power(2000) - power(1999), power(1999)));
}

//! The integer engine leaves every operator usable at compile time
TEST(Arith, ConstantEvaluation) {
  using T = E4M3; // bias 7, so an exponent field of 7 is 1.0
  constexpr T ONE = T::from_bits(0x38);
  constexpr T TWO = T::from_bits(0x40);
  static_assert((ONE + ONE).to_bits() == TWO.to_bits());
  static_assert((TWO - ONE).to_bits() == ONE.to_bits());
  static_assert((TWO * ONE).to_bits() == TWO.to_bits());
  static_assert((TWO / TWO).to_bits() == ONE.to_bits());
  SUCCEED();
}

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
