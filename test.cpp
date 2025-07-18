// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "minifloat.hpp"
#include <gtest/gtest.h>

using namespace skymizer::minifloat; // NOLINT(google-build-using-namespace)

namespace {
/// Test floating-point identity like Object.is in JavaScript
///
/// This is necessary because NaN != NaN in C++.  We also want to differentiate
/// -0 from +0.  Using this functor, NaNs are considered identical to each
/// other, while +0 and -0 are considered different.
bool same_double(double x, double y) {
  return bit_cast<std::uint64_t>(x) == bit_cast<std::uint64_t>(y) || (x != x && y != y);
}

/// Test floating-point identity like Object.is in JavaScript
///
/// See also `same_double`.
template <int E, int M, NanStyle N, int B, SubnormalStyle D>
bool same_mini(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  return x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
}

/// Comparison result similar to `x <=> y` in C++20
///
/// - +2 if `x > y`
/// - -2 if `x < y`
/// -  0 if `x == y` or not comparable
template <typename T> int compare(T x, T y) { return (x > y) - (x < y); }

/// Iterate over all possible values of a minifloat type `T`
template <typename T, typename Predicate> bool for_all(Predicate pred) {
  constexpr unsigned END = 1U << (T::EXPONENT_BITS + T::MANTISSA_BITS + 1);

  for (unsigned i = 0; i < END; ++i) {
    if (!pred(T::from_bits(i)))
      return false;
  }
  return true;
}

template <typename Checker> void test_selected_types() {
  EXPECT_TRUE((Checker::template check<2, 5, NanStyle::IEEE>)());
  EXPECT_TRUE((Checker::template check<2, 5, NanStyle::FN>)());
  EXPECT_TRUE((Checker::template check<2, 5, NanStyle::FNUZ>)());

  EXPECT_TRUE((Checker::template check<3, 4, NanStyle::IEEE>)());
  EXPECT_TRUE((Checker::template check<3, 4, NanStyle::FN>)());
  EXPECT_TRUE((Checker::template check<3, 4, NanStyle::FNUZ>)());

  EXPECT_TRUE((Checker::template check<4, 3, NanStyle::IEEE>)());
  EXPECT_TRUE((Checker::template check<4, 3, NanStyle::FN>)());
  EXPECT_TRUE((Checker::template check<4, 3, NanStyle::FNUZ>)());

  EXPECT_TRUE((Checker::template check<4, 3, NanStyle::IEEE, 11>)());
  EXPECT_TRUE((Checker::template check<4, 3, NanStyle::FN, 11>)());
  EXPECT_TRUE((Checker::template check<4, 3, NanStyle::FNUZ, 11>)());

  EXPECT_TRUE((Checker::template check<5, 2, NanStyle::FN>)());
  EXPECT_TRUE((Checker::template check<5, 2, NanStyle::FNUZ>)());
  EXPECT_TRUE((Checker::template check<5, 7, NanStyle::FN>)());
  EXPECT_TRUE((Checker::template check<5, 7, NanStyle::FNUZ>)());

  EXPECT_TRUE((Checker::template check<6, 1, NanStyle::FN>)());
  EXPECT_TRUE((Checker::template check<6, 1, NanStyle::FNUZ>)());

  EXPECT_TRUE((Checker::template check<7, 0, NanStyle::FN>)());
  EXPECT_TRUE((Checker::template check<7, 0, NanStyle::FNUZ>)());

  EXPECT_TRUE((Checker::template check<5, 7, NanStyle::FN>)());
}

struct CheckCopying {
  template <int E, int M, NanStyle N, int B = default_bias(E)> static bool check() {
    Minifloat<E, M, N, B> a{2.0F};
    Minifloat<E, M, N, B> b = a;
    Minifloat<E, M, N, B> c;
    c = b;
    return c == a;
  }
};

struct CheckEquality {
  template <int E, int M, NanStyle N, int B = default_bias(E)> static bool check() {
    using T = Minifloat<E, M, N, B>;
    constexpr float FIXED_POINT = M == 0 ? -2.0F : -3.0F;

    EXPECT_EQ(T{FIXED_POINT}.to_float(), FIXED_POINT);
    EXPECT_EQ(T{FIXED_POINT}.to_double(), FIXED_POINT);

    EXPECT_EQ(T{0.0F}, T{-0.0F});
    EXPECT_EQ(T{0.0F}.to_bits() == T{-0.0F}.to_bits(), N == NanStyle::FNUZ);

    EXPECT_TRUE(T{NAN}.is_nan());
    EXPECT_TRUE((std::isnan)(T{NAN}.to_float()));
    EXPECT_TRUE((std::isnan)(T{NAN}.to_double()));

    return for_all<T>([](T x) { return (x != x) == x.is_nan(); });
  }
};

struct CheckUnarySign {
  template <int E, int M, NanStyle N, int B = default_bias(E)> static bool check() {
    using T = Minifloat<E, M, N, B>;

    return T{0.0F} == -T{0.0F} &&
           for_all<T>([](T x) { return same_mini(x, +x) && same_mini(x, - -x); });
  }
};

struct CheckComparison {
  template <int E, int M, NanStyle N, int B = default_bias(E)> static bool check() {
    using T = Minifloat<E, M, N, B>;

    return for_all<T>([](T x) {
      return for_all<T>([x](T y) {
        return compare(x, y) == compare(x.to_float(), y.to_float()) &&
               compare(x, y) == compare(x.to_double(), y.to_double());
      });
    });
  }
};

struct CheckClassification {
  template <int E, int M, NanStyle N, int B = default_bias(E)> static bool check() {
    using T = Minifloat<E, M, N, B>;

    return for_all<T>([](T x) {
      const int category = x.is_nan() << FP_NAN |             //
                           x.is_infinite() << FP_INFINITE |   //
                           !x << FP_ZERO |                    //
                           x.is_subnormal() << FP_SUBNORMAL | //
                           x.is_normal() << FP_NORMAL;
      return category == 1 << x.classify();
    });
  }
};

struct CheckIdentityConversion {
  template <int E, int M, NanStyle N, int B = default_bias(E)> static bool check() {
    using T = Minifloat<E, M, N, B>;
    constexpr bool HAS_NEG_ZERO = N != NanStyle::FNUZ;

    EXPECT_EQ(bit_cast<std::uint32_t>(T{0.0F}.to_float()), 0U);
    EXPECT_EQ(bit_cast<std::uint64_t>(T{0.0F}.to_double()), 0U);
    EXPECT_EQ(bit_cast<std::uint32_t>(T{-0.0F}.to_float()), HAS_NEG_ZERO * 0x8000'0000);
    EXPECT_EQ(bit_cast<std::uint64_t>(T{-0.0F}.to_double()), HAS_NEG_ZERO * 0x8000'0000'0000'0000);

    return for_all<T>([](T x) {
      return same_mini(x, T::from_bits(x.to_bits())) && same_mini(x, T{x.to_float()}) &&
             same_double(x.to_float(), x.to_double());
    });
  }
};

template <SubnormalStyle D, int E, int M, NanStyle N, int B>
bool check_subnormal_conversion(Minifloat<E, M, N, B, SubnormalStyle::Precise> prec) {
  static_assert(M > 0);
  static_assert(D != SubnormalStyle::Precise);

  using T = Minifloat<E, M, N, B, D>;
  using Bits = typename T::Storage;

  const T conv(prec.to_float());

  if (prec.signbit() != conv.signbit() && (N != NanStyle::FNUZ || conv.to_bits() != 0))
    return false;

  constexpr Bits THRESHOLD = 1U << M;
  const Bits magnitude = prec.abs().to_bits();

  if (magnitude == 0 || magnitude >= THRESHOLD)
    return prec.to_bits() == conv.to_bits() || (prec.is_nan() && conv.is_nan());

  if constexpr (D == SubnormalStyle::Reserved) {
    const Bits magnitude = conv.abs().to_bits();
    return magnitude == 0 || magnitude == 1U << M;
  }

  return T::from_bits(0) <= conv.abs() && conv.abs() <= T::from_bits(1U << M);
}

struct CheckSubnormalConversion {
  template <int E, int M, NanStyle N, int B = default_bias(E)> static bool check() {
    using T = Minifloat<E, M, N, B>;

    if constexpr (M > 0) {
      return for_all<T>([](T x) {
        return check_subnormal_conversion<SubnormalStyle::Reserved>(x) &&
               check_subnormal_conversion<SubnormalStyle::Fast>(x);
      });
    }
    return true;
  }
};

template <typename Operation> struct CheckExactArithmetics {
  template <int E, int M, NanStyle N, int B = default_bias(E)> static bool check() {
    using T = Minifloat<E, M, N, B>;

    return for_all<T>([op = Operation{}](T x) {
      return for_all<T>([op, x](T y) {
        const T z = op(x, y);
        const T answer{op(x.to_double(), y.to_double())};
        return same_mini(z, answer);
      });
    });
  }
};

template <int E, int M, NanStyle N = NanStyle::FN> bool test_snowball_sum() {
  using T = Minifloat<E, M, N>;
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

template <int E, int M> void test_finite_bits(float x, unsigned bits) {
  EXPECT_EQ((Minifloat<E, M>{x}.to_bits()), bits);
  EXPECT_EQ((Minifloat<E, M, NanStyle::FN>{x}.to_bits()), bits);
  EXPECT_EQ((Minifloat<E, M, NanStyle::FNUZ>{x}.to_bits()), bits);
}
} // namespace

TEST(SkymizerMinifloat, TestFiniteBits) {
  test_finite_bits<3, 4>(2.0F, 0x40);
  test_finite_bits<4, 3>(2.0F, 0x40);
  test_finite_bits<5, 2>(2.0F, 0x40);
  test_finite_bits<5, 7>(2.0F, 0b0'10000'0000000);

  test_finite_bits<3, 4>(1.0F, 0b0'011'0000);
  test_finite_bits<4, 3>(1.0F, 0b0'0111'000);
  test_finite_bits<5, 2>(1.0F, 0b0'01111'00);
  test_finite_bits<5, 7>(1.0F, 0b0'01111'0000000);

  test_finite_bits<3, 4>(-1.25F, 0b1'011'0100);
  test_finite_bits<4, 3>(-1.25F, 0b1'0111'010);
  test_finite_bits<5, 2>(-1.25F, 0b1'01111'01);
  test_finite_bits<5, 7>(-1.25F, 0b1'01111'0100000);
}

TEST(SkymizerMinifloat, TestSnowballSum) {
  test_snowball_sum<2, 11>();
  test_snowball_sum<3, 11>();
  test_snowball_sum<4, 11>();
  test_snowball_sum<2, 12>();
  test_snowball_sum<3, 12>();
}

TEST(SkymizerMinifloat, TestCopying) { test_selected_types<CheckCopying>(); }
TEST(SkymizerMinifloat, TestEquality) { test_selected_types<CheckEquality>(); }
TEST(SkymizerMinifloat, TestUnarySign) { test_selected_types<CheckUnarySign>(); }
TEST(SkymizerMinifloat, TestComparison) { test_selected_types<CheckComparison>(); }
TEST(SkymizerMinifloat, TestClassification) { test_selected_types<CheckClassification>(); }
TEST(SkymizerMinifloat, TestIdentityConversion) { test_selected_types<CheckIdentityConversion>(); }

TEST(SkymizerMinifloat, TestSubnormalConversion) {
  test_selected_types<CheckSubnormalConversion>();
}

TEST(SkymizerMinifloat, TestExactAddition) {
  test_selected_types<CheckExactArithmetics<std::plus<>>>();
}

TEST(SkymizerMinifloat, TestExactSubtraction) {
  test_selected_types<CheckExactArithmetics<std::minus<>>>();
}

TEST(SkymizerMinifloat, TestExactMultiplication) {
  test_selected_types<CheckExactArithmetics<std::multiplies<>>>();
}

TEST(SkymizerMinifloat, TestExactDivision) {
  test_selected_types<CheckExactArithmetics<std::divides<>>>();
}
