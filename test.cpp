// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "minifloat.hpp"
#include <gtest/gtest.h>

using namespace skymizer::minifloat;

namespace {
template <typename> struct Trait;

template <int E_, int M_, NanStyle N_, int B_, SubnormalStyle D_>
struct Trait<Minifloat<E_, M_, N_, B_, D_>> {
  static const int E = E_;
  static const int M = M_;
  static const NanStyle N = N_;
  static const int B = B_;
  static const SubnormalStyle D = D_;
};
} // namespace

template <typename T, typename F>
static void iterate(F f) {
  const unsigned END = 1U << (Trait<T>::E + Trait<T>::M + 1);

  for (unsigned i = 0; i < END; ++i)
    f(T::from_bits(i));
}

template <typename T>
static T identity(T x) { return x; }

/** \brief Test floating-point identity like Object.is in JavaScript
  *
  * This is necessary because NaN != NaN in C++.  We also want to differentiate
  * -0 from +0.  Using this functor, NaNs are considered identical to each
  * other, while +0 and -0 are considered different. 
  */
static const struct {
  bool operator()(double x, double y) const {
    using detail::bit_cast;
    return bit_cast<std::uint64_t>(x) == bit_cast<std::uint64_t>(y) || (x != x && y != y);
  }

  bool operator()(float x, float y) const {
    using detail::bit_cast;
    return bit_cast<std::uint32_t>(x) == bit_cast<std::uint32_t>(y) || (x != x && y != y);
  }

  template <int E, int M, NanStyle N, int B, SubnormalStyle D>
  bool operator()(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) const {
    return x.bits() == y.bits() || (x.isnan() && y.isnan());
  }
} are_identical;

using E4M3B11 = Minifloat<4, 3, NanStyle::IEEE, 11>;
using E4M3B11FN = Minifloat<4, 3, NanStyle::FN, 11>;
using E4M3B11FNUZ = Minifloat<4, 3, NanStyle::FNUZ, 11>;

#define MAKE_TESTS_FOR_SELECTED_TYPES(Suite, CALLBACK) \
TEST(Suite, e3m4) { CALLBACK<E3M4>(); } \
TEST(Suite, e3m4fn) { CALLBACK<E3M4FN>(); } \
TEST(Suite, e3m4fnuz) { CALLBACK<E3M4FNUZ>(); } \
TEST(Suite, e4m3) { CALLBACK<E4M3>(); } \
TEST(Suite, e4m3fn) { CALLBACK<E4M3FN>(); } \
TEST(Suite, e4m3fnuz) { CALLBACK<E4M3FNUZ>(); } \
TEST(Suite, e4m3b11) { CALLBACK<E4M3B11>(); } \
TEST(Suite, e4m3b11fn) { CALLBACK<E4M3B11FN>(); } \
TEST(Suite, e4m3b11fnuz) { CALLBACK<E4M3B11FNUZ>(); } \
TEST(Suite, e5m2) { CALLBACK<E5M2>(); } \
TEST(Suite, e5m2fn) { CALLBACK<E5M2FN>(); } \
TEST(Suite, e5m2fnuz) { CALLBACK<E5M2FNUZ>(); } \
TEST(Suite, e4m4fn) { CALLBACK<E4M4FN>(); } \
TEST(Suite, e4m5fn) { CALLBACK<E4M5FN>(); } \
TEST(Suite, e5m4fn) { CALLBACK<E5M4FN>(); }

template <int E, int M>
static void test_finite_bits(float x, unsigned bits) {
  EXPECT_EQ((Minifloat<E, M>{x}.bits()), bits);
  EXPECT_EQ((Minifloat<E, M, NanStyle::FN>{x}.bits()), bits);
  EXPECT_EQ((Minifloat<E, M, NanStyle::FNUZ>{x}.bits()), bits);
}

TEST(SanityCheck, finite_bits) {
  test_finite_bits<3, 4>(2.0f, 0x40);
  test_finite_bits<4, 3>(2.0f, 0x40);
  test_finite_bits<5, 2>(2.0f, 0x40);
  test_finite_bits<5, 7>(2.0f, 0b0'10000'0000000);

  test_finite_bits<3, 4>(1.0f, 0b0'011'0000);
  test_finite_bits<4, 3>(1.0f, 0b0'0111'000);
  test_finite_bits<5, 2>(1.0f, 0b0'01111'00);
  test_finite_bits<5, 7>(1.0f, 0b0'01111'0000000);

  test_finite_bits<3, 4>(-1.25f, 0b1'011'0100);
  test_finite_bits<4, 3>(-1.25f, 0b1'0111'010);
  test_finite_bits<5, 2>(-1.25f, 0b1'01111'01);
  test_finite_bits<5, 7>(-1.25f, 0b1'01111'0100000);
}

template <typename T>
static void test_equality() {
  EXPECT_EQ(identity<float>(T{-3.0f}), -3.0f);
  EXPECT_EQ(identity<double>(T{-3.0}), -3.0);
  EXPECT_EQ(T{0.0f}, T{-0.0f});
  EXPECT_EQ(T{0.0f}.bits() == T{-0.0f}.bits(), Trait<T>::N == NanStyle::FNUZ);
  EXPECT_TRUE(T{NAN}.isnan());
  EXPECT_TRUE((std::isnan)(identity<float>(T{NAN})));
  EXPECT_TRUE((std::isnan)(identity<double>(T{NAN})));
  iterate<T>([](T x){ EXPECT_EQ(x != x, x.isnan()); });
}

MAKE_TESTS_FOR_SELECTED_TYPES(EqualityCheck, test_equality)

template <typename T>
static int compare(T x, T y) {
  return (x > y) - (x < y);
}

template <typename T>
static void test_unary_sign() {
  EXPECT_EQ(T{0.0f}, -T{0.0f});

  iterate<T>([](T x){
    EXPECT_PRED2(are_identical, x, +x);
    EXPECT_PRED2(are_identical, x, - -x);
  });
}

MAKE_TESTS_FOR_SELECTED_TYPES(UnarySignCheck, test_unary_sign)

template <typename T>
static void test_comparison() {
  iterate<T>([](T x){
    iterate<T>([x](T y){
      EXPECT_EQ(compare(x, y), compare(identity<float>(x), identity<float>(y)));
      EXPECT_EQ(compare(x, y), compare(identity<double>(x), identity<double>(y)));
    });
  });
}

MAKE_TESTS_FOR_SELECTED_TYPES(ComparisonCheck, test_comparison)

template <typename T>
static void test_identity_conversion() {
  using detail::bit_cast;
  const bool IS_FNUZ = Trait<T>::N == NanStyle::FNUZ;

  EXPECT_EQ(bit_cast<std::uint32_t>(identity<float>(T{0.0f})), 0);
  EXPECT_EQ(bit_cast<std::uint64_t>(identity<double>(T{0.0f})), 0);
  EXPECT_EQ(bit_cast<std::uint32_t>(identity<float>(T{-0.0f})), !IS_FNUZ * 0x8000'0000);
  EXPECT_EQ(bit_cast<std::uint64_t>(identity<double>(T{-0.0f})), !IS_FNUZ * 0x8000'0000'0000'0000);

  iterate<T>([](T x){
    EXPECT_PRED2(are_identical, x, T::from_bits(x.bits()));
    EXPECT_PRED2(are_identical, x, T{identity<float>(x)});
    EXPECT_PRED2(are_identical, identity<float>(x), identity<double>(x));
  });
}

MAKE_TESTS_FOR_SELECTED_TYPES(IdentityConversionCheck, test_identity_conversion)

template <SubnormalStyle D, int E, int M, NanStyle N, int B>
static void test_subnormal_conversion(Minifloat<E, M, N, B, SubnormalStyle::Precise> x) {
  using T = Minifloat<E, M, N, B, D>;
  using Bits = typename T::StorageType;

  const T y(identity<float>(x));
  EXPECT_TRUE(x.signbit() == y.signbit() || (N == NanStyle::FNUZ && !y.bits()));

  const Bits THRESHOLD = 1U << M;
  const Bits magnitude = x.abs().bits();

  if (magnitude == 0 || magnitude >= THRESHOLD) {
    EXPECT_TRUE(x.bits() == y.bits() || (x.isnan() && y.isnan()));
    return;
  }

  if constexpr (D == SubnormalStyle::Reserved) {
    const Bits magnitude = y.abs().bits();
    EXPECT_TRUE(magnitude == 0 || magnitude == 1U << M);
    return;
  }
  EXPECT_LE(T::from_bits(0), y.abs());
  EXPECT_LE(y.abs(), T::from_bits(1U << M));
}

template <typename T>
static void test_subnormal_conversion() {
  iterate<T>([](T x){
    test_subnormal_conversion<SubnormalStyle::Reserved>(x);
    test_subnormal_conversion<SubnormalStyle::Fast>(x);
  });
}

MAKE_TESTS_FOR_SELECTED_TYPES(SubnormalConversionCheck, test_subnormal_conversion)

template <typename T, typename F>
static void test_exact_arithmetics(F op) {
  iterate<T>([op](T x){
    iterate<T>([op, x](T y){
      const auto z = op(x, y);
      const double precise = op(identity<double>(x), identity<double>(y));
      EXPECT_PRED2(are_identical, static_cast<T>(z), static_cast<T>(precise));
      EXPECT_PRED2(are_identical, z, static_cast<decltype(z)>(precise));
    });
  });
}

template <typename T>
static void test_exact_addition() {
  test_exact_arithmetics<T>(std::plus<>());
}

template <typename T>
static void test_exact_subtraction() {
  test_exact_arithmetics<T>(std::minus<>());
}

template <typename T>
static void test_exact_multiplication() {
  test_exact_arithmetics<T>(std::multiplies<>());
}

template <typename T>
static void test_exact_division() {
  test_exact_arithmetics<T>(std::divides<>());
}

MAKE_TESTS_FOR_SELECTED_TYPES(AdditionCheck, test_exact_addition)
MAKE_TESTS_FOR_SELECTED_TYPES(SubtractionCheck, test_exact_subtraction)
MAKE_TESTS_FOR_SELECTED_TYPES(MultiplicationCheck, test_exact_multiplication)
MAKE_TESTS_FOR_SELECTED_TYPES(DivisionCheck, test_exact_division)

template <int E, int M, NanStyle N = NanStyle::FN>
static void test_snowball_addition() {
  using T = Minifloat<E, M, N>;
  using Bits = typename T::StorageType;

  const Bits STEP = 1U << M;
  const Bits SIGNIFICAND = STEP - 1U;
  const Bits EXPONENT = 1U << (E + M - 1);
  const Bits GREATER = EXPONENT | SIGNIFICAND;

  for (Bits lesser = SIGNIFICAND; lesser <= GREATER; lesser += STEP) {
    const T x = T::from_bits(GREATER);
    const T y = T::from_bits(lesser);

    const double xx = x;
    const double yy = y;
    EXPECT_PRED2(are_identical, static_cast<T>(x + y), static_cast<T>(xx + yy));
  }
}

TEST(SnowballAdditionCheck, e2m11) { test_snowball_addition<2, 11>(); }
TEST(SnowballAdditionCheck, e3m11) { test_snowball_addition<3, 11>(); }
TEST(SnowballAdditionCheck, e4m11) { test_snowball_addition<4, 11>(); }
TEST(SnowballAdditionCheck, e2m12) { test_snowball_addition<2, 12>(); }
TEST(SnowballAdditionCheck, e3m12) { test_snowball_addition<3, 12>(); }