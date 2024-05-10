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

template <typename T, unsigned STEP = 1, typename F>
static void iterate(F f) {
  const unsigned END = 1U << (Trait<T>::E + Trait<T>::M + 1);

  for (unsigned i = 0; i < END; i += STEP)
    f(T::from_bits(i));
}

template <typename T>
static T id(T x) { return x; }

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

#define RUN_ON_SELECTED_TYPES(F) do { \
  F<E3M4>(); \
  F<E3M4FN>(); \
  F<E3M4FNUZ>(); \
  F<E4M3>(); \
  F<E4M3FN>(); \
  F<E4M3FNUZ>(); \
  F<E4M3B11>(); \
  F<E4M3B11FN>(); \
  F<E4M3B11FNUZ>(); \
  F<E5M2>(); \
  F<E5M2FN>(); \
  F<E5M2FNUZ>(); \
  F<E5M7>(); \
  F<E5M7FN>(); \
  F<E5M7FNUZ>(); \
} while (false)

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
  EXPECT_EQ(id<float>(T{-3.0f}), -3.0f);
  EXPECT_EQ(id<double>(T{-3.0}), -3.0);
  EXPECT_EQ(T{0.0f}, T{-0.0f});
  EXPECT_EQ(T{0.0f}.bits() == T{-0.0f}.bits(), Trait<T>::N == NanStyle::FNUZ);
  EXPECT_TRUE(T{NAN}.isnan());
  EXPECT_TRUE((std::isnan)(id<float>(T{NAN})));
  EXPECT_TRUE((std::isnan)(id<double>(T{NAN})));
  iterate<T>([](T x){ EXPECT_EQ(x != x, x.isnan()); });
}

TEST(SanityCheck, equality) {
  RUN_ON_SELECTED_TYPES(test_equality);
}

template <typename T>
static int compare(T x, T y) {
  return (x > y) - (x < y);
}

template <typename T>
static void test_comparison() {
  const unsigned X_STEP = 38 << (Trait<T>::E + Trait<T>::M) >> 15 | 1;
  const unsigned Y_STEP = 49 << (Trait<T>::E + Trait<T>::M) >> 15 | 1;

  iterate<T, X_STEP>([](T x){
    iterate<T, Y_STEP>([x](T y){
      EXPECT_EQ(compare(x, y), compare(id<float>(x), id<float>(y)));
      EXPECT_EQ(compare(x, y), compare(id<double>(x), id<double>(y)));
    });
  });
}

TEST(SanityCheck, comparison) {
  RUN_ON_SELECTED_TYPES(test_comparison);
}

template <typename T>
static void test_unary_sign() {
  EXPECT_EQ(T{0.0f}, -T{0.0f});

  iterate<T>([](T x){
    EXPECT_PRED2(are_identical, x, +x);
    EXPECT_PRED2(are_identical, x, - -x);
  });
}

TEST(SanityCheck, unary_sign) {
  RUN_ON_SELECTED_TYPES(test_unary_sign);
}

template <typename T>
static void test_identity_conversion() {
  using detail::bit_cast;
  const bool IS_FNUZ = Trait<T>::N == NanStyle::FNUZ;

  EXPECT_EQ(bit_cast<std::uint32_t>(id<float>(T{0.0f})), 0);
  EXPECT_EQ(bit_cast<std::uint64_t>(id<double>(T{0.0f})), 0);
  EXPECT_EQ(bit_cast<std::uint32_t>(id<float>(T{-0.0f})), !IS_FNUZ * 0x8000'0000);
  EXPECT_EQ(bit_cast<std::uint64_t>(id<double>(T{-0.0f})), !IS_FNUZ * 0x8000'0000'0000'0000);

  iterate<T>([](T x){
    EXPECT_PRED2(are_identical, x, T::from_bits(x.bits()));
    EXPECT_PRED2(are_identical, x, T{id<float>(x)});
    EXPECT_PRED2(are_identical, id<float>(x), id<double>(x));
  });
}

TEST(ConversionCheck, identity) {
  RUN_ON_SELECTED_TYPES(test_identity_conversion);
}

template <SubnormalStyle D, int E, int M, NanStyle N, int B>
static void test_subnormal_conversion(Minifloat<E, M, N, B, SubnormalStyle::Precise> x) {
  using T = Minifloat<E, M, N, B, D>;
  using Bits = typename T::StorageType;

  const T y(x);
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

TEST(ConversionCheck, subnormal) {
  RUN_ON_SELECTED_TYPES(test_subnormal_conversion);
}

template <typename T, typename F>
static void test_exact_arithmetics(F op) {
  const unsigned X_STEP = 38 << (Trait<T>::E + Trait<T>::M) >> 15 | 1;
  const unsigned Y_STEP = 49 << (Trait<T>::E + Trait<T>::M) >> 15 | 1;

  iterate<T, X_STEP>([op](T x){
    iterate<T, Y_STEP>([op, x](T y){
      const auto z = op(x, y);
      const double precise = op(id<double>(x), id<double>(y));
      EXPECT_PRED2(are_identical, static_cast<T>(z), static_cast<T>(precise));
      EXPECT_PRED2(are_identical, z, static_cast<decltype(z)>(precise));
    });
  });
}

template <typename T>
static void test_exact_arithmetics() {
  test_exact_arithmetics<T>(std::plus<>());
  test_exact_arithmetics<T>(std::minus<>());
  test_exact_arithmetics<T>(std::multiplies<>());
  test_exact_arithmetics<T>(std::divides<>());
}

TEST(ArithmeticCheck, exact) {
  RUN_ON_SELECTED_TYPES(test_exact_arithmetics);
}