// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "support.hpp"

#include <type_traits>

using namespace minifloat_test;      // NOLINT(google-build-using-namespace)
using namespace skymizer::minifloat; // NOLINT(google-build-using-namespace)

namespace {
using NoexceptCheck = IEEE<3, 4>;
static_assert(std::is_nothrow_default_constructible_v<NoexceptCheck>);
static_assert(std::is_nothrow_constructible_v<NoexceptCheck, float>);
static_assert(std::is_nothrow_constructible_v<NoexceptCheck, double>);
static_assert(std::is_nothrow_copy_constructible_v<NoexceptCheck>);
static_assert(std::is_nothrow_move_constructible_v<NoexceptCheck>);
static_assert(std::is_nothrow_copy_assignable_v<NoexceptCheck>);
static_assert(std::is_nothrow_move_assignable_v<NoexceptCheck>);
static_assert(std::is_nothrow_destructible_v<NoexceptCheck>);

static_assert(NoexceptCheck::from_bits(0) == NoexceptCheck::from_bits(0));
static_assert(NoexceptCheck::from_bits(1) != NoexceptCheck::from_bits(2));
static_assert(NoexceptCheck::from_bits(1) < NoexceptCheck::from_bits(2));
static_assert(NoexceptCheck::from_bits(1) <= NoexceptCheck::from_bits(2));
static_assert(NoexceptCheck::from_bits(2) > NoexceptCheck::from_bits(1));
static_assert(NoexceptCheck::from_bits(2) >= NoexceptCheck::from_bits(1));
// A NaN loses every comparison, at compile time as much as at run time
static_assert(!(NoexceptCheck::quiet_NaN() == NoexceptCheck::quiet_NaN()));

enum class Style { Finite, IEEE, FN, FNUZ };

//! Textbook value of a bit pattern, independent of the library
template <Style S, typename T> double oracle(unsigned bits) {
  constexpr int E = T::EXPONENT_BITS;
  constexpr int M = T::MANTISSA_BITS;
  constexpr int B = T::BIAS;

  const unsigned magnitude = bits & ((1U << (E + M)) - 1U);
  const double sign = bits >> (E + M) ? -1.0 : 1.0;
  const unsigned exponent = magnitude >> M;
  const unsigned mantissa = magnitude & ((1U << M) - 1U);

  if constexpr (S == Style::FNUZ)
    if (bits == 1U << (E + M))
      return NAN;
  if constexpr (S == Style::FN)
    if (magnitude == (1U << (E + M)) - 1U)
      return NAN;
  if constexpr (S == Style::IEEE)
    if (exponent == (1U << E) - 1U)
      return mantissa ? NAN : sign * HUGE_VAL;

  if (!exponent)
    return sign * std::ldexp(mantissa, 1 - B - M);
  return sign * std::ldexp(mantissa + (1U << M), static_cast<int>(exponent) - B - M);
}

template <Style S, typename T> void check_oracle() {
  constexpr unsigned END = 1U << (T::EXPONENT_BITS + T::MANTISSA_BITS + 1);

  for (unsigned bits = 0; bits < END; ++bits) {
    const T x = T::from_bits(static_cast<typename T::Storage>(bits));
    ASSERT_TRUE(same_double(x.to_double(), oracle<S, T>(bits)))
        << describe<T>() << " bits=" << bits << " got=" << x.to_double()
        << " want=" << oracle<S, T>(bits);
  }
}

//! All four layers agree on a finite value they can represent
template <int E, int M> void test_finite_bits(float x, unsigned bits) {
  EXPECT_EQ((Finite<E, M>{x}.to_bits()), bits);
  EXPECT_EQ((IEEE<E, M>{x}.to_bits()), bits);
  EXPECT_EQ((FN<E, M>{x}.to_bits()), bits);
  EXPECT_EQ((FNUZ<E, M, default_bias(E)>{x}.to_bits()), bits);
}
} // namespace

TEST(Spec, FiniteBits) {
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

TEST(Spec, EveryBitPatternMatchesFormula) {
  check_oracle<Style::Finite, E2M1FN>();
  check_oracle<Style::Finite, E2M3FN>();
  check_oracle<Style::Finite, E3M2FN>();
  check_oracle<Style::Finite, Finite<5, 2>>();
  check_oracle<Style::Finite, Finite<7, 0>>();
  check_oracle<Style::IEEE, E3M4>();
  check_oracle<Style::IEEE, E4M3>();
  check_oracle<Style::IEEE, E5M2>();
  check_oracle<Style::IEEE, E5M10>();
  check_oracle<Style::IEEE, E8M7>();
  check_oracle<Style::IEEE, IEEE<4, 3, 11>>();
  check_oracle<Style::IEEE, IEEE<9, 3>>();
  check_oracle<Style::FN, E4M3FN>();
  check_oracle<Style::FN, FN<2, 1>>();
  check_oracle<Style::FN, FN<7, 0>>();
  check_oracle<Style::FN, FN<5, 7>>();
  check_oracle<Style::FNUZ, E4M3FNUZ>();
  check_oracle<Style::FNUZ, E4M3B11FNUZ>();
  check_oracle<Style::FNUZ, E5M2FNUZ>();
  check_oracle<Style::FNUZ, FNUZ<7, 0>>();
  check_oracle<Style::FNUZ, FNUZ<5, 7>>();

  check_oracle<Style::IEEE, IEEE<11, 4>>();
  check_oracle<Style::IEEE, IEEE<12, 3>>();
  check_oracle<Style::IEEE, IEEE<12, 3, 1000>>();
  check_oracle<Style::FN, FN<12, 3>>();
  check_oracle<Style::FNUZ, FNUZ<12, 3>>();
  check_oracle<Style::Finite, Finite<12, 3>>();
  check_oracle<Style::IEEE, IEEE<2, 13>>();
}

TEST(Spec, AliasRanges) {
  EXPECT_EQ(E2M1FN::max().to_double(), 6.0);
  EXPECT_EQ(E2M3FN::max().to_double(), 7.5);
  EXPECT_EQ(E3M2FN::max().to_double(), 28.0);
  static_assert(!std::numeric_limits<E2M1FN>::has_quiet_NaN);
  static_assert(!std::numeric_limits<E2M1FN>::has_infinity);
  static_assert(!std::numeric_limits<E2M3FN>::has_quiet_NaN);
  static_assert(!std::numeric_limits<E3M2FN>::has_infinity);

  EXPECT_EQ((FN<2, 1>::max().to_double()), 4.0);
  static_assert(std::numeric_limits<FN<2, 1>>::has_quiet_NaN);

  static_assert(E4M3FNUZ::BIAS == 8);
  static_assert(E5M2FNUZ::BIAS == 16);
  EXPECT_EQ(E4M3FNUZ::max().to_double(), 240.0);
  EXPECT_EQ(E5M2FNUZ::max().to_double(), 57344.0);
  EXPECT_EQ(E4M3B11FNUZ::max().to_double(), 30.0);

  EXPECT_EQ(E3M4::max().to_double(), 15.5);
  EXPECT_EQ(E4M3::max().to_double(), 240.0);
  EXPECT_EQ(E4M3FN::max().to_double(), 448.0);
  EXPECT_EQ(E5M2::max().to_double(), 57344.0);
  EXPECT_EQ(E5M10::max().to_double(), 65504.0);
  EXPECT_EQ(E8M7::max().to_double(), std::ldexp(255.0, 120));
}

TEST(Spec, NumericLimits) {
  using T = E5M10;
  using L = std::numeric_limits<T>;

  static_assert(L::is_specialized);
  static_assert(L::is_signed);
  static_assert(!L::is_integer);
  static_assert(!L::is_exact);
  static_assert(L::has_infinity);
  static_assert(L::has_quiet_NaN);
  static_assert(!L::has_signaling_NaN);
  static_assert(!L::has_denorm_loss);
  static_assert(L::round_style == std::round_to_nearest);
  static_assert(L::is_iec559);
  static_assert(L::is_bounded);
  static_assert(!L::is_modulo);
  static_assert(L::radix == 2);
  static_assert(L::digits == T::MANTISSA_DIGITS);
  static_assert(!L::traps);
  static_assert(!L::tinyness_before);

  EXPECT_EQ(L::min(), T::min());
  EXPECT_EQ(L::max(), T::max());
  EXPECT_EQ(L::lowest(), -T::max());
  EXPECT_EQ(L::denorm_min(), T::true_min());
  EXPECT_EQ(L::epsilon().to_double(), std::ldexp(1.0, 1 - L::digits));
  EXPECT_NE(T{1.0F} + L::epsilon(), T{1.0F});
  EXPECT_EQ(L::round_error().to_double(), 0.5);
  EXPECT_TRUE(std::isinf(L::infinity().to_double()));
  EXPECT_TRUE(L::quiet_NaN().is_nan());
  EXPECT_TRUE(L::signaling_NaN().is_nan());

  static_assert(!std::numeric_limits<E4M3FN>::has_infinity);
  static_assert(!std::numeric_limits<E4M3FN>::is_iec559);
  EXPECT_TRUE(std::numeric_limits<E4M3FN>::quiet_NaN().is_nan());
  static_assert(std::numeric_limits<FN<7, 0>>::has_denorm == std::denorm_absent);
  static_assert(std::numeric_limits<E5M2>::has_denorm == std::denorm_present);

  static_assert(std::numeric_limits<E5M10>::max_exponent10 == 4);
  static_assert(std::numeric_limits<E8M7>::max_exponent10 == 38);
  static_assert(std::numeric_limits<E4M3FN>::max_exponent10 == 2);
  static_assert(std::numeric_limits<E2M1FN>::max_exponent10 == 0);
  static_assert(std::numeric_limits<E3M2FN>::max_exponent10 == 1);

  // Wherever the format spends the all-ones magnitude, its maximum finite
  // value is short of the top of the binade `MAX_EXP` names, and a decimal
  // exponent read off `MAX_EXP` alone comes out one too high.
  static_assert(std::numeric_limits<FN<5, 2>>::max_exponent10 == 4);
  static_assert(std::numeric_limits<FN<7, 0>>::max_exponent10 == 18);
  static_assert(std::numeric_limits<FNUZ<7, 0>>::max_exponent10 == 18);
  static_assert(std::numeric_limits<Finite<7, 0>>::max_exponent10 == 19);
  static_assert(std::numeric_limits<FN<4, 0, 8>>::max_exponent10 == 1);
  EXPECT_EQ((FN<4, 0, 8>::max().to_double()), 64.0);
}

TEST(Spec, DefaultConstructionIsZero) {
  E3M4 x;
  EXPECT_EQ(x.to_bits(), 0u);
  EXPECT_FALSE(static_cast<bool>(x));
  EXPECT_FALSE(x.signbit());
}
