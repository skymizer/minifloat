// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "minifloat.hpp"
#include <cassert>
#include <gtest/gtest.h>
#include <string>
#include <unordered_set>

using namespace skymizer::minifloat; // NOLINT(google-build-using-namespace)

namespace {
// Compile-time check: the noexcept annotations on the public API propagate to
// the standard "is_nothrow_*" traits that STL containers use to pick faster
// move / value paths.
using NoexceptCheck = IEEE<3, 4>;
static_assert(std::is_nothrow_default_constructible_v<NoexceptCheck>);
static_assert(std::is_nothrow_constructible_v<NoexceptCheck, float>);
static_assert(std::is_nothrow_constructible_v<NoexceptCheck, double>);
static_assert(std::is_nothrow_copy_constructible_v<NoexceptCheck>);
static_assert(std::is_nothrow_move_constructible_v<NoexceptCheck>);
static_assert(std::is_nothrow_copy_assignable_v<NoexceptCheck>);
static_assert(std::is_nothrow_move_assignable_v<NoexceptCheck>);
static_assert(std::is_nothrow_destructible_v<NoexceptCheck>);

// Comparison operators are constexpr.
static_assert(NoexceptCheck::from_bits(0) == NoexceptCheck::from_bits(0));
static_assert(NoexceptCheck::from_bits(1) != NoexceptCheck::from_bits(2));
static_assert(NoexceptCheck::from_bits(1) < NoexceptCheck::from_bits(2));

//! Test floating-point identity like Object.is in JavaScript
//!
//! This is necessary because NaN != NaN in C++.  We also want to differentiate
//! -0 from +0.  Using this functor, NaNs are considered identical to each
//! other, while +0 and -0 are considered different.
bool same_double(double x, double y) {
  return bit_cast<std::uint64_t>(x) == bit_cast<std::uint64_t>(y) || (x != x && y != y);
}

//! Test floating-point identity like Object.is in JavaScript
//!
//! See also `same_double`.
template <class Format> bool same_mini(Minifloat<Format> x, Minifloat<Format> y) {
  return x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
}

//! Comparison result similar to `x <=> y` in C++20
//!
//! - +2 if `x > y`
//! - -2 if `x < y`
//! -  0 if `x == y` or not comparable
template <typename T> int compare(T x, T y) { return (x > y) - (x < y); }

//! Name a type in a failure message, since the checkers run over a type list
template <typename T> std::string describe() {
  return "E" + std::to_string(T::EXPONENT_BITS) + "M" + std::to_string(T::MANTISSA_BITS) +
         " B=" + std::to_string(T::BIAS) + (T::HAS_INF ? " inf" : "") + (T::HAS_NAN ? " nan" : "") +
         (T::HAS_NEG_ZERO ? " -0" : "");
}

//! Iterate over all possible values of a minifloat type `T`
template <typename T, typename Predicate> bool for_all(Predicate pred) {
  constexpr unsigned END = 1U << (T::EXPONENT_BITS + T::MANTISSA_BITS + 1);

  for (unsigned i = 0; i < END; ++i) {
    if (!pred(T::from_bits(static_cast<typename T::Storage>(i))))
      return false;
  }
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

//! Types for the linear checkers: every format layer, both biases, the full
//! width range, plus every public alias shape.
template <typename Checker> void test_selected_types() {
  check_each<
      Checker, //
      Finite<2, 1>, Finite<2, 3>, Finite<3, 2>, Finite<3, 4>, Finite<5, 2>, Finite<7, 0>,
      IEEE<2, 5>, IEEE<3, 4>, IEEE<4, 3>, IEEE<4, 3, 11>, IEEE<5, 2>, IEEE<5, 7>, IEEE<5, 10>,
      IEEE<8, 7>, //
      FN<2, 5>, FN<3, 4>, FN<4, 3>, FN<4, 3, 11>, FN<5, 2>, FN<5, 7>, FN<6, 1>, FN<7, 0>,
      FNUZ<2, 5>, FNUZ<3, 4>, FNUZ<4, 3>, FNUZ<4, 3, 11>, FNUZ<5, 2>, FNUZ<5, 7>, FNUZ<6, 1>,
      FNUZ<7, 0>>();
}

//! Types for the quadratic checkers, which visit every ordered pair of values
template <typename Checker> void test_paired_types() {
  check_each<
      Checker, //
      Finite<2, 1>, Finite<2, 3>, Finite<3, 2>, Finite<3, 4>, Finite<5, 2>, IEEE<2, 5>, IEEE<3, 4>,
      IEEE<4, 3>, IEEE<4, 3, 11>, IEEE<5, 2>, IEEE<5, 5>, FN<2, 5>, FN<3, 4>, FN<4, 3>,
      FN<4, 3, 11>, FN<5, 2>, FN<5, 5>, FN<6, 1>, FN<7, 0>, FNUZ<2, 5>, FNUZ<3, 4>, FNUZ<4, 3>,
      FNUZ<4, 3, 11>, FNUZ<5, 2>, FNUZ<5, 5>, FNUZ<6, 1>, FNUZ<7, 0>>();
}

struct CheckCopying {
  template <typename T> static bool check() {
    T a{2.0F};
    T b = a;
    T c;
    c = b;
    return c == a;
  }
};

struct CheckEquality {
  template <typename T> static bool check() {
    constexpr float FIXED_POINT = T::MANTISSA_BITS == 0 ? -2.0F : -3.0F;

    EXPECT_EQ(T{FIXED_POINT}.to_float(), FIXED_POINT);
    EXPECT_EQ(T{FIXED_POINT}.to_double(), FIXED_POINT);

    EXPECT_EQ(T{0.0F}, T{-0.0F});
    EXPECT_EQ(T{0.0F}.to_bits() == T{-0.0F}.to_bits(), !T::HAS_NEG_ZERO);

    if constexpr (T::HAS_NAN) {
      EXPECT_TRUE(T{NAN}.is_nan());
      EXPECT_TRUE((std::isnan)(T{NAN}.to_float()));
      EXPECT_TRUE((std::isnan)(T{NAN}.to_double()));
    }

    return for_all<T>([](T x) { return (x != x) == x.is_nan(); });
  }
};

struct CheckUnarySign {
  template <typename T> static bool check() {
    return T{0.0F} == -T{0.0F} &&
           for_all<T>([](T x) { return same_mini(x, +x) && same_mini(x, - -x); });
  }
};

struct CheckComparison {
  template <typename T> static bool check() {
    return for_all<T>([](T x) {
      return for_all<T>([x](T y) {
        return compare(x, y) == compare(x.to_float(), y.to_float()) &&
               compare(x, y) == compare(x.to_double(), y.to_double());
      });
    });
  }
};

struct CheckClassification {
  // Safe on all platforms, fast on glibc
  constexpr static int to_shift(int category) {
    switch (category) {
    case FP_NAN:
      return 0;
    case FP_INFINITE:
      return 1;
    case FP_ZERO:
      return 2;
    case FP_SUBNORMAL:
      return 3;
    case FP_NORMAL:
      return 4;
    default:
      assert(!"Invalid floating-point category");
      return category;
    }
  }

  template <typename T> static bool check() {
    return for_all<T>([](T x) {
      const int category = x.is_nan() << to_shift(FP_NAN) |             //
                           x.is_infinite() << to_shift(FP_INFINITE) |   //
                           !x << to_shift(FP_ZERO) |                    //
                           x.is_subnormal() << to_shift(FP_SUBNORMAL) | //
                           x.is_normal() << to_shift(FP_NORMAL);
      return category == 1 << to_shift(x.classify());
    });
  }
};

struct CheckIdentityConversion {
  template <typename T> static bool check() {
    EXPECT_EQ(bit_cast<std::uint32_t>(T{0.0F}.to_float()), 0U);
    EXPECT_EQ(bit_cast<std::uint64_t>(T{0.0F}.to_double()), 0U);
    EXPECT_EQ(bit_cast<std::uint32_t>(T{-0.0F}.to_float()), T::HAS_NEG_ZERO * 0x8000'0000);
    EXPECT_EQ(
        bit_cast<std::uint64_t>(T{-0.0F}.to_double()), T::HAS_NEG_ZERO * 0x8000'0000'0000'0000
    );

    return for_all<T>([](T x) {
      return same_mini(x, T::from_bits(x.to_bits())) && same_mini(x, T{x.to_float()}) &&
             same_double(x.to_float(), x.to_double());
    });
  }
};

struct CheckIntegerDecodeReconstruction {
  template <typename T> static bool check() {
    return for_all<T>([](T x) {
      const auto parts = integer_decode(x);

      if (x.is_nan())
        return parts.sign == 0 && parts.mantissa == 0u && parts.exponent == 0;

      const double integer = parts.sign * static_cast<std::int64_t>(parts.mantissa);
      const double y = std::ldexp(integer, +parts.exponent);

      if (x.is_infinite()) {
        const T reconstructed{y};
        return reconstructed.is_infinite() && reconstructed.signbit() == x.signbit();
      }

      return y == static_cast<double>(x.to_float());
    });
  }
};

template <typename Operation> struct CheckExactArithmetics {
  template <typename T> static bool check() {
    return for_all<T>([op = Operation{}](T x) {
      return for_all<T>([op, x](T y) {
        const double reference = op(x.to_double(), y.to_double());

        // Feeding a NaN to a format without one violates its precondition.
        if (!T::HAS_NAN && (std::isnan)(reference))
          return true;

        return same_mini(op(x, y), T{reference});
      });
    });
  }
};

//! How a bit pattern reads by the book, independently of the library
enum struct Style { Finite, IEEE, FN, FNUZ };

template <Style S, typename T> double oracle(unsigned bits) {
  constexpr int E = T::EXPONENT_BITS;
  constexpr int M = T::MANTISSA_BITS;
  constexpr int B = T::BIAS;

  const unsigned magnitude = bits & ((1U << (E + M)) - 1U);
  const double sign = bits >> (E + M) ? -1.0 : 1.0;
  const unsigned exponent = magnitude >> M;
  const unsigned mantissa = magnitude & ((1U << M) - 1U);

  if (S == Style::FNUZ && bits == 1U << (E + M))
    return NAN;

  if (S == Style::FN && magnitude == (1U << (E + M)) - 1U)
    return NAN;

  if (S == Style::IEEE && exponent == (1U << E) - 1U)
    return mantissa ? NAN : sign * HUGE_VAL;

  if (exponent == 0)
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

//! All four layers agree on a finite value they can all represent
template <int E, int M> void test_finite_bits(float x, unsigned bits) {
  EXPECT_EQ((Finite<E, M>{x}.to_bits()), bits);
  EXPECT_EQ((IEEE<E, M>{x}.to_bits()), bits);
  EXPECT_EQ((FN<E, M>{x}.to_bits()), bits);
  EXPECT_EQ((FNUZ<E, M, default_bias(E)>{x}.to_bits()), bits);
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

TEST(SkymizerMinifloat, TestOracle) {
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
}

TEST(SkymizerMinifloat, TestAliasRanges) {
  // OCP MX types: no NaN, no infinity, every pattern is a number.
  EXPECT_EQ(E2M1FN::max().to_double(), 6.0);
  EXPECT_EQ(E2M3FN::max().to_double(), 7.5);
  EXPECT_EQ(E3M2FN::max().to_double(), 28.0);
  static_assert(!std::numeric_limits<E2M1FN>::has_quiet_NaN);
  static_assert(!std::numeric_limits<E2M1FN>::has_infinity);
  static_assert(!std::numeric_limits<E2M3FN>::has_quiet_NaN);
  static_assert(!std::numeric_limits<E3M2FN>::has_infinity);

  // The FN suffix is LLVM's name for a format, not a property of the `FN`
  // template: `FN<2, 1>` reserves the all-ones magnitude, `E2M1FN` does not.
  EXPECT_EQ((FN<2, 1>::max().to_double()), 4.0);
  EXPECT_TRUE((std::numeric_limits<FN<2, 1>>::has_quiet_NaN));

  // FNUZ defaults to a bias one greater than IEEE's, as in LLVM.
  static_assert(E4M3FNUZ::BIAS == 8);
  static_assert(E5M2FNUZ::BIAS == 16);
  EXPECT_EQ(E4M3FNUZ::max().to_double(), 240.0);
  EXPECT_EQ(E5M2FNUZ::max().to_double(), 57344.0);
  EXPECT_EQ(E4M3B11FNUZ::max().to_double(), 30.0);

  EXPECT_EQ(E3M4::max().to_double(), 15.5);
  EXPECT_EQ(E4M3::max().to_double(), 240.0);
  EXPECT_EQ(E4M3FN::max().to_double(), 448.0);
  EXPECT_EQ(E5M2::max().to_double(), 57344.0);
  EXPECT_EQ(E5M10::max().to_double(), 65504.0);               // binary16
  EXPECT_EQ(E8M7::max().to_double(), std::ldexp(255.0, 120)); // bfloat16
}

TEST(SkymizerMinifloat, TestNumericLimits) {
  using T = E5M10; // half-precision shape (binary16)
  using L = std::numeric_limits<T>;

  static_assert(L::is_specialized);
  static_assert(L::is_signed);
  static_assert(!L::is_integer);
  static_assert(L::has_infinity);
  static_assert(L::has_quiet_NaN);
  static_assert(L::is_iec559);
  static_assert(L::radix == 2);
  static_assert(L::digits == T::MANTISSA_DIGITS);

  EXPECT_EQ(L::min(), T::min());
  EXPECT_EQ(L::max(), T::max());
  EXPECT_EQ(L::lowest(), -T::max());
  EXPECT_EQ(L::denorm_min(), T::true_min());

  // epsilon: 1.0 + epsilon must be representable and distinct from 1.0.
  EXPECT_EQ(L::epsilon().to_double(), std::ldexp(1.0, 1 - L::digits));
  EXPECT_NE(T{1.0F} + L::epsilon(), T{1.0F});

  EXPECT_EQ(L::round_error().to_double(), 0.5);
  EXPECT_TRUE(std::isinf(L::infinity().to_double()));
  EXPECT_TRUE(L::quiet_NaN().is_nan());

  // FN style has no infinity but still has NaN.
  static_assert(!std::numeric_limits<E4M3FN>::has_infinity);
  static_assert(!std::numeric_limits<E4M3FN>::is_iec559);
  EXPECT_TRUE(std::numeric_limits<E4M3FN>::quiet_NaN().is_nan());

  // A format with no mantissa bit has no subnormals to speak of.
  static_assert(std::numeric_limits<FN<7, 0>>::has_denorm == std::denorm_absent);
  static_assert(std::numeric_limits<E5M2>::has_denorm == std::denorm_present);
}

TEST(SkymizerMinifloat, TestIntegerInterop) {
  using T = E5M2;
  EXPECT_EQ(T{3}.to_float(), 3.0F);
  EXPECT_EQ(T{-7}.to_float(), -7.0F);
  EXPECT_EQ(T{0u}.to_float(), 0.0F);
  EXPECT_EQ(T{8L}.to_float(), 8.0F);

  // Conversion back to integer truncates toward zero.
  EXPECT_EQ(static_cast<int>(T{3.5F}), 3);
  EXPECT_EQ(static_cast<int>(T{-3.5F}), -3);
  EXPECT_EQ(static_cast<long>(T{0.0F}), 0L);

  // bool conversion still routes through operator bool() (true if nonzero).
  EXPECT_TRUE(static_cast<bool>(T{1.0F}));
  EXPECT_FALSE(static_cast<bool>(T{0.0F}));
}

TEST(SkymizerMinifloat, TestCompoundAssignment) {
  using T = E5M2; // covers a wider range than E3M4
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

TEST(SkymizerMinifloat, TestStdHash) {
  using T = E3M4;
  std::hash<T> h;
  // +0 and -0 compare equal, so their hashes must agree.
  EXPECT_EQ(h(T{0.0F}), h(T{-0.0F}));
  // Distinct nonzero values should hash to distinct slots in this small space.
  std::unordered_set<T> s{T{1.0F}, T{2.0F}, T{-1.0F}, T{0.0F}, T{-0.0F}};
  EXPECT_EQ(s.size(), 4u);
}

TEST(SkymizerMinifloat, TestDefaultConstructionIsZero) {
  // bits_ is value-initialized; a default-constructed Minifloat must read as
  // a positive zero rather than being indeterminate UB.
  using T = E3M4;
  T x;
  EXPECT_EQ(x.to_bits(), 0u);
  EXPECT_FALSE(static_cast<bool>(x));
  EXPECT_FALSE(x.signbit());
}

TEST(SkymizerMinifloat, TestWideExponentRange) {
  // A format whose exponent range reaches past double's on both ends: neither
  // conversion is exact, so both go through the ldexp path. Regression for a
  // 0.1.0 bug where the zero exponent field of a host zero or subnormal was
  // read as an ordinary exponent, and to_double() rebuilt garbage bits.
  using T = IEEE<12, 3>;
  static_assert(!T::HAS_EXACT_F32_CONVERSION);
  static_assert(!T::HAS_EXACT_F64_CONVERSION);

  EXPECT_EQ(T::from_bits(0).to_double(), 0.0);
  EXPECT_EQ(T{0.0F}.to_bits(), 0u);
  EXPECT_EQ(T{-0.0F}.to_bits(), 1u << 15);
  EXPECT_EQ(T{0.0}.to_bits(), 0u);
  EXPECT_EQ(T{1.0F}.to_bits(), unsigned{T::BIAS} << 3);
  EXPECT_EQ(T{1.0}.to_bits(), unsigned{T::BIAS} << 3);
  EXPECT_EQ(T::from_bits(unsigned{T::BIAS} << 3).to_double(), 1.0);
  EXPECT_EQ(T::from_bits((unsigned{T::BIAS} + 1) << 3).to_double(), 2.0);
  EXPECT_EQ(T::max().to_double(), HUGE_VAL);   // exponent beyond double
  EXPECT_EQ(T::from_bits(1).to_double(), 0.0); // magnitude beneath double
  EXPECT_TRUE(T::from_bits(0xFFFF).is_nan());

  // Subnormal host sources are normal here, and must not be read as if their
  // zero exponent field were an ordinary exponent.
  EXPECT_EQ(T{FLT_TRUE_MIN}.to_bits(), (unsigned{T::BIAS} - 149) << 3);
  EXPECT_EQ(T{FLT_TRUE_MIN}.to_double(), static_cast<double>(FLT_TRUE_MIN));
  EXPECT_EQ(T{0x1p-1070}.to_bits(), (unsigned{T::BIAS} - 1070) << 3);
  EXPECT_EQ(T{0x1p-1070}.to_double(), 0x1p-1070);

  // Reaches under float's normal range but not double's, so the float encoder
  // scales its source and the double encoder does not.
  using U = IEEE<9, 3>;
  static_assert(!U::HAS_EXACT_F32_CONVERSION);
  static_assert(U::HAS_EXACT_F64_CONVERSION);
  EXPECT_EQ(U{0.0F}.to_bits(), 0u);
  EXPECT_EQ(U{-0.0F}.to_bits(), 1u << 12);
  EXPECT_EQ(U{1.0F}.to_float(), 1.0F);
  EXPECT_EQ(U{FLT_MIN}.to_bits(), (unsigned{U::BIAS} - 126) << 3);
  EXPECT_EQ(U{FLT_TRUE_MIN}.to_bits(), (unsigned{U::BIAS} - 149) << 3);
  EXPECT_EQ(U{FLT_TRUE_MIN}.to_double(), static_cast<double>(FLT_TRUE_MIN));
  EXPECT_EQ(U{static_cast<double>(FLT_TRUE_MIN)}.to_bits(), (unsigned{U::BIAS} - 149) << 3);
}

TEST(SkymizerMinifloat, TestSnowballSum) {
  EXPECT_TRUE((test_snowball_sum<2, 11>()));
  EXPECT_TRUE((test_snowball_sum<3, 11>()));
  EXPECT_TRUE((test_snowball_sum<4, 11>()));
  EXPECT_TRUE((test_snowball_sum<2, 12>()));
  EXPECT_TRUE((test_snowball_sum<3, 12>()));
}

TEST(SkymizerMinifloat, TestCopying) { test_selected_types<CheckCopying>(); }
TEST(SkymizerMinifloat, TestEquality) { test_selected_types<CheckEquality>(); }
TEST(SkymizerMinifloat, TestUnarySign) { test_selected_types<CheckUnarySign>(); }
TEST(SkymizerMinifloat, TestClassification) { test_selected_types<CheckClassification>(); }
TEST(SkymizerMinifloat, TestIdentityConversion) { test_selected_types<CheckIdentityConversion>(); }

TEST(SkymizerMinifloat, TestIntegerDecodeReconstruction) {
  test_selected_types<CheckIntegerDecodeReconstruction>();
}

TEST(SkymizerMinifloat, TestComparison) { test_paired_types<CheckComparison>(); }

TEST(SkymizerMinifloat, TestExactAddition) {
  test_paired_types<CheckExactArithmetics<std::plus<>>>();
}

TEST(SkymizerMinifloat, TestExactSubtraction) {
  test_paired_types<CheckExactArithmetics<std::minus<>>>();
}

TEST(SkymizerMinifloat, TestExactMultiplication) {
  test_paired_types<CheckExactArithmetics<std::multiplies<>>>();
}

TEST(SkymizerMinifloat, TestExactDivision) {
  test_paired_types<CheckExactArithmetics<std::divides<>>>();
}
