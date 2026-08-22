// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "support.hpp"

using namespace minifloat_test;      // NOLINT(google-build-using-namespace)
using namespace skymizer::minifloat; // NOLINT(google-build-using-namespace)

namespace {
struct CheckToFloat {
  template <typename T> static bool check() {
    EXPECT_TRUE(same_float(T{0.0F}.to_float(), 0.0F));
    EXPECT_TRUE(same_float(T{-0.0F}.to_float(), T::HAS_NEG_ZERO ? -0.0F : 0.0F));

    return !T::HAS_EXACT_F32_CONVERSION ||
           for_all<T>([](T x) { return same_mini(T{x.to_float()}, x); });
  }
};

struct CheckToDouble {
  template <typename T> static bool check() {
    EXPECT_TRUE(same_double(T{0.0}.to_double(), 0.0));
    EXPECT_TRUE(same_double(T{-0.0}.to_double(), T::HAS_NEG_ZERO ? -0.0 : 0.0));

    return !T::HAS_EXACT_F64_CONVERSION ||
           for_all<T>([](T x) { return same_mini(T{x.to_double()}, x); });
  }
};

struct CheckFloatAgreement {
  template <typename T> static bool check() {
    return !T::HAS_EXACT_F32_CONVERSION || for_all<T>([](T x) {
      return same_double(static_cast<double>(x.to_float()), x.to_double());
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
      return same_mini(x, T::from_bits(x.to_bits())) &&
             (!T::HAS_EXACT_F32_CONVERSION || same_mini(x, T{x.to_float()})) &&
             (!T::HAS_EXACT_F32_CONVERSION || same_double(x.to_float(), x.to_double()));
    });
  }
};

struct CheckIntegerDecodeReconstruction {
  template <typename T> static bool check() {
    return for_all<T>([](T x) {
      const auto parts = integer_decode(x);
      if (x.is_nan())
        return parts.sign == 0 && parts.mantissa == 0u && parts.exponent == 0;
      if (parts.sign != 1 && parts.sign != -1)
        return false;

      const double integer = static_cast<double>(parts.sign) * parts.mantissa;
      const double reconstructed = std::ldexp(integer, +parts.exponent);
      if (x.is_infinite()) {
        const T y{reconstructed};
        return y.is_infinite() && y.signbit() == x.signbit();
      }
      return same_double(reconstructed, x.to_double());
    });
  }
};

struct CheckExplicitCasts {
  template <typename T> static bool check() {
    return for_all<T>([](T x) {
      return same_float(static_cast<float>(x), x.to_float()) &&
             same_double(static_cast<double>(x), x.to_double());
    });
  }
};
} // namespace

TEST(Convert, ToFloat) { test_all_types<CheckToFloat>(); }

TEST(Convert, ToDouble) { test_all_types<CheckToDouble>(); }

TEST(Convert, FloatAndDoubleAgree) { test_all_types<CheckFloatAgreement>(); }

TEST(Convert, IdentityRoundTrips) { test_all_types<CheckIdentityConversion>(); }

TEST(Convert, ExactnessConstants) {
  static_assert(E5M10::HAS_EXACT_F32_CONVERSION);
  static_assert(E5M10::HAS_EXACT_F64_CONVERSION);
  static_assert(E8M7::HAS_EXACT_F32_CONVERSION);
  static_assert(E8M7::HAS_EXACT_F64_CONVERSION);
  static_assert(E4M3FN::HAS_EXACT_F32_CONVERSION);
  static_assert(E4M3FN::HAS_EXACT_F64_CONVERSION);

  static_assert(!IEEE<11, 4>::HAS_EXACT_F32_CONVERSION);
  static_assert(IEEE<11, 4>::HAS_EXACT_F64_CONVERSION);
  static_assert(!IEEE<12, 3>::HAS_EXACT_F32_CONVERSION);
  static_assert(!IEEE<12, 3>::HAS_EXACT_F64_CONVERSION);
  static_assert(IEEE<2, 13>::HAS_EXACT_F32_CONVERSION);
  static_assert(IEEE<2, 13>::HAS_EXACT_F64_CONVERSION);
}

TEST(Convert, ExplicitCastsMatchNamedConversions) { test_all_types<CheckExplicitCasts>(); }

TEST(Convert, IntegerDecodeReconstruction) { test_all_types<CheckIntegerDecodeReconstruction>(); }

//! `BF<32>` matches every non-NaN bit; NaN payloads canonicalize but class and
//! sign remain
TEST(Convert, BF32MatchesFloatBitsAndNaNSemantics) {
  const auto matches = [](std::uint32_t bits) {
    const float value = bit_cast<float>(bits);
    const BF<32> encoded{value};
    const float decoded = BF<32>::from_bits(bits).to_float();

    if ((std::isnan)(value))
      return encoded.is_nan() && encoded.signbit() == (std::signbit)(value) &&
             (std::isnan)(decoded) && (std::signbit)(decoded) == (std::signbit)(value);
    return encoded.to_bits() == bits && bit_cast<std::uint32_t>(decoded) == bits;
  };

  for (const std::uint32_t bits : {
           0U,
           UINT32_C(0x80000000),
           UINT32_C(0x00000001),
           UINT32_C(0x007FFFFF),
           UINT32_C(0x00800000),
           UINT32_C(0x7F7FFFFF),
           UINT32_C(0x7F800000),
           UINT32_C(0xFF800000),
           UINT32_C(0x7FC00000),
           UINT32_C(0xFFC00000),
       })
    EXPECT_TRUE(matches(bits)) << bits;

  Lcg random{UINT64_C(0x123456789ABCDEF0)};
  for (unsigned i = 0; i < 1U << 16; ++i) {
    const std::uint32_t bits = random.next();
    ASSERT_TRUE(matches(bits)) << bits;
  }
}

//! `detail::exp2i` stands in for `std::exp2` on the inexact conversion paths
//!
//! Its two boundaries are the ones an off-by-one in the field arithmetic moves:
//! the least subnormal `double` and the largest finite power of two.
TEST(Convert, Exp2i) {
  for (int x = -1200; x <= 1200; ++x)
    EXPECT_TRUE(same_double(detail::exp2i(x), std::exp2(static_cast<double>(x))))
        << "exp2i(" << x << ')';

  EXPECT_EQ(detail::exp2i(-1074), std::numeric_limits<double>::denorm_min());
  EXPECT_EQ(detail::exp2i(-1075), 0.0);
  EXPECT_EQ(detail::exp2i(1023), std::ldexp(1.0, 1023));
  EXPECT_EQ(detail::exp2i(1024), HUGE_VAL);
}

TEST(Convert, IntegerInterop) {
  using T = E5M2;
  EXPECT_EQ(T{3}.to_float(), 3.0F);
  EXPECT_EQ(T{-7}.to_float(), -7.0F);
  EXPECT_EQ(T{0u}.to_float(), 0.0F);
  EXPECT_EQ(T{8L}.to_float(), 8.0F);

  EXPECT_EQ(static_cast<int>(T{3.5F}), 3);
  EXPECT_EQ(static_cast<int>(T{-3.5F}), -3);
  EXPECT_EQ(static_cast<long>(T{0.0F}), 0L);
  EXPECT_TRUE(static_cast<bool>(T{1.0F}));
  EXPECT_FALSE(static_cast<bool>(T{0.0F}));
}

TEST(Convert, WideExponentRange) {
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
  EXPECT_EQ(T::max().to_double(), HUGE_VAL);
  EXPECT_EQ(T::from_bits(1).to_double(), 0.0);
  EXPECT_TRUE(T::from_bits(0xFFFF).is_nan());

  EXPECT_EQ(T{FLT_TRUE_MIN}.to_bits(), (unsigned{T::BIAS} - 149) << 3);
  EXPECT_EQ(T{FLT_TRUE_MIN}.to_double(), static_cast<double>(FLT_TRUE_MIN));
  EXPECT_EQ(T{0x1p-1070}.to_bits(), (unsigned{T::BIAS} - 1070) << 3);
  EXPECT_EQ(T{0x1p-1070}.to_double(), 0x1p-1070);

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
