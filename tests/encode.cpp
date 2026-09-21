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
struct CheckRoundingBoundaries {
  template <typename T> static bool check() {
    for (double x : rounding_inputs<T>()) {
      const T expected = reference_encode<T>(x);
      const T actual{x};
      if (actual.to_bits() != expected.to_bits()) {
        ADD_FAILURE() << describe<T>() << " from_double(" << x << "): got bits "
                      << +actual.to_bits() << ", expected " << +expected.to_bits();
        return false;
      }

      const float narrow = static_cast<float>(x);
      if (!(std::isnan)(x) && same_double(static_cast<double>(narrow), x)) {
        const T via_float{narrow};
        if (via_float.to_bits() != actual.to_bits()) {
          ADD_FAILURE() << describe<T>() << " float and double encoders disagree for " << x;
          return false;
        }
      }
    }
    return true;
  }
};

struct CheckRandomFloatPatterns {
  template <typename T> static bool check() {
    Lcg random{UINT64_C(0x123456789ABCDEF0)};
    for (unsigned i = 0; i < 1U << 14; ++i) {
      const float x = bit_cast<float>(random.next());
      if (!T::HAS_NAN && (std::isnan)(x))
        continue;

      const double wide = static_cast<double>(x);
      const T expected = reference_encode<T>(wide);
      const T actual{x};
      if (actual.to_bits() != expected.to_bits() || T{wide}.to_bits() != actual.to_bits()) {
        ADD_FAILURE() << describe<T>() << " encoders disagree for float bits "
                      << bit_cast<std::uint32_t>(x);
        return false;
      }
    }
    return true;
  }
};

struct CheckBfDoubleEncoding {
  template <typename T> static bool check() {
    Lcg random{UINT64_C(0xC371A8045FDF602B)};
    for (unsigned i = 0; i < 1U << 12; ++i) {
      // Raw doubles cover overflow and underflow; adjacent BF codes put the
      // normal shortcut and the subnormal fallback on their rounding ties.
      const auto raw = (static_cast<std::uint64_t>(random.next()) << 32) | random.next();
      const double x = bit_cast<double>(raw);
      if (T{x}.to_bits() != reference_encode<T>(x).to_bits())
        return false;
      const auto code = random.next() % static_cast<std::uint32_t>(T::max().to_bits());
      const double lo = from_code<T>(code).to_double();
      const double hi = from_code<T>(code + 1).to_double();
      const double tie = (lo + hi) * 0.5;
      for (double candidate : {std::nextafter(tie, 0.0), tie, std::nextafter(tie, HUGE_VAL)})
        for (double sign : {-1.0, 1.0}) {
          const double value = sign * candidate;
          if (T{value}.to_bits() != reference_encode<T>(value).to_bits())
            return false;
        }
    }
    return true;
  }
};

struct CheckBfFloatEncoding {
  template <typename T> static bool check() {
    using Word = typename T::Storage;
    constexpr int DROP = 23 - T::MANTISSA_BITS;
    constexpr int PADDING = std::numeric_limits<Word>::digits - 9 - T::MANTISSA_BITS;
    std::vector<std::pair<std::uint32_t, Word>> cases;
    const auto add = [&cases](std::uint32_t bits) {
      // Form expectations before enabling DAZ: the independent oracle widens
      // its float input to double and would otherwise lose host subnormals.
      cases.emplace_back(bits, reference_encode<T>(bit_cast<float>(bits)).to_bits());
    };
    for (auto bits :
         {0U, 0x80000000U, 1U, 0x80000001U, 0x007fffffU, 0x00800000U, 0x7f7fffffU, 0xff7fffffU,
          0x7f800000U, 0xff800000U, 0x7f800001U, 0xff800001U, 0x7fffffffU, 0xffffffffU})
      add(bits);
    Lcg random{UINT64_C(0xA63F940D5E2187BC)};
    for (unsigned i = 0; i < 1U << 12; ++i) {
      const auto bits = random.next();
      add(bits);
      if constexpr (DROP != 0) {
        const auto tie = (bits & (UINT32_MAX << DROP)) | (UINT32_C(1) << (DROP - 1));
        add(tie - 1);
        add(tie);
        add(tie + 1);
      }
    }
    const auto check = [&cases, PADDING] {
      for (const auto &sample : cases) {
        volatile std::uint32_t cell = sample.first;
        const T value{bit_cast<float>(static_cast<std::uint32_t>(cell))};
        if (value.to_bits() != sample.second ||
            bit_cast<Word>(value) != static_cast<Word>(sample.second << PADDING))
          return false;
      }
      return true;
    };
    for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
      const HostEnvironment saved;
      if (std::fesetround(mode) != 0 || !check())
        return false;
      if (set_flush_to_zero() && !check())
        return false;
    }
    return true;
  }
};

template <int N> void expect_float_path_matches_generic(std::uint64_t stride) {
  constexpr std::uint64_t END = UINT64_C(1) << 32;

  for (std::uint64_t code = 0; code < END; code += stride) {
    const auto bits = static_cast<std::uint32_t>(code);
    const float x = bit_cast<float>(bits);
    const auto actual = BF<N>{x}.to_bits();
    const auto expected = (bits & (UINT32_MAX >> 1)) > UINT32_C(0x7F800000)
                              ? static_cast<typename BF<N>::Storage>(
                                    BF<N>::quiet_NaN().to_bits() | (bits >> 31) << (N - 1)
                                )
                              : BF<N>{static_cast<double>(x)}.to_bits();

    if (actual != expected) {
      ADD_FAILURE() << "BF<" << N << "> mismatch for float bits " << bits;
      return;
    }
  }
}

template <typename T> bool float_encoder_matches_reference(float x) {
  if constexpr (!T::HAS_NAN)
    if ((std::isnan)(x))
      return true;

  return T{x}.to_bits() == reference_encode<T>(static_cast<double>(x)).to_bits();
}
} // namespace

TEST(Encode, RoundsEveryBoundaryCorrectly) { test_all_types<CheckRoundingBoundaries>(); }

TEST(Encode, RandomFloatSweep) { test_all_types<CheckRandomFloatPatterns>(); }

TEST(Encode, BfDoubleEncodingMatchesIndependentOracle) { test_bf_types<CheckBfDoubleEncoding>(); }

TEST(Encode, BfFloatEncodingIgnoresHostEnvironment) { test_bf_types<CheckBfFloatEncoding>(); }

TEST(Encode, BFFloatFastPathMatchesGenericPath) {
  constexpr std::uint64_t END = UINT64_C(1) << 32;
  constexpr std::uint64_t STRIDE = (END >> 20) | 1U;

  expect_float_path_matches_generic<16>(1);
  expect_float_path_matches_generic<20>(STRIDE);
  expect_float_path_matches_generic<24>(STRIDE);
}

TEST(Encode, GeneralizedFloatFastPathMatchesReference) {
  const auto failing = find_failing_pair<E5M10>([](E5M10 high, E5M10 low) {
    const auto bits = static_cast<std::uint32_t>(high.to_bits()) << 16 | low.to_bits();
    const float x = bit_cast<float>(bits);
    return float_encoder_matches_reference<E4M3>(x) && float_encoder_matches_reference<E4M3FN>(x) &&
           float_encoder_matches_reference<E4M3FNUZ>(x) &&
           float_encoder_matches_reference<E2M1FN>(x) && float_encoder_matches_reference<E5M10>(x);
  });

  if (failing)
    ADD_FAILURE() << "float bits " << (failing->first << 16 | failing->second);
}
