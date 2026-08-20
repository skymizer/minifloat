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
} // namespace

TEST(Encode, RoundsEveryBoundaryCorrectly) { test_all_types<CheckRoundingBoundaries>(); }

TEST(Encode, RandomFloatSweep) { test_all_types<CheckRandomFloatPatterns>(); }
