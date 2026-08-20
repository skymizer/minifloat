// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "support.hpp"

#include <cassert>
#include <unordered_set>

using namespace minifloat_test;      // NOLINT(google-build-using-namespace)
using namespace skymizer::minifloat; // NOLINT(google-build-using-namespace)

namespace {
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

struct CheckUnary {
  template <typename T> static bool check() {
    return T{0.0F} == -T{0.0F} && for_all<T>([](T x) {
             const T absolute = x.abs();
             return same_mini(x, +x) && same_mini(x, - -x) &&
                    (x.is_nan() ? absolute.is_nan() : !absolute.signbit());
           });
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

struct CheckHashConsistency {
  template <typename T> static bool check() {
    const std::hash<T> hash;
    return for_all<T>([&hash](T x) {
      const auto x_hash = hash(x);
      return for_all<T>([&hash, x, x_hash](T y) { return x != y || x_hash == hash(y); });
    });
  }
};
} // namespace

TEST(Ops, Copying) { test_all_types<CheckCopying>(); }

TEST(Ops, Equality) { test_all_types<CheckEquality>(); }

TEST(Ops, UnarySignAndAbs) { test_all_types<CheckUnary>(); }

TEST(Ops, Comparison) { test_paired_types<CheckComparison>(); }

TEST(Ops, Classification) { test_all_types<CheckClassification>(); }

TEST(Ops, HashConsistentWithEquality) {
  test_small_types<CheckHashConsistency>();

  using T = E3M4;
  const std::hash<T> hash;
  EXPECT_EQ(hash(T{0.0F}), hash(T{-0.0F}));
  std::unordered_set<T> values{T{1.0F}, T{2.0F}, T{-1.0F}, T{0.0F}, T{-0.0F}};
  EXPECT_EQ(values.size(), 4u);
}
