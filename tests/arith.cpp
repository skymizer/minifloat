// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "support.hpp"

#include <array>
#include <functional>
#include <utility>
#include <vector>

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
//! Retain a float reference for BF16 and binary16 under the default host
//! environment. BF's implementation now uses double, so this also compares
//! against a different intermediate width rather than repeating its route.
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
    if (!T::HAS_EXACT_F64_CONVERSION || 2 * T::MANTISSA_DIGITS + 2 > DBL_MANT_DIG)
      return true;

    Lcg random{UINT64_C(0x0FEDCBA987654321)};
    constexpr auto MASK = bit_mask(T::EXPONENT_BITS + T::MANTISSA_BITS + 1);
    for (unsigned i = 0; i < 1U << 13; ++i) {
      // `MASK` is captured for MSVC's sake; see the `SIGN` capture below.
      const auto draw = [&random, MASK] {
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
  // A 30-bit significand shifted by more overflows `int64_t` once the 32-bit
  // shapes run through here; 31 keeps their sum under 2**62.  The different
  // window and sticky substitute below are deliberately unlike the engine,
  // and both round alike.
  constexpr int ALIGN_CAP = 31;
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
      // `MASK` is captured for MSVC's sake; see the `SIGN` capture below.
      const auto draw = [&random, MASK] {
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

inline float opaque_float(std::uint32_t bits) noexcept {
  volatile std::uint32_t cell = bits;
  return bit_cast<float>(static_cast<std::uint32_t>(cell));
}

using Bits4 = std::array<std::uint32_t, 4>;

//! All four operators of one pair, through the library and through the host
struct Sweep {
  std::vector<Bits4> mini;
  std::vector<Bits4> host;
};

//! Operands that separate the rounding modes, and operands FTZ and DAZ erase
//!
//! The drawn pairs are normal, so no NaN payload can move for reasons that have
//! nothing to do with the environment, and every quotient has a divisor.  Four
//! binades either side of zero leaves room for a product to overflow, which is
//! a stable answer, and none to underflow, which is why the subnormal cases are
//! written out.
inline std::vector<std::pair<std::uint32_t, std::uint32_t>> environment_pairs() {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs{
      // One plus half an ulp: ties to even keeps the one, and no other mode does.
      {UINT32_C(0x3F800000), UINT32_C(0x33800000)},
      // A subnormal product, a subnormal difference, and a subnormal operand:
      // what FTZ erases on the way out and DAZ on the way in.
      {UINT32_C(0x00800000), UINT32_C(0x3F000000)},
      {UINT32_C(0x00800000), UINT32_C(0x007FFFFF)},
      {UINT32_C(0x00000001), UINT32_C(0x00000002)},
  };
  Lcg random{UINT64_C(0x0FF32EE24DD16CC0)};
  const auto draw = [&random] {
    const std::uint32_t bits = random.next();
    return (bits & UINT32_C(0x807FFFFF)) | ((bits >> 23 & 0x7F) + 64) << 23;
  };

  for (int i = 0; i < 1 << 12; ++i)
    pairs.emplace_back(draw(), draw());

  return pairs;
}

inline Sweep sweep(const std::vector<std::pair<std::uint32_t, std::uint32_t>> &pairs) {
  const auto bits = [](float x) { return bit_cast<std::uint32_t>(x); };
  Sweep result;
  result.mini.reserve(pairs.size());
  result.host.reserve(pairs.size());

  for (const auto &pair : pairs) {
    const std::uint32_t x = pair.first;
    const std::uint32_t y = pair.second;
    using T = BF<32>;
    result.mini.push_back(
        {static_cast<std::uint32_t>((opaque<T>(x) + opaque<T>(y)).to_bits()),
         static_cast<std::uint32_t>((opaque<T>(x) - opaque<T>(y)).to_bits()),
         static_cast<std::uint32_t>((opaque<T>(x) * opaque<T>(y)).to_bits()),
         static_cast<std::uint32_t>((opaque<T>(x) / opaque<T>(y)).to_bits())}
    );
    result.host.push_back(
        {bits(opaque_float(x) + opaque_float(y)), bits(opaque_float(x) - opaque_float(y)),
         bits(opaque_float(x) * opaque_float(y)), bits(opaque_float(x) / opaque_float(y))}
    );
  }
  return result;
}

} // namespace

TEST(Arith, MatchesHostRoundTrip) { test_paired_types<CheckHostArithmetic>(); }

//! Every ordered pair of binary16 and BF16 against a float reference.
//!
//! These formats have enough guard bits in binary32 for the comparison under
//! the default environment. The other 16-bit formats either need more precision
//! or exceed its range. BF16's double implementation keeps this exhaustive
//! comparison against an independent intermediate width.
TEST(Arith, EveryPairMatchesFloatRoundTrip) {
  expect_all_pairs<E5M10>(float_arithmetic_matches<E5M10>);
  expect_all_pairs<E8M7>(float_arithmetic_matches<E8M7>);
}

//! BF32 agrees with native float in the default environment. Its double
//! implementation must still give the same answer after integer rounding.
TEST(Arith, BF32MatchesFloatArithmetic) {
  Lcg random{UINT64_C(0x0FF32EE24DD16CC0)};

  for (unsigned i = 0; i < 1U << 22; ++i) {
    const BF<32> x = BF<32>::from_bits(random.next());
    const BF<32> y = BF<32>::from_bits(random.next());
    ASSERT_TRUE(float_arithmetic_matches(x, y)) << +x.to_bits() << ", " << +y.to_bits();
  }
}

//! Numeric results ignore the caller's rounding and subnormal modes.
//!
//! BF32's double intermediate and integer rounding must retain ties-to-even
//! where a direct float operation follows the host. Native float is the control:
//! if it does not move, the platform ignored the request and a pass is vacuous.
TEST(Arith, IgnoresHostEnvironment) {
  const HostEnvironment saved;
  const auto pairs = environment_pairs();
  const Sweep nearest = sweep(pairs);

  const auto check = [&pairs, &nearest](const char *what) {
    static const char *const OPERATORS[] = {"+", "-", "*", "/"};
    const Sweep disturbed = sweep(pairs);

    for (std::size_t i = 0; i < pairs.size(); ++i)
      for (std::size_t op = 0; op < 4; ++op)
        ASSERT_EQ(disturbed.mini[i][op], nearest.mini[i][op])
            << what << ": " << std::hex << pairs[i].first << ' ' << OPERATORS[op] << ' '
            << pairs[i].second;

    EXPECT_NE(disturbed.host, nearest.host)
        << what << " moved no native float; this platform leaves the test vacuous";
  };

  ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
  check("FE_UPWARD");
  ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
  check("FE_DOWNWARD");
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);

  if (set_flush_to_zero())
    check("FTZ and DAZ");
}

//! Exact conversions do not borrow the caller's subnormal mode
TEST(Convert, ExactConversionsIgnoreFlushToZero) {
  const HostEnvironment saved;
  using F = FN<7, 23, 120>;
  using D = FN<10, 1, 1023>;

  const auto check = [] {
    EXPECT_EQ(bit_cast<std::uint32_t>(opaque<F>(1).to_float()), UINT32_C(0x00000080));
    EXPECT_EQ(bit_cast<std::uint32_t>(opaque<F>(0x40000001).to_float()), UINT32_C(0x80000080));
    EXPECT_EQ(bit_cast<std::uint32_t>(opaque<F>(0x40000000).to_float()), UINT32_C(0x80000000));
    EXPECT_EQ(bit_cast<std::uint32_t>(opaque<F>(0x007FFFFF).to_float()), UINT32_C(0x03FFFFFE));
    EXPECT_EQ(bit_cast<std::uint64_t>(opaque<D>(1).to_double()), UINT64_C(0x0008000000000000));
  };

  check();
  if (set_flush_to_zero())
    check();
}

TEST(Arith, WideFormatsMatchHostRoundTrip) { test_wide_types<CheckWideHostArithmetic>(); }

TEST(Arith, CorrectlyRoundedSmallFormats) { test_small_types<CheckExactSmallArithmetic>(); }

TEST(Arith, CorrectlyRoundedWideFormats) { test_wide_types<CheckExactWideArithmetic>(); }

namespace {
struct CheckBfHardware {
  template <typename T> static bool check() {
    static_assert(T::IS_BFLOAT);
    constexpr T ONE{1};
    constexpr T TWO{2};
    static_assert((ONE + ONE).to_bits() == TWO.to_bits());
    static_assert((TWO - ONE).to_bits() == ONE.to_bits());
    static_assert((TWO * TWO).to_bits() == T{4}.to_bits());
    static_assert((TWO / TWO).to_bits() == ONE.to_bits());

    Lcg random{UINT64_C(0x97401953AC1DF035)};
    for (unsigned i = 0; i < 1U << 12; ++i) {
      const T x = opaque<T>(random.next());
      const T y = opaque<T>(random.next());
      if (!check_exact_pair(x, y))
        return false;
    }
    const T edge[] = {T{},      -T{},      ONE,      -ONE,     T::true_min(), -T::true_min(),
                      T::min(), -T::min(), T::max(), -T::max()};
    for (T x : edge)
      for (T y : edge)
        if (!check_exact_pair(x, y))
          return false;
    return CheckSpecialLadder::check<T>();
  }
};
} // namespace

TEST(Arith, BfHardwareMatchesExactOracleInEveryEnvironment) {
  const HostEnvironment saved;
  for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    const HostEnvironment before_mode;
    ASSERT_EQ(std::fesetround(mode), 0);
    test_bf_types<CheckBfHardware>();
    if (set_flush_to_zero())
      test_bf_types<CheckBfHardware>();
  }
}

namespace {
template <class> struct FormatOf;
template <class Format> struct FormatOf<Minifloat<Format>> {
  using type = Format;
};

//! The integer rounding behind BF arithmetic on normal operands
//!
//! Its whole domain is the positive normal binary64 range such a result can
//! reach, so it is checked there directly against the independent encoder:
//! random magnitudes from well below half the least subnormal to past
//! overflow, and both sides of the tie above sampled codes -- subnormal,
//! normal, the seam between them, and the carry into infinity.
struct CheckBfRoundMagnitude {
  template <typename T> static bool check() {
    using Format = typename FormatOf<T>::type;
    const auto matches = [](double x) {
      const auto actual = detail::bf_round_magnitude<Format>(bit_cast<std::uint64_t>(x));
      if (actual == reference_encode<T>(x).to_bits())
        return true;
      ADD_FAILURE() << describe<T>() << " magnitude " << x;
      return false;
    };
    const auto max = static_cast<std::uint32_t>(T::max().to_bits());
    const auto tie_above = [max](std::uint32_t code) {
      const double lo = from_code<T>(code).to_double();
      const double hi = code < max ? from_code<T>(code + 1).to_double()
                                   : 2 * lo - from_code<T>(code - 1).to_double();
      return (lo + hi) * 0.5;
    };
    const auto around = [&matches](double tie) {
      return matches(std::nextafter(tie, 0.0)) && matches(tie) &&
             matches(std::nextafter(tie, HUGE_VAL));
    };

    Lcg random{UINT64_C(0x5B0E6A2D98C4F173)};
    for (unsigned i = 0; i < 1U << 12; ++i) {
      const auto fraction = ((static_cast<std::uint64_t>(random.next()) << 32) | random.next()) &
                            ((UINT64_C(1) << 52) - 1U);
      const auto exponent = static_cast<std::uint64_t>(1023 - 310 + random.next() % 600);
      if (!matches(bit_cast<double>(exponent << 52 | fraction)))
        return false;
    }
    for (std::uint32_t code : {0U, 1U, (1U << T::MANTISSA_BITS) - 1U, 1U << T::MANTISSA_BITS, max})
      if (!around(tie_above(code)))
        return false;
    for (unsigned i = 0; i < 1U << 12; ++i)
      if (!around(tie_above(random.next() % (max + 1U))))
        return false;
    return true;
  }
};
} // namespace

TEST(Arith, BfRoundMagnitudeMatchesIndependentOracle) { test_bf_types<CheckBfRoundMagnitude>(); }

TEST(Arith, BfNormalDivisionAvoidsFloatDoubleRounding) {
  const auto x = opaque<BF<24>>(0x3fc6cc);
  const auto y = opaque<BF<24>>(0x3f8f47);
  EXPECT_EQ((x / y).to_bits(), 0x3fb199U);
}

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

  using D = IEEE<6, 25>;
  constexpr D DIVIDEND = D::from_bits(1);
  constexpr D DIVISOR = D::from_bits((1U << 26) - 1U);
  static_assert((D::from_bits(0) / D::from_bits(1)).to_bits() == 0);
  static_assert((DIVIDEND / DIVISOR).to_bits() == 0x0A00'0001);

  using P = IEEE<2, 29>;
  constexpr P PRODUCT = P::from_bits((1U << 30) - 1U);
  static_assert((PRODUCT * PRODUCT).to_bits() == 0x5FFF'FFFE);

  using LowBias = Finite<30, 0, 0>;
  static_assert((LowBias::max() * LowBias::max()).to_bits() == LowBias::max().to_bits());
  using HighBias = Finite<2, 1, 1073741824>;
  static_assert((HighBias::true_min() * HighBias::true_min()).to_bits() == 0);
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
