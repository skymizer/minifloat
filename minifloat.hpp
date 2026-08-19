// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2026 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SKYMIZER_MINIFLOAT_HPP
#define SKYMIZER_MINIFLOAT_HPP

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

#if __cplusplus >= 202002L
#include <bit>
#else
#include <cstring>
#endif

// The `gnu::const` and `gnu::pure` attributes are GCC/Clang extensions. Other
// compilers (notably MSVC under strict warning flags) may warn on the
// attribute namespace. Wrap them so we degrade to a no-op elsewhere.
#if defined(__GNUC__) || defined(__clang__)
#define SKYMIZER_MINIFLOAT_CONST [[gnu::const]]
#define SKYMIZER_MINIFLOAT_PURE [[gnu::pure]]
#else
#define SKYMIZER_MINIFLOAT_CONST
#define SKYMIZER_MINIFLOAT_PURE
#endif

//! Namespace for Skymizer
namespace skymizer {

//! Namespace for the minifloat library
namespace minifloat {

//! Backport of C++20 std::bit_cast
template <typename To, typename From>
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST To bit_cast(const From &from) noexcept {
  static_assert(sizeof(To) == sizeof(From));
  static_assert(std::is_trivially_copyable_v<To>);
  static_assert(std::is_trivially_copyable_v<From>);

#if __cplusplus >= 202002L
  return std::bit_cast<To>(from);
#else
  static_assert(std::is_trivially_constructible_v<To>);
  To to;
  std::memcpy(&to, &from, sizeof(To));
  return to;
#endif
}

//! Default exponent bias, the IEEE 754 one
constexpr int default_bias(int exponent_width) { return (1 << (exponent_width - 1)) - 1; }

namespace detail {

//! Unsigned integer type as wide as the host floating-point type `Float`
template <typename Float>
using BitsOf =
    std::conditional_t<sizeof(Float) == sizeof(std::uint32_t), std::uint32_t, std::uint64_t>;

//! Round a *normal* (or infinite, or NaN) host float to `M` mantissa bits
//!
//! Ties go to even.  The result keeps the host format, so the caller still has
//! to encode it.  Subnormal and zero inputs must be scaled into the normal
//! range first — their exponent field does not mean what this bit trick
//! assumes.
template <int M, typename Float>
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST Float round_normal_to_mantissa(Float x) noexcept {
  using Bits = BitsOf<Float>;
  constexpr int MANT_DIG = std::numeric_limits<Float>::digits;

  static_assert(M < MANT_DIG);
  static_assert(std::numeric_limits<Float>::radix == 2);
  static_assert(std::numeric_limits<Float>::is_iec559);

  const auto bits = bit_cast<Bits>(x);
  const auto ulp = Bits{1} << (MANT_DIG - 1 - M);
  const auto bias = static_cast<Bits>(ulp / 2 - !(bits & ulp));
  return bit_cast<Float>(static_cast<Bits>((bits + bias) & ~(ulp - 1)));
}

//! What a bit pattern denotes
enum struct Kind { Zero, Subnormal, Normal, Infinite, NaN };

//! Plain scientific reading of the magnitude field
//!
//! Every magnitude `(e << M) | m` denotes `(1 + m / 2**M) * 2**(e - B)`.  This
//! layer owns the bit space and nothing else: no zero, no subnormal, no
//! reserved code point.  The layers above reinterpret parts of it.
template <int E, int M, int B> struct ScientificFormat {
  static_assert(E + M < 16);
  static_assert(E >= 2);
  static_assert(M >= 0);

  using Storage = std::conditional_t<(E + M < 8), std::uint_least8_t, std::uint_least16_t>;

  static constexpr Storage MAG_MASK = static_cast<Storage>((1U << (E + M)) - 1U);
  static constexpr Storage SIGN_MASK = static_cast<Storage>(1U << (E + M));

  //! Least exponent of a representable magnitude, `FLT_MIN_EXP` style
  static constexpr int MIN_EXP = 1 - B;
  //! One past the greatest exponent, `FLT_MAX_EXP` style
  static constexpr int MAX_EXP = (1 << E) - B;

  static constexpr Kind kind(Storage) noexcept { return Kind::Normal; }
};

//! `ScientificFormat` with row 0 spent on zero and subnormals
//!
//! Magnitudes below `2**M` denote `m * 2**(MIN_EXP - 1 - M)`, so the format
//! gains a zero (and a negative zero) and loses the `2**-B` scientific row.
//! Nothing is reserved for infinity or NaN, hence *finite*.
template <int E, int M, int B> struct FiniteFormat {
  using Inner = ScientificFormat<E, M, B>;
  using Storage = typename Inner::Storage;

  static constexpr int EXPONENT_BITS = E, MANTISSA_BITS = M, BIAS = B;
  static constexpr Storage MAG_MASK = Inner::MAG_MASK, SIGN_MASK = Inner::SIGN_MASK;

  //! Row 0 no longer denotes `2**-B`, so the least normal exponent moves up
  static constexpr int MIN_EXP = Inner::MIN_EXP + 1;
  static constexpr int MAX_EXP = Inner::MAX_EXP;

  static constexpr bool HAS_INF = false, HAS_NAN = false, HAS_NEG_ZERO = true;

  static constexpr Storage MAX_FINITE_MAG = MAG_MASK;
  static constexpr Storage OVERFLOW_MAG = MAG_MASK;

  static constexpr bool is_nan(Storage) noexcept { return false; }

  static constexpr Kind kind(Storage bits) noexcept {
    const Storage mag = bits & MAG_MASK;
    if (mag == 0)
      return Kind::Zero;
    if (mag < (Storage{1} << M))
      return Kind::Subnormal;
    return Inner::kind(mag);
  }
};

//! `FiniteFormat` with the top exponent row reserved, as in IEEE 754
//!
//! A zero mantissa there denotes an infinity, any other value a NaN.
template <int E, int M, int B> struct IeeeFormat {
  using Inner = FiniteFormat<E, M, B>;
  using Storage = typename Inner::Storage;

  static_assert(M > 0, "IEEE 754 needs a mantissa bit to tell infinity from NaN");

  static constexpr int EXPONENT_BITS = E, MANTISSA_BITS = M, BIAS = B;
  static constexpr Storage MAG_MASK = Inner::MAG_MASK, SIGN_MASK = Inner::SIGN_MASK;

  static constexpr int MIN_EXP = Inner::MIN_EXP;
  //! The top row is not finite, so the exponent range loses its last step
  static constexpr int MAX_EXP = Inner::MAX_EXP - 1;

  static constexpr bool HAS_INF = true, HAS_NAN = true, HAS_NEG_ZERO = true;

  static constexpr Storage INF_MAG = static_cast<Storage>(MAG_MASK << M & MAG_MASK);
  static constexpr Storage MAX_FINITE_MAG = static_cast<Storage>(INF_MAG - 1U);
  static constexpr Storage OVERFLOW_MAG = INF_MAG;
  static constexpr Storage NAN_BITS = static_cast<Storage>(MAG_MASK << (M - 1) & MAG_MASK);

  static constexpr bool is_nan(Storage bits) noexcept { return (bits & MAG_MASK) > INF_MAG; }

  static constexpr Kind kind(Storage bits) noexcept {
    const Storage mag = bits & MAG_MASK;
    if (mag > INF_MAG)
      return Kind::NaN;
    if (mag == INF_MAG)
      return Kind::Infinite;
    return Inner::kind(mag);
  }
};

//! `FiniteFormat` with the all-ones magnitude reserved for NaN
//!
//! `FN` as in LLVM/MLIR: `F` for finite (there is no infinity), `N` for a
//! special NaN encoding.
template <int E, int M, int B> struct FnFormat {
  using Inner = FiniteFormat<E, M, B>;
  using Storage = typename Inner::Storage;

  static constexpr int EXPONENT_BITS = E, MANTISSA_BITS = M, BIAS = B;
  static constexpr Storage MAG_MASK = Inner::MAG_MASK, SIGN_MASK = Inner::SIGN_MASK;

  static constexpr int MIN_EXP = Inner::MIN_EXP;
  static constexpr int MAX_EXP = Inner::MAX_EXP;

  static constexpr bool HAS_INF = false, HAS_NAN = true, HAS_NEG_ZERO = true;

  static constexpr Storage NAN_BITS = MAG_MASK;
  static constexpr Storage MAX_FINITE_MAG = static_cast<Storage>(MAG_MASK - 1U);
  static constexpr Storage OVERFLOW_MAG = MAX_FINITE_MAG;

  static constexpr bool is_nan(Storage bits) noexcept { return (bits & MAG_MASK) == MAG_MASK; }

  static constexpr Kind kind(Storage bits) noexcept {
    const Storage mag = bits & MAG_MASK;
    return mag == MAG_MASK ? Kind::NaN : Inner::kind(mag);
  }
};

//! `FiniteFormat` with the negative zero reserved for NaN
//!
//! `FNUZ` as in LLVM/MLIR: `F` for finite, `N` for a special NaN encoding,
//! `UZ` for unsigned zero.  Since the sole NaN is the would-be negative zero,
//! it is the one format whose NaN test needs the sign bit.
template <int E, int M, int B> struct FnuzFormat {
  using Inner = FiniteFormat<E, M, B>;
  using Storage = typename Inner::Storage;

  static constexpr int EXPONENT_BITS = E, MANTISSA_BITS = M, BIAS = B;
  static constexpr Storage MAG_MASK = Inner::MAG_MASK, SIGN_MASK = Inner::SIGN_MASK;

  static constexpr int MIN_EXP = Inner::MIN_EXP;
  static constexpr int MAX_EXP = Inner::MAX_EXP;

  static constexpr bool HAS_INF = false, HAS_NAN = true, HAS_NEG_ZERO = false;

  static constexpr Storage NAN_BITS = SIGN_MASK;
  static constexpr Storage MAX_FINITE_MAG = MAG_MASK;
  static constexpr Storage OVERFLOW_MAG = MAG_MASK;

  static constexpr bool is_nan(Storage bits) noexcept { return bits == SIGN_MASK; }

  static constexpr Kind kind(Storage bits) noexcept {
    return bits == SIGN_MASK ? Kind::NaN : Inner::kind(bits & MAG_MASK);
  }
};

} // namespace detail

//! Configurable signed floating-point type up to 16 bits
//!
//! `Format` is one of the layered policies in `detail`, each of which
//! reinterprets one part of the bit space in terms of the layer below it.
//! Spell types through the `Finite`, `IEEE`, `FN`, and `FNUZ` alias templates
//! rather than naming a policy directly.
template <class Format> class Minifloat {
  static constexpr int E = Format::EXPONENT_BITS;
  static constexpr int M = Format::MANTISSA_BITS;
  static constexpr int B = Format::BIAS;

public:
  using Storage = typename Format::Storage;

  static constexpr int EXPONENT_BITS = E;
  static constexpr int MANTISSA_BITS = M;
  static constexpr int MANTISSA_DIGITS = M + 1;
  static constexpr int BIAS = B;
  static constexpr int MAX_EXP = Format::MAX_EXP;
  static constexpr int MIN_EXP = Format::MIN_EXP;
  static constexpr Storage ABS_MASK = Format::MAG_MASK;

  //! Does this format reserve code points for infinities?
  static constexpr bool HAS_INF = Format::HAS_INF;
  //! Does this format reserve code points for NaNs?
  static constexpr bool HAS_NAN = Format::HAS_NAN;
  //! Does this format distinguish &minus;0.0 from +0.0?
  static constexpr bool HAS_NEG_ZERO = Format::HAS_NEG_ZERO;

  static constexpr bool HAS_EXACT_F32_CONVERSION =
      FLT_MANT_DIG >= MANTISSA_DIGITS && FLT_MAX_EXP >= MAX_EXP && FLT_MIN_EXP <= MIN_EXP &&
      std::numeric_limits<float>::radix == 2 && std::numeric_limits<float>::is_iec559;

  static constexpr bool HAS_EXACT_F64_CONVERSION =
      DBL_MANT_DIG >= MANTISSA_DIGITS && DBL_MAX_EXP >= MAX_EXP && DBL_MIN_EXP <= MIN_EXP &&
      std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::is_iec559;

  static constexpr bool USE_FLT_ADD = FLT_MANT_DIG >= 2 * MANTISSA_DIGITS && //
                                      (FLT_MAX_EXP > MAX_EXP) &&             //
                                      (FLT_MIN_EXP < MIN_EXP);

  static constexpr bool USE_FLT_MUL = FLT_MANT_DIG >= 2 * MANTISSA_DIGITS &&
                                      FLT_MAX_EXP >= 2 * MAX_EXP &&
                                      FLT_MIN_EXP - 1 <= 2 * (MIN_EXP - 1);

private:
  Storage bits_{};

  //! Encode a host float, rounding to nearest with ties to even
  //!
  //! A NaN input needs `Format::HAS_NAN`; see the constructors.
  template <typename Float>
  [[nodiscard]] SKYMIZER_MINIFLOAT_CONST static Storage bits_from(Float x) noexcept {
    using Bits = detail::BitsOf<Float>;
    using Int = std::make_signed_t<Bits>;
    constexpr int MANT_DIG = std::numeric_limits<Float>::digits;
    constexpr int SRC_MIN_EXP = std::numeric_limits<Float>::min_exponent;

    const auto sign = static_cast<unsigned>(std::signbit(x)) << (E + M);

    if ((std::isnan)(x)) {
      if constexpr (Format::HAS_NAN)
        return static_cast<Storage>(sign | Format::NAN_BITS);
      else // Precondition violation; saturate rather than emit a wild pattern.
        return static_cast<Storage>(sign | Format::MAX_FINITE_MAG);
    }

    Float normalized = x;
    Int offset = 0;

    // A zero or subnormal source has a zero exponent field, which the linear
    // magnitude below misreads. Only formats reaching under the source's own
    // normal range ever get here, and for them the scaling is exact.
    if constexpr (MIN_EXP < SRC_MIN_EXP) {
      if (!(std::abs(x) >= (std::numeric_limits<Float>::min)())) {
        if (x == Float{0})
          return static_cast<Storage>(Format::HAS_NEG_ZERO * sign);

        normalized = x * static_cast<Float>(Bits{1} << MANT_DIG);
        offset = Int{MANT_DIG} << M;
      }
    }

    const auto bits = bit_cast<Bits>(detail::round_normal_to_mantissa<M>(normalized));
    const Int diff = Int{MIN_EXP - SRC_MIN_EXP} * (Int{1} << M) + offset;
    const Int magnitude = static_cast<Int>(bits << 1 >> (MANT_DIG - M)) - diff;

    if (magnitude < Int{1} << M) {
      // The scale stays double: it overflows `float` for the wider formats.
      const auto ticks =
          static_cast<Storage>(std::nearbyint(std::abs(x) * std::exp2(MANTISSA_DIGITS - MIN_EXP)));
      return static_cast<Storage>((Format::HAS_NEG_ZERO || ticks) * sign | ticks);
    }
    return static_cast<Storage>(sign | std::min<Int>(magnitude, Format::OVERFLOW_MAG));
  }

  //! Exact reconstruction into `Float`
  //!
  //! Requires the matching `HAS_EXACT_*_CONVERSION`.
  template <typename Float> [[nodiscard]] SKYMIZER_MINIFLOAT_PURE Float to_exact() const noexcept {
    using Bits = detail::BitsOf<Float>;
    constexpr int MANT_DIG = std::numeric_limits<Float>::digits;
    constexpr int DST_MIN_EXP = std::numeric_limits<Float>::min_exponent;

    const Float sign = signbit() ? Float{-1} : Float{1};
    const auto magnitude = static_cast<Bits>(bits_ & ABS_MASK);

    if constexpr (Format::HAS_NAN)
      if (Format::is_nan(bits_))
        return std::copysign(std::numeric_limits<Float>::quiet_NaN(), sign);

    if constexpr (Format::HAS_INF)
      if (magnitude == Format::INF_MAG)
        return std::copysign(std::numeric_limits<Float>::infinity(), sign);

    if (magnitude < Bits{1} << M)
      return magnitude *
             std::copysign(std::exp2(static_cast<Float>(MIN_EXP - MANTISSA_DIGITS)), sign);

    const auto shifted = static_cast<Bits>(magnitude << (MANT_DIG - MANTISSA_DIGITS));
    const auto bias = static_cast<Bits>(Bits{MIN_EXP - DST_MIN_EXP} << (MANT_DIG - 1));
    const auto sign_bit =
        static_cast<Bits>(Bits{signbit()} << (std::numeric_limits<Bits>::digits - 1));
    return bit_cast<Float>(static_cast<Bits>(sign_bit | (shifted + bias)));
  }

public:
  Minifloat() = default;

  explicit Minifloat(float x) noexcept : bits_(bits_from(x)) {
    assert((HAS_NAN || !(std::isnan)(x)) && "this minifloat format cannot represent a NaN");
  }

  explicit Minifloat(double x) noexcept : bits_(bits_from(x)) {
    assert((HAS_NAN || !(std::isnan)(x)) && "this minifloat format cannot represent a NaN");
  }

  //! Construct from any non-bool integer type by routing through double.
  //! `bool` is excluded so it routes through `operator bool()` instead and
  //! does not collide with that overload.
  template <
      typename Int,
      std::enable_if_t<
          std::is_integral_v<Int> && !std::is_same_v<std::remove_cv_t<Int>, bool>, int> = 0>
  explicit Minifloat(Int x) noexcept : bits_(bits_from(static_cast<double>(x))) {}

  static constexpr Minifloat from_bits(Storage bits) noexcept {
    Minifloat result;
    result.bits_ = static_cast<Storage>(bits & (Format::MAG_MASK | Format::SIGN_MASK));
    return result;
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_CONST static Minifloat from_float(float x) noexcept {
    return Minifloat{x};
  }
  [[nodiscard]] SKYMIZER_MINIFLOAT_CONST static Minifloat from_double(double x) noexcept {
    return Minifloat{x};
  }

  //! Minimum positive value, which is probably subnormal
  //!
  //! This can be normal when bitwidth is low.  Therefore, it is named after
  //! `FLT_TRUE_MIN` instead of `numeric_limits::denorm_min()`.
  [[nodiscard]] static constexpr Minifloat true_min() noexcept { return from_bits(1); }

  /// Minimum positive normal value
  [[nodiscard]] static constexpr Minifloat min() noexcept {
    return from_bits(static_cast<Storage>(1U << M));
  }

  /// Maximum finite value
  [[nodiscard]] static constexpr Minifloat max() noexcept {
    return from_bits(Format::MAX_FINITE_MAG);
  }

  //! Positive infinity, or `+0.0` for the formats that have none
  [[nodiscard]] static constexpr Minifloat infinity() noexcept {
    if constexpr (Format::HAS_INF)
      return from_bits(Format::INF_MAG);
    else
      return from_bits(0);
  }

  //! Quiet NaN, or `+0.0` for the formats that have none
  [[nodiscard]] static constexpr Minifloat quiet_NaN() noexcept {
    if constexpr (Format::HAS_NAN)
      return from_bits(Format::NAN_BITS);
    else
      return from_bits(0);
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr Storage to_bits() const noexcept { return bits_; }

  //! Sign bit
  //!
  //! Note for `FNUZ`: the sole NaN representation has the sign bit set, so
  //! `signbit()` returns `true` for a FNUZ NaN even though there is no
  //! negative-zero counterpart to compare it to. Callers that filter by
  //! `signbit()` should test `is_nan()` first when working with FNUZ.
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool signbit() const noexcept {
    return (bits_ & Format::SIGN_MASK) != 0;
  }

  //! Check if the number is nonzero
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr explicit operator bool() const noexcept {
    if constexpr (!Format::HAS_NEG_ZERO)
      return bits_ != 0;

    return (bits_ & ABS_MASK) != 0;
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool is_nan() const noexcept {
    return Format::is_nan(bits_);
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool is_infinite() const noexcept {
    if constexpr (Format::HAS_INF)
      return (bits_ & ABS_MASK) == Format::INF_MAG;

    return false;
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool is_finite() const noexcept {
    return !is_nan() && !is_infinite();
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool is_normal() const noexcept {
    return is_finite() && (bits_ & ABS_MASK) >= (1U << M);
  }

  //! Check if the number is nonzero subnormal
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool is_subnormal() const noexcept {
    const Storage magnitude = bits_ & ABS_MASK;
    return 0 < magnitude && magnitude < (1U << M);
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr int classify() const noexcept {
    const detail::Kind kind = Format::kind(bits_);
    if (kind == detail::Kind::NaN)
      return FP_NAN;
    if (kind == detail::Kind::Infinite)
      return FP_INFINITE;
    if (kind == detail::Kind::Subnormal)
      return FP_SUBNORMAL;
    if (kind == detail::Kind::Zero)
      return FP_ZERO;
    return FP_NORMAL;
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr Minifloat abs() const noexcept {
    // A FNUZ NaN is the would-be negative zero; clearing its sign would turn
    // it into a zero.
    if constexpr (!Format::HAS_NEG_ZERO)
      if (!(bits_ & ABS_MASK))
        return *this;

    return from_bits(static_cast<Storage>(bits_ & ABS_MASK));
  }

  //! Explicit conversion to float
  //!
  //! The lossy branch goes through double.  Conversion to double is lossy only
  //! when the exponent range is too wide, and in that case a second conversion
  //! to float is safe.
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE float to_float() const noexcept {
    if constexpr (HAS_EXACT_F32_CONVERSION)
      return to_exact<float>();

    return static_cast<float>(to_double());
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE explicit operator float() const noexcept {
    return to_float();
  }

  //! Conversion to double
  //!
  //! When `HAS_EXACT_F64_CONVERSION` holds, the result is exact; otherwise the
  //! exponent range overflows to `HUGE_VAL` or underflows toward zero.
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE double to_double() const noexcept {
    if constexpr (HAS_EXACT_F64_CONVERSION)
      return to_exact<double>();

    const double sign = signbit() ? -1.0 : 1.0;
    const auto magnitude = static_cast<std::uint32_t>(bits_ & ABS_MASK);

    if constexpr (Format::HAS_NAN)
      if (Format::is_nan(bits_))
        return std::copysign(NAN, sign);

    if constexpr (Format::HAS_INF)
      if (magnitude == Format::INF_MAG)
        return std::copysign(HUGE_VAL, sign);

    if (magnitude < 1U << M)
      return std::copysign(std::ldexp(magnitude, MIN_EXP - MANTISSA_DIGITS), sign);

    const auto significand = static_cast<std::uint32_t>((magnitude & ((1U << M) - 1U)) | 1U << M);
    const int exponent = static_cast<int>(magnitude >> M) - B;
    return std::copysign(std::ldexp(significand, exponent - M), sign);
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE explicit operator double() const noexcept {
    return to_double();
  }

  //! Truncating conversion to any non-bool integer type. Out-of-range values
  //! invoke the host's float-to-integer truncation, matching what
  //! `static_cast<Int>(double)` would do — for NaN and infinity this is
  //! implementation-defined per the C++ standard.
  template <
      typename Int,
      std::enable_if_t<
          std::is_integral_v<Int> && !std::is_same_v<std::remove_cv_t<Int>, bool>, int> = 0>
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE explicit operator Int() const noexcept {
    return static_cast<Int>(to_double());
  }

  Minifloat &operator+=(Minifloat y) noexcept { return *this = *this + y; }
  Minifloat &operator-=(Minifloat y) noexcept { return *this = *this - y; }
  Minifloat &operator*=(Minifloat y) noexcept { return *this = *this * y; }
  Minifloat &operator/=(Minifloat y) noexcept { return *this = *this / y; }
};

namespace detail {
//! Are both operands zero, in a format where that can happen with unequal bits?
template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr bool
are_different_zeroes(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  if constexpr (!Format::HAS_NEG_ZERO)
    return false;

  return ((x.to_bits() | y.to_bits()) & Format::MAG_MASK) == 0;
}
} // namespace detail

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr bool
operator==(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  return (x.to_bits() == y.to_bits() && !x.is_nan()) || detail::are_different_zeroes(x, y);
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr bool
operator!=(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  return !(x == y);
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr bool
operator<(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  const auto a = x.to_bits();
  const auto b = y.to_bits();
  const bool sign = ((a | b) & Format::SIGN_MASK) != 0;

  if (x.is_nan() || y.is_nan() || detail::are_different_zeroes(x, y))
    return false;

  return sign ? a > b : a < b;
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr bool
operator<=(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  const auto a = x.to_bits();
  const auto b = y.to_bits();
  const bool sign = ((a | b) & Format::SIGN_MASK) != 0;

  if (x.is_nan() || y.is_nan())
    return false;

  if (detail::are_different_zeroes(x, y))
    return true;

  return sign ? a >= b : a <= b;
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr bool
operator>(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  return y < x;
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr bool
operator>=(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  return y <= x;
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format> operator+(Minifloat<Format> x) noexcept {
  return x;
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format> operator-(Minifloat<Format> x) noexcept {
  // Flipping the sign of a FNUZ zero would produce its NaN.
  if constexpr (!Format::HAS_NEG_ZERO)
    if (!(x.to_bits() & Format::MAG_MASK))
      return x;

  return Minifloat<Format>::from_bits(
      static_cast<typename Format::Storage>(x.to_bits() ^ Format::SIGN_MASK)
  );
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST Minifloat<Format>
operator+(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  if constexpr (Minifloat<Format>::USE_FLT_ADD)
    return Minifloat<Format>{x.to_float() + y.to_float()};

  return Minifloat<Format>{x.to_double() + y.to_double()};
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST Minifloat<Format>
operator-(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  if constexpr (Minifloat<Format>::USE_FLT_ADD)
    return Minifloat<Format>{x.to_float() - y.to_float()};

  return Minifloat<Format>{x.to_double() - y.to_double()};
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST Minifloat<Format>
operator*(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  if constexpr (Minifloat<Format>::USE_FLT_MUL)
    return Minifloat<Format>{x.to_float() * y.to_float()};

  return Minifloat<Format>{x.to_double() * y.to_double()};
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST Minifloat<Format>
operator/(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  return Minifloat<Format>{x.to_double() / y.to_double()};
}

//! Mantissa, base 2 exponent, and sign as integer
//!
//! A finite original value can be reconstructed as
//! `sign * mantissa * 2**exponent`.
//!
//! See also `integer_decode`.
struct IntegerDecode {
  std::uint64_t mantissa;
  std::int16_t exponent;
  std::int8_t sign;
};

//! Decode the argument into mantissa, exponent, and sign
//!
//! NaN inputs produce the sentinel `{0, 0, 0}` (sign zero) — non-NaN values
//! always have sign `+1` or `-1`, so `sign == 0` unambiguously signals NaN.
//! As in Rust's `FloatCore::integer_decode`, an infinity produces the integer
//! triple immediately beyond the largest finite value. Reconstructing that
//! finite value and converting it back to the same Minifloat type yields
//! infinity.
//!
//! **Additional promise**: LSB of `mantissa` aligns with ULP of a normal `x`.
//!
//! See Rust
//! [`num::traits::float::FloatCore::integer_decode`](https://docs.rs/num/0.4.3/num/traits/float/trait.FloatCore.html#tymethod.integer_decode).
template <class Format> IntegerDecode integer_decode(Minifloat<Format> x) noexcept {
  if (x.is_nan())
    return {0, 0, 0};

  constexpr int E = Format::EXPONENT_BITS;
  constexpr int M = Format::MANTISSA_BITS;
  constexpr int BIAS = M + Format::BIAS;
  const auto bit_mask = [](int width) { return width > 0 ? UINT32_MAX >> (32 - width) : 0; };

  const auto bits = x.to_bits();
  const int sign = (bits & Format::SIGN_MASK) ? -1 : 1;
  const int exponent = bits >> M & bit_mask(E);

  const std::uint32_t payload = bits & bit_mask(M);
  const std::uint32_t mantissa = exponent == 0 ? payload << 1 : payload | (UINT32_C(1) << M);

  return {
      mantissa,
      static_cast<std::int16_t>(exponent - BIAS),
      static_cast<std::int8_t>(sign),
  };
}

//! Finite-only format: every bit pattern is a number, none is NaN or infinity
template <int E, int M, int B = default_bias(E)>
using Finite = Minifloat<detail::FiniteFormat<E, M, B>>;

//! IEEE 754 format: the top exponent row holds the infinities and the NaNs
template <int E, int M, int B = default_bias(E)>
using IEEE = Minifloat<detail::IeeeFormat<E, M, B>>;

//! LLVM/MLIR `FN` format: no infinity, all-ones magnitude is NaN
template <int E, int M, int B = default_bias(E)> using FN = Minifloat<detail::FnFormat<E, M, B>>;

//! LLVM/MLIR `FNUZ` format: no infinity, no &minus;0.0, that code point is NaN
//!
//! The default bias is one greater than `default_bias(E)`, as in LLVM.
template <int E, int M, int B = default_bias(E) + 1>
using FNUZ = Minifloat<detail::FnuzFormat<E, M, B>>;

// Aliases for the formats LLVM's APFloat knows at 16 bits or fewer, without
// its `FloatN` prefix. The `FN` suffix is LLVM's name for the format, not a
// promise about NaN: the OCP MX types below (FP4 E2M1, FP6 E2M3 and E3M2) have
// no NaN at all, so they are `Finite`. The `FN` *template* always means
// "all-ones magnitude is NaN", hence `FN<2, 1>` differs from `E2M1FN`.
using E2M1FN = Finite<2, 1>; //!< OCP MX FP4
using E2M3FN = Finite<2, 3>; //!< OCP MX FP6
using E3M2FN = Finite<3, 2>; //!< OCP MX FP6
using E3M4 = IEEE<3, 4>;
using E4M3 = IEEE<4, 3>;
using E4M3FN = FN<4, 3>;
using E4M3FNUZ = FNUZ<4, 3>;
using E4M3B11FNUZ = FNUZ<4, 3, 11>;
using E5M2 = IEEE<5, 2>;
using E5M2FNUZ = FNUZ<5, 2>;
using E5M10 = IEEE<5, 10>; //!< IEEE 754 binary16
using E8M7 = IEEE<8, 7>;   //!< bfloat16

} // namespace minifloat

using minifloat::Minifloat;

} // namespace skymizer

namespace std {
//! Hash specialization for `Minifloat` so it can be used in unordered
//! containers. Positive and negative zero are normalized so that values that
//! compare equal hash equally.
template <class Format> struct hash<::skymizer::minifloat::Minifloat<Format>> {
  size_t operator()(::skymizer::minifloat::Minifloat<Format> x) const noexcept {
    auto bits = x.to_bits();
    if ((bits & Format::MAG_MASK) == 0)
      bits = 0;
    return static_cast<size_t>(bits);
  }
};

//! Standard `numeric_limits` specialization for `Minifloat`. Mirrors the
//! traits/methods that built-in floating types provide so generic numeric code
//! (algorithms, type-erased wrappers, math libraries) can introspect a
//! Minifloat just like `float` or `double`.
template <class Format> struct numeric_limits<::skymizer::minifloat::Minifloat<Format>> {
private:
  using T = ::skymizer::minifloat::Minifloat<Format>;

  // Bit pattern for 2^k. Saturates to zero when k is below the subnormal
  // range; saturates to max() when k overflows the exponent. Used to derive
  // epsilon() and round_error().
  static constexpr T pow2(int k) noexcept {
    const int biased = k + T::BIAS;
    if (biased >= 1)
      return T::from_bits(static_cast<typename T::Storage>(biased) << T::MANTISSA_BITS);
    if (biased + T::MANTISSA_BITS >= 1)
      return T::from_bits(typename T::Storage{1} << (biased + T::MANTISSA_BITS - 1));
    return T::from_bits(0);
  }

public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = T::HAS_INF;
  static constexpr bool has_quiet_NaN = T::HAS_NAN;
  static constexpr bool has_signaling_NaN = false;
  static constexpr float_denorm_style has_denorm =
      T::MANTISSA_BITS > 0 ? denorm_present : denorm_absent;
  static constexpr bool has_denorm_loss = false;
  static constexpr float_round_style round_style = round_to_nearest;
  //! Only the IEEE layer reserves both an infinity and a NaN
  static constexpr bool is_iec559 = T::HAS_INF && T::HAS_NAN;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int radix = 2;
  static constexpr int digits = T::MANTISSA_DIGITS;
  static constexpr int digits10 = (digits - 1) * 30103 / 100000;
  static constexpr int max_digits10 = digits * 30103 / 100000 + 2;
  static constexpr int min_exponent = T::MIN_EXP;
  static constexpr int max_exponent = T::MAX_EXP;
  static constexpr int min_exponent10 = (T::MIN_EXP - 1) * 30103 / 100000;
  static constexpr int max_exponent10 = T::MAX_EXP * 30103 / 100000;
  static constexpr bool traps = false;
  static constexpr bool tinyness_before = false;

  static constexpr T min() noexcept { return T::min(); }
  static constexpr T max() noexcept { return T::max(); }
  static constexpr T lowest() noexcept { return -T::max(); }
  static constexpr T denorm_min() noexcept { return T::true_min(); }
  static constexpr T epsilon() noexcept { return pow2(1 - digits); }
  static constexpr T round_error() noexcept { return pow2(-1); }
  static constexpr T infinity() noexcept { return T::infinity(); }
  static constexpr T quiet_NaN() noexcept { return T::quiet_NaN(); }
  static constexpr T signaling_NaN() noexcept { return T::quiet_NaN(); }
};
} // namespace std

#undef SKYMIZER_MINIFLOAT_CONST
#undef SKYMIZER_MINIFLOAT_PURE

#endif
