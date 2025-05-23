// This file is part of the minifloat project of Skymizer.
//
// Copyright (C) 2024-2025 Chen-Pang He <jdh8@skymizer.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SKYMIZER_MINIFLOAT_HPP
#define SKYMIZER_MINIFLOAT_HPP

#include <algorithm>
#include <limits>
#include <type_traits>
#include <cmath>
#include <cstdint>

/// Namespace for Skymizer
namespace skymizer {

/// Namespace for the minifloat library
namespace minifloat {

namespace detail {

/// Backport of C++20 std::bit_cast
template <typename To, typename From>
[[nodiscard, gnu::const]]
auto bit_cast(const From &from) noexcept -> std::enable_if_t<
  std::is_trivially_copyable<To>::value &&
  std::is_trivially_copyable<From>::value &&
  sizeof(To) == sizeof(From), To>
{
#if defined(__has_builtin) && __has_builtin(__builtin_bit_cast)
  return __builtin_bit_cast(To, from);
#else
  union { From _; To to; } caster = {from};
  return caster.to;
#endif
}

/// Round a float to the nearest float with the given significand width
///
/// \tparam M - Significand (mantissa) width
template <int M>
[[nodiscard, gnu::const]]
float round_to_mantissa(float x) {
  static_assert(std::numeric_limits<float>::radix == 2);
  static_assert(M < std::numeric_limits<float>::digits);
  static_assert(std::numeric_limits<float>::is_iec559);

  const auto bits = bit_cast<std::uint32_t>(x);
  const auto ulp = std::uint32_t{1} << (std::numeric_limits<float>::digits - 1 - M);
  const auto bias = ulp / 2 - !(bits & ulp);
  return bit_cast<float>((bits + bias) & ~(ulp - 1));
}

/// Round a float to the nearest float with the given significand width
///
/// \tparam M - Significand (mantissa) width
template <int M>
[[nodiscard, gnu::const]]
double round_to_mantissa(double x) {
  static_assert(std::numeric_limits<double>::radix == 2);
  static_assert(M < std::numeric_limits<double>::digits);
  static_assert(std::numeric_limits<double>::is_iec559);

  const auto bits = bit_cast<std::uint64_t>(x);
  const auto ulp = std::uint64_t{1} << (std::numeric_limits<double>::digits - 1 - M);
  const auto bias = ulp / 2 - !(bits & ulp);
  return bit_cast<double>((bits + bias) & ~(ulp - 1));
}

} // namespace detail

/// A fast native type borrowed for minifloat arithmetics
///
/// \tparam M - Significand (mantissa) width
///
/// We want to emulate addition and multiplication of floating point number
/// type T with a wider type U.  If the significand of U is more than twice as
/// wide as T, double rounding is not a problem.  The proof is left as an
/// exercise to the reader.
template <int M>
using BitTrueGroupArithmeticType = std::conditional_t<
  M <= std::numeric_limits<float>::digits / 2 - 1,
  float, double>;

/// Default bias for a given exponent width
///
/// \tparam E - Exponent width
template <int E>
using DefaultBias = std::integral_constant<int, (1 << (E - 1)) - 1>;

/// NaN encoding style
///
/// The variants follow [LLVM/MLIR naming conventions][llvm] derived from
/// their differences to [IEEE 754][ieee].
///
/// [llvm]: https://llvm.org/doxygen/structllvm_1_1APFloatBase.html
/// [ieee]: https://en.wikipedia.org/wiki/IEEE_754
enum struct NanStyle {
  /// IEEE 754 NaN encoding
  ///
  /// The maximum exponent is reserved for non-finite numbers.  The zero
  /// mantissa stands for infinity, while any other value represents a NaN.
  IEEE,

  /// `FN` suffix as in LLVM/MLIR
  ///
  /// `F` is for finite, `N` for a special NaN encoding.  There are no
  /// infinities.  The maximum magnitude is reserved for NaNs, where the
  /// exponent and mantissa are all ones.
  FN,

  /// `FNUZ` suffix as in LLVM/MLIR
  ///
  /// `F` is for finite, `N` for a special NaN encoding, `UZ` for unsigned
  /// zero.  There are no infinities.  The negative zero (&minus;0.0)
  /// representation is reserved for NaN.  As a result, there is only one
  /// (+0.0) unsigned zero.
  FNUZ,
};

/// Subnormal handling style
enum struct SubnormalStyle {
  /// IEEE 754 subnormal numbers
  ///
  /// Subnormal numbers have the smallest exponent (same as the smallest normal
  /// numbers) but without the implicit leading bit.  Subnormal numbers provide
  /// a smooth transition from zero to normal numbers.
  Precise,

  /// Do not use subnormal representations
  ///
  /// I (jdh8) do not recommend this option, but one of our users requested it.
  /// If you want to reserve floating-point representations, I strongly suggest
  /// using NaN boxing instead.
  Reserved,

  /// I am speed
  ///
  /// Subnormal numbers are valid representations, but they are not
  /// guaranteed to be precise.  This is useful for fast emulation of
  /// subnormal numbers.
  ///
  /// The fast representations still have a correct sign and a magnitude between
  /// zero and the smallest positive normal number.
  Fast,
};

/// Configurable signed floating point type
///
/// \tparam E - Exponent width
/// \tparam M - Significand (mantissa) width
/// \tparam N - NaN encoding style
/// \tparam B - Exponent bias
/// \tparam D - Subnormal (denormal) encoding style
///
/// Constraints:
/// - E > 0
/// - M >= 0 (M > 0 if N is `NanStyle::IEEE`)
/// - E + M < 16
template <int E, int M,
  NanStyle N = NanStyle::IEEE,
  int B = DefaultBias<E>::value,
  SubnormalStyle D = SubnormalStyle::Precise>
class Minifloat;

template <int EXPONENT_BITS, int MANTISSA_BITS, NanStyle NAN_STYLE, int BIAS, SubnormalStyle SUBNORMAL_STYLE>
class Minifloat {
public:
  static constexpr int E = EXPONENT_BITS;
  static constexpr int M = MANTISSA_BITS;
  static constexpr NanStyle N = NAN_STYLE;
  static constexpr int B = BIAS;
  static constexpr SubnormalStyle D = SUBNORMAL_STYLE;

  static_assert(E > 0);
  static_assert(M >= 0);
  static_assert(E + M < 16);

  using StorageType = std::conditional_t<E + M < 8, std::uint_least8_t, std::uint_least16_t>;
  static const int RADIX = 2;
  static const int MANTISSA_DIGITS = M + 1;
  static const int MAX_EXP = (1 << E) - B - int{N == NanStyle::IEEE};
  static const int MIN_EXP = 2 - B;
  static const StorageType ABS_MASK = (1U << (E + M)) - 1U;

  static const bool HAS_EXACT_F32_CONVERSION =
    std::numeric_limits<float>::radix == RADIX &&
    std::numeric_limits<float>::digits >= MANTISSA_DIGITS &&
    std::numeric_limits<float>::max_exponent >= MAX_EXP &&
    std::numeric_limits<float>::min_exponent <= MIN_EXP &&
    std::numeric_limits<float>::is_iec559;
  
  static const bool HAS_EXACT_F64_CONVERSION =
    std::numeric_limits<double>::radix == RADIX &&
    std::numeric_limits<double>::digits >= MANTISSA_DIGITS &&
    std::numeric_limits<double>::max_exponent >= MAX_EXP &&
    std::numeric_limits<double>::min_exponent <= MIN_EXP &&
    std::numeric_limits<double>::is_iec559;

private:
  StorageType bits_;

  [[nodiscard, gnu::const]]
  static constexpr StorageType inf_bits() noexcept {
    const StorageType max = (UINT32_C(1) << (E + M)) - 1;

    if constexpr (N == NanStyle::IEEE)
      return max << M & max;

    return max - (N == NanStyle::FN);
  }

  [[nodiscard, gnu::const]]
  static constexpr StorageType nan_bits() noexcept {
    const StorageType n0 = UINT32_C(1) << (E + M);
    const StorageType max = n0 - 1;

    if constexpr (N == NanStyle::FNUZ)
      return n0;

    if constexpr (N == NanStyle::FN || M == 0)
      return max;

    return max << (M - 1) & max;
  }

  [[nodiscard, gnu::const]]
  static StorageType to_bits(float x) noexcept {
    const auto bits = detail::bit_cast<std::uint32_t>(detail::round_to_mantissa<M>(x));
    const auto sign = bits >> 31 << (E + M);

    if (x != x) 
      return sign | nan_bits();

    const auto diff = std::int32_t{MIN_EXP - std::numeric_limits<float>::min_exponent} << M;
    const auto magnitude = static_cast<std::int32_t>(bits << 1 >> (std::numeric_limits<float>::digits - M)) - diff;

    if (magnitude < 1 << M) {
      if constexpr (D == SubnormalStyle::Fast)
        return magnitude <= 0 ? (N != NanStyle::FNUZ) * sign : sign | magnitude;

      if constexpr (D == SubnormalStyle::Reserved)
        return magnitude <= 1 << M >> 1 ? (N != NanStyle::FNUZ) * sign : sign | 1 << M;

      const StorageType ticks = std::rint(std::abs(x) * std::exp2(MANTISSA_DIGITS - MIN_EXP));
      return (N != NanStyle::FNUZ || ticks) * sign | ticks;
    }
    return sign | std::min<std::int32_t>(magnitude, inf_bits());
  }

  [[nodiscard, gnu::const]]
  static StorageType to_bits(double x) noexcept {
    const auto bits = detail::bit_cast<std::uint64_t>(detail::round_to_mantissa<M>(x));
    const auto sign = bits >> 63 << (E + M);

    if (x != x) 
      return sign | nan_bits();

    const auto diff = std::int64_t{MIN_EXP - std::numeric_limits<double>::min_exponent} << M;
    const auto magnitude = static_cast<std::int64_t>(bits << 1 >> (std::numeric_limits<double>::digits - M)) - diff;

    if (magnitude < 1 << M) {
      if constexpr (D == SubnormalStyle::Fast)
        return magnitude <= 0 ? (N != NanStyle::FNUZ) * sign : sign | magnitude;

      if constexpr (D == SubnormalStyle::Reserved)
        return magnitude <= 1 << M >> 1 ? (N != NanStyle::FNUZ) * sign : sign | 1 << M;

      const StorageType ticks = std::rint(std::abs(x) * std::exp2(MANTISSA_DIGITS - MIN_EXP));
      return (N != NanStyle::FNUZ || ticks) * sign | ticks;
    }
    return sign | std::min<std::int64_t>(magnitude, inf_bits());
  }

public:
  Minifloat() = default;
  explicit Minifloat(float x) : bits_(to_bits(x)) {};
  explicit Minifloat(double x) : bits_(to_bits(x)) {};

  static constexpr Minifloat from_bits(StorageType bits) noexcept {
    const unsigned mask = (1U << (E + M + 1)) - 1U;
    Minifloat result;
    result.bits_ = bits & mask;
    return result;
  }

  [[nodiscard, gnu::pure]] constexpr StorageType bits() const noexcept { return bits_; }
  [[nodiscard, gnu::pure]] constexpr bool sign() const noexcept { return bits_ >> (E + M); }

  [[nodiscard, gnu::pure]]
  constexpr bool isnan() const noexcept {
    if constexpr (N == NanStyle::FNUZ)
      return bits_ == ABS_MASK + 1U;

    if constexpr (N == NanStyle::FN)
      return (bits_ & ABS_MASK) == ABS_MASK;

    return (bits_ & ABS_MASK) > inf_bits();
  }

  [[nodiscard, gnu::pure]]
  constexpr bool signbit() const noexcept {
    return bits_ >> (E + M) & 1;
  }

  [[nodiscard, gnu::pure]]
  constexpr Minifloat abs() const noexcept {
    const StorageType magnitude = bits_ & ABS_MASK;
    if (N == NanStyle::FNUZ && !magnitude) return *this;
    return from_bits(magnitude);
  }

  /// Explicit conversion to float
  ///
  /// The lossy branch makes use of conversion to double.  Conversion to double is
  /// lossy only when then exponent width is too large.  In this case, a second
  /// conversion to float is safe.
  [[nodiscard, gnu::pure]]
  float to_float() const noexcept {
    if constexpr (!HAS_EXACT_F32_CONVERSION)
      return static_cast<double>(*this);

    const float sgn = sign() ? -1.0F : 1.0F;
    const std::uint32_t magnitude = bits_ & ABS_MASK;

    if (isnan())
      return std::copysign(NAN, sgn);

    if (N == NanStyle::IEEE && magnitude == inf_bits())
      return std::copysign(HUGE_VALF, sgn);

    if (D == SubnormalStyle::Precise && magnitude < 1 << M)
      return magnitude * std::copysign(std::exp2f(MIN_EXP - MANTISSA_DIGITS), sgn);

    const std::uint32_t shifted = magnitude << (std::numeric_limits<float>::digits - MANTISSA_DIGITS);
    const std::uint32_t diff = MIN_EXP - std::numeric_limits<float>::min_exponent;
    const std::uint32_t bias = diff << (std::numeric_limits<float>::digits - 1);
    return detail::bit_cast<float>(sign() << 31 | (shifted + bias));
  }

  [[nodiscard, gnu::pure]]
  explicit operator float() const noexcept { return to_float(); }

  /// Implicit lossless conversion to double
  ///
  /// The conversion is only enabled if it is proven to be lossless at compile
  /// time.  If the conversion is lossy, the user must explicitly cast to double.
  [[nodiscard, gnu::pure]]
  std::enable_if_t<HAS_EXACT_F64_CONVERSION, double> to_double() const noexcept {
    const double sgn = sign() ? -1.0 : 1.0;
    const std::uint64_t magnitude = bits_ & ABS_MASK;

    if (isnan())
      return std::copysign(NAN, sgn);

    if (N == NanStyle::IEEE && magnitude == inf_bits())
      return std::copysign(HUGE_VAL, sgn);

    if (D == SubnormalStyle::Precise && magnitude < 1 << M)
      return magnitude * std::copysign(std::exp2(MIN_EXP - MANTISSA_DIGITS), sgn);

    const std::uint64_t shifted = magnitude << (std::numeric_limits<double>::digits - MANTISSA_DIGITS);
    const std::uint64_t diff = MIN_EXP - std::numeric_limits<double>::min_exponent;
    const std::uint64_t bias = diff << (std::numeric_limits<double>::digits - 1);
    return detail::bit_cast<double>(std::uint64_t{sign()} << 63 | (shifted + bias));
  }
  
  /// Explicit lossy conversion to double
  ///
  /// This variant assumes that the conversion is lossy only when the exponent
  /// is out of range.
  template <bool INEXACT = !HAS_EXACT_F64_CONVERSION>
  [[nodiscard, gnu::pure]]
  std::enable_if_t<INEXACT, double> to_double() const noexcept {
    static_assert(std::numeric_limits<double>::radix == RADIX);
    static_assert(std::numeric_limits<double>::digits >= MANTISSA_DIGITS);
    static_assert(std::numeric_limits<double>::is_iec559);

    const double sgn = sign() ? -1.0 : 1.0;
    const std::uint64_t magnitude = bits_ & ABS_MASK;

    if (isnan())
      return std::copysign(NAN, sgn);

    if (N == NanStyle::IEEE && magnitude == inf_bits())
      return std::copysign(HUGE_VAL, sgn);

    if (magnitude >= static_cast<std::uint64_t>(std::numeric_limits<double>::max_exponent + B) << M)
      return std::copysign(HUGE_VAL, sgn);

    if (D == SubnormalStyle::Precise && magnitude < 1 << M)
      return std::copysign(std::ldexp(magnitude, MIN_EXP - MANTISSA_DIGITS), sgn);

    if (static_cast<int>(magnitude >> M) < std::numeric_limits<double>::min_exponent + B) {
      const std::uint64_t significand = (magnitude & ((1U << M) - 1)) | 1U << M;
      const int exponent = static_cast<int>(magnitude >> M) - B;
      return std::copysign(std::ldexp(significand, exponent - M), sgn);
    }

    const std::uint64_t shifted = magnitude << (std::numeric_limits<double>::digits - (E + M));
    const std::uint64_t diff = MIN_EXP - std::numeric_limits<double>::min_exponent;
    const std::uint64_t bias = diff << (std::numeric_limits<double>::digits - 1);
    return detail::bit_cast<double>(std::uint64_t{sign()} << 63 | (shifted + bias));
  }

  [[nodiscard, gnu::pure]]
  explicit operator double() const noexcept { return to_double(); }
};

namespace detail {
template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
constexpr bool are_different_zeroes(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  const auto a = x.bits();
  const auto b = y.bits();

  if constexpr (N == NanStyle::FNUZ)
    return false;

  return ((a | b) & Minifloat<E, M, N, B, D>::ABS_MASK) == 0;
}
} // namespace detail

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator==(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  return (x.bits() == y.bits() && !x.isnan()) || detail::are_different_zeroes(x, y);
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator!=(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  return !(x == y);
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator<(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  const auto a = x.bits();
  const auto b = y.bits();
  const bool sign = (a | b) >> (E + M) & 1;

  if (x.isnan() || y.isnan() || detail::are_different_zeroes(x, y))
    return false;

  return sign ? a > b : a < b;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator<=(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  const auto a = x.bits();
  const auto b = y.bits();
  const bool sign = (a | b) >> (E + M) & 1;

  if (x.isnan() || y.isnan())
    return false;

  if (detail::are_different_zeroes(x, y))
    return true;

  return sign ? a >= b : a <= b;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator>(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  return y < x;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator>=(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  return y <= x;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
constexpr Minifloat<E, M, N, B, D> operator+(Minifloat<E, M, N, B, D> x) noexcept {
  return x;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
constexpr Minifloat<E, M, N, B, D> operator-(Minifloat<E, M, N, B, D> x) noexcept {
  constexpr auto ABS_MASK = Minifloat<E, M, N, B, D>::ABS_MASK;
  if (N == NanStyle::FNUZ && (x.bits() & ABS_MASK) == 0) return x;
  return Minifloat<E, M, N, B, D>::from_bits(x.bits() ^ (ABS_MASK + 1));
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
BitTrueGroupArithmeticType<M> operator+(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  BitTrueGroupArithmeticType<M> a(x);
  BitTrueGroupArithmeticType<M> b(y);
  return a + b;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
BitTrueGroupArithmeticType<M> operator-(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  BitTrueGroupArithmeticType<M> a(x);
  BitTrueGroupArithmeticType<M> b(y);
  return a - b;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
BitTrueGroupArithmeticType<M> operator*(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  BitTrueGroupArithmeticType<M> a(x);
  BitTrueGroupArithmeticType<M> b(y);
  return a * b;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
double operator/(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  return x.to_double() / y.to_double();
}

#define SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, MANT) \
  using E##EXP##M##MANT = Minifloat<EXP, MANT>; \
  using E##EXP##M##MANT##FN = Minifloat<EXP, MANT, NanStyle::FN>; \
  using E##EXP##M##MANT##FNUZ = Minifloat<EXP, MANT, NanStyle::FNUZ>;

#define SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(EXP) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 0) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 1) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 2) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 3) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 4) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 5) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 6) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 7) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 8) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 9) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 10) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 11) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 12) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 13) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 14) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 15)

SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(1)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(2)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(3)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(4)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(5)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(6)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(7)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(8)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(9)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(10)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(11)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(12)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(13)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(14)
SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(15)

} // namespace minifloat

using minifloat::Minifloat;

} // namespace skymizer

#endif