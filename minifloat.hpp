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
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#if __cplusplus >= 202002L
#include <bit>
#else
#include <cstring>
#endif

/// Namespace for Skymizer
namespace skymizer {

/// Namespace for the minifloat library
namespace minifloat {

/// Backport of C++20 std::bit_cast
template <typename To, typename From>
[[nodiscard, gnu::const]]
To bit_cast(const From &from) {
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

template <int M>
[[nodiscard, gnu::const]]
float round_normal_float_to_mantissa(float x) {
  static_assert(M < FLT_MANT_DIG);
  static_assert(std::numeric_limits<float>::radix == 2);
  static_assert(std::numeric_limits<float>::is_iec559);

  const auto bits = bit_cast<std::uint32_t>(x);
  const auto ulp = std::uint32_t{1} << (FLT_MANT_DIG - 1 - M);
  const auto bias = ulp / 2 - !(bits & ulp);
  return bit_cast<float>((bits + bias) & ~(ulp - 1));
}

template <int M>
[[nodiscard, gnu::const]]
double round_normal_double_to_mantissa(double x) {
  static_assert(M < DBL_MANT_DIG);
  static_assert(std::numeric_limits<double>::radix == 2);
  static_assert(std::numeric_limits<double>::is_iec559);

  const auto bits = bit_cast<std::uint64_t>(x);
  const auto ulp = std::uint64_t{1} << (DBL_MANT_DIG - 1 - M);
  const auto bias = ulp / 2 - !(bits & ulp);
  return bit_cast<double>((bits + bias) & ~(ulp - 1));
}

constexpr int default_bias(int exponent_width) { return (1 << (exponent_width - 1)) - 1; }

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

/// FP classification helper for `Minifloat`
template <NanStyle> struct FpClassifier;

/// Configurable signed floating point type
///
/// \tparam E - Exponent width
/// \tparam M - Mantissa (significand) width
/// \tparam N - NaN encoding style
/// \tparam B - Exponent bias
/// \tparam D - Subnormal (denormal) encoding style
///
/// Constraints:
/// - E + M < 16
/// - E >= 2
/// - M >= 0 (M > 0 if N is `NanStyle::IEEE`) (∞ ≠ NaN)
/// - D = `SubnormalStyle::Precise` if M = 0
template <
    int E, int M, NanStyle N = NanStyle::IEEE, int B = default_bias(E),
    SubnormalStyle D = SubnormalStyle::Precise>
class Minifloat {
  friend FpClassifier<N>;

public:
  static constexpr int EXPONENT_BITS = E;
  static constexpr int MANTISSA_BITS = M;
  static constexpr NanStyle NAN_STYLE = N;
  static constexpr int BIAS = B;
  static constexpr SubnormalStyle SUBNORMAL_STYLE = D;

  static_assert(E + M < 16);
  static_assert(E >= 2);
  static_assert(M >= 0);
  static_assert(M > 0 || N != NanStyle::IEEE);
  static_assert(M > 0 || D == SubnormalStyle::Precise);

  using Storage = std::conditional_t<(E + M < 8), std::uint_least8_t, std::uint_least16_t>;
  static constexpr int RADIX = 2;
  static constexpr int MANTISSA_DIGITS = M + 1;
  static constexpr int MAX_EXP = (1 << E) - B - int{N == NanStyle::IEEE};
  static constexpr int MIN_EXP = 2 - B;
  static constexpr Storage ABS_MASK = (1U << (E + M)) - 1U;

  static constexpr bool HAS_EXACT_F32_CONVERSION =
      FLT_MANT_DIG >= MANTISSA_DIGITS && FLT_MAX_EXP >= MAX_EXP && FLT_MIN_EXP <= MIN_EXP &&
      std::numeric_limits<float>::radix == RADIX && std::numeric_limits<float>::is_iec559;

  static constexpr bool HAS_EXACT_F64_CONVERSION =
      DBL_MANT_DIG >= MANTISSA_DIGITS && DBL_MAX_EXP >= MAX_EXP && DBL_MIN_EXP <= MIN_EXP &&
      std::numeric_limits<double>::radix == RADIX && std::numeric_limits<double>::is_iec559;

  static constexpr bool USE_FLT_ADD = FLT_MANT_DIG >= 2 * MANTISSA_DIGITS && //
                                      (FLT_MAX_EXP > MAX_EXP) &&             //
                                      (FLT_MIN_EXP < MIN_EXP);

  static constexpr bool USE_FLT_MUL = FLT_MANT_DIG >= 2 * MANTISSA_DIGITS &&
                                      FLT_MAX_EXP >= 2 * MAX_EXP &&
                                      FLT_MIN_EXP - 1 <= 2 * (MIN_EXP - 1);

private:
  Storage bits_;

  static constexpr Storage HUGE_REPR = [] {
    const Storage max = (UINT32_C(1) << (E + M)) - 1;

    if constexpr (N == NanStyle::IEEE)
      return max << M & max;

    return max - (N == NanStyle::FN);
  }();

  static constexpr Storage NAN_REPR = []() -> Storage {
    const Storage signbit = UINT32_C(1) << (E + M);
    const Storage max = signbit - 1;

    if constexpr (N == NanStyle::FNUZ)
      return signbit;

    if constexpr (N == NanStyle::IEEE)
      return max << (M - 1) & max;

    return max;
  }();

  [[nodiscard, gnu::const]]
  static Storage bits_from_float(float x) {
    const auto bits = bit_cast<std::uint32_t>(round_normal_float_to_mantissa<M>(x));
    const auto sign = bits >> 31 << (E + M);

    if (x != x)
      return sign | NAN_REPR;

    const auto diff = std::int32_t{MIN_EXP - FLT_MIN_EXP} << M;
    const auto magnitude = static_cast<std::int32_t>(bits << 1 >> (FLT_MANT_DIG - M)) - diff;

    if (magnitude < 1 << M) {
      if constexpr (D == SubnormalStyle::Fast)
        return magnitude <= 0 ? (N != NanStyle::FNUZ) * sign : sign | magnitude;

      if constexpr (D == SubnormalStyle::Reserved)
        return magnitude <= 1 << M >> 1 ? (N != NanStyle::FNUZ) * sign : sign | 1 << M;

      const Storage ticks = std::rint(std::abs(x) * std::exp2(MANTISSA_DIGITS - MIN_EXP));
      return (N != NanStyle::FNUZ || ticks) * sign | ticks;
    }
    return sign | std::min<std::int32_t>(magnitude, HUGE_REPR);
  }

  [[nodiscard, gnu::const]]
  static Storage bits_from_double(double x) {
    const auto bits = bit_cast<std::uint64_t>(round_normal_double_to_mantissa<M>(x));
    const auto sign = bits >> 63 << (E + M);

    if (x != x)
      return sign | NAN_REPR;

    const auto diff = std::int64_t{MIN_EXP - DBL_MIN_EXP} << M;
    const auto magnitude = static_cast<std::int64_t>(bits << 1 >> (DBL_MANT_DIG - M)) - diff;

    if (magnitude < 1 << M) {
      if constexpr (D == SubnormalStyle::Fast)
        return magnitude <= 0 ? (N != NanStyle::FNUZ) * sign : sign | magnitude;

      if constexpr (D == SubnormalStyle::Reserved)
        return magnitude <= 1 << M >> 1 ? (N != NanStyle::FNUZ) * sign : sign | 1 << M;

      const Storage ticks = std::rint(std::abs(x) * std::exp2(MANTISSA_DIGITS - MIN_EXP));
      return (N != NanStyle::FNUZ || ticks) * sign | ticks;
    }
    return sign | std::min<std::int64_t>(magnitude, HUGE_REPR);
  }

public:
  Minifloat() = default;
  explicit Minifloat(float x) : bits_(bits_from_float(x)) {};
  explicit Minifloat(double x) : bits_(bits_from_double(x)) {};

  static constexpr Minifloat from_bits(Storage bits) {
    const unsigned mask = (1U << (E + M + 1)) - 1U;
    Minifloat result;
    result.bits_ = bits & mask;
    return result;
  }

  [[nodiscard, gnu::const]] static Minifloat from_float(float x) { return Minifloat{x}; }
  [[nodiscard, gnu::const]] static Minifloat from_double(double x) { return Minifloat{x}; }

  [[nodiscard, gnu::pure]] constexpr Storage to_bits() const { return bits_; }
  [[nodiscard, gnu::pure]] constexpr bool signbit() const { return bits_ >> (E + M) & 1; }

  /// Check if the number is nonzero
  [[nodiscard, gnu::pure]]
  constexpr explicit operator bool() const {
    if constexpr (N == NanStyle::FNUZ)
      return bits_ != 0;

    return (bits_ & ABS_MASK) != 0;
  }

  [[nodiscard, gnu::pure]]
  constexpr bool is_nan() const {
    if constexpr (N == NanStyle::FNUZ)
      return bits_ == ABS_MASK + 1U;

    if constexpr (N == NanStyle::FN)
      return (bits_ & ABS_MASK) == ABS_MASK;

    return (bits_ & ABS_MASK) > HUGE_REPR;
  }

  [[nodiscard, gnu::pure]]
  constexpr bool is_infinite() const {
    return N == NanStyle::IEEE && (bits_ & ABS_MASK) == HUGE_REPR;
  }

  [[nodiscard, gnu::pure]]
  constexpr bool is_finite() const {
    if constexpr (N == NanStyle::IEEE)
      return (bits_ & ABS_MASK) < HUGE_REPR;

    return !is_nan();
  }

  [[nodiscard, gnu::pure]]
  constexpr bool is_normal() const {
    return is_finite() && (bits_ & ABS_MASK) >= (1U << M);
  }

  /// Check if the number is nonzero subnormal
  [[nodiscard, gnu::pure]]
  constexpr bool is_subnormal() const {
    const Storage abs_bits = bits_ & ABS_MASK;
    return 0 < abs_bits && abs_bits < (1U << M);
  }

  [[nodiscard, gnu::pure]]
  constexpr int classify() const {
    return FpClassifier<N>::classify(*this);
  }

  [[nodiscard, gnu::pure]]
  constexpr Minifloat abs() const {
    const Storage magnitude = bits_ & ABS_MASK;

    if (N == NanStyle::FNUZ && !magnitude)
      return *this;

    return from_bits(magnitude);
  }

  /// Explicit conversion to float
  ///
  /// The lossy branch makes use of conversion to double.  Conversion to double
  /// is lossy only when then exponent width is too large.  In this case, a
  /// second conversion to float is safe.
  [[nodiscard, gnu::pure]]
  float to_float() const {
    if constexpr (!HAS_EXACT_F32_CONVERSION)
      return to_double();

    const float sign = signbit() ? -1.0F : 1.0F;
    const std::uint32_t magnitude = bits_ & ABS_MASK;

    if (is_nan())
      return std::copysign(NAN, sign);

    if (N == NanStyle::IEEE && magnitude == HUGE_REPR)
      return std::copysign(HUGE_VALF, sign);

    if (D == SubnormalStyle::Precise && magnitude < 1 << M)
      return magnitude * std::copysign(std::exp2f(MIN_EXP - MANTISSA_DIGITS), sign);

    const std::uint32_t shifted = magnitude << (FLT_MANT_DIG - MANTISSA_DIGITS);
    const std::uint32_t diff = MIN_EXP - FLT_MIN_EXP;
    const std::uint32_t bias = diff << (FLT_MANT_DIG - 1);
    return bit_cast<float>(signbit() << 31 | (shifted + bias));
  }

  [[nodiscard, gnu::pure]]
  explicit operator float() const {
    return to_float();
  }

  /// Implicit lossless conversion to double
  ///
  /// The conversion is only enabled if it is proven to be lossless at compile
  /// time.  If the conversion is lossy, the user must explicitly cast to
  /// double.
  [[nodiscard, gnu::pure]]
  std::enable_if_t<HAS_EXACT_F64_CONVERSION, double> to_double() const {
    const double sign = signbit() ? -1.0 : 1.0;
    const std::uint64_t magnitude = bits_ & ABS_MASK;

    if (is_nan())
      return std::copysign(NAN, sign);

    if (N == NanStyle::IEEE && magnitude == HUGE_REPR)
      return std::copysign(HUGE_VAL, sign);

    if (D == SubnormalStyle::Precise && magnitude < 1 << M)
      return magnitude * std::copysign(std::exp2(MIN_EXP - MANTISSA_DIGITS), sign);

    const std::uint64_t shifted = magnitude << (DBL_MANT_DIG - MANTISSA_DIGITS);
    const std::uint64_t diff = MIN_EXP - DBL_MIN_EXP;
    const std::uint64_t bias = diff << (DBL_MANT_DIG - 1);
    return bit_cast<double>(std::uint64_t{signbit()} << 63 | (shifted + bias));
  }

  /// Explicit lossy conversion to double
  ///
  /// This variant assumes that the conversion is lossy only when the exponent
  /// is out of range.
  template <bool INEXACT = !HAS_EXACT_F64_CONVERSION>
  [[nodiscard, gnu::pure]]
  std::enable_if_t<INEXACT, double> to_double() const {
    static_assert(DBL_MANT_DIG >= MANTISSA_DIGITS);
    static_assert(std::numeric_limits<double>::radix == RADIX);
    static_assert(std::numeric_limits<double>::is_iec559);

    const double sign = signbit() ? -1.0 : 1.0;
    const std::uint64_t magnitude = bits_ & ABS_MASK;

    if (is_nan())
      return std::copysign(NAN, sign);

    if (N == NanStyle::IEEE && magnitude == HUGE_REPR)
      return std::copysign(HUGE_VAL, sign);

    if (magnitude >= static_cast<std::uint64_t>(DBL_MAX_EXP + B) << M)
      return std::copysign(HUGE_VAL, sign);

    if (D == SubnormalStyle::Precise && magnitude < 1 << M)
      return std::copysign(std::ldexp(magnitude, MIN_EXP - MANTISSA_DIGITS), sign);

    if (static_cast<int>(magnitude >> M) < DBL_MIN_EXP + B) {
      const std::uint64_t significand = (magnitude & ((1U << M) - 1)) | 1U << M;
      const int exponent = static_cast<int>(magnitude >> M) - B;
      return std::copysign(std::ldexp(significand, exponent - M), sign);
    }

    const std::uint64_t shifted = magnitude << (DBL_MANT_DIG - (E + M));
    const std::uint64_t diff = MIN_EXP - DBL_MIN_EXP;
    const std::uint64_t bias = diff << (DBL_MANT_DIG - 1);
    return bit_cast<double>(std::uint64_t{signbit()} << 63 | (shifted + bias));
  }

  [[nodiscard, gnu::pure]]
  explicit operator double() const {
    return to_double();
  }
};

template <> struct FpClassifier<NanStyle::IEEE> {
  template <int E, int M, int B, SubnormalStyle D>
  static constexpr int classify(Minifloat<E, M, NanStyle::IEEE, B, D> x) {
    const decltype(x.ABS_MASK) bits = x.to_bits() & x.ABS_MASK;

    if (bits > x.HUGE_REPR)
      return FP_NAN;

    if (bits == x.HUGE_REPR)
      return FP_INFINITE;

    if (bits >= 1U << M)
      return FP_NORMAL;

    if (bits > 0)
      return FP_SUBNORMAL;

    return FP_ZERO;
  }
};

template <> struct FpClassifier<NanStyle::FN> {
  template <int E, int M, int B, SubnormalStyle D>
  static constexpr int classify(Minifloat<E, M, NanStyle::FN, B, D> x) {
    static_assert(x.NAN_REPR == x.ABS_MASK);
    const decltype(x.ABS_MASK) bits = x.to_bits() & x.ABS_MASK;

    if (bits == x.ABS_MASK)
      return FP_NAN;

    if (bits >= 1U << M)
      return FP_NORMAL;

    if (bits > 0)
      return FP_SUBNORMAL;

    return FP_ZERO;
  }
};

template <> struct FpClassifier<NanStyle::FNUZ> {
  template <int E, int M, int B, SubnormalStyle D>
  static constexpr int classify(Minifloat<E, M, NanStyle::FNUZ, B, D> x) {
    static_assert(x.NAN_REPR == x.ABS_MASK + 1U);
    const decltype(x.ABS_MASK) bits = x.to_bits() & x.ABS_MASK;

    if (x.is_nan())
      return FP_NAN;

    if (bits >= 1U << M)
      return FP_NORMAL;

    if (bits > 0)
      return FP_SUBNORMAL;

    return FP_ZERO;
  }
};

namespace detail {
template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
constexpr bool are_different_zeroes(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  const auto a = x.to_bits();
  const auto b = y.to_bits();

  if constexpr (N == NanStyle::FNUZ)
    return false;

  return ((a | b) & Minifloat<E, M, N, B, D>::ABS_MASK) == 0;
}
} // namespace detail

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator==(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  return (x.to_bits() == y.to_bits() && !x.is_nan()) || detail::are_different_zeroes(x, y);
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator!=(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  return !(x == y);
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator<(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  const auto a = x.to_bits();
  const auto b = y.to_bits();
  const bool sign = (a | b) >> (E + M) & 1;

  if (x.is_nan() || y.is_nan() || detail::are_different_zeroes(x, y))
    return false;

  return sign ? a > b : a < b;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator<=(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  const auto a = x.to_bits();
  const auto b = y.to_bits();
  const bool sign = (a | b) >> (E + M) & 1;

  if (x.is_nan() || y.is_nan())
    return false;

  if (detail::are_different_zeroes(x, y))
    return true;

  return sign ? a >= b : a <= b;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator>(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  return y < x;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator>=(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  return y <= x;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
constexpr Minifloat<E, M, N, B, D> operator+(Minifloat<E, M, N, B, D> x) {
  return x;
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
constexpr Minifloat<E, M, N, B, D> operator-(Minifloat<E, M, N, B, D> x) {
  constexpr auto ABS_MASK = Minifloat<E, M, N, B, D>::ABS_MASK;
  if (N == NanStyle::FNUZ && (x.to_bits() & ABS_MASK) == 0)
    return x;

  return Minifloat<E, M, N, B, D>::from_bits(x.to_bits() ^ (ABS_MASK + 1));
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
Minifloat<E, M, N, B, D> operator+(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  if constexpr (Minifloat<E, M, N, B, D>::USE_FLT_ADD)
    return Minifloat<E, M, N, B, D>{x.to_float() + y.to_float()};

  return Minifloat<E, M, N, B, D>{x.to_double() + y.to_double()};
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
Minifloat<E, M, N, B, D> operator-(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  if constexpr (Minifloat<E, M, N, B, D>::USE_FLT_ADD)
    return Minifloat<E, M, N, B, D>{x.to_float() - y.to_float()};

  return Minifloat<E, M, N, B, D>{x.to_double() - y.to_double()};
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
Minifloat<E, M, N, B, D> operator*(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  if constexpr (Minifloat<E, M, N, B, D>::USE_FLT_MUL)
    return Minifloat<E, M, N, B, D>{x.to_float() * y.to_float()};

  return Minifloat<E, M, N, B, D>{x.to_double() * y.to_double()};
}

template <int E, int M, NanStyle N, int B, SubnormalStyle D>
[[gnu::const]]
Minifloat<E, M, N, B, D> operator/(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) {
  return Minifloat<E, M, N, B, D>{x.to_double() / y.to_double()};
}

#define SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, MANT)                                                     \
  using E##EXP##M##MANT = Minifloat<EXP, MANT>;                                                    \
  using E##EXP##M##MANT##FN = Minifloat<EXP, MANT, NanStyle::FN>;                                  \
  using E##EXP##M##MANT##FNUZ = Minifloat<EXP, MANT, NanStyle::FNUZ>;

#define SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(EXP)                                                     \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 0)                                                              \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 1)                                                              \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 2)                                                              \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 3)                                                              \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 4)                                                              \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 5)                                                              \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 6)                                                              \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 7)                                                              \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 8)                                                              \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 9)                                                              \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 10)                                                             \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 11)                                                             \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 12)                                                             \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 13)                                                             \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 14)                                                             \
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