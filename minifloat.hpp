//===- minifloat.hpp ------------------------------------------------------===//
//
// This file is part of the minifloat project of Skymizer.
//
// This file is distributed under the New BSD License.
// See LICENSE for details.
//
//===----------------------------------------------------------------------===//
#ifndef SKYMIZER_MINIFLOAT_HPP
#define SKYMIZER_MINIFLOAT_HPP

#include <algorithm>
#include <limits>
#include <type_traits>
#include <cmath>
#include <cstdint>

namespace skymizer {

namespace minifloat {

namespace detail {
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

/** \brief Round a float to the nearest float with the given significand width
  * \tparam M - Significand (mantissa) width
  */
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
} // namespace detail

/** \brief A fast native type borrowed for minifloat arithmetics
  * \tparam M - Significand (mantissa) width
  *
  * We want to emulate addition and multiplication of floating point number
  * type T with a wider type U.  If the significand of U is more than twice as
  * wide as T, double rounding is not a problem.  The proof is left as an
  * exercise to the reader.
  */
template <int M>
using BitTrueGroupArithmeticType = std::conditional_t<
  M <= std::numeric_limits<float>::digits / 2 - 1,
  float, double>;

/** \brief Default bias for a given exponent width
  * \tparam E - Exponent width
  */
template <int E>
using DefaultBias = std::integral_constant<int, (1 << (E - 1)) - 1>;

enum struct NaNStyle {
  IEEE, ///< IEEE 754 NaNs and infinities
  FN,   ///< The NaNs have all magnitude (non-sign) bits set
  FNUZ, ///< The NaN is -0
};

enum struct SubnormalStyle {
  Precise,  ///< IEEE 754 subnormal numbers
  Reserved, ///< Do not use subnormal representations
  Fast,     ///< I am speed
};

/** \brief Configurable signed floating point type
  * \tparam E - Exponent width
  * \tparam M - Significand (mantissa) width
  * \tparam N - NaN encoding style
  * \tparam B - Exponent bias
  * \tparam D - Subnormal (denormal) encoding style
  *
  * Constraints:
  * - E > 0
  * - M >= 0
  * - E + M < 16
  */
template <int E, int M,
  NaNStyle N = NaNStyle::IEEE,
  int B = DefaultBias<E>::value,
  SubnormalStyle D = SubnormalStyle::Precise>
class Minifloat {
public:
  static_assert(E > 0);
  static_assert(M >= 0);
  static_assert(E + M < 16);
  typedef std::conditional_t<E + M < 8, std::uint_least8_t, std::uint_least16_t> StorageType;

  static const int RADIX = 2;
  static const int MANTISSA_DIGITS = M + 1;
  static const int MAX_EXP = (1 << E) - B - (N == NaNStyle::IEEE);
  static const int MIN_EXP = 2 - B;

private:
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

  static const StorageType ABS_MASK = (1U << (E + M)) - 1U;
  StorageType _bits;

  [[nodiscard, gnu::const]]
  static constexpr StorageType _inf_bits() noexcept {
    const StorageType MAX = (UINT32_C(1) << (E + M)) - 1;

    if constexpr (N == NaNStyle::IEEE)
      return MAX << M & MAX;

    return MAX - (N == NaNStyle::FN);
  }

  [[nodiscard, gnu::const]]
  static constexpr StorageType _nan_bits() noexcept {
    const StorageType N0 = UINT32_C(1) << (E + M);
    const StorageType MAX = N0 - 1;

    if constexpr (N == NaNStyle::FNUZ)
      return N0;

    if constexpr (N == NaNStyle::FN || M == 0)
      return MAX;

    return MAX << (M - 1) & MAX;
  }

  [[nodiscard, gnu::const]]
  static StorageType _to_bits(float x) noexcept {
    const auto bits = detail::bit_cast<std::uint32_t>(detail::round_to_mantissa<M>(x));
    const auto sign = bits >> 31 << (E + M);

    if (x != x) 
      return sign | _nan_bits();

    const std::int32_t diff = MIN_EXP - std::numeric_limits<float>::min_exponent;
    const std::int32_t bias = diff << (std::numeric_limits<float>::digits - 1);
    const std::int32_t magnitude = (bits & INT32_MAX) - bias;

    if (magnitude < std::int32_t{1} << (std::numeric_limits<float>::digits - 1)) {
      if constexpr (D == SubnormalStyle::Precise) {
        const StorageType ticks = std::rint(std::abs(x) * std::exp2f(MANTISSA_DIGITS) * std::exp2f(-MIN_EXP));
        return (N != NaNStyle::FNUZ || ticks) * sign | ticks;
      }
      if (magnitude <= std::int32_t{1} << (std::numeric_limits<float>::digits - 2))
        return (N != NaNStyle::FNUZ) * sign;
      return sign | 1 << M;
    }
    const int shift = std::numeric_limits<float>::digits - MANTISSA_DIGITS;
    return sign | std::min<std::int32_t>(magnitude >> shift, _inf_bits());
  }

public:
  Minifloat() = default;
  explicit Minifloat(float x) : _bits(_to_bits(x)) {};

  static constexpr Minifloat from_bits(StorageType bits) noexcept {
    Minifloat result;
    result._bits = bits;
    return result;
  }

  [[nodiscard, gnu::const]] StorageType bits() const noexcept { return _bits; }
  [[nodiscard, gnu::const]] bool sign() const noexcept { return _bits >> (E + M); }

  [[nodiscard, gnu::const]]
  constexpr bool isnan() const noexcept {
    if constexpr (N == NaNStyle::FNUZ)
      return _bits == ABS_MASK + 1U;

    if constexpr (N == NaNStyle::FN)
      return (_bits & ABS_MASK) == ABS_MASK;

    return (_bits & ABS_MASK) > _inf_bits();
  }

  [[nodiscard, gnu::const]]
  constexpr bool signbit() const noexcept {
    return _bits >> (E + M) & 1;
  }

  [[nodiscard, gnu::const]]
  constexpr Minifloat abs() const noexcept {
    return from_bits(_bits & ABS_MASK);
  }

  /** \brief Implicit lossless conversion to float
    *
    * The conversion is only enabled if it is proven to be lossless at compile
    * time.  If the conversion is lossy, the user must explicitly cast to float.
    */
  template <bool ENABLE = HAS_EXACT_F32_CONVERSION>
  [[nodiscard, gnu::const]]
  operator std::enable_if_t<ENABLE, float>() const noexcept {
    const float sgn = sign() ? -1.0f : 1.0f;
    const std::uint32_t magnitude = _bits & ABS_MASK;

    if (isnan())
      return std::copysign(NAN, sgn);

    if (N == NaNStyle::IEEE && magnitude == _inf_bits())
      return std::copysign(HUGE_VALF, sgn);

    if (D == SubnormalStyle::Precise && magnitude < 1 << M)
      return magnitude * std::copysign(std::exp2f(MIN_EXP - MANTISSA_DIGITS), sgn);

    const std::uint32_t shifted = magnitude << (std::numeric_limits<float>::digits - MANTISSA_DIGITS);
    const std::uint32_t diff = MIN_EXP - std::numeric_limits<float>::min_exponent;
    const std::uint32_t bias = diff << (std::numeric_limits<float>::digits - 1);
    return detail::bit_cast<float>(sign() << 31 | (shifted + bias));
  }

  /** \brief Explicit lossy conversion to float
    *
    * This variant makes use of conversion to double.  Conversion to double is
    * lossy only when then exponent width is too large.  In this case, a second
    * conversion to float is safe.
    */
  template <bool ENABLE = !HAS_EXACT_F32_CONVERSION, std::enable_if_t<ENABLE> * = nullptr>
  [[nodiscard, gnu::const]]
  explicit operator float() const noexcept {
    return static_cast<double>(*this);
  }

  /** \brief Implicit lossless conversion to double
    *
    * The conversion is only enabled if it is proven to be lossless at compile
    * time.  If the conversion is lossy, the user must explicitly cast to double.
    */
  template <bool ENABLE = HAS_EXACT_F64_CONVERSION>
  [[nodiscard, gnu::const]]
  operator std::enable_if_t<ENABLE, double>() const noexcept {
    const double sgn = sign() ? -1.0 : 1.0;
    const std::uint64_t magnitude = _bits & ABS_MASK;

    if (isnan())
      return std::copysign(NAN, sgn);

    if (N == NaNStyle::IEEE && magnitude == _inf_bits())
      return std::copysign(HUGE_VAL, sgn);

    if (D == SubnormalStyle::Precise && magnitude < 1 << M)
      return magnitude * std::copysign(std::exp2(MIN_EXP - MANTISSA_DIGITS), sgn);

    const std::uint64_t shifted = magnitude << (std::numeric_limits<double>::digits - MANTISSA_DIGITS);
    const std::uint64_t diff = MIN_EXP - std::numeric_limits<double>::min_exponent;
    const std::uint64_t bias = diff << (std::numeric_limits<double>::digits - 1);
    return detail::bit_cast<double>(std::uint64_t{sign()} << 63 | (shifted + bias));
  }
  
  /** \brief Explicit lossy conversion to double
    *
    * This variant assumes that the conversion is lossy only when the exponent
    * is out of range.
    */
  template <bool ENABLE = !HAS_EXACT_F64_CONVERSION, std::enable_if_t<ENABLE> * = nullptr>
  [[nodiscard, gnu::const]]
  explicit operator double() const noexcept {
    static_assert(std::numeric_limits<double>::radix == RADIX);
    static_assert(std::numeric_limits<double>::digits >= MANTISSA_DIGITS);
    static_assert(std::numeric_limits<double>::is_iec559);

    const double sgn = sign() ? -1.0 : 1.0;
    const std::uint64_t magnitude = _bits & ABS_MASK;

    if (isnan())
      return std::copysign(NAN, sgn);

    if (N == NaNStyle::IEEE && magnitude == _inf_bits())
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
};

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator==(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  const auto a = x.bits();
  const auto b = y.bits();
  const decltype(a) ABS_MASK = (1U << (E + M)) - 1U;
  return (a == b && !x.isnan()) || (N != NaNStyle::FNUZ && !((a | b) & ABS_MASK));
}

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator!=(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  return !(x == y);
}

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator<(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  const auto a = x.bits();
  const auto b = y.bits();
  const bool sign = (a | b) >> (E + M) & 1;
  const decltype(a) ABS_MASK = (1U << (E + M)) - 1U;

  if (x.isnan() || y.isnan())
    return false;

  if (N != NaNStyle::FNUZ && !((a | b) & ABS_MASK))
    return false;

  return sign ? a > b : a < b;
}

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator<=(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  const auto a = x.bits();
  const auto b = y.bits();
  const bool sign = (a | b) >> (E + M) & 1;
  const decltype(a) ABS_MASK = (1U << (E + M)) - 1U;

  if (x.isnan() || y.isnan())
    return false;

  if (N != NaNStyle::FNUZ && !((a | b) & ABS_MASK))
    return true;

  return sign ? a >= b : a <= b;
}

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator>(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  return y < x;
}

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
bool operator>=(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  return y <= x;
}

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
constexpr Minifloat<E, M, N, B, D> operator+(Minifloat<E, M, N, B, D> x) noexcept {
  return x;
}

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
constexpr Minifloat<E, M, N, B, D> operator-(Minifloat<E, M, N, B, D> x) noexcept {
  return Minifloat<E, M, N, B, D>::from_bits(x.bits() ^ (1U << (E + M)));
}

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
BitTrueGroupArithmeticType<M> operator+(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  BitTrueGroupArithmeticType<M> a = x;
  BitTrueGroupArithmeticType<M> b = y;
  return a + b;
}

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
BitTrueGroupArithmeticType<M> operator-(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  BitTrueGroupArithmeticType<M> a = x;
  BitTrueGroupArithmeticType<M> b = y;
  return a - b;
}

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
BitTrueGroupArithmeticType<M> operator*(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  BitTrueGroupArithmeticType<M> a = x;
  BitTrueGroupArithmeticType<M> b = y;
  return a * b;
}

template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
[[gnu::const]]
double operator/(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) noexcept {
  double a = x;
  double b = y;
  return a / b;
}

#define SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, MANT)                       \
  typedef Minifloat<EXP, MANT>                 E##EXP##M##MANT;      \
  typedef Minifloat<EXP, MANT, NaNStyle::FN>   E##EXP##M##MANT##FN;  \
  typedef Minifloat<EXP, MANT, NaNStyle::FNUZ> E##EXP##M##MANT##FNUZ;

#define SKYMIZER_MINIFLOAT_TYPEDEFS_ALL_M(EXP) \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 0)          \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 1)          \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 2)          \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 3)          \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 4)          \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 5)          \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 6)          \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 7)          \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 8)          \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 9)          \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 10)         \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 11)         \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 12)         \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 13)         \
  SKYMIZER_MINIFLOAT_TYPEDEFS(EXP, 14)         \
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