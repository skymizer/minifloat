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
#if __has_builtin(__builtin_bit_cast)
  return __builtin_bit_cast(To, from);
#else
  union { From _; To to; } caster = {from};
  return caster.to;
#endif
}

/** \brief Round a float to the nearest float with the given significand width
  * \tparam M - Significand (mantissa) width
  */
template <unsigned M>
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
template <unsigned M>
using BitTrueGroupArithmeticType = std::conditional_t<
  M <= std::numeric_limits<float>::digits / 2 - 1,
  float, double>;

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

/** \brief Configurable floating point type
  * \tparam S - Signedness
  * \tparam E - Exponent width
  * \tparam M - Significand (mantissa) width
  * \tparam B - Exponent bias
  * \tparam N - NaN encoding style
  * \tparam D - Subnormal (denormal) encoding style
  *
  * Constraints:
  * - S + E + M <= 16
  * - FNUZ must be signed
  */
template <bool S, unsigned E, unsigned M,
  int B = (1 << (E - 1)) - 1,
  NaNStyle N = NaNStyle::IEEE,
  SubnormalStyle D = SubnormalStyle::Precise>
class Minifloat {
public:
  static_assert(N != NaNStyle::FNUZ || S);
  static_assert(S + E + M <= 16);
  typedef std::conditional_t<S + E + M <= 8, std::uint_least8_t, std::uint_least16_t> StorageType;

  static const unsigned RADIX = 2;
  static const unsigned MANTISSA_DIGITS = M + 1;
  static const int MAX_EXP = (1 << E) - B - (N == NaNStyle::IEEE);
  static const int MIN_EXP = 2 - B;

private:
  StorageType _bits;

  [[nodiscard, gnu::const]]
  static constexpr StorageType _inf_bits() {
    const StorageType MAX = (UINT32_C(1) << (E + M)) - 1;

    if constexpr (N == NaNStyle::IEEE)
      return MAX << M & MAX;

    return MAX - (N == NaNStyle::FN);
  }

  [[nodiscard, gnu::const]]
  static constexpr StorageType _nan_bits() {
    const StorageType N0 = UINT32_C(1) << (E + M);
    const StorageType MAX = N0 - 1;

    if constexpr (N == NaNStyle::FNUZ)
      return N0;

    if constexpr (N == NaNStyle::FN || M == 0)
      return MAX;

    return MAX << (M - 1) & MAX;
  }

  [[nodiscard, gnu::const]]
  static StorageType _to_bits(float x) {
    const auto bits = detail::bit_cast<std::uint32_t>(detail::round_to_mantissa<M>(x));
    const auto sign = S ? bits >> 31 << (E + M) : 0;

    if (x != x) 
      return sign | _nan_bits();

    if (!S && x <= 0)
      return 0;

    const std::int32_t diff = MIN_EXP - std::numeric_limits<float>::min_exponent;
    const std::int32_t bias = diff << (std::numeric_limits<float>::digits - 1);
    const std::int32_t magnitude = (bits & INT32_MAX) - bias;

    if (magnitude < std::int32_t{1} << (std::numeric_limits<float>::digits - 1)) {
      if constexpr (D == SubnormalStyle::Precise) {
        const StorageType ticks = std::rint(std::ldexp(std::abs(x), MANTISSA_DIGITS - MIN_EXP));
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

  [[nodiscard, gnu::const]]
  StorageType bits() const noexcept { return _bits; }

  [[nodiscard, gnu::const]]
  bool is_nan() const noexcept {
    const unsigned MAGNITUDE_BITS = E + M;
    const StorageType MAX_MAGNITUDE = (1U << MAGNITUDE_BITS) - 1U;

    if constexpr (N == NaNStyle::FNUZ)
      return _bits == MAX_MAGNITUDE + 1U;

    if constexpr (N == NaNStyle::FN)
      return (_bits & MAX_MAGNITUDE) == MAX_MAGNITUDE;

    return (_bits & MAX_MAGNITUDE) > ((1U << E) - 1U) << M;
  }

  /** \brief Implicit lossless conversion to float
    *
    * The conversion is only enabled if it is proven to be lossless at compile
    * time.  If the conversion is lossy, the user must explicitly cast to float.
    */
  template <bool ENABLE =
    std::numeric_limits<float>::radix == RADIX &&
    std::numeric_limits<float>::digits >= MANTISSA_DIGITS &&
    std::numeric_limits<float>::max_exponent >= MAX_EXP &&
    std::numeric_limits<float>::min_exponent <= MIN_EXP &&
    std::numeric_limits<float>::is_iec559>
  [[nodiscard, gnu::const]]
  operator std::enable_if_t<ENABLE, float>() const noexcept {
    const unsigned MAGNITUDE_BITS = E + M;
    const std::uint32_t MAX_MAGNITUDE = (1U << MAGNITUDE_BITS) - 1U;
    const std::uint32_t sign = S && _bits >> MAGNITUDE_BITS;

    if (is_nan())
      return std::copysign(NAN, sign ? -1.0f : 1.0f);

    const std::uint32_t magnitude = (_bits & MAX_MAGNITUDE) << (std::numeric_limits<float>::digits - MAGNITUDE_BITS);
    const std::uint32_t diff = MIN_EXP - std::numeric_limits<float>::min_exponent;
    const std::uint32_t bias = diff << (std::numeric_limits<float>::digits - 1);
    return detail::bit_cast<float>(sign << 31 | (magnitude + bias));
  }

  /** \brief Implicit lossless conversion to double
    *
    * The conversion is only enabled if it is proven to be lossless at compile
    * time.  If the conversion is lossy, the user must explicitly cast to double.
    */
  template <bool ENABLE =
    std::numeric_limits<double>::radix == RADIX &&
    std::numeric_limits<double>::digits >= MANTISSA_DIGITS &&
    std::numeric_limits<double>::max_exponent >= MAX_EXP &&
    std::numeric_limits<double>::min_exponent <= MIN_EXP &&
    std::numeric_limits<double>::is_iec559>
  [[nodiscard, gnu::const]]
  operator std::enable_if_t<ENABLE, double>() const noexcept {
    const unsigned MAGNITUDE_BITS = E + M;
    const std::uint64_t MAX_MAGNITUDE = (1U << MAGNITUDE_BITS) - 1U;
    const std::uint64_t sign = S && _bits >> MAGNITUDE_BITS;

    if (is_nan())
      return std::copysign(NAN, sign ? -1.0 : 1.0);

    const std::uint64_t magnitude = (_bits & MAX_MAGNITUDE) << (std::numeric_limits<double>::digits - MAGNITUDE_BITS);
    const std::uint64_t diff = MIN_EXP - std::numeric_limits<double>::min_exponent;
    const std::uint64_t bias = diff << (std::numeric_limits<double>::digits - 1);
    return detail::bit_cast<double>(sign << 63 | (magnitude + bias));
  }

  /** \brief Explicit lossy conversion to float
    *
    * This variant makes use of the lossless implicit conversion to double.
    */
  [[nodiscard, gnu::const]]
  explicit operator std::enable_if_t<
    !std::is_convertible<Minifloat, float>::value &&
    std::is_convertible<Minifloat, double>::value,
    float>() const noexcept
  {
    return static_cast<double>(*this);
  }

  [[nodiscard, gnu::const]]
  explicit operator std::enable_if_t<
    !std::is_convertible<Minifloat, float>::value &&
    !std::is_convertible<Minifloat, double>::value,
    float>() const noexcept;
  
  [[nodiscard, gnu::const]]
  explicit operator std::enable_if_t<!std::is_convertible_v<Minifloat, double>, double>() const noexcept;
};

template <bool S, unsigned E, unsigned M, int B, NaNStyle N, SubnormalStyle D>
[[gnu::const]]
Minifloat<S, E, M, B, N, D> operator+(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  typedef BitTrueGroupArithmeticType<M> ArithmeticType;
  return static_cast<ArithmeticType>(x) + static_cast<ArithmeticType>(y);
}

template <bool S, unsigned E, unsigned M, int B, NaNStyle N, SubnormalStyle D>
[[gnu::const]]
Minifloat<S, E, M, B, N, D> operator-(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  typedef BitTrueGroupArithmeticType<M> ArithmeticType;
  return static_cast<ArithmeticType>(x) - static_cast<ArithmeticType>(y);
}

template <bool S, unsigned E, unsigned M, int B, NaNStyle N, SubnormalStyle D>
[[gnu::const]]
Minifloat<S, E, M, B, N, D> operator*(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  typedef BitTrueGroupArithmeticType<M> ArithmeticType;
  return static_cast<ArithmeticType>(x) * static_cast<ArithmeticType>(y);
}

template <bool S, unsigned E, unsigned M, int B, NaNStyle N, SubnormalStyle D>
[[gnu::const]]
Minifloat<S, E, M, B, N, D> operator/(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  return static_cast<double>(x) / static_cast<double>(y);
}

} // namespace minifloat

using minifloat::Minifloat;

} // namespace skymizer

#endif