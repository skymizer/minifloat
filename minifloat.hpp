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

#include <cfloat>
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
} // namespace detail

/** \brief Internal storage type for minifloats
  * \tparam S - Signedness
  * \tparam E - Exponent width
  * \tparam M - Significand (mantissa) width
  */
template <bool S, unsigned E, unsigned M>
using StorageType = std::enable_if_t<S + E + M <= 16,
  std::conditional_t<S + E + M <= 8, std::uint_least8_t, std::uint_least16_t>>;

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

} // namespace minifloat

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
  minifloat::NaNStyle N = minifloat::NaNStyle::IEEE,
  minifloat::SubnormalStyle D = minifloat::SubnormalStyle::Precise>
class Minifloat
{
  minifloat::StorageType<S, E, M> _bits;

public:
  static const unsigned RADIX = 2;
  static const unsigned MANTISSA_DIGITS = M + 1;
  static const int MAX_EXP = (1 << E) - 1 - B;
  static const int MIN_EXP = 2 - B;

  Minifloat() = default;
  explicit Minifloat(float);
  explicit Minifloat(double);

  [[nodiscard, gnu::const]]
  operator float() const noexcept {
    static_assert(RADIX == std::numeric_limits<float>::radix);
    static_assert(MANTISSA_DIGITS <= std::numeric_limits<float>::digits);
    static_assert(MAX_EXP <= std::numeric_limits<float>::max_exponent);
    static_assert(MIN_EXP >= std::numeric_limits<float>::min_exponent);
    static_assert(std::numeric_limits<float>::is_iec559);

    const unsigned MAGNITUDE_BITS = E + M;
    const std::uint32_t MAX_MAGNITUDE = (1U << MAGNITUDE_BITS) - 1U;
    const std::uint32_t sign = S && _bits >> MAGNITUDE_BITS;
    const std::uint32_t magnitude = _bits & MAX_MAGNITUDE;

    if (N == minifloat::NaNStyle::FNUZ && _bits == MAX_MAGNITUDE + 1U)
      return NAN;

    if (N == minifloat::NaNStyle::FN && magnitude == MAX_MAGNITUDE)
      return sign ? -NAN : NAN;

    const std::uint32_t shifted = magnitude << (std::numeric_limits<float>::digits - MAGNITUDE_BITS);

    if (N == minifloat::NaNStyle::IEEE && magnitude > ((1U << E) - 1U) << M)
      return minifloat::detail::bit_cast<float>(sign << 31 | 0x7FC00000 | shifted);

    const std::uint32_t diff = MIN_EXP - std::numeric_limits<float>::min_exponent;
    const std::uint32_t bias = diff << (std::numeric_limits<float>::digits - 1);
    return minifloat::detail::bit_cast<float>(sign << 31 | (shifted + bias));
  }

  [[nodiscard, gnu::const]]
  operator double() const noexcept {
    static_assert(RADIX == std::numeric_limits<double>::radix);
    static_assert(MANTISSA_DIGITS <= std::numeric_limits<double>::digits);
    static_assert(MAX_EXP <= std::numeric_limits<double>::max_exponent);
    static_assert(MIN_EXP >= std::numeric_limits<double>::min_exponent);
    static_assert(std::numeric_limits<double>::is_iec559);

    const unsigned MAGNITUDE_BITS = E + M;
    const std::uint64_t MAX_MAGNITUDE = (1U << MAGNITUDE_BITS) - 1U;
    const std::uint64_t sign = S && _bits >> MAGNITUDE_BITS;
    const std::uint64_t magnitude = _bits & MAX_MAGNITUDE;

    if (N == minifloat::NaNStyle::FNUZ && _bits == MAX_MAGNITUDE + 1U)
      return NAN;

    if (N == minifloat::NaNStyle::FN && magnitude == MAX_MAGNITUDE)
      return sign ? -NAN : NAN;

    const std::uint64_t shifted = magnitude << (std::numeric_limits<double>::digits - MAGNITUDE_BITS);

    if (N == minifloat::NaNStyle::IEEE && magnitude > ((1U << E) - 1U) << M)
      return minifloat::detail::bit_cast<double>(sign << 63 | 0x7FF8000000000000 | shifted);

    const std::uint64_t diff = MIN_EXP - std::numeric_limits<double>::min_exponent;
    const std::uint64_t bias = diff << (std::numeric_limits<double>::digits - 1);
    return minifloat::detail::bit_cast<double>(sign << 63 | (shifted + bias));
  }
};

template <bool S, unsigned E, unsigned M, int B, minifloat::NaNStyle N, minifloat::SubnormalStyle D>
[[gnu::const]]
Minifloat<S, E, M, B, N, D> operator+(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  typedef minifloat::BitTrueGroupArithmeticType<M> ArithmeticType;
  return static_cast<ArithmeticType>(x) + static_cast<ArithmeticType>(y);
}

template <bool S, unsigned E, unsigned M, int B, minifloat::NaNStyle N, minifloat::SubnormalStyle D>
[[gnu::const]]
Minifloat<S, E, M, B, N, D> operator-(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  typedef minifloat::BitTrueGroupArithmeticType<M> ArithmeticType;
  return static_cast<ArithmeticType>(x) - static_cast<ArithmeticType>(y);
}

template <bool S, unsigned E, unsigned M, int B, minifloat::NaNStyle N, minifloat::SubnormalStyle D>
[[gnu::const]]
Minifloat<S, E, M, B, N, D> operator*(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  typedef minifloat::BitTrueGroupArithmeticType<M> ArithmeticType;
  return static_cast<ArithmeticType>(x) * static_cast<ArithmeticType>(y);
}

template <bool S, unsigned E, unsigned M, int B, minifloat::NaNStyle N, minifloat::SubnormalStyle D>
[[gnu::const]]
Minifloat<S, E, M, B, N, D> operator/(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  return static_cast<double>(x) / static_cast<double>(y);
}

} // namespace skymizer

#endif