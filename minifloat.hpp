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

#include <type_traits>

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

/** \brief A fast native type for floating point arithmetic
  * \tparam M - Significand (mantissa) width
  *
  * We want to emulate addition and multiplication of floating point number
  * type T with a wider type U.  If the significand of U is more than twice as
  * wide as T, double rounding is not a problem.  The proof is left as an
  * exercise to the reader.
  */
template <unsigned M>
using BitTrueGroupArithmeticType = std::conditional_t<M <= 11, float, double>;

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
public:
  Minifloat() = default;
  Minifloat(float);
  Minifloat(double);

  operator float() const noexcept;
  operator double() const noexcept;
};

template <bool S, unsigned E, unsigned M, int B, minifloat::NaNStyle N, minifloat::SubnormalStyle D>
Minifloat<S, E, M, B, N, D> operator+(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  typedef minifloat::BitTrueGroupArithmeticType<M> ArithmeticType;
  return static_cast<ArithmeticType>(x) + static_cast<ArithmeticType>(y);
}

template <bool S, unsigned E, unsigned M, int B, minifloat::NaNStyle N, minifloat::SubnormalStyle D>
Minifloat<S, E, M, B, N, D> operator-(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  typedef minifloat::BitTrueGroupArithmeticType<M> ArithmeticType;
  return static_cast<ArithmeticType>(x) - static_cast<ArithmeticType>(y);
}

template <bool S, unsigned E, unsigned M, int B, minifloat::NaNStyle N, minifloat::SubnormalStyle D>
Minifloat<S, E, M, B, N, D> operator*(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  typedef minifloat::BitTrueGroupArithmeticType<M> ArithmeticType;
  return static_cast<ArithmeticType>(x) * static_cast<ArithmeticType>(y);
}

template <bool S, unsigned E, unsigned M, int B, minifloat::NaNStyle N, minifloat::SubnormalStyle D>
Minifloat<S, E, M, B, N, D> operator/(const Minifloat<S, E, M, B, N, D> &x, const Minifloat<S, E, M, B, N, D> &y) noexcept {
  return static_cast<double>(x) / static_cast<double>(y);
}

} // namespace skymizer

#endif