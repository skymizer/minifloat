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

// `log2_floor` reaches for MSVC's 64-bit bit scan below. The intrinsic is
// declared here rather than by including <intrin.h>: it is the compiler's own,
// exactly as `__builtin_clzll` is, and the header is not part of the C++17
// standard library this file otherwise confines itself to. VS 2019 16.5
// (`_MSC_VER` 1925) is where `__builtin_is_constant_evaluated` arrives, and
// `_WIN64` is where a 64-bit scan does -- 32-bit MSVC has only the 32-bit one.
#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER >= 1925 && defined(_WIN64)
#define SKYMIZER_MINIFLOAT_BSR64
extern "C" unsigned char _BitScanReverse64(unsigned long *, unsigned __int64);
#pragma intrinsic(_BitScanReverse64)
#endif

//! Namespace for Skymizer
namespace skymizer {

//! Namespace for the minifloat library
namespace minifloat {

//! Backport of C++20 std::bit_cast
//!
//! `PURE`, not `CONST`.  `[[gnu::const]]` promises the result depends on the
//! argument *values* and nothing else, which for a reference parameter means
//! the address rather than what is at it -- GCC's own documentation says a
//! function that examines what a pointer argument points to must not be
//! declared `const`.  GCC 11.4 at `-O1` takes the promise: with `CONST` here,
//! `BF<32>{2.0F}.to_bits()` folds to zero and `BF<32>{2.0F} * BF<32>{3.0F}` to
//! a NaN, while `-O0`, `-O2` and `-O3` come out right.  Nothing in the suite
//! can catch that, `make check` and the CMake route both building optimized.
template <typename To, typename From>
[[nodiscard]] SKYMIZER_MINIFLOAT_PURE To bit_cast(const From &from) noexcept {
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

//! A number as sign times significand times two to the exponent
//!
//! The significand is an integer, so a triple stays exact where no host float
//! can.  Arithmetic hands one to `from_parts`, which is where every rounding
//! happens; its lowest bit may be sticky.
struct Parts {
  bool negative;
  std::uint64_t significand;
  int exponent;
};

//! Index of the highest set bit; the argument must be nonzero
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST constexpr int log2_floor(std::uint64_t x) noexcept {
#if __cplusplus >= 202002L
  return 63 - std::countl_zero(x);
#elif defined(__GNUC__) || defined(__clang__)
  return 63 - __builtin_clzll(x);
#else
  // MSVC lands here in every standard: its `__cplusplus` stays at 199711L
  // without `/Zc:__cplusplus`, which CMake does not pass, so `std::countl_zero`
  // above is out of reach.
#ifdef SKYMIZER_MINIFLOAT_BSR64
  // `_BitScanReverse64` is not a constant expression, and `from_parts` needs
  // one.  `__builtin_is_constant_evaluated` is how both hold at once: MSVC
  // exposes it as a compiler intrinsic in every standard mode, not as the
  // C++20 library entity -- microsoft/STL declares its own
  // `_Is_constant_evaluated` on it outside the `_HAS_CXX20` guard.
  if (!__builtin_is_constant_evaluated()) {
    unsigned long index = 0;
    _BitScanReverse64(&index, x);
    return static_cast<int>(index);
  }
#endif
  // The constant-evaluated path, and the run-time one on 32-bit MSVC.  Six
  // halvings, not a shift per bit: a sum or quotient can arrive here with a
  // 63-bit significand, and this is on the path of every operator.
  int result = 0;
  if ((x >> 32) != 0) {
    x >>= 32;
    result += 32;
  }
  if ((x >> 16) != 0) {
    x >>= 16;
    result += 16;
  }
  if ((x >> 8) != 0) {
    x >>= 8;
    result += 8;
  }
  if ((x >> 4) != 0) {
    x >>= 4;
    result += 4;
  }
  if ((x >> 2) != 0) {
    x >>= 2;
    result += 2;
  }
  return result + static_cast<int>(x >> 1);
#endif
}

//! A host float taken apart into integer fields
//!
//! `finite` says whether `significand` and `exponent` describe a magnitude.
//! Where it does not, the input was an infinity if `significand` is zero and a
//! NaN otherwise -- the payload, which is what tells the two apart.
struct Decomposed {
  bool negative;
  bool finite;
  std::uint64_t significand;
  int exponent;
};

//! Take a host float apart without leaving the integer domain
//!
//! The magnitude is `significand * 2**exponent` with no hidden bits, subnormal
//! inputs included.  One `bit_cast` answers everything `std::signbit`,
//! `std::isnan` and `std::isinf` answer one at a time, and a `float` stops
//! widening to `double` just to have its fields read.
template <typename Float>
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST Decomposed decompose(Float x) noexcept {
  static_assert(std::numeric_limits<Float>::radix == 2);
  static_assert(std::numeric_limits<Float>::is_iec559);

  using Bits = BitsOf<Float>;
  constexpr int MANT_DIG = std::numeric_limits<Float>::digits;
  constexpr int MIN_EXP = std::numeric_limits<Float>::min_exponent;
  constexpr int RESERVED_FIELD = 2 * std::numeric_limits<Float>::max_exponent - 1;
  constexpr Bits SIGN = Bits{1} << (std::numeric_limits<Bits>::digits - 1);

  const auto bits = bit_cast<Bits>(x);
  const auto magnitude = static_cast<Bits>(bits & ~SIGN);
  const auto field = static_cast<int>(magnitude >> (MANT_DIG - 1));
  const auto fraction = static_cast<std::uint64_t>(magnitude & ((Bits{1} << (MANT_DIG - 1)) - 1));
  const bool negative = bits != magnitude;

  if (field == RESERVED_FIELD)
    return {negative, false, fraction, 0};
  if (field == 0)
    return {negative, true, fraction, MIN_EXP - MANT_DIG};
  return {
      negative,
      true,
      fraction | UINT64_C(1) << (MANT_DIG - 1),
      field + MIN_EXP - 1 - MANT_DIG,
  };
}

//! Two to the power of an integer, exactly where `double` can hold it
//!
//! Spelling a power of two as a libm call leaves the compiler free not to fold
//! it, and `std::ldexp` costs a call even where its exponent is a literal.
//! Building the field directly needs neither, and reaches the subnormal results
//! a naive `1 << (52 + field)` would not.  Out of range it saturates to
//! infinity or to zero, which is what both callers want at either end.
//!
//! Not `constexpr`: `bit_cast` is only constexpr from C++20, and C++17 is the
//! floor.  A literal argument folds regardless, the body being `inline` and in
//! view; the attribute below buys nothing here and is carried for consistency
//! with the rest of `detail`.
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST inline double exp2i(int x) noexcept {
  const int field = x + (DBL_MAX_EXP - 1);

  if (field >= 2 * DBL_MAX_EXP - 1)
    return HUGE_VAL;
  if (field > 0)
    return bit_cast<double>(static_cast<std::uint64_t>(field) << (DBL_MANT_DIG - 1));
  if (field >= 2 - DBL_MANT_DIG)
    return bit_cast<double>(UINT64_C(1) << (DBL_MANT_DIG - 2 + field));
  return 0.0;
}

//! Round `significand * 2**exponent` to a multiple of `2**target`
//!
//! Ties go to even, which is what IEEE 754 rounds to by default.  Working on
//! an integer significand keeps this exact for exponents far outside the range
//! of any host float.
//!
//! The `shift >= 64` guard covers right shifts only.  A left shift is a caller
//! contract: every significand handed in is below 2**63, so it has room in an
//! `int64_t`.
//!
//! `parity_offset` maps the retained bit's parity to the destination code's.
//! It matters only when the format has no mantissa bit, where the code's low
//! bit is the exponent field's and the significand is always 1.
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST constexpr std::int64_t round_to_scale(
    std::uint64_t significand, int exponent, int target, bool parity_offset = false
) noexcept {
  const int shift = target - exponent;

  if (shift <= 0)
    return static_cast<std::int64_t>(significand << -shift);
  if (shift >= 64)
    return 0;

  const std::uint64_t dropped = significand & ((UINT64_C(1) << shift) - 1U);
  const std::uint64_t kept = significand >> shift;
  const std::uint64_t half = UINT64_C(1) << (shift - 1);
  const bool odd = ((kept & 1) != 0) != parity_offset;
  const bool round_up = dropped > half || (dropped == half && odd);
  return static_cast<std::int64_t>(kept + static_cast<std::uint64_t>(round_up));
}

//! Widest exponent gap an aligned sum spans
//!
//! An addend further below the other than this cannot move it at all: the
//! other is a representable value, and anything under half its ULP rounds
//! straight back to it.  For p-bit significands the cap must be at least p + 1,
//! and no more than 62 - p keeps the aligned sum inside an `int64_t`.  32 meets
//! both bounds through p = 30: two 30-bit significands and this shift make a
//! 63-bit sum at worst.
constexpr int ALIGN_CAP = 32;

//! One addend at the common exponent, signed
//!
//! An addend below that exponent is one `ALIGN_CAP` has already ruled out.
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST constexpr std::int64_t
align(bool negative, std::uint64_t significand, int exponent, int base) noexcept {
  const auto magnitude =
      exponent >= base ? static_cast<std::int64_t>(significand << (exponent - base)) : 0;
  return negative ? -magnitude : magnitude;
}

//! Sum of two signed magnitudes, exact enough to round
//!
//! The caller is responsible for the significands fitting in 30 bits, which
//! every minifloat does.
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST constexpr Parts add_parts(Parts x, Parts y) noexcept {
  const int top = x.exponent > y.exponent ? x.exponent : y.exponent;
  const int bottom = x.exponent < y.exponent ? x.exponent : y.exponent;
  const int base = top - bottom > ALIGN_CAP ? top - ALIGN_CAP : bottom;

  const std::int64_t sum = align(x.negative, x.significand, x.exponent, base) +
                           align(y.negative, y.significand, y.exponent, base);

  return {
      // Cancellation yields +0 unless both addends were negative.
      sum != 0 ? sum < 0 : (x.negative && y.negative),
      static_cast<std::uint64_t>(sum < 0 ? -sum : sum),
      base,
  };
}

//! Quotient of two magnitudes, exact enough to round
//!
//! The remainder collapses into the lowest bit of the quotient, the sticky bit
//! every divider keeps.  A fixed shift leaves too few bits when a subnormal
//! dividend meets a full-width divisor, so the dividend is normalized to bit
//! 62.  That yields at least 63 - p quotient bits for p-bit significands, enough
//! to round through p = 30, while keeping the quotient below 2**63.  The caller
//! is responsible for a nonzero divisor and for both significands fitting in
//! 30 bits.
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST constexpr Parts div_parts(
    bool negative, std::uint64_t significand, int exponent, std::uint64_t rhs_significand,
    int rhs_exponent
) noexcept {
  const int shift = 62 - log2_floor(significand | 1U);
  const std::uint64_t numerator = significand << shift;
  const std::uint64_t quotient = numerator / rhs_significand;
  const std::uint64_t remainder = numerator % rhs_significand;

  return {
      negative,
      quotient | static_cast<std::uint64_t>(remainder != 0),
      exponent - rhs_exponent - shift,
  };
}

//! What a bit pattern denotes
enum struct Kind { Zero, Subnormal, Normal, Infinite, NaN };

//! Plain scientific reading of the magnitude field
//!
//! Every magnitude `(e << M) | m` denotes `(1 + m / 2**M) * 2**(e - B)`.  This
//! layer owns the bit space and nothing else: no zero, no subnormal, no
//! reserved code point.  The layers above reinterpret parts of it.
template <int E, int M, int B> struct ScientificFormat {
  static_assert(E + M < 32);
  static_assert(E >= 2);
  static_assert(E <= 30);
  static_assert(M >= 0);

  using Storage = std::conditional_t<
      (E + M < 8), std::uint_least8_t,
      std::conditional_t<(E + M < 16), std::uint_least16_t, std::uint_least32_t>>;

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

//! Split a finite code into exact `(sign, significand, exponent)`
//!
//! The inverse of `from_parts` where the value is representable: `significand`
//! carries the implicit bit where the code has one, and `exponent` is the ULP
//! scale of the code.  There is no special-value check — an IEEE infinity
//! decodes to significand `1 << M`, and multiplication and division use
//! `significand == 0` as their zero test after decoding unconditionally.
template <class Format>
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST constexpr Parts to_parts(typename Format::Storage bits
) noexcept {
  constexpr int M = Format::MANTISSA_BITS;
  const auto magnitude = static_cast<std::uint64_t>(bits & Format::MAG_MASK);
  const auto field = static_cast<int>(magnitude >> M);
  const std::uint64_t fraction = magnitude & ((UINT64_C(1) << M) - 1U);
  const bool negative = (bits & Format::SIGN_MASK) != 0;

  if (field == 0)
    return {negative, fraction, 1 - Format::BIAS - M};
  return {negative, fraction | UINT64_C(1) << M, field - Format::BIAS - M};
}

//! Correctly rounded code for `sign * significand * 2**exponent`
//!
//! This is where every rounding in the library happens.  The triple is an
//! exact number, or one whose lowest bit is sticky, so it can come from a
//! `double` or from arithmetic of its own.
template <class Format>
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST constexpr typename Format::Storage from_parts(Parts parts
) noexcept {
  using Storage = typename Format::Storage;
  constexpr int M = Format::MANTISSA_BITS;
  const auto sign_bit = static_cast<Storage>(parts.negative ? Format::SIGN_MASK : Storage{0});

  // Without a negative zero, signing a zero spells NaN.
  if (parts.significand == 0)
    return Format::HAS_NEG_ZERO ? sign_bit : Storage{0};

  // The exponent of the value, which is in [2**e, 2**(e+1)).
  const int e = parts.exponent + log2_floor(parts.significand);

  std::int64_t magnitude = 0;
  if (e < Format::MIN_EXP - 1) {
    // Subnormal numbers all share the ULP of the smallest one, so their code
    // *is* the rounded multiple of that ULP.
    magnitude = round_to_scale(parts.significand, parts.exponent, Format::MIN_EXP - 1 - M);
  } else {
    // Rounding to `M + 1` digits may carry into the implicit bit.  That lands
    // on the next exponent field with a zero mantissa, which is exactly where
    // the extra ULP belongs.  The code trails the rounded significand by
    // `(e + B - 1) << M`, whose parity is the tie-break's only correction.
    const bool parity_offset = M == 0 && (e + Format::BIAS) % 2 == 0;
    const std::int64_t rounded =
        round_to_scale(parts.significand, parts.exponent, e - M, parity_offset);
    magnitude = (static_cast<std::int64_t>(e + Format::BIAS) << M) + rounded - (INT64_C(1) << M);
  }

  const auto code = static_cast<Storage>(std::min<std::int64_t>(magnitude, Format::OVERFLOW_MAG));
  // A value that rounds to zero drops its sign for the same reason a zero does.
  return static_cast<Storage>(code | ((Format::HAS_NEG_ZERO || code != 0) ? sign_bit : Storage{0}));
}

//! The shape's fields differ from a host float only in mantissa width
//!
//! Matching exponent range and IEEE special values make conversion a mantissa
//! shift.  The host checks license reading `Float` through its integer layout.
template <class Format, typename Float> constexpr bool shares_host_exponent() noexcept {
  using Bits = BitsOf<Float>;

  return Format::HAS_INF && Format::HAS_NAN && Format::HAS_NEG_ZERO &&
         Format::MANTISSA_BITS < std::numeric_limits<Float>::digits &&
         Format::MIN_EXP == std::numeric_limits<Float>::min_exponent &&
         Format::MAX_EXP == std::numeric_limits<Float>::max_exponent &&
         sizeof(Float) == sizeof(Bits) &&
         Format::EXPONENT_BITS + std::numeric_limits<Float>::digits ==
             std::numeric_limits<Bits>::digits &&
         std::numeric_limits<Float>::radix == 2 && std::numeric_limits<Float>::is_iec559;
}

//! The shape *is* `float`, so a round trip through one is the identity
//!
//! `shares_host_exponent` excludes `FN<8, 23>` and `Finite<8, 23>` on special
//! values, and `IEEE<7, 23>` on exponent range.  Matching precision leaves
//! `IEEE<8, 23>` -- `BF<32>` -- as the only admitted shape.
template <class Format> constexpr bool is_host_float() noexcept {
  return shares_host_exponent<Format, float>() && Format::MANTISSA_BITS + 1 == FLT_MANT_DIG;
}

} // namespace detail

//! Configurable signed floating-point type up to 32 bits
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

  //! Is this type `float`, bit for bit?
  //!
  //! Where it holds, `to_float` and construction from a `float` are the
  //! identity, and an array of these is an array of `float`.  Arithmetic is
  //! *not*: the operators stay on the integer engine, because a shape that is
  //! a `float` is also a shape whose FPU answer moves with the caller's
  //! rounding mode and MXCSR, and this library rounds to nearest either way.
  //! `detail::is_host_float` is the predicate and says which shapes miss it and
  //! why; `benches/arith.cpp` reads this to pick the host type it compares the
  //! shape against, so the two cannot drift apart.
  static constexpr bool IS_HOST_FLOAT = detail::is_host_float<Format>();

  static constexpr bool HAS_EXACT_F64_CONVERSION =
      DBL_MANT_DIG >= MANTISSA_DIGITS && DBL_MAX_EXP >= MAX_EXP && DBL_MIN_EXP <= MIN_EXP &&
      std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::is_iec559;

private:
  Storage bits_{};

  //! Encode a host float, rounding to nearest with ties to even
  //!
  //! A NaN input needs `Format::HAS_NAN`; see the constructors.  A host input
  //! with the same exponent field rounds by discarding mantissa bits directly;
  //! every other finite value is decomposed exactly and rounded once by
  //! `detail::from_parts`, so a shape whose exponent range outruns `double`'s is
  //! served as exactly as any other.
  template <typename Float>
  [[nodiscard]] SKYMIZER_MINIFLOAT_CONST static Storage bits_from(Float x) noexcept {
    if constexpr (detail::shares_host_exponent<Format, Float>()) {
      using Bits = detail::BitsOf<Float>;
      constexpr int SHIFT = std::numeric_limits<Float>::digits - 1 - M;
      const Bits bits = bit_cast<Bits>(x);

      // Rounding a NaN can erase its retained payload and spell infinity.
      if ((bits & ((std::numeric_limits<Bits>::max)() >> 1)) > static_cast<Bits>(Format::INF_MAG)
                                                                 << SHIFT)
        return static_cast<Storage>(((bits >> SHIFT) & Format::SIGN_MASK) | Format::NAN_BITS);

      if constexpr (SHIFT == 0) {
        return static_cast<Storage>(bits);
      } else {
        const Bits bias = (Bits{1} << (SHIFT - 1)) - 1U + ((bits >> SHIFT) & 1U);
        return static_cast<Storage>((bits + bias) >> SHIFT);
      }
    }

    const detail::Decomposed parts = detail::decompose(x);
    const auto sign = static_cast<Storage>(parts.negative ? Format::SIGN_MASK : Storage{0});

    if (!parts.finite) {
      // A zero payload under a reserved exponent is an infinity, and nothing
      // else is.
      if (parts.significand == 0)
        return static_cast<Storage>(sign | Format::OVERFLOW_MAG);
      if constexpr (Format::HAS_NAN)
        return static_cast<Storage>(sign | Format::NAN_BITS);
      else // Precondition violation; saturate rather than emit a wild pattern.
        return static_cast<Storage>(sign | Format::MAX_FINITE_MAG);
    }

    return detail::from_parts<Format>({parts.negative, parts.significand, parts.exponent});
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
             std::copysign(static_cast<Float>(detail::exp2i(MIN_EXP - MANTISSA_DIGITS)), sign);

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
    if constexpr (detail::shares_host_exponent<Format, float>())
      return bit_cast<float>(
          static_cast<detail::BitsOf<float>>(bits_) << (FLT_MANT_DIG - 1 - M)
      );

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
    if constexpr (detail::shares_host_exponent<Format, double>())
      return bit_cast<double>(
          static_cast<detail::BitsOf<double>>(bits_) << (DBL_MANT_DIG - 1 - M)
      );

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

    const bool subnormal = magnitude < 1U << M;
    const auto significand =
        subnormal ? magnitude
                  : static_cast<std::uint32_t>((magnitude & ((1U << M) - 1U)) | 1U << M);
    const int exponent =
        subnormal ? MIN_EXP - MANTISSA_DIGITS : static_cast<int>(magnitude >> M) - B - M;

    // Splitting the scale keeps either factor inside `double`'s exponent range,
    // so the first product is exact wherever the value is representable at all
    // and the second rounds at most once.  A single factor would flush to zero
    // or to infinity long before the product does, which is the whole reason
    // this branch exists.  These two multiplies are one of the three places a
    // caller's rounding mode can still reach a result; `arithmetic.md` names
    // all three, and all three are conversions this shape cannot make exactly.
    const int head = exponent < DBL_MIN_EXP - 1   ? DBL_MIN_EXP - 1
                     : exponent > DBL_MAX_EXP - 1 ? DBL_MAX_EXP - 1
                                                  : exponent;
    return sign * significand * detail::exp2i(head) * detail::exp2i(exponent - head);
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

namespace detail {
//! The result of an invalid operation
//!
//! A format without a NaN saturates one to `max()`, as encoding a NaN does.
//! The sign of a default NaN means nothing, so this one is positive — nominally
//! so for FNUZ, whose NaN *is* the sign-bit pattern.
template <class Format> SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format> invalid() noexcept {
  if constexpr (Format::HAS_NAN)
    return Minifloat<Format>::from_bits(Format::NAN_BITS);
  else
    return Minifloat<Format>::from_bits(Format::MAX_FINITE_MAG);
}

//! The overflow result with the sign an operation worked out
//!
//! An infinity where the format has one, its maximum finite value otherwise.
template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format> huge(bool negative) noexcept {
  using Storage = typename Format::Storage;
  const auto sign = static_cast<Storage>(negative ? Format::SIGN_MASK : Storage{0});
  return Minifloat<Format>::from_bits(static_cast<Storage>(Format::OVERFLOW_MAG | sign));
}

//! `x + y`, or `x - y` when `flip` is set
//!
//! Subtraction is addition with the subtrahend's sign flipped.  Flipping it
//! here, in the one place that reads a sign, is what spares a format without a
//! negative zero the guard its unary minus needs: there is no intermediate
//! value to keep representable, only a bool to invert.  Both callers pass a
//! literal, so the flag folds away before anything is emitted.
template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format>
add_impl(Minifloat<Format> x, Minifloat<Format> y, bool flip) noexcept {
  if (x.is_nan() || y.is_nan())
    return invalid<Format>();

  if (x.is_infinite() && y.is_infinite()) {
    // Two infinities agree only when their signs do; what the other case ought
    // to be is exactly the question.
    return x.signbit() == (y.signbit() != flip) ? x : invalid<Format>();
  }
  // An infinity outweighs anything finite added to it.
  if (x.is_infinite())
    return x;
  if (y.is_infinite()) {
    // Reachable only where the format has infinities, and there a negation is
    // the bare XOR it looks like.
    return flip ? -y : y;
  }

  Parts rhs = to_parts<Format>(y.to_bits());
  rhs.negative = rhs.negative != flip;
  const Parts sum = add_parts(to_parts<Format>(x.to_bits()), rhs);
  return Minifloat<Format>::from_bits(from_parts<Format>(sum));
}
} // namespace detail

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format>
operator+(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  return detail::add_impl(x, y, false);
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format>
operator-(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  return detail::add_impl(x, y, true);
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format>
operator*(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  if (x.is_nan() || y.is_nan())
    return detail::invalid<Format>();

  const bool negative = x.signbit() != y.signbit();
  const detail::Parts lhs = detail::to_parts<Format>(x.to_bits());
  const detail::Parts rhs = detail::to_parts<Format>(y.to_bits());

  if (x.is_infinite() || y.is_infinite()) {
    // An infinity scaled by zero is the invalid one; the significands say
    // which operand is the zero.
    if (lhs.significand == 0 || rhs.significand == 0)
      return detail::invalid<Format>();
    return detail::huge<Format>(negative);
  }
  // Two significands of at most 30 bits multiply exactly.
  const detail::Parts product{
      negative, lhs.significand * rhs.significand, lhs.exponent + rhs.exponent};
  return Minifloat<Format>::from_bits(detail::from_parts<Format>(product));
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format>
operator/(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  if (x.is_nan() || y.is_nan())
    return detail::invalid<Format>();

  const bool negative = x.signbit() != y.signbit();
  const detail::Parts lhs = detail::to_parts<Format>(x.to_bits());
  const detail::Parts rhs = detail::to_parts<Format>(y.to_bits());

  if (x.is_infinite()) {
    // Infinity over infinity is the invalid one.
    return y.is_infinite() ? detail::invalid<Format>() : detail::huge<Format>(negative);
  }
  if (y.is_infinite())
    return Minifloat<Format>::from_bits(detail::from_parts<Format>({negative, 0, 0}));

  if (rhs.significand == 0) {
    // Zero over zero is the invalid one; anything else over zero overflows
    // every exponent there is.
    if (lhs.significand == 0)
      return detail::invalid<Format>();
    return detail::huge<Format>(negative);
  }
  const detail::Parts quotient =
      detail::div_parts(negative, lhs.significand, lhs.exponent, rhs.significand, rhs.exponent);
  return Minifloat<Format>::from_bits(detail::from_parts<Format>(quotient));
}

//! Mantissa, base 2 exponent, and sign as integer
//!
//! A finite original value can be reconstructed as
//! `sign * mantissa * 2**exponent`.
//!
//! See also `integer_decode`.
struct IntegerDecode {
  std::uint64_t mantissa;
  std::int32_t exponent;
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
      static_cast<std::int32_t>(exponent - BIAS),
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

//! Brain float with `N` total bits
//!
//! `BF<16>` is `E8M7`, `BF<19>` has TensorFloat-32's layout, and `BF<32>` has
//! `float`'s layout.
template <int N> using BF = IEEE<8, N - 9>;

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

  // Decimal digits per bit, to more places than 30103 / 100000 gives. The two
  // exponent10 members below are the only ones that need it.
  static constexpr double LOG10_2 = 0.30102999566398119521373889472449;

  // constexpr floor, which `std::floor` is not in C++17 and which truncation
  // is not: both signs occur here, since a large bias puts `max()` under 1 and
  // a bias at or below zero puts `min()` above it.
  static constexpr int floored(double x) noexcept {
    const auto truncated = static_cast<int>(x);
    return truncated - (truncated > x);
  }

  // floor(log10(max())), read off the maximal finite code rather than off
  // `MAX_EXP`. The two part company wherever a format spends the top of its
  // exponent range on something that is not a number: `FN<5, 2>` tops out at
  // 2^17 * (1 - 2^-2), which is under 10^5 although `MAX_EXP` is 17, and
  // `FN<E, 0>` loses its whole top row to the one NaN. Reading the code also
  // makes the bias fall out for free, wherever a caller supplies an odd one.
  static constexpr int exact_max_exponent10() noexcept {
    // log2(1 - 2^-p) indexed by precision p, so that a maximal finite
    // magnitude is 2^BINADE * (1 - 2^-PRECISION). Index 0 is unreachable —
    // a maximum's precision is at least 1 — and its entry is a leftover.
    // `E + M < 32` bounds the precision at 30. Shared with the minifloat-rs
    // sibling, which spells the same table `detail::LOG2_SIGNIFICAND`.
    constexpr double LOG2_SIGNIFICAND[31] = {
        -2.0,
        -1.0,
        -4.15037499278843813e-1,
        -1.92645077942395881e-1,
        -9.31094043914814651e-2,
        -4.58036896131247886e-2,
        -2.27200765000835289e-2,
        -1.13153132278341461e-2,
        -5.64656314114206272e-3,
        -2.82051906237866263e-3,
        -1.40957025467135363e-3,
        -7.04612976589372706e-4,
        -3.52263471629021385e-4,
        -1.76120984274024062e-4,
        -8.80578045800263834e-5,
        -4.40282304417772115e-5,
        -2.20139472639555020e-5,
        -1.10069316433851864e-5,
        -5.50345532462453928e-6,
        -2.75172503805526697e-6,
        -1.37586186296463416e-6,
        -6.87930767466723669e-7,
        -3.43965342729483034e-7,
        -1.71982661113774261e-7,
        -8.59913279941456218e-8,
        -4.29956633563874719e-8,
        -2.14978315180224060e-8,
        -1.07489157189683711e-8,
        -5.37445784947347765e-9,
        -2.68722892223406186e-9,
        -1.34361446049136169e-9,
    };

    constexpr unsigned MAG = Format::MAX_FINITE_MAG;
    constexpr unsigned MAN_MASK = (1U << T::MANTISSA_BITS) - 1U;
    // The significand is one bit shorter wherever the all-ones magnitude is
    // spent elsewhere; `M == 0` falls out of an empty mask matching itself.
    constexpr int PRECISION = T::MANTISSA_BITS + ((MAG & MAN_MASK) == MAN_MASK);
    // `E >= 2` leaves the maximal code's exponent field at 2 or more, so this
    // is always the binade of a normal value.
    constexpr int BINADE = static_cast<int>(MAG >> T::MANTISSA_BITS) - T::BIAS + 1;

    return floored((BINADE + LOG2_SIGNIFICAND[PRECISION]) * LOG10_2);
  }

  // ceil(log10(min())) in the `FLT_MIN_10_EXP` sense: the least power of ten
  // that is still a normal value. `(MIN_EXP - 1) * 30103 / 100000` was a
  // ceiling only for as long as it truncated toward zero from below, and
  // `MIN_EXP` is `2 - BIAS`, so a bias at or below zero lifts the dividend
  // above zero and turns that same truncation into a floor.
  static constexpr int exact_min_exponent10() noexcept {
    return -floored(-(T::MIN_EXP - 1) * LOG10_2);
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
  static constexpr int min_exponent10 = exact_min_exponent10();
  static constexpr int max_exponent10 = exact_max_exponent10();
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
