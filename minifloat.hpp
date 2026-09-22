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

// Keeps a rarely taken path from counting against its caller's inlining
// budget; see `bf_arithmetic`.
#if defined(__GNUC__) || defined(__clang__)
#define SKYMIZER_MINIFLOAT_NOINLINE [[gnu::noinline]]
#elif defined(_MSC_VER)
#define SKYMIZER_MINIFLOAT_NOINLINE __declspec(noinline)
#else
#define SKYMIZER_MINIFLOAT_NOINLINE
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
//! infinity or to zero, which is what integral conversion's range check wants
//! at either end.
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

//! Round an integer significand directly into a host IEEE representation
template <typename Float>
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST inline Float host_from_parts(Parts parts) noexcept {
  using Bits = BitsOf<Float>;
  constexpr int DIGITS = std::numeric_limits<Float>::digits;
  constexpr int FRACTION_BITS = DIGITS - 1;
  constexpr int MIN_EXP = std::numeric_limits<Float>::min_exponent;
  constexpr int MAX_EXP = std::numeric_limits<Float>::max_exponent;
  constexpr Bits INF_BITS = static_cast<Bits>(2 * MAX_EXP - 1) << FRACTION_BITS;

  static_assert(std::numeric_limits<Float>::radix == 2);
  static_assert(std::numeric_limits<Float>::is_iec559);
  static_assert(sizeof(Float) == sizeof(Bits));

  const Bits sign = static_cast<Bits>(parts.negative) << (std::numeric_limits<Bits>::digits - 1);
  if (parts.significand == 0)
    return bit_cast<Float>(sign);

  const int e = parts.exponent + log2_floor(parts.significand);
  Bits magnitude = 0;
  if (e < MIN_EXP - 1) {
    magnitude =
        static_cast<Bits>(round_to_scale(parts.significand, parts.exponent, MIN_EXP - DIGITS));
  } else if (e >= MAX_EXP) {
    magnitude = INF_BITS;
  } else {
    const auto rounded =
        static_cast<Bits>(round_to_scale(parts.significand, parts.exponent, e - FRACTION_BITS));
    magnitude = (static_cast<Bits>(e - MIN_EXP + 1) << FRACTION_BITS) + rounded;
    magnitude = std::min(magnitude, INF_BITS);
  }
  return bit_cast<Float>(static_cast<Bits>(sign | magnitude));
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
  static_assert(
      std::int64_t{(1U << E) - 1U} - B <= (std::numeric_limits<int>::max)() / 2,
      "bias is too small for exponent arithmetic"
  );
  static_assert(
      std::int64_t{1} - B - M >= (std::numeric_limits<int>::min)() / 2,
      "bias is too large for exponent arithmetic"
  );

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
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST constexpr Parts
to_parts(typename Format::Storage bits) noexcept {
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
[[nodiscard]] SKYMIZER_MINIFLOAT_CONST constexpr typename Format::Storage
from_parts(Parts parts) noexcept {
  using Storage = typename Format::Storage;
  constexpr int M = Format::MANTISSA_BITS;
  const auto sign_bit = static_cast<Storage>(parts.negative ? Format::SIGN_MASK : Storage{0});

  // Without a negative zero, signing a zero spells NaN.
  if (parts.significand == 0)
    return Format::HAS_NEG_ZERO ? sign_bit : Storage{0};

  // The exponent of the value, which is in [2**e, 2**(e+1)).
  const int e = parts.exponent + log2_floor(parts.significand);
  constexpr int FLOOR = Format::MIN_EXP - 1;
  const int clamped_e = std::max(e, FLOOR);
  // Subnormal codes and normal exponent rows meet at the same rounding scale.
  // M == 0 still takes its tie parity from the normal exponent field.
  const bool parity_offset = M == 0 && e >= FLOOR && (e + Format::BIAS) % 2 == 0;
  const std::int64_t magnitude =
      round_to_scale(parts.significand, parts.exponent, clamped_e - M, parity_offset) +
      (static_cast<std::int64_t>(clamped_e - FLOOR) << M);

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

//! The shape has `float`'s layout; only NaN payload canonicalization changes bits
//!
//! `shares_host_exponent` excludes `FN<8, 23>` and `Finite<8, 23>` on special
//! values, and `IEEE<7, 23>` on exponent range.  Matching precision leaves
//! `IEEE<8, 23>` -- `BF<32>` -- as the only admitted shape.
template <class Format> constexpr bool is_host_float() noexcept {
  return shares_host_exponent<Format, float>() && Format::MANTISSA_BITS + 1 == FLT_MANT_DIG;
}

//! Packed format codes, with unused high bits cleared at the boundary.
template <class Format> class PackedStorage {
  using Bits = typename Format::Storage;
  Bits bits_{};

public:
  constexpr PackedStorage() = default;

  static constexpr PackedStorage from_bits(Bits bits) noexcept {
    PackedStorage result;
    result.bits_ = static_cast<Bits>(bits & (Format::MAG_MASK | Format::SIGN_MASK));
    return result;
  }

  constexpr Bits to_bits() const noexcept { return bits_; }
};

//! BF codes aligned with the most significant bit of their storage word.
//!
//! The format is part of the type, so padding and precision cannot disagree.
//! Only factories and exact widening can create a value; low padding bits
//! always stay zero. BF16 occupies two bytes, BF17 through BF32 four.
template <class Format> class BfStorage {
  static_assert(shares_host_exponent<Format, float>());
  template <class> friend class BfStorage;
  using Bits = typename Format::Storage;
  static constexpr int WIDTH = std::numeric_limits<Bits>::digits;
  static constexpr int SHIFT = WIDTH - 9 - Format::MANTISSA_BITS;
  Bits bits_{};

public:
  constexpr BfStorage() = default;

  template <class Other, std::enable_if_t<(Other::MANTISSA_BITS < Format::MANTISSA_BITS), int> = 0>
  explicit constexpr BfStorage(BfStorage<Other> other) noexcept
      : bits_(
            static_cast<Bits>(static_cast<Bits>(other.bits_) << (WIDTH - BfStorage<Other>::WIDTH))
        ) {}

  static constexpr BfStorage from_bits(Bits bits) noexcept {
    BfStorage result;
    result.bits_ = static_cast<Bits>(bits << SHIFT);
    return result;
  }

  //! Round directly into the aligned word, without a packed-code intermediate.
  static BfStorage from_float(float x) noexcept {
    constexpr int DROP = 23 - Format::MANTISSA_BITS;
    auto bits = bit_cast<std::uint32_t>(x);
    if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
      bits = (bits & UINT32_C(0x80000000)) | UINT32_C(0x7fc00000);
    } else if constexpr (DROP != 0) {
      bits += (UINT32_C(1) << (DROP - 1)) - 1U + ((bits >> DROP) & 1U);
    }
    BfStorage result;
    result.bits_ = static_cast<Bits>((bits >> (32 - WIDTH)) & (UINT32_MAX << SHIFT));
    return result;
  }

  constexpr Bits to_bits() const noexcept { return static_cast<Bits>(bits_ >> SHIFT); }
  constexpr std::uint32_t float_bits() const noexcept {
    return static_cast<std::uint32_t>(bits_) << (32 - WIDTH);
  }
};

//! Keep constant expressions on the integer engine in every supported dialect.
constexpr bool is_constant_evaluated() noexcept {
#if __cplusplus >= 202002L
  return std::is_constant_evaluated();
#elif (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 9) ||                               \
    (defined(_MSC_VER) && _MSC_VER >= 1925)
  return __builtin_is_constant_evaluated();
#elif defined(__clang__)
#if __has_builtin(__builtin_is_constant_evaluated)
  return __builtin_is_constant_evaluated();
#else
  return true;
#endif
#else
  return true;
#endif
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

  //! IEEE binary32's exponent field and a reduced (or equal) precision.
  static constexpr bool IS_BFLOAT = detail::shares_host_exponent<Format, float>();

  static constexpr bool HAS_EXACT_F32_CONVERSION =
      FLT_MANT_DIG >= MANTISSA_DIGITS && FLT_MAX_EXP >= MAX_EXP && FLT_MIN_EXP <= MIN_EXP &&
      std::numeric_limits<float>::radix == 2 && std::numeric_limits<float>::is_iec559;

  //! Is this type `float`, bit for bit?
  //!
  //! Where it holds, `to_float` is the identity, while construction from a
  //! `float` preserves non-NaN bits and canonicalizes NaN payloads. Arithmetic
  //! uses the BF family's wider double intermediate and integer rounding;
  //! doing the operation directly in float would inherit the host environment.
  static constexpr bool IS_HOST_FLOAT = detail::is_host_float<Format>();

  static constexpr bool HAS_EXACT_F64_CONVERSION =
      DBL_MANT_DIG >= MANTISSA_DIGITS && DBL_MAX_EXP >= MAX_EXP && DBL_MIN_EXP <= MIN_EXP &&
      std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::is_iec559;

private:
  template <class> friend class Minifloat;
  using Representation =
      std::conditional_t<IS_BFLOAT, detail::BfStorage<Format>, detail::PackedStorage<Format>>;
  Representation storage_{};

  //! Encode a host float, rounding to nearest with ties to even
  //!
  //! A NaN input needs `Format::HAS_NAN`; see the constructors.  A host input
  //! with the same exponent field rounds by discarding mantissa bits directly.
  //! Other exact, narrow shapes rebase the source field and round its integer
  //! code; the remaining values are decomposed exactly and rounded once by
  //! `detail::from_parts`.
  template <typename Float>
  [[nodiscard]] SKYMIZER_MINIFLOAT_CONST static Storage bits_from(Float x) noexcept {
    if constexpr (IS_BFLOAT && std::is_same_v<Float, double>) {
      // Normal BF values need only mantissa rounding and an exponent rebase.
      // Subnormals, overflow and special values use the general integer path.
      constexpr std::uint64_t MIN = UINT64_C(0x3810000000000000);
      constexpr std::uint64_t END = UINT64_C(0x47f0000000000000);
      constexpr int SHIFT = 52 - M;
      const auto bits = bit_cast<std::uint64_t>(x);
      const auto magnitude = bits & UINT64_C(0x7fffffffffffffff);
      if (magnitude - MIN < END - MIN) {
        const auto bias = (UINT64_C(1) << (SHIFT - 1)) - 1U + ((magnitude >> SHIFT) & 1U);
        const auto code = ((magnitude + bias) >> SHIFT) - (UINT64_C(896) << M);
        return static_cast<Storage>(code | ((bits >> 63) << (M + 8)));
      }
    }

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

    constexpr bool EXACT_CONVERSION =
        std::is_same_v<Float, float> ? HAS_EXACT_F32_CONVERSION : HAS_EXACT_F64_CONVERSION;
    constexpr bool SUBNORMALS_ROUND_TO_ZERO =
        MIN_EXP - MANTISSA_DIGITS >= std::numeric_limits<Float>::min_exponent;
    if constexpr (EXACT_CONVERSION && SUBNORMALS_ROUND_TO_ZERO) {
      using Bits = detail::BitsOf<Float>;
      constexpr int DIGITS = std::numeric_limits<Float>::digits;
      constexpr int FRACTION_BITS = DIGITS - 1;
      constexpr int RESERVED_FIELD = 2 * std::numeric_limits<Float>::max_exponent - 1;
      constexpr Bits SIGN = Bits{1} << (std::numeric_limits<Bits>::digits - 1);

      const Bits bits = bit_cast<Bits>(x);
      const Bits magnitude = bits & ~SIGN;
      const auto sign = static_cast<Storage>((bits & SIGN) != 0 ? Format::SIGN_MASK : Storage{0});
      const int field = static_cast<int>(magnitude >> FRACTION_BITS);
      const std::uint64_t fraction = magnitude & ((Bits{1} << FRACTION_BITS) - 1U);

      if (field == RESERVED_FIELD) {
        if (fraction == 0)
          return static_cast<Storage>(sign | Format::OVERFLOW_MAG);
        if constexpr (Format::HAS_NAN)
          return static_cast<Storage>(sign | Format::NAN_BITS);
        else
          return static_cast<Storage>(sign | Format::MAX_FINITE_MAG);
      }

      // This tier's floor gate makes every source subnormal less than half the
      // destination's true minimum.
      if (field == 0)
        return Format::HAS_NEG_ZERO ? sign : Storage{0};

      const int rebased = field + std::numeric_limits<Float>::min_exponent - MIN_EXP;
      const int shift = std::min(63, DIGITS - MANTISSA_DIGITS + std::max(1 - rebased, 0));
      const std::uint64_t unrounded =
          (static_cast<std::uint64_t>(std::max(rebased, 1)) << FRACTION_BITS) | fraction;
      const std::uint64_t bias =
          shift == 0 ? 0 : (UINT64_C(1) << (shift - 1)) - 1U + ((unrounded >> shift) & 1U);
      const std::uint64_t rounded = shift == 0 ? unrounded : (unrounded + bias) >> shift;
      const auto code =
          static_cast<Storage>(std::min(rounded, static_cast<std::uint64_t>(Format::OVERFLOW_MAG)));
      return static_cast<Storage>(code | ((Format::HAS_NEG_ZERO || code != 0) ? sign : Storage{0}));
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
    const auto magnitude = static_cast<Bits>(to_bits() & ABS_MASK);

    if constexpr (Format::HAS_NAN)
      if (Format::is_nan(to_bits()))
        return std::copysign(std::numeric_limits<Float>::quiet_NaN(), sign);

    if constexpr (Format::HAS_INF)
      if (magnitude == Format::INF_MAG)
        return std::copysign(std::numeric_limits<Float>::infinity(), sign);

    const auto sign_bit =
        static_cast<Bits>(Bits{signbit()} << (std::numeric_limits<Bits>::digits - 1));

    if (magnitude < Bits{1} << M) {
      if (magnitude == 0)
        return bit_cast<Float>(sign_bit);

      // Build host fields directly; scaling in `Float` would inherit FTZ/DAZ.
      const int leading = detail::log2_floor(magnitude);
      const int exponent = MIN_EXP - MANTISSA_DIGITS + leading;

      if (exponent < DST_MIN_EXP - 1) {
        const int shift = exponent - leading - DST_MIN_EXP + MANT_DIG;
        return bit_cast<Float>(static_cast<Bits>(sign_bit | (magnitude << shift)));
      }

      const auto shifted = static_cast<Bits>(magnitude << (MANT_DIG - 1 - leading));
      const auto bias = static_cast<Bits>(exponent - DST_MIN_EXP + 1) << (MANT_DIG - 1);
      return bit_cast<Float>(static_cast<Bits>(sign_bit | (shifted + bias)));
    }

    const auto shifted = static_cast<Bits>(magnitude << (MANT_DIG - MANTISSA_DIGITS));
    const auto bias = static_cast<Bits>(Bits{MIN_EXP - DST_MIN_EXP} << (MANT_DIG - 1));
    return bit_cast<Float>(static_cast<Bits>(sign_bit | (shifted + bias)));
  }

  //! Rounded host reconstruction for a shape that is not exact in `Float`
  template <typename Float>
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE Float to_inexact() const noexcept {
    using Bits = detail::BitsOf<Float>;
    constexpr int FRACTION_BITS = std::numeric_limits<Float>::digits - 1;
    constexpr Bits INF_BITS = static_cast<Bits>(2 * std::numeric_limits<Float>::max_exponent - 1)
                              << FRACTION_BITS;
    constexpr Bits NAN_BITS = INF_BITS | Bits{1} << (FRACTION_BITS - 1);
    const auto sign = static_cast<Bits>(Bits{signbit()} << (std::numeric_limits<Bits>::digits - 1));
    const auto magnitude = static_cast<Storage>(to_bits() & ABS_MASK);

    if constexpr (Format::HAS_NAN)
      if (Format::is_nan(to_bits()))
        return bit_cast<Float>(static_cast<Bits>(sign | NAN_BITS));

    if constexpr (Format::HAS_INF)
      if (magnitude == Format::INF_MAG)
        return bit_cast<Float>(static_cast<Bits>(sign | INF_BITS));

    if (magnitude == 0)
      return bit_cast<Float>(sign);

    // Wide formats spend most of their code space beyond the host's range.
    // Normal source rows can settle those two cases without normalization.
    if (magnitude >= Storage{1} << M) {
      const int e = static_cast<int>(magnitude >> M) - B;
      if (e < std::numeric_limits<Float>::min_exponent - std::numeric_limits<Float>::digits - 1)
        return bit_cast<Float>(sign);
      if (e >= std::numeric_limits<Float>::max_exponent)
        return bit_cast<Float>(static_cast<Bits>(sign | INF_BITS));
    }

    return detail::host_from_parts<Float>(detail::to_parts<Format>(to_bits()));
  }

public:
  Minifloat() = default;

  explicit Minifloat(float x) noexcept {
    if constexpr (IS_BFLOAT)
      storage_ = Representation::from_float(x);
    else
      storage_ = Representation::from_bits(bits_from(x));
    assert((HAS_NAN || !(std::isnan)(x)) && "this minifloat format cannot represent a NaN");
  }

  explicit Minifloat(double x) noexcept : storage_(Representation::from_bits(bits_from(x))) {
    assert((HAS_NAN || !(std::isnan)(x)) && "this minifloat format cannot represent a NaN");
  }

  //! Exact BF widening, with no conversion through a host numeric type.
  template <
      class Other,
      std::enable_if_t<
          IS_BFLOAT && detail::shares_host_exponent<Other, float>() && (Other::MANTISSA_BITS < M),
          int> = 0>
  explicit constexpr Minifloat(Minifloat<Other> other) noexcept : storage_(other.storage_) {}

  //! Construct from any integer type, rounding once.
  template <typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
  explicit constexpr Minifloat(Int x) noexcept {
    if constexpr (std::is_same_v<std::remove_cv_t<Int>, bool>) {
      storage_ = Representation::from_bits(
          detail::from_parts<Format>({false, static_cast<std::uint64_t>(x), 0})
      );
    } else {
      using Unsigned = std::make_unsigned_t<Int>;
      bool negative = false;
      auto magnitude = static_cast<Unsigned>(x);
      if constexpr (std::is_signed_v<Int>) {
        negative = x < 0;
        if (negative)
          magnitude = Unsigned{0} - magnitude;
      }

      std::uint64_t significand = 0;
      int exponent = 0;
      if constexpr (std::numeric_limits<Unsigned>::digits <= 63) {
        significand = static_cast<std::uint64_t>(magnitude);
      } else {
        bool sticky = false;
        const auto limit = static_cast<Unsigned>((std::numeric_limits<std::int64_t>::max)());
        while (magnitude > limit) {
          sticky = sticky || (magnitude & Unsigned{1}) != 0;
          magnitude >>= 1;
          ++exponent;
        }
        significand = static_cast<std::uint64_t>(magnitude) | static_cast<std::uint64_t>(sticky);
      }

      storage_ =
          Representation::from_bits(detail::from_parts<Format>({negative, significand, exponent}));
    }
  }

  static constexpr Minifloat from_bits(Storage bits) noexcept {
    Minifloat result;
    result.storage_ = Representation::from_bits(bits);
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
  [[nodiscard]] static constexpr Minifloat(min)() noexcept {
    return from_bits(static_cast<Storage>(1U << M));
  }

  /// Maximum finite value
  [[nodiscard]] static constexpr Minifloat(max)() noexcept {
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

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr Storage to_bits() const noexcept {
    return storage_.to_bits();
  }

  //! Sign bit
  //!
  //! Note for `FNUZ`: the sole NaN representation has the sign bit set, so
  //! `signbit()` returns `true` for a FNUZ NaN even though there is no
  //! negative-zero counterpart to compare it to. Callers that filter by
  //! `signbit()` should test `is_nan()` first when working with FNUZ.
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool signbit() const noexcept {
    return (to_bits() & Format::SIGN_MASK) != 0;
  }

  //! Check if the number is nonzero
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr explicit operator bool() const noexcept {
    if constexpr (!Format::HAS_NEG_ZERO)
      return to_bits() != 0;

    return (to_bits() & ABS_MASK) != 0;
  }

  //! BF tests its aligned word, as a host float would.  Packing it first costs
  //! a shift per test, after which GCC no longer shares one comparison between
  //! `is_nan` and `is_infinite` in a classification loop.
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool is_nan() const noexcept {
    if constexpr (IS_BFLOAT)
      return (storage_.float_bits() & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000);

    return Format::is_nan(to_bits());
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool is_infinite() const noexcept {
    if constexpr (IS_BFLOAT)
      return (storage_.float_bits() & UINT32_C(0x7fffffff)) == UINT32_C(0x7f800000);

    if constexpr (Format::HAS_INF)
      return (to_bits() & ABS_MASK) == Format::INF_MAG;

    return false;
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool is_finite() const noexcept {
    return !is_nan() && !is_infinite();
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool is_normal() const noexcept {
    return is_finite() && (to_bits() & ABS_MASK) >= (1U << M);
  }

  //! Check if the number is nonzero subnormal
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr bool is_subnormal() const noexcept {
    const Storage magnitude = to_bits() & ABS_MASK;
    return 0 < magnitude && magnitude < (1U << M);
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE constexpr int classify() const noexcept {
    const detail::Kind kind = Format::kind(to_bits());
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
      if (!(to_bits() & ABS_MASK))
        return *this;

    return from_bits(static_cast<Storage>(to_bits() & ABS_MASK));
  }

  //! Explicit conversion to float
  //!
  //! The lossy branch rounds directly into `float`'s integer fields, so it does
  //! not inherit the caller's rounding mode or flush-to-zero setting.
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE float to_float() const noexcept {
    if constexpr (IS_BFLOAT)
      return bit_cast<float>(storage_.float_bits());

    if constexpr (HAS_EXACT_F32_CONVERSION)
      return to_exact<float>();

    return to_inexact<float>();
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE explicit operator float() const noexcept {
    return to_float();
  }

  //! Conversion to double
  //!
  //! When `HAS_EXACT_F64_CONVERSION` holds, the result is exact.  Otherwise its
  //! integer fields are rounded once, overflowing to infinity or underflowing
  //! toward zero without host floating-point arithmetic.
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE double to_double() const noexcept {
    if constexpr (IS_BFLOAT) {
      const auto bits = storage_.float_bits();
      const auto magnitude = bits & UINT32_C(0x7fffffff);
      // Normal floats and infinities widen exactly in any host environment.
      if (magnitude - UINT32_C(0x00800000) <= UINT32_C(0x7f000000))
        return static_cast<double>(bit_cast<float>(bits));
      if (magnitude < UINT32_C(0x00800000)) {
        // Never feed a subnormal float to the FPU: DAZ could erase it.
        // Both the integer significand and the scaled double are exact;
        // the signed scale also preserves negative zero in every mode.
        const double scale = bits >> 31 ? -0x1p-149 : 0x1p-149;
        return static_cast<double>(static_cast<std::int32_t>(magnitude)) * scale;
      }
      return std::copysign(std::numeric_limits<double>::quiet_NaN(), signbit() ? -1.0 : 1.0);
    }

    if constexpr (detail::shares_host_exponent<Format, double>())
      return bit_cast<double>(
          static_cast<detail::BitsOf<double>>(to_bits()) << (DBL_MANT_DIG - 1 - M)
      );

    if constexpr (HAS_EXACT_F64_CONVERSION)
      return to_exact<double>();

    return to_inexact<double>();
  }

  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE explicit operator double() const noexcept {
    return to_double();
  }

  //! Truncating conversion to any non-bool integer type.
  //!
  //! NaN converts to zero. Values outside the destination range, including
  //! infinities, saturate; negative values saturate to zero for unsigned types.
  //! `bool` stays on the dedicated nonzero conversion above.
  template <
      typename Int,
      std::enable_if_t<
          std::is_integral_v<Int> && !std::is_same_v<std::remove_cv_t<Int>, bool>, int> = 0>
  [[nodiscard]] SKYMIZER_MINIFLOAT_PURE explicit operator Int() const noexcept {
    if (is_nan())
      return Int{0};

    const double value = to_double();
    const double limit = detail::exp2i(std::numeric_limits<Int>::digits);
    if (value >= limit)
      return (std::numeric_limits<Int>::max)();

    if constexpr (std::is_signed_v<Int>) {
      if (value <= -limit)
        return (std::numeric_limits<Int>::min)();
    } else if (value <= 0) {
      return Int{0};
    }

    return static_cast<Int>(value);
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
template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format>
mul_impl(Minifloat<Format> x, Minifloat<Format> y) noexcept {
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
      negative, lhs.significand * rhs.significand, lhs.exponent + rhs.exponent
  };
  return Minifloat<Format>::from_bits(detail::from_parts<Format>(product));
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format>
div_impl(Minifloat<Format> x, Minifloat<Format> y) noexcept {
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

//! A BF intermediate has at most 24-bit operands and exponents in [-149, 127].
//! Binary64 can carry every product exactly and has ample exponent range for
//! every nonzero result. Addition and division retain enough guard digits for
//! one final integer rounding, even under directed host rounding. Exact zero
//! sums and invalid results get the library's own signs and NaN code.
enum class BfOp { Add, Sub, Mul, Div };

template <BfOp Op> SKYMIZER_MINIFLOAT_CONST double bf_apply(double a, double b) noexcept {
  if constexpr (Op == BfOp::Add)
    return a + b;
  if constexpr (Op == BfOp::Sub)
    return a - b;
  if constexpr (Op == BfOp::Mul)
    return a * b;
  return a / b;
}

//! Round a binary64 magnitude of at least FLT_MIN to a BF code
//!
//! A mantissa rounding and an exponent rebase, ties to even.  The carry out
//! of the largest finite value lands on infinity's code, and so does every
//! larger magnitude, infinity's included; a NaN's must not come here.
template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr typename Format::Storage
bf_round_normal(std::uint64_t magnitude) noexcept {
  constexpr int M = Format::MANTISSA_BITS;
  constexpr int SHIFT = DBL_MANT_DIG - 1 - M;
  const auto bias = (UINT64_C(1) << (SHIFT - 1)) - 1U + ((magnitude >> SHIFT) & 1U);
  const auto code = ((magnitude + bias) >> SHIFT) - (UINT64_C(896) << M);
  // Overflow starts at the tie above the largest finite value, whose odd
  // significand rounds it up: half a BF unit below 2**FLT_MAX_EXP.  Deciding
  // on the magnitude keeps this a select.  As min(code, INF_MAG) it became a
  // minimum, which Clang turned back into a branch inside loops, where random
  // operands mispredict it.
  constexpr auto OVERFLOW =
      (static_cast<std::uint64_t>(DBL_MAX_EXP - 1 + FLT_MAX_EXP) << (DBL_MANT_DIG - 1)) -
      (UINT64_C(1) << (SHIFT - 1));
  return static_cast<typename Format::Storage>(magnitude >= OVERFLOW ? Format::INF_MAG : code);
}

//! Round a binary64 magnitude below FLT_MIN to a BF code, ties to even
//!
//! Counts in units of the least subnormal, 2**(FLT_MIN_EXP - 1 - M), by
//! shifting the significand as far as the exponent says.  A carry out of the
//! subnormals lands on the least normal code.
template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr typename Format::Storage
bf_round_subnormal(std::uint64_t magnitude) noexcept {
  constexpr int M = Format::MANTISSA_BITS;
  // The value is significand * 2**(exponent - DOUBLE_BIAS), where the bias
  // also scales the significand to an integer.
  constexpr int DOUBLE_BIAS = DBL_MAX_EXP - 1 + DBL_MANT_DIG - 1;
  constexpr int UNIT = DOUBLE_BIAS + (FLT_MIN_EXP - 1) - M;
  // A binade shifted by 63 lies below half a unit, and so does everything
  // under it, zero and binary64 subnormals included.  Giving those that
  // binade's exponent keeps the shift in range with a select on the
  // magnitude; clamping the shift itself compiled to a branch that random
  // operands mispredict.
  constexpr auto FLOOR = static_cast<std::uint64_t>(UNIT - 63) << (DBL_MANT_DIG - 1);
  const int exponent =
      magnitude < FLOOR ? UNIT - 63 : static_cast<int>(magnitude >> (DBL_MANT_DIG - 1));
  const auto significand =
      (magnitude & ((UINT64_C(1) << (DBL_MANT_DIG - 1)) - 1U)) | UINT64_C(1) << (DBL_MANT_DIG - 1);
  const int shift = UNIT - exponent;
  const auto bias = (UINT64_C(1) << (shift - 1)) - 1U + ((significand >> shift) & 1U);
  return static_cast<typename Format::Storage>((significand + bias) >> shift);
}

//! Round a binary64 magnitude, short of NaN, to a BF code
template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr typename Format::Storage
bf_round_magnitude(std::uint64_t magnitude) noexcept {
  if (magnitude >= UINT64_C(0x3810000000000000))
    return bf_round_normal<Format>(magnitude);
  return bf_round_subnormal<Format>(magnitude);
}

//! BF arithmetic with a zero, subnormal, infinite or NaN operand
template <BfOp Op, class Format>
SKYMIZER_MINIFLOAT_NOINLINE SKYMIZER_MINIFLOAT_CONST Minifloat<Format>
bf_arithmetic_special(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  using T = Minifloat<Format>;
  const double value = bf_apply<Op>(x.to_double(), y.to_double());
  const auto bits = bit_cast<std::uint64_t>(value);
  if ((bits & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000))
    return T::quiet_NaN();
  if constexpr (Op == BfOp::Add || Op == BfOp::Sub) {
    if ((bits << 1) == 0) {
      const bool negative = x.signbit() && (y.signbit() != (Op == BfOp::Sub));
      return T::from_bits(negative ? Format::SIGN_MASK : 0);
    }
  }
  return T{value};
}

template <BfOp Op, class Format>
SKYMIZER_MINIFLOAT_CONST Minifloat<Format>
bf_arithmetic(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  using T = Minifloat<Format>;
  static_assert(T::IS_BFLOAT);
  static_assert(DBL_MANT_DIG >= 2 * T::MANTISSA_DIGITS + 3);
  static_assert(DBL_MIN_EXP <= 2 * (FLT_MIN_EXP - FLT_MANT_DIG));
  static_assert(DBL_MAX_EXP >= FLT_MAX_EXP - FLT_MIN_EXP + FLT_MANT_DIG);

  // A zero or subnormal operand goes out of line: DAZ could erase the
  // latter, and zeros bring the signed-zero rules.  Every other operand --
  // normal, infinite or NaN -- widens exactly on any host, and the argument
  // above `BfOp` leaves a binary64 result that is a NaN, an infinity, or
  // normal unless it is zero.  Keeping the rest out of line lets callers
  // still inline this.
  const auto fx = bit_cast<std::uint32_t>(x.to_float());
  const auto fy = bit_cast<std::uint32_t>(y.to_float());
  if (!(fx & UINT32_C(0x7f800000)) || !(fy & UINT32_C(0x7f800000)))
    return bf_arithmetic_special<Op>(x, y);

  const auto bits = bit_cast<std::uint64_t>(bf_apply<Op>(
      static_cast<double>(bit_cast<float>(fx)), static_cast<double>(bit_cast<float>(fy))
  ));
  const auto magnitude = bits & UINT64_C(0x7fffffffffffffff);
  const auto sign = static_cast<typename Format::Storage>(bits >> 63 ? Format::SIGN_MASK : 0U);
  constexpr auto MIN = UINT64_C(0x3810000000000000);
  constexpr auto INF = UINT64_C(0x7ff0000000000000);
  if (magnitude - MIN <= INF - MIN)
    return T::from_bits(
        static_cast<typename Format::Storage>(bf_round_normal<Format>(magnitude) | sign)
    );
  if (magnitude > INF)
    return T::quiet_NaN();
  // Nonzero addends cancel exactly to +0.  A quotient over infinity is the
  // other zero, and it takes the host's sign, which no rounding mode affects.
  if constexpr (Op == BfOp::Add || Op == BfOp::Sub)
    if (magnitude == 0)
      return T::from_bits(0);
  return T::from_bits(
      static_cast<typename Format::Storage>(bf_round_subnormal<Format>(magnitude) | sign)
  );
}
} // namespace detail

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format>
operator+(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  if constexpr (Minifloat<Format>::IS_BFLOAT)
    if (!detail::is_constant_evaluated())
      return detail::bf_arithmetic<detail::BfOp::Add>(x, y);
  return detail::add_impl(x, y, false);
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format>
operator-(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  if constexpr (Minifloat<Format>::IS_BFLOAT)
    if (!detail::is_constant_evaluated())
      return detail::bf_arithmetic<detail::BfOp::Sub>(x, y);
  return detail::add_impl(x, y, true);
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format>
operator*(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  if constexpr (Minifloat<Format>::IS_BFLOAT)
    if (!detail::is_constant_evaluated())
      return detail::bf_arithmetic<detail::BfOp::Mul>(x, y);
  return detail::mul_impl(x, y);
}

template <class Format>
SKYMIZER_MINIFLOAT_CONST constexpr Minifloat<Format>
operator/(Minifloat<Format> x, Minifloat<Format> y) noexcept {
  if constexpr (Minifloat<Format>::IS_BFLOAT)
    if (!detail::is_constant_evaluated())
      return detail::bf_arithmetic<detail::BfOp::Div>(x, y);
  return detail::div_impl(x, y);
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
    if (biased > static_cast<int>(Format::MAX_FINITE_MAG >> T::MANTISSA_BITS))
      return (T::max)();
    if (biased >= 1)
      return T::from_bits(static_cast<typename T::Storage>(biased) << T::MANTISSA_BITS);
    if (biased + T::MANTISSA_BITS >= 1)
      return T::from_bits(typename T::Storage{1} << (biased + T::MANTISSA_BITS - 1));
    return T::from_bits(0);
  }

  // A fraction with 96 binary places. Decimal exponent limits reach roughly
  // +/- 10^9, where even a correctly rounded double logarithm can cross an
  // integer. Use integer intervals instead; long double is only double on MSVC.
  struct LogFraction {
    std::uint32_t high;
    std::uint64_t low;
  };

  // floor(exponent * logarithm - correction), with both fractions scaled by
  // 2^96. Three 32-bit products avoid a nonstandard 128-bit integer dependency.
  static constexpr int log_floor(int exponent, LogFraction logarithm, LogFraction correction) {
    const auto n =
        static_cast<std::uint64_t>(exponent < 0 ? -static_cast<std::int64_t>(exponent) : exponent);
    const auto bottom = static_cast<std::uint32_t>(logarithm.low) * n;
    const auto middle = (logarithm.low >> 32) * n + (bottom >> 32);
    auto high = static_cast<std::int64_t>(logarithm.high * n + (middle >> 32));
    auto low = (middle << 32) | static_cast<std::uint32_t>(bottom);
    if (exponent < 0) {
      high = -high - (low != 0);
      low = std::uint64_t{0} - low;
    }
    high -= static_cast<std::int64_t>(correction.high) + (low < correction.low);
    constexpr auto RADIX = INT64_C(1) << 32;
    return static_cast<int>(high / RADIX - (high % RADIX < 0));
  }

  // floor(log10(2^Exponent * (1 - 2^-Precision))). Precision zero omits
  // the significand correction, and precision one reduces to 2^(Exponent-1).
  // Constants are lower bounds; adding one unit gives the upper bound.
  // docs/decimal-limits.md records their generation and whole-domain check.
  template <int Exponent, int Precision> static constexpr int decimal_exponent() {
    constexpr LogFraction LOG10_2{UINT32_C(0x4d104d42), UINT64_C(0x7de7fbcc47c4acd6)};
    // -log10(1 - 2^-p), indexed by p. The first two cases need no correction.
    constexpr LogFraction CORRECTIONS[31] = {
        {UINT32_C(0x00000000), UINT64_C(0x0000000000000000)},
        {UINT32_C(0x00000000), UINT64_C(0x0000000000000000)},
        {UINT32_C(0x1ffbfc2b), UINT64_C(0xbc780375837c4b0b)},
        {UINT32_C(0x0ed88f6b), UINT64_C(0xb355fa196e1e0dda)},
        {UINT32_C(0x072ce3f3), UINT64_C(0x362ff6da5aca518d)},
        {UINT32_C(0x0387a106), UINT64_C(0xef09881397f0ba9b)},
        {UINT32_C(0x01c03a80), UINT64_C(0xae5e05382d51f71b)},
        {UINT32_C(0x00df3b5e), UINT64_C(0xbbda7e186b65af39)},
        {UINT32_C(0x006f65a8), UINT64_C(0x75f672f0a347b623)},
        {UINT32_C(0x0037a4e0), UINT64_C(0x8b7fa04961d6b4d2)},
        {UINT32_C(0x001bcef5), UINT64_C(0x18e29611a506bc65)},
        {UINT32_C(0x000de69b), UINT64_C(0xf8f58005dfc20fa8)},
        {UINT32_C(0x0006f316), UINT64_C(0x5e90f44aa39c5303)},
        {UINT32_C(0x0003797d), UINT64_C(0x48ac878f8968ca07)},
        {UINT32_C(0x0001bcbb), UINT64_C(0x2acb14e53d57da94)},
        {UINT32_C(0x0000de5c), UINT64_C(0xb706384ddbaf140a)},
        {UINT32_C(0x00006f2e), UINT64_C(0x23ebb6cdf12726eb)},
        {UINT32_C(0x00003797), UINT64_C(0x04100ff69b6d68a4)},
        {UINT32_C(0x00001bcb), UINT64_C(0x7e8e96dbf0662f6c)},
        {UINT32_C(0x00000de5), UINT64_C(0xbe68ef5db7f99bee)},
        {UINT32_C(0x000006f2), UINT64_C(0xdefce0b1becf7c00)},
        {UINT32_C(0x00000379), UINT64_C(0x6f708a9a767866a6)},
        {UINT32_C(0x000001bc), UINT64_C(0xb7b4cbddbccbdada)},
        {UINT32_C(0x000000de), UINT64_C(0x5bd98793024346d5)},
        {UINT32_C(0x0000006f), UINT64_C(0x2dec8c328a8827b3)},
        {UINT32_C(0x00000037), UINT64_C(0x96f6383387ab9aa9)},
        {UINT32_C(0x0000001b), UINT64_C(0xcb7b18a054716bc0)},
        {UINT32_C(0x0000000d), UINT64_C(0xe5bd8b71ce5fd512)},
        {UINT32_C(0x00000006), UINT64_C(0xf2dec5815039b948)},
        {UINT32_C(0x00000003), UINT64_C(0x796f62b2c25f5132)},
        {UINT32_C(0x00000001), UINT64_C(0xbcb7b155e7c045d8)},
    };
    constexpr int E = Exponent - (Precision == 1);
    constexpr auto C = CORRECTIONS[Precision];
    constexpr int LOWER =
        log_floor(E, {LOG10_2.high, LOG10_2.low + (E < 0)}, {C.high, C.low + (Precision > 1)});
    constexpr int UPPER = log_floor(E, {LOG10_2.high, LOG10_2.low + (E >= 0)}, C);
    static_assert(LOWER == UPPER, "decimal exponent interval crosses an integer");
    return LOWER;
  }

  // floor(log10(max())), read off the maximal finite code rather than off
  // `MAX_EXP`. The two part company wherever a format spends the top of its
  // exponent range on something that is not a number: `FN<5, 2>` tops out at
  // 2^17 * (1 - 2^-2), which is under 10^5 although `MAX_EXP` is 17, and
  // `FN<E, 0>` loses its whole top row to the one NaN. Reading the code also
  // makes the bias fall out for free, wherever a caller supplies an odd one.
  static constexpr int exact_max_exponent10() noexcept {
    constexpr unsigned MAG = Format::MAX_FINITE_MAG;
    constexpr unsigned MAN_MASK = (1U << T::MANTISSA_BITS) - 1U;
    // The significand is one bit shorter wherever the all-ones magnitude is
    // spent elsewhere; `M == 0` falls out of an empty mask matching itself.
    constexpr int PRECISION = T::MANTISSA_BITS + ((MAG & MAN_MASK) == MAN_MASK);
    // `E >= 2` leaves the maximal code's exponent field at 2 or more, so this
    // is always the binade of a normal value.
    constexpr int BINADE = static_cast<int>(MAG >> T::MANTISSA_BITS) - T::BIAS + 1;

    return decimal_exponent<BINADE, PRECISION>();
  }

  // ceil(log10(min())) in the `FLT_MIN_10_EXP` sense: the least power of ten
  // that is still a normal value. `(MIN_EXP - 1) * 30103 / 100000` was a
  // ceiling only for as long as it truncated toward zero from below, and
  // `MIN_EXP` is `2 - BIAS`, so a bias at or below zero lifts the dividend
  // above zero and turns that same truncation into a floor.
  static constexpr int exact_min_exponent10() noexcept {
    return -decimal_exponent<1 - T::MIN_EXP, 0>();
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

  static constexpr T(min)() noexcept { return (T::min)(); }
  static constexpr T(max)() noexcept { return (T::max)(); }
  static constexpr T lowest() noexcept { return -(T::max)(); }
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
