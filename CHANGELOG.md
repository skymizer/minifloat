# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with Cargo-style compatibility before 1.0: `0.y.z` releases may break
compatibility when `y` changes, while changes to `z` remain compatible.

## [0.2.1] - 2026-08-25

### Changed

- Exact narrow host inputs now encode by rebasing and rounding their integer
  fields, and arithmetic shares one subnormal/normal rounding path.
- Lossy `to_float` and `to_double` conversions now build host fields directly,
  so they always round to nearest-even and ignore directed rounding and
  flush-to-zero settings.

## [0.2.0] - 2026-08-24

Rewrite of the header as layered formats over one engine. This is a breaking
change: the `Minifloat<E, M, NanStyle, Bias, SubnormalStyle>` signature and the
generated typedef grid are gone.

### Added

- `Finite<E, M, B>`, a format where every bit pattern is a number: no infinity,
  no NaN. This is what the OCP MX types need, and it is the layer the other
  three build on.
- The OCP MX element formats `E2M1FN` (FP4), `E2M3FN` and `E3M2FN` (FP6), which
  0.1.0 could not express.
- `IEEE<E, M, B>`, `FN<E, M, B>`, and `FNUZ<E, M, B>` as the named format
  templates, replacing the `NanStyle` parameter.
- Type aliases for LLVM APFloat's set of formats at 16 bits or fewer, without
  its `FloatN` prefix: `E2M1FN`, `E2M3FN`, `E3M2FN`, `E3M4`, `E4M3`, `E4M3FN`,
  `E4M3FNUZ`, `E4M3B11FNUZ`, `E5M2`, `E5M2FNUZ`, `E5M10` (binary16), and `E8M7`
  (bfloat16).
- `BF<N>` for the 10- through 32-bit brain-float family: an 8-bit exponent over
  `N - 9` mantissa bits. `BF<16>` is `E8M7`, and `BF<32>` has `float`'s layout.
- Formats through 32 total bits, stored in `uint_least32_t` where needed.
- `HAS_INF`, `HAS_NAN`, and `HAS_NEG_ZERO` as public constants of every type.
- `IS_HOST_FLOAT`, true for a type that *is* `float` bit for bit — matching
  precision, exponent range and non-finite semantics, which among the shapes
  this library admits means `IEEE<8, 23>` and nothing else. It is a stronger
  question than `HAS_EXACT_F32_CONVERSION`, which only asks whether a `float`
  can hold every value. It reports a fact about the layout and selects nothing:
  `to_float` is the identity where it holds, while construction from a `float`
  preserves every non-NaN bit pattern and canonicalizes NaN payloads. Arithmetic
  is the same integer engine every other shape gets.
- `docs/arithmetic.md` and `docs/benchmarking.md`, recording the decisions
  behind the integer route and the protocol every measured claim has to meet,
  and `CLAUDE.md` as the routing table to them.
- `Arith.IgnoresHostEnvironment`, which puts 4100 operand pairs through all four
  operators under `FE_UPWARD`, `FE_DOWNWARD` and MXCSR's FTZ and DAZ bits and
  requires every answer to equal the one the default environment gives, with a
  native `float` computed beside it as the control so that a platform ignoring
  the request cannot make the pass vacuous.
- Exhaustive sweeps over all 2³² ordered pairs of `E5M10` and `E8M7`: the four
  operators against a `float` round trip, and the comparisons against both host
  types. The other quadratic checks stop at 11 bits; these are the two benchmark
  shapes routed through `float` by the 2*p* + 2 rule, so they get the exhaustive
  referee. `BF<32>` is on that route as well — it *is* `float`, so IEEE 754
  rounds it correctly with nothing to narrow — and takes 2²² sampled pairs,
  its own pair space being 2⁶⁴.
- Exhaustive code sweeps through 20 bits and odd-stride samples above that,
  sampled wide-format exact-arithmetic and eligible host-round-trip checks
  reaching `IEEE<6, 25>` and `IEEE<2, 29>` at the *p* = 30 ceiling the engine is
  argued correct to, and a `BF<32>` check that matches sampled non-NaN bit
  patterns in both directions and preserves NaN class and sign while payloads
  canonicalize.
- A benchmark page at <https://skymizer.github.io/minifloat/dev/bench/>,
  refreshed by `.github/workflows/bench.yml` on every push to `main` from
  `./bench --json`. One shared-runner GCC sample per commit is a tripwire for a
  2× cliff, not a measurement: `docs/benchmarking.md` says what it is not, and
  nothing read off it enters a commit body or this file.

### Changed

- The library is now one engine, `Minifloat<Format>`, over a chain of format
  policies that each reinterpret one part of the bit space in terms of the layer
  below: scientific → finite → {IEEE, FN, FNUZ}. Adding a format is a policy,
  not a class.
- `FNUZ` defaults to a bias one greater than `default_bias(E)`, matching LLVM:
  `E4M3FNUZ` has bias 8 and `E5M2FNUZ` bias 16. `FN` and `IEEE` are unchanged.
- Converting a NaN to a format that has none (`Finite`) is a documented
  precondition violation. Debug builds assert; release builds saturate.
- `+`, `-`, `*`, and `/` are computed on integer significands and rounded once,
  instead of being evaluated in a host `float` or `double` and rounded back.
  Every operator is now correctly rounded for every declared shape, and the four
  of them are `constexpr`. Encoding a host float shares that one rounding path.
  No shape is exempt, including `BF<32>`, which *is* `float` bit for bit: an FPU
  rounds each operator once, but it rounds the way the caller's rounding mode
  says and flushes subnormals the way the caller's `MXCSR` says, so no operator
  reaches one. An answer therefore never moves with the floating-point
  environment a caller left behind — which is a promise about the operators and
  about encoding, not about `to_double` for a shape whose exponent range exceeds
  `double`'s, where the conversion is already documented as lossy.
- Addition's alignment window is 32 binades instead of 46, and division
  normalizes its dividend to bit 62 instead of using a fixed 46-bit shift. These
  bounds keep the integer engine correctly rounded through 30-bit significands.
- `IntegerDecode::exponent` is now `int32_t`, so formats with exponent fields of
  16 bits or wider decode without overflowing.
- Conversion to and from a host float no longer calls libm. Powers of two are
  built from the exponent field by `detail::exp2i` instead of `std::exp2` and
  `std::ldexp`, neither of which the compilers reliably folded even where the
  argument was a literal, and a host float is taken apart by one `bit_cast`
  instead of `std::signbit` plus `std::isnan` plus `std::isinf` plus a widening
  to `double`. Construction from a `float` costs 0.71x of what it did under
  Clang and 0.83x under GCC, and `to_float`/`to_double` for a shape with no
  exact host conversion costs about a quarter. `docs/arithmetic.md` has the
  measurements and the null this round also produced.
- `BF<10>` through `BF<31>` now convert to and from `float` by shifting their
  shared exponent field instead of taking the generic decomposition path.
  `BF<20>` and `BF<24>` conversion costs 0.18x–0.34x across GCC and Clang; their
  arithmetic and `double` conversions are unchanged. Construction still
  canonicalizes NaNs, while `to_float` now preserves a stored NaN payload.
- Default-biased `IEEE<11, 1>` through `IEEE<11, 20>` now take the same
  field-shift route to and from `double`. Their arithmetic is unchanged;
  construction canonicalizes NaNs, while `to_double` preserves a stored NaN
  payload.
- `detail::log2_floor` finds the top bit by halving where neither
  `std::countl_zero` nor `__builtin_clzll` is reachable, instead of shifting one
  bit at a time. That branch is MSVC's in every standard, since its `__cplusplus`
  stays at 199711L without `/Zc:__cplusplus`, and it is on the path of every
  operator and every host-float conversion through `from_parts`. The MSVC leg's
  exhaustive 2³² sweeps went from 783 s to 594 s.
- `detail::log2_floor` then goes further on 64-bit MSVC, calling
  `_BitScanReverse64` outside constant evaluation and keeping the halving search
  for inside it. The discriminator is `__builtin_is_constant_evaluated`, which
  MSVC exposes as a compiler intrinsic in every standard mode rather than as the
  C++20 library entity, so the C++17 leg reaches it; the intrinsic is declared in
  the header and pinned with `#pragma intrinsic` rather than pulled in with
  `<intrin.h>`, which would be a dependency the library does not otherwise take.
  32-bit MSVC, which has no 64-bit bit scan, keeps the halving search. The MSVC
  leg's sweeps went from 594 s to 508 s at the median of six runs each. Note
  that GitHub's Windows pool is bimodal — the same six runs also drew a host
  where those figures are 352 s and 292 s — so the two thirds of the original
  783 s that survives the halving search is largely which host the job landed
  on, not codegen: on a fast draw this leg is level with GCC/Linux.

- An invalid operation now returns the format's own NaN, or its maximum finite
  value where the format has none, rather than inheriting a host NaN's sign:
  `0 / 0` in a `Finite` format used to be &minus;`max()` on x86 and +`max()` on
  ARM. Invalid means a NaN operand, opposite infinities added, infinity minus
  itself, infinity times zero, zero over zero, and infinity over infinity.

### Removed

- `Minifloat<E, M, NanStyle, Bias, SubnormalStyle>`, `NanStyle`,
  `SubnormalStyle` (`Precise` / `Fast` / `Reserved`), and `FpClassifier`. There
  is no compatibility alias.
- The 301 macro-generated `E<x>M<y>[FN|FNUZ]` typedefs, replaced by the alias
  list above plus the format templates.
- `round_normal_float_to_mantissa` and `round_normal_double_to_mantissa`, which
  were implementation details. The integer rounding path replaced them.
- `USE_FLT_ADD` and `USE_FLT_MUL`, public constants that selected the host type
  an operator evaluated in. There is no host route left to select, for any
  shape, whether or not a constant says so.

### Fixed

- Exact conversion of a Minifloat subnormal now constructs the destination
  fields directly. It no longer multiplies in the host type, where FTZ or DAZ
  could turn an exactly representable value into zero.
- Lossy `to_float()` and `to_double()` conversions now return a stored signed
  zero before scaling. With a large negative bias, the old scale split could
  otherwise evaluate zero times infinity and produce a NaN.
- Integral construction now rounds directly from the integer magnitude instead
  of first rounding through `double`; `BF<32>{9007199791611905}` now produces
  `0x5A000001` in one rounding. `bool` construction is unambiguous and uses the
  same direct integer path. Integral conversion now maps NaN to zero and
  saturates infinity and out-of-range values instead of invoking undefined
  behaviour.
- Format biases are constrained to the range in which the integer engine's
  exponent arithmetic cannot overflow.
- `std::numeric_limits<T>::epsilon()` and `round_error()` now saturate to
  `max()` when their power of two lies above a format's finite exponent range,
  rather than wrapping the exponent field.
- `detail::bit_cast` was declared `[[gnu::const]]`, which promises the result
  depends on the argument *values* alone — for a reference parameter, the
  address rather than what is at it. GCC's own documentation forbids it for a
  function that examines what a pointer argument points to, and GCC 11.4 at
  `-O1` takes the promise: `BF<32>{2.0F}.to_bits()` folded to zero and
  `BF<32>{2.0F} * BF<32>{3.0F}` to a NaN, while `-O0`, `-O2` and `-O3` came out
  right. It is `[[gnu::pure]]` now. The attribute has been wrong since 0.1.0;
  what made it reachable was this round's `bit_cast` conversion fast path.
  Neither `make check` nor the CMake route can see it, both building optimized.
- `std::numeric_limits<T>::max_exponent10` read the top of the exponent range
  and ignored the significand of `max()`, so it came out one too high for every
  format that spends the all-ones magnitude on something that is not a number.
  `FN<5, 2>` said 5 for a maximum of 98304, and `FN<7, 0>` and `FNUZ<7, 0>` said
  19 for maxima below 10¹⁹. It is now derived from the maximal finite code.
- `std::numeric_limits<T>::min_exponent10` truncated toward zero, which is the
  ceiling the standard asks for only while the value is negative. A bias at or
  below zero — legal while it remains inside the exponent-arithmetic bound —
  puts `min()` above 1 and turns that truncation into a floor:
  `Finite<4, 3, 0>::min()` is 2, and the member said 0 where 1 is the least
  power of ten inside the normal range.
- NaN is no longer detected with `x != x`, which a compiler is free to fold
  away; the encoder reads the exponent field and the payload out of one
  `bit_cast` instead (#3).
- `to_double()` rebuilt garbage bits for formats whose exponent range exceeds
  `double`'s, such as `IEEE<12, 3>`; it now scales in two in-range steps.
- Constructing such a format from a host zero or subnormal read the zero
  exponent field as an ordinary exponent, so `IEEE<12, 3>{0.0F}` returned the
  maximum finite value instead of zero. The same expression also shifted a
  negative value left, which is undefined behaviour.
- Arithmetic on a shape whose exponent range exceeds `double`'s lost results the
  shape represents: `IEEE<12, 3>` squared 2⁻¹⁰⁰⁰ to zero and 2¹⁰⁰⁰ to infinity.

## [0.1.0] - 2026-08-20

Initial public release.

### Added

- Header-only `Minifloat` template supporting configurable exponent width,
  mantissa width, bias, NaN encoding, and subnormal handling for formats up to
  16 bits.
- IEEE-style, finite-only (`FN`), and unsigned-zero finite-only (`FNUZ`) type
  aliases, including `E4M3`, `E4M3FN`, and `E5M2`.
- Explicit conversion to and from host floating-point and integer types,
  arithmetic and comparison operators, classification, bit-level round trips,
  and `integer_decode`.
- `std::numeric_limits` and `std::hash` specializations.
- CMake build, installation, version discovery, and exported
  `skymizer::minifloat` target.
- Tests for GCC/Linux, Apple Clang/macOS, and MSVC/Windows with a C++17 minimum,
  plus C++20, sanitizer, and installed-consumer coverage.

### Known limitations

- `-ffast-math` and equivalent finite-math modes are unsupported because they
  can discard required NaN, infinity, signed-zero, and subnormal semantics.
- Host `float` and `double` types must use IEEE 754 binary32 and binary64
  representations.

[0.2.1]: https://github.com/skymizer/minifloat/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/skymizer/minifloat/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/skymizer/minifloat/commits/0.1.0
