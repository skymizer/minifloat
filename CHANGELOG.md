# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with Cargo-style compatibility before 1.0: `0.y.z` releases may break
compatibility when `y` changes, while changes to `z` remain compatible.

## [Unreleased] — 0.2.0-dev

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
- `HAS_INF`, `HAS_NAN`, and `HAS_NEG_ZERO` as public constants of every type.

### Changed

- The library is now one engine, `Minifloat<Format>`, over a chain of format
  policies that each reinterpret one part of the bit space in terms of the layer
  below: scientific → finite → {IEEE, FN, FNUZ}. Adding a format is a policy,
  not a class.
- `FNUZ` defaults to a bias one greater than `default_bias(E)`, matching LLVM:
  `E4M3FNUZ` has bias 8 and `E5M2FNUZ` bias 16. `FN` and `IEEE` are unchanged.
- Converting a NaN to a format that has none (`Finite`) is a documented
  precondition violation. Debug builds assert; release builds saturate.

### Removed

- `Minifloat<E, M, NanStyle, Bias, SubnormalStyle>`, `NanStyle`,
  `SubnormalStyle` (`Precise` / `Fast` / `Reserved`), and `FpClassifier`. There
  is no compatibility alias.
- The 301 macro-generated `E<x>M<y>[FN|FNUZ]` typedefs, replaced by the alias
  list above plus the format templates.
- `round_normal_float_to_mantissa` and `round_normal_double_to_mantissa`, which
  were implementation details; they are now one function in `detail`.

### Fixed

- NaN is detected with `std::isnan` rather than `x != x`, which a compiler is
  free to fold away (#3).
- `to_double()` rebuilt garbage bits for formats whose exponent range exceeds
  `double`'s, such as `IEEE<12, 3>`; it now goes through `std::ldexp`.
- Constructing such a format from a host zero or subnormal read the zero
  exponent field as an ordinary exponent, so `IEEE<12, 3>{0.0F}` returned the
  maximum finite value instead of zero. The same expression also shifted a
  negative value left, which is undefined behaviour.

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

[Unreleased]: https://github.com/skymizer/minifloat/compare/0.1.0...HEAD
[0.1.0]: https://github.com/skymizer/minifloat/commits/0.1.0
