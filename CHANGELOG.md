# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with Cargo-style compatibility before 1.0: `0.y.z` releases may break
compatibility when `y` changes, while changes to `z` remain compatible.

## [Unreleased]

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
