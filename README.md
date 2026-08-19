Minifloat
=========
C++ template library for minifloats dedicated to [@skymizer][skymizer]

[skymizer]: https://github.com/skymizer

This header-only C++ library provides emulation of minifloats up to 16 bits.
All numeric conversions are explicit so that rounding, overflow, and other
representation changes stay visible at call sites.

## Quick start

```cpp
#include <minifloat.hpp>

using skymizer::Minifloat;
using skymizer::minifloat::NanStyle;
using skymizer::minifloat::SubnormalStyle;

// IEEE-style E4M3 and E5M2 shapes are predefined.
using skymizer::minifloat::E4M3;
using skymizer::minifloat::E5M2;

E4M3 a{1.5F};
E4M3 b{0.25F};
E4M3 c = a + b;          // 1.75 in E4M3
float f = c.to_float();  // explicit, lossy if inexact

// Finite-only E4M3 and custom shapes are also available.
using skymizer::minifloat::E4M3FN;
using FP8 = Minifloat<4, 3, NanStyle::FN>;
```

Helpful entry points:

- `Minifloat<E, M, NanStyle, Bias, SubnormalStyle>` — the main class.
- `from_bits(bits)` / `to_bits()` — round-trip the storage representation.
- `to_float()` / `to_double()` — explicit conversion to the host types.
- `integer_decode(x)` — `(mantissa, exponent, sign)` triple; `sign == 0` is
  the NaN sentinel.
- `std::numeric_limits<Minifloat<...>>` — `min`, `max`, `lowest`, `epsilon`,
  `round_error`, `infinity`, `quiet_NaN`, `denorm_min`, plus the usual traits.
- `std::hash<Minifloat<...>>` — usable as a key in unordered containers.

## Design

The library requires explicit construction from numeric types and explicit
conversion back through `to_float()`, `to_double()`, or `static_cast`. The
`HAS_EXACT_F32_CONVERSION` and `HAS_EXACT_F64_CONVERSION` traits report whether
the corresponding host type can represent every value of a Minifloat type,
but do not make the conversion implicit.

The NaN encoding (`NanStyle`) and subnormal handling (`SubnormalStyle`) are
template parameters so the same Minifloat template covers IEEE-style 754
encodings, the LLVM/MLIR `FN` and `FNUZ` variants, and trade-offs between
precise / fast / reserved subnormal handling.

## Dependencies

- C++17 standard library
- IEEE 754 binary32 `float` and binary64 `double`

Version 0.1.0 is tested with GCC on Linux, Apple Clang on macOS, and MSVC on
Windows. C++20 is tested in addition to the C++17 minimum. `-ffast-math` and
equivalent finite-math modes are unsupported because they may discard the NaN,
infinity, signed-zero, and subnormal semantics that this library preserves.

### Additional dependencies for testing

- A supported C++ compiler
- `make` or CMake (≥ 3.14)
- Google Test

### Building with CMake

```sh
cmake -B build -DSKYMIZER_MINIFLOAT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Downstream projects can either `add_subdirectory(minifloat)` or install the
package:

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build
cmake --install build
```

An installed package supports Cargo-style pre-1.0 compatibility: `0.1.x`
releases are compatible, while breaking changes advance to `0.2.0`. Consume it
from CMake with:

```cmake
find_package(skymizer-minifloat 0.1 CONFIG REQUIRED)
target_link_libraries(your-target PRIVATE skymizer::minifloat)
```
