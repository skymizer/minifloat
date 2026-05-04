Minifloat
=========
C++ template library for minifloats dedicated to [@skymizer][skymizer]

[skymizer]: https://github.com/skymizer

This header-only C++ library provides emulation of minifloats up to 16 bits.
In this library, implicit conversions are proven lossless at compile time.
This design prevents accidental loss of precision and allows the user to
perform potentially lossy conversions explicitly.

## Quick start

```cpp
#include <minifloat.hpp>

using skymizer::Minifloat;
using skymizer::minifloat::NanStyle;
using skymizer::minifloat::SubnormalStyle;

// Standard FP8 E4M3 (IEEE NaN style) and E5M2 shapes are predefined.
using skymizer::minifloat::E4M3;
using skymizer::minifloat::E5M2;

E4M3 a{1.5F};
E4M3 b{0.25F};
E4M3 c = a + b;          // 1.75 in E4M3
float f = c.to_float();  // explicit, lossy if inexact

// Custom shapes, e.g. an LLVM/MLIR-style FP8 with FN NaN encoding:
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

The library treats *implicit* conversions as *proven lossless*: a conversion
template only kicks in when the compiler can statically prove (via the
`HAS_EXACT_F32_CONVERSION` / `HAS_EXACT_F64_CONVERSION` traits) that no
precision is lost. If a conversion would be lossy, the user must call
`to_float()` / `to_double()` (or use `static_cast`) to opt in.

The NaN encoding (`NanStyle`) and subnormal handling (`SubnormalStyle`) are
template parameters so the same Minifloat template covers IEEE-style 754
encodings, the LLVM/MLIR `FN` and `FNUZ` variants, and trade-offs between
precise / fast / reserved subnormal handling.

## Dependencies

- C++17 standard library
- Native floating-point arithmetics (`float` and `double`)

### Additional dependencies for testing

- GCC-compatible compiler (including Clang/LLVM)
- Un*x `make` or CMake (≥ 3.14)
- Google Test

### Building with CMake

```sh
cmake -B build -DSKYMIZER_MINIFLOAT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Downstream projects can either `add_subdirectory(minifloat)` or install the
package and `find_package(skymizer-minifloat)`, then link `skymizer::minifloat`.