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

using skymizer::minifloat::E4M3;
using skymizer::minifloat::E5M2;

E4M3 a{1.5F};
E4M3 b{0.25F};
E4M3 c = a + b;          // 1.75 in E4M3
float f = c.to_float();  // explicit, lossy if inexact
```

A format is one of four templates over an exponent width `E`, a mantissa width
`M`, and an exponent bias `B`, each spending the bit space a little
differently:

| Template          | Reserved code points                       | Default bias        |
| ----------------- | ------------------------------------------ | ------------------- |
| `Finite<E, M, B>` | none — every pattern is a number           | `default_bias(E)`     |
| `IEEE<E, M, B>`   | top exponent row: infinities and NaNs      | `default_bias(E)`     |
| `FN<E, M, B>`     | all-ones magnitude: NaN                    | `default_bias(E)`     |
| `FNUZ<E, M, B>`   | the &minus;0.0 slot: NaN                   | `default_bias(E) + 1` |

Constraints: `E + M < 16`, `E >= 2`, `M >= 0`, and `M > 0` for `IEEE`.

## Type aliases

The named types are the formats LLVM's [`APFloat`][apfloat] knows at 16 bits or
fewer, spelled without its `FloatN` prefix.

[apfloat]: https://llvm.org/doxygen/structllvm_1_1APFloatBase.html

| Alias          | Definition       | LLVM name              | Max finite |
| -------------- | ---------------- | ---------------------- | ---------- |
| `E2M1FN`       | `Finite<2, 1>`   | `Float4E2M1FN`         | 6          |
| `E2M3FN`       | `Finite<2, 3>`   | `Float6E2M3FN`         | 7.5        |
| `E3M2FN`       | `Finite<3, 2>`   | `Float6E3M2FN`         | 28         |
| `E3M4`         | `IEEE<3, 4>`     | `Float8E3M4`           | 15.5       |
| `E4M3`         | `IEEE<4, 3>`     | `Float8E4M3`           | 240        |
| `E4M3FN`       | `FN<4, 3>`       | `Float8E4M3FN`         | 448        |
| `E4M3FNUZ`     | `FNUZ<4, 3>`     | `Float8E4M3FNUZ`       | 240        |
| `E4M3B11FNUZ`  | `FNUZ<4, 3, 11>` | `Float8E4M3B11FNUZ`    | 30         |
| `E5M2`         | `IEEE<5, 2>`     | `Float8E5M2`           | 57344      |
| `E5M2FNUZ`     | `FNUZ<5, 2>`     | `Float8E5M2FNUZ`       | 57344      |
| `E5M10`        | `IEEE<5, 10>`    | `IEEEhalf` (binary16)  | 65504      |
| `E8M7`         | `IEEE<8, 7>`     | `BFloat` (bfloat16)    | ≈ 3.39 · 10³⁸  |

The `FN` suffix in an alias is LLVM's name for that format, not a promise about
NaN: the OCP MX types `E2M1FN`, `E2M3FN`, and `E3M2FN` have no NaN at all, which
is why they are `Finite`. The `FN` *template* always means "the all-ones
magnitude is NaN", so `FN<2, 1>` is a different type from `E2M1FN` — it tops out
at 4 rather than 6.

The unsigned `UE4M3` of the OCP MX specification is `E4M3FN` restricted to a
zero sign bit; this library has no unsigned layout of its own.

Helpful entry points:

- `from_bits(bits)` / `to_bits()` — round-trip the storage representation.
- `to_float()` / `to_double()` — explicit conversion to the host types.
- `integer_decode(x)` — `(mantissa, exponent, sign)` triple; `sign == 0` is
  the NaN sentinel.
- `std::numeric_limits<...>` — `min`, `max`, `lowest`, `epsilon`,
  `round_error`, `infinity`, `quiet_NaN`, `denorm_min`, plus the usual traits.
- `std::hash<...>` — usable as a key in unordered containers.

## Design

The library is one engine, `Minifloat<Format>`, over a chain of format policies.
Each policy reinterprets one part of the bit space in terms of the layer below
it: `ScientificFormat` reads every magnitude as `(1 + m/2^M)·2^(e−B)`;
`FiniteFormat` spends row 0 on zero and subnormals; and `IeeeFormat`,
`FnFormat`, and `FnuzFormat` each reserve their own code points on top of that.
A layer only *uses* the one below — static calls and a template parameter, no
inheritance and no contained object — so adding a format is a policy of about
forty lines rather than a class with thirty members.

The alternative, wrapping a `Finite` value inside an `IEEE` value, would have
each wrapper re-implement roughly twenty members, because infinities, NaN, and
the &minus;0.0 slot reach into every conversion, all six comparisons, every
classifier, and arithmetic (which must round through the *outer* type). Policies
keep one responsibility per layer without making each layer a value type.

The engine consults only orthogonal, format-declared properties — `HAS_INF`,
`HAS_NAN`, `HAS_NEG_ZERO`, the reserved code points, and a `kind` classifier
that each layer answers for its own code points before delegating inward. Every
branch on them is `if constexpr`, so the generated code for a concrete format
carries no trace of the layering.

Construction from numeric types and conversion back through `to_float()`,
`to_double()`, or `static_cast` are explicit. The `HAS_EXACT_F32_CONVERSION` and
`HAS_EXACT_F64_CONVERSION` traits report whether the corresponding host type can
represent every value of a Minifloat type, but do not make the conversion
implicit.

Rounding is to nearest, ties to even. Arithmetic is correctly rounded: every
operator works out a result exact enough to round, on integer significands, and
rounds it once. Multiplication of two significands is exact; addition aligns
both addends and sums them in an `int64_t`; division keeps 46 quotient bits and
folds the remainder into a sticky bit. No host float takes part, so a shape
whose exponent range outruns `double`'s is served like any other — `IEEE<12, 3>`
squares 2⁻¹⁰⁰⁰ to 2⁻²⁰⁰⁰ rather than to zero — and an invalid operation yields
the format's own NaN, or its maximum finite value where it has none, instead of
whatever sign the host's default NaN happened to carry.

Correctness, not speed, is why the host route is gone; it is not the slower one
either, though by how much depends on the compiler. `benches/arith.cpp` times
both routes over the same operands, and [docs/arithmetic.md](docs/arithmetic.md)
has the numbers and what they do and do not license.

## Design notes

Standing decisions, with the measurements and the rejected alternatives behind
them, for anyone working on the library rather than using it:

- [docs/arithmetic.md](docs/arithmetic.md) — the integer route, the two
  deliberately inexact tails and why neither can change a rounding, the
  independent oracle, and the open questions.
- [docs/benchmarking.md](docs/benchmarking.md) — what a number from this
  repository has to survive before it is quoted.

## Dependencies

- C++17 standard library
- IEEE 754 binary32 `float` and binary64 `double`

The library is tested with GCC on Linux, Apple Clang on macOS, and MSVC on
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

`make check` builds and runs the same suite through the plain Makefile, and
`make run-bench` builds `benches/arith.cpp` and runs it pinned to one core.

Downstream projects can either `add_subdirectory(minifloat)` or install the
package:

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build
cmake --install build
```

An installed package supports Cargo-style pre-1.0 compatibility: `0.2.x`
releases are compatible, while breaking changes advance to `0.3.0`. Consume it
from CMake with:

```cmake
find_package(skymizer-minifloat 0.2 CONFIG REQUIRED)
target_link_libraries(your-target PRIVATE skymizer::minifloat)
```
