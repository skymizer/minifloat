Minifloat
=========
C++ template library for minifloats dedicated to @skymizer

This header-only C++ library provides emulation of minifloats up to 16 bits.
In this library, implicit conversions are proven lossless at compile time.
This design prevents accidental loss of precision and allows the user to
perform potentially lossy conversions explicitly.

## Dependencies

- C++17 standard library
- Native floating-point arithmetics (`float` and `double`)

### Additional dependencies for testing

- GCC-compatible compiler (including Clang/LLVM)
- Un*x `make`
- Google Test