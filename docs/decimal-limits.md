# Decimal exponent limits

`numeric_limits<T>::min_exponent10` is the ceiling of `log10(min())`;
`max_exponent10` is the floor of `log10(max())`. The admitted biases put their
binary exponents near ±2³⁰. A double product can then round across the integer
whose side these functions need to distinguish:

```
198096465 * log10(2) = 59632978.000000002598316594477929...
146964308 * log10(2) = 44240664.999999996879600048755869...
```

`long double` does not fix this portably: MSVC gives it double's precision.
The header instead uses fractions with 96 binary places. It multiplies the
binary exponent by a bounded `log10(2)` and subtracts a bounded
`-log10(1 - 2^-p)` for the maximal significand. Precision one reduces to a
power of two, including the exact boundary at 1; precision zero denotes an
uncorrected power of two. Three 32-bit products and carries suffice, without
compiler-specific 128-bit integers. Signed division is explicitly floored,
without relying on right shifts of negative integers.

Every constant is a lower bound; one unit above it is an upper bound. Each
instantiation computes both bounds and asserts that their floors agree. The
following independent integer check also verifies this for **every** binary
exponent of magnitude at most `2^30 + 32` and every admitted precision. The
extra 32 covers the mantissa adjustment at the bias bounds. It sums the floor
differences with Euclidean division instead of enumerating billions of values.
Each difference is nonnegative, so a zero sum proves agreement everywhere.

Run this with Python 3 to regenerate the constants and repeat the domain check;
Python is not needed to build or use the library.

```python
from decimal import Decimal, localcontext


def floor_sum(n, modulus, a, b):
    """Sum floor((a*i + b)/modulus) for 0 <= i < n."""
    result = 0
    while True:
        qa, a = divmod(a, modulus)
        qb, b = divmod(b, modulus)
        result += qa * n * (n - 1) // 2 + qb * n
        top = a * n + b
        if top < modulus:
            return result
        n, b = divmod(top, modulus)
        modulus, a = a, modulus


with localcontext() as context:
    context.prec = 100
    scale = 1 << 96
    limit = (1 << 30) + 32
    logarithm = int(Decimal(2).log10() * scale)
    corrections = [0, 0] + [
        int(-(1 - Decimal(2) ** -p).log10() * scale)
        for p in range(2, 31)
    ]
    for value in [logarithm] + corrections:
        print(f"{{0x{value >> 64:08x}, 0x{value & ((1 << 64) - 1):016x}}},")
    for p in [0] + list(range(2, 31)):
        correction = corrections[p]
        upper_correction = correction + (p != 0)
        for sign in [1, -1]:
            lower = sign * (logarithm if sign > 0 else logarithm + 1)
            upper = sign * (logarithm + 1 if sign > 0 else logarithm)
            difference = floor_sum(limit, scale, upper, upper - correction)
            difference -= floor_sum(limit, scale, lower, lower - upper_correction)
            assert difference == 0, (p, sign, difference)
        # At exponent zero, a corrected value is strictly between zero and one.
        assert (-upper_correction // scale) == (-correction // scale)
    print("All exponent intervals agree.")
```
