# The arithmetic

*Every operator works out a result exact enough to round correctly, on integer
significands, and rounds it once.  Nothing else in this document is as
important as that sentence.*

## Integers, not a hardware float

0.1.0 evaluated `+`, `-`, `*`, `/` by widening both operands to a host float,
letting the FPU work, and rounding the answer back.  That is only ever as good
as the host float's reach, and a declared shape can outrun it: `IEEE<12, 3>`
squares 2<sup>&minus;1000</sup> to zero and 2<sup>1000</sup> to infinity through
a `double`, though it represents both answers exactly.

The route now is `to_parts` → an exact integer computation → `from_parts`:

- **multiply** — two significands of at most 30 bits multiply exactly in a
  `std::uint64_t`, exponents add.  Genuinely exact.
- **add** — `detail::add_parts` aligns both addends on the lower exponent and
  sums them signed in an `std::int64_t`.  Addends more than `detail::ALIGN_CAP`
  = 32 binades apart drop the smaller one.  For precision *p* the cap must be at
  least *p* + 1 so the dropped value rounds straight back to the larger, and at
  most 62 &minus; *p* so the aligned sum fits `std::int64_t`; 32 meets both
  bounds through *p* = 30.
- **divide** — `detail::div_parts` normalizes the dividend to bit 62 before the
  integer division, then folds the remainder into the quotient's lowest bit as
  sticky.  A fixed shift leaves too few quotient bits for a subnormal dividend
  over a full-width divisor.  Normalization yields at least 63 &minus; *p* bits,
  enough for the *p* + 2 rounding bits through *p* = 30, and keeps the quotient
  below 2<sup>63</sup>.

Addition's dropped addend and division's sticky remainder are deliberately
inexact tails.  Neither can change a rounding, which is why the thesis says
*exact enough to round*.

Normalizing the dividend is not free.  Against the preceding kernel, on an
otherwise idle Ryzen 9 7950X3D under 15 alternating passes pinned to core 2 on
2026-08-23, division took 1.056x under GCC 11.4 and 1.097x under Clang 14.  Every
one of the 14 existing rows lay beyond its compiler's unchanged-`mul` control
band (0.963x–1.002x and 0.982x–1.008x).  The fixed-width divider is incorrectly
rounded for admitted wide significands, so correctness buys that measured
regression.

`detail::from_parts` is the only place *arithmetic* rounds, ties to even, by
way of `detail::round_to_scale`.  One rounding, so no intermediate can lose
what the format is able to hold, and a shape whose exponent range overruns
`double`'s is served as exactly as any other.  The library's one other rounding
is genuinely elsewhere: `to_double` splitting its scale into two in-range
factors once a shape's exponent leaves `double`'s, where the second multiply
rounds.

That rounding is also where a caller's floating-point environment still reaches,
and it is worth naming the whole of it rather than discovering it a third time.
Two sites emit a host arithmetic instruction, both on conversions the shape
cannot make exactly: `to_double`'s scale split, for the six shapes without
`HAS_EXACT_F64_CONVERSION`; and `to_float`'s
`static_cast<float>(to_double())` fallback, for the nine shapes that take
neither the field shift nor `to_exact`.  A rounding mode or an FTZ bit moves
those.  It moves nothing else: the four operators, the comparisons,
`from_parts`, integral construction and exact host conversions contain no host
arithmetic.  `Arith.IgnoresHostEnvironment` pins the operators, and
`Convert.ExactConversionsIgnoreFlushToZero` pins the exact conversion branch
that once multiplied subnormals.  The `BF<32>` section below is what happens
when this is forgotten.  `bits_from` rounds a host float *in* either by dropping
mantissa bits directly when its exponent field matches the source type's, or by
rebasing and rounding its integer fields when the destination is exact in the
source and its true minimum is above the source's subnormal range.  The latter
gate makes every source subnormal round to zero, leaving the remaining narrow
shapes to skip both `decompose` and `from_parts`.  Other inputs decompose exactly
and hand the triple to that same `from_parts`.  Integral construction hands an
integer significand directly to `from_parts`, and `to_exact` constructs host
fields directly.  None of these routes performs host arithmetic or reads the
caller's floating-point mode.

Across the 12 unary `from` rows that take the rebased-field tier, after over
before was 0.943x under GCC 11.4 and 0.818x under Clang 14.  These are the
geomeans, not per-shape claims: code placement left the six unchanged `from`
rows spread across 0.947x–1.007x and 0.899x–1.328x.  The `soft` controls
centered at 1.005x and 1.009x, and `neg`/`abs` at 1.005x and 0.998x.  Ryzen 9
7950X3D, 2026-08-24, minimum of 15 alternating passes pinned to core 2 under
the protocol in [benchmarking.md](benchmarking.md).

Subtraction does not build a negated operand.  `detail::add_impl(x, y, flip)`
inverts the right sign where `add_parts` already has it as a `bool`; `operator+`
passes `false` and `operator-` passes `true`, both literals, so the flag folds
away before anything is emitted.  The minifloat-rs sibling found this: `x + -y`
made a format without a negative zero pay for `operator-`'s guard on every
subtraction, and the intermediate had no reason to exist.  This library was
ported with the finding already applied, so it has never had the gap.

The special cases stopped borrowing the host's at the same time.
`detail::invalid` yields the format's NaN, or its maximum finite magnitude where
the format has none, rather than whatever sign the host's default NaN carried —
`0 / 0` in a `Finite` format used to be &minus;`max()` on x86 and +`max()` on
ARM, which is a portability bug the 0.1.0 test suite could not see because it
compared against the same host float.

## Zero-mantissa formats need `parity_offset`

`round_to_scale` takes a `parity_offset` flag that the Rust sibling has no use
for, because its roster stops at exponent width 6 and never declares a format
with `M == 0`.  This library's `Finite<7, 0>`, `FN<7, 0>` and `FNUZ<7, 0>` do.

Ties to even is a question about the *stored code*, not about the significand.
Where `M > 0` the two coincide: the significand's low bit is the code's low bit.
Where `M == 0` the rounded significand is always exactly 1, so the significand's
low bit is a constant, and the bit that has to be even is the exponent field's.
`from_parts` computes `M == 0 && (e + Format::BIAS) % 2 == 0` and hands it to
`round_to_scale`, which flips the parity it reads.  The old bit-trick encoder
carried this for free; an integer significand has to be told.

## Why there is no hardware route left to choose but one

The speed is a bonus.  The reason is correctness: a host float cannot referee a
shape it cannot hold, so keeping it would have meant keeping a route that is
wrong for exactly the shapes this library exists to support.  The shape that
needs no refereeing because it *is* a host float looked like the exception for
one round; the section below is why it is not one.

But it is worth knowing what the bonus is, and here the answer depends on the
compiler and on the operator — it is not one number.  `benches/arith.cpp` times
each operator twice over the same operands — once as the library computes it,
once the way a caller would fake it. At commit `c045c04`, on an idle Ryzen 7
8700F, 2026-08-21, under the protocol in
[benchmarking.md](benchmarking.md):

| | wins | geomean | add | sub | mul | div |
| --- | --- | --- | --- | --- | --- | --- |
| Clang 22.1.8 | 37 of 56 | 1.044x | 0.948x | 0.914x | 1.254x | 1.093x |
| GCC 16.1.1 | 23 of 56 | 0.966x | 0.852x | 0.862x | 1.143x | 1.035x |

Multiplication and division win every row under Clang.  Under GCC they win on
the geomean but not everywhere, and on two different splits: `mul` takes 9 of
14, losing only narrow shapes, while `div` takes 6 of 14 and loses `E5M10` at
0.87x as readily as `E4M3FN` at 0.84x.

Addition and subtraction split two ways, and separating them is what makes the
table usable.  `E8M7` and `E11M4` — the two widest exponent ranges benched —
give both to the host route under *both* compilers, 0.56x to 0.86x.  That is
the library's own shape rather than a back end's: a wide exponent range is the
distance `align` has to shift across, and an FPU does that in its exponent
field for free.  `E2M13` is the opposite corner and both compilers agree on it
too, the other way: it wins every one of its four operators, 1.11x to 1.87x
under GCC and 1.35x to 1.61x under Clang.  It is the one shape whose host route
pays for a `double` round trip — `route` sends it there because 2*p* + 2 is 30
digits — without the integer route paying a wide exponent range for it.

Between those two corners the compilers part.  Clang stays near even, 0.89x to
1.11x, while GCC gives the host route every narrow shape except `FNUZ`, which it
wins at 1.09x to 1.10x.  Same source, same box; the difference is in the back
end, and the next section has the diagnosis.

The brain floats keep the split rather than settling it.  On a
Ryzen 9 7950X3D, 2026-08-23, under 15 alternating A/B passes pinned to core 2,
each binary taking its own minimum over 30 internal passes, and with
`host / soft` above one favouring the integer route:

| | BF20 | BF24 | BF32 | all 12 operator rows |
| --- | --- | --- | --- | --- |
| GCC 11.4 | 1.029x | 1.031x | 0.768x | 0.934x |
| Clang 14 | 0.889x | 0.857x | 0.649x | 0.791x |

`BF20` and `BF24` are timed against a `double` and `BF32` against a `float`.
The `BF32` column is superseded: it was measured against a constructor that
still decomposed, and the section below has the shape's current figures.  Of
the two that remain, both compilers give addition and subtraction to the host
route and multiplication to the integer engine.  Division favours the integer
route under GCC and the host under Clang, so neither speed nor one compiler
licenses a second engine.

The whole table is re-measured rather than spliced, so `BF20` and `BF24` are
not the 1.009x, 1.007x, 0.841x and 0.842x first recorded on 2026-08-23.
Nothing about those two shapes changed, including their route; the drift is
inside the control band this build pair's 68 unchanged rows give, 0.958x–1.045x
under GCC and 0.940x–1.164x under Clang.  Under GCC they are near enough to
parity that the aggregate settles nothing either way, which is the same reading
the earlier numbers got.

The box was not idle for this one.  Load average was 1.15 at the start and
17.02 and 14.33 at the ends of the two legs — a shared machine, other users'
work, not the benchmark's own.  Minimum-of-N absorbs that, since interference
only ever adds time, and the 68 unchanged rows agree across the pair at a
geomean of 0.999x under GCC and 1.007x under Clang.  The bands quoted above are
wider than [benchmarking.md](benchmarking.md)'s usual for the same reason, and
the per-row readings are taken against them rather than against a fixed floor.

## GCC's addition gap is Clang vectorizing `add_parts`

*The two `align` calls are a two-lane operation, and Clang 14 packs them into
one.  GCC 11.4 does not.  Nothing else in `operator+` accounts for the split.*

`operator+` decomposes into four cumulative stages, and timing them one on top
of another says which one the compilers disagree about.  On an idle core 2 of a
Ryzen 9 7950X3D, 2026-08-23, `-O3 -march=native -DNDEBUG`, minimum of 30 passes
over 1024 operand pairs, each column the nanoseconds that stage *adds* to the
one left of it:

| `E5M2` | `to_parts` &times;2 | `add_parts` | `from_parts` | non-finite ladder |
| --- | --- | --- | --- | --- |
| GCC 11.4 | 1.107 | +1.095 | +1.634 | +0.311 |
| Clang 14 | 1.034 | +0.549 | +2.023 | &minus;0.222 |

`to_parts` is a tie, `from_parts` is a GCC *win* by a quarter, and `add_parts`
costs GCC twice what it costs Clang.  The same shape of answer at `E4M3`
(+1.033 against +0.551), `E5M10` (+1.031 against +0.474) and `E8M7` (+1.290
against +0.878).

The disassembly says why.  Clang loads both operand codes as one 16-bit load,
moves them into an `xmm`, and runs both `to_parts` and both `align` calls in
parallel lanes — `vpsllvq` for the two variable shifts, a masked `vpsubq` for
the two sign negations, `vpshufd` plus `vpaddq` to fold the two aligned addends
together.  GCC emits two scalar `shlx` / `neg` / `cmovs` chains and an `add`.
That is the plan's first suspect confirmed and its stated mechanism refuted: the
two `align` calls *are* where the time goes, but as a missed SLP vectorization,
not as a missed if-conversion.

Taking the vector ISA away confirms it and then reverses it:

| `add_parts`, ns added | `-march=native` | `-march=x86-64-v3` | `-march=x86-64-v2` |
| --- | --- | --- | --- |
| GCC 11.4 | +1.095 | +1.007 | +1.041 |
| Clang 14 | +0.549 | +0.794 | +1.335 |

GCC is flat, because it never vectorized.  Clang's advantage is bought with
AVX-512 on Zen 4, shrinks to AVX2, and inverts once the packing is worth less
than the shuffles — at `x86-64-v2` Clang's `add_parts` is the slower one.  So
this is not a back end getting `operator+` wrong.  It is one back end finding a
two-lane operation in the source and the other not, and the finding is worth
what the host's vector width is worth.

Two things this closes off.  Since GCC's loss is not a defect to be fixed in
the source, the addition gap does not shrink on its own, and a `+`/`-` route
decision cannot wait on it.  And the non-finite ladder is not the story: it
costs about 0.3 ns under GCC and nothing measurable under Clang, which the
ladder column above shows directly.

That is worth stating because the ladder was the first hypothesis, and it
looked strong.  Inside `benches/arith.cpp`, GCC's `E4M3` soft addition ran
4.775 ns against `E4M3FN`'s 3.968 — same *E*, same *M*, same bias, an identical
integer kernel, and only the ladder between them.  A standalone binary holding
*E*, *M* and bias fixed and varying only the format layer found no such
ordering under either compiler; GCC's `IEEE` row came out *faster* than its
`Finite` one.  Comparing two shapes' rows inside the big benchmark binary
compares two placements as much as two bodies, which is
[benchmarking.md](benchmarking.md)'s warning arriving in a new disguise: the
rows a shape's own ratio is built from are adjacent and comparable, and rows
belonging to different shapes are not.

Rewriting the ladder as one `is_finite` guard was measured and rejected; the
before-and-after is on the `scratch/add-nonfinite-guard` ref, along with the
stage harness these numbers come from.

## `BF<32>` is `float`, and takes the integer engine anyway

*The one shape where a host route costs no rounding error — withdrawn, because
rounding error is not all a host route costs.*

`route`'s 2*p* + 2 rule is Figueroa's bound on *narrowing* a wide intermediate,
and `BF<32>` narrows nothing: `IEEE<8, 23>` has `float`'s precision, exponent
range and non-finite semantics, so IEEE 754 rounds each operator once, to
exactly the digits the shape stores.  `detail::is_host_float` is that predicate
and `Minifloat::IS_HOST_FLOAT` publishes it; the comment on the predicate says
why `FN<8, 23>`, `Finite<8, 23>` and `IEEE<7, 23>` all miss it.  For one round
that was read as a licence to give the shape's four operators an FPU
instruction.  It is not.  Rounding *once* is a different promise from rounding
to *nearest*, and an FPU is not a function of its operands: it is a function of
its operands and of a control register the caller owns.

Four readings against `c1f8b6f`, the last commit that had the route, on a Ryzen
9 7950X3D under Ubuntu 22.04, 2026-08-24, with `bit_cast`'s attribute corrected
first so that its own `-O1` fold is not read as one of these.  The first
reading is the whole case; the rest are how it was found and what it survives.

- `std::fesetround(FE_UPWARD)`, then `BF<32>::from_bits(0x3F800000) +
  BF<32>::from_bits(0x33800000)` — one plus half an ulp — answered
  `0x3F800001` at `-O0` and `0x3F800000` from `-O1` up, under *both* compilers.
  Same expression, same header, same environment, two answers, and which one a
  caller gets is the optimizer's business.  Neither is defensible as a
  contract: `0x3F800000` is ties to even, which is what every other shape
  answers and what a constant evaluation gives on that same header, and
  `0x3F800001` is what the caller asked for.  A result that moves with `-O`
  keeps neither promise.
- Put the operands where the fold cannot reach them — out of line, or through
  `volatile` — and the second answer is the only answer: `0x3F800001` under
  both compilers at `-O1`, `-O2` and `-O3` alike.  That is the defect with no
  compiler cleverness in it at all, the FPU honouring `FE_UPWARD` and the
  library breaking its own rounding contract.  The reading above is the same
  defect wearing a constant fold, which is why it looked like a disagreement
  between compilers before the operands were made opaque.
- Two such calls separated by the `fesetround` answer the same value twice
  under Clang 14 at `-O3`.  `[[gnu::const]]` licenses exactly that, and it was
  a false promise for as long as the route existed — but deleting it is not the
  cheap fix it looks like.  A copy of the header with
  `SKYMIZER_MINIFLOAT_CONST` defined empty is bit-identical on this probe under
  both compilers.  Without `#pragma STDC FENV_ACCESS` — which GCC 11.4 does not
  implement, and which Clang rejects outright in a translation unit built with
  `-ffast-math` — the default floating-point model already permits the reuse.
- Compile a translation unit *without* `-ffast-math` and link it *with*
  `-ffast-math`: the driver pulls in `crtfastmath.o`, which sets FTZ and DAZ
  before `main`.  `BF<32>::from_bits(0x00800000) * BF<32>{0.5F}` is then zero
  under both compilers, where `0x00400000` is a code the shape has and the
  integer engine returns.  No `fesetround`, no pragma, nothing undefined
  anywhere in the program — one flag on somebody's link line, and one shape out
  of 54 quietly loses its subnormals while the other 53 stay right.  MXCSR is
  outside the standard, so *the caller was already in undefined behaviour*
  never covered this half.

A header cannot promise *the `float` it is* without owning `#pragma STDC
FENV_ACCESS`, `-frounding-math`, the optimization level and the user's link
line; and it cannot promise *correctly rounded* with an FPU in the loop.  So the
four operators go back through the integer engine; `detail::as_host_float`,
`detail::from_host_float` and `detail::in_constant_expression` go with them,
having no other caller; and `[[gnu::const]]` on the operators is true again.
`to_float` stays a `bit_cast`; the `float` constructor also only inspects bits,
though it canonicalizes NaN payloads.  Neither reads a control register.

What is withdrawn is the library *choosing* the FPU for a caller who cannot be
asked.  The caller can still choose it, and is the only party in a position to
know whether its own floating-point environment is safe: `IS_HOST_FLOAT` is
public, `to_float()` is the identity, and construction back preserves every
non-NaN bit pattern while canonicalizing NaN payloads, so
`if constexpr (T::IS_HOST_FLOAT) out[i] = T{a[i].to_float() + b[i].to_float()};`
compiles to SSE float adds at `BF<32>` under both compilers — vectorized under
Clang — while the same template at `BF<16>` emits no float instruction at all.
That is what the trait is for now.

`Arith.IgnoresHostEnvironment` is the referee.  It puts 4100 operand pairs
through all four operators under `FE_UPWARD`, `FE_DOWNWARD` and FTZ+DAZ, and
requires each answer to equal the one the same sweep gave under the default
environment — the contract restated as a property, rather than a table of
hand-picked results to drift out of date.  A native `float` computed beside
each pair is the control: where *it* does not move either, the platform ignored
the request and a pass would be vacuous.  Both arms reload their operands
through `volatile`, which is not decoration.  Without it Clang answers the
second sweep from the first, exactly as the third reading above predicts, and
the test passes on the broken header.

What the withdrawal costs, as `host / soft` from the ratio table — both routes
timed in one binary over one operand array, so the figure is self-contained:

| | add | sub | mul | div |
| --- | --- | --- | --- | --- |
| GCC 11.4 | 0.194x | 0.193x | 0.452x | 0.390x |
| Clang 14 | 0.117x | 0.118x | 0.307x | 0.356x |

Ryzen 9 7950X3D, `taskset -c 2`, 2026-08-24, reproducing to three digits across
two runs two minutes apart at load average 10.4 and 1.9.  The integer engine
costs five times a `float` on GCC's addition and eight and a half on Clang's,
and that is what the contract is worth paying.  The route is unadopted, not
lost: it is commit `9f5bbd1`, whose body carries its own interleaved A/B —
0.131x and 0.083x on addition, after over before — and rebuilds from there.

Two earlier readings survive the withdrawal and are worth keeping.  The first is
that the shape's column read 1.194x and 0.887x until 2026-08-23, and all of that
difference was in the baseline: applying the 2*p* + 2 rule to a shape that
narrows nothing bought it a `double` and a software re-encode to emulate
arithmetic a `float` performs exactly, costing the baseline 2.1x on addition
under GCC and 1.5x under Clang, which the integer engine then collected.  Timed
against a real `float` instead it lost addition and subtraction at 0.474x and
0.449x under GCC against 0.434x and 0.432x under Clang — and against a `float`
the *constructor* also `bit_cast`s, at the 0.194x and 0.117x above.

The second is the doubt recorded beside them.  The integer engine appeared to
keep multiplication at 1.449x and division at 1.128x under GCC, but both wins
were against a `Minifloat<IEEE<8, 23>>{float}` that decomposed and re-encoded a
value the constructor was entitled to `bit_cast`.  With the constructor casting,
the host arm takes both: 0.452x and 0.390x above.  The doubt was justified and
the wins were the baseline's.

What none of this licenses is a general `+`/`-` route for `E8M7`- or
`E11M4`-class shapes, which lose to a host float by as much.  Those need a
conversion to be correct about, and so need the DAZ-safe gate
`MIN_EXP - MANTISSA_DIGITS >= FLT_MIN_EXP` rather than
`HAS_EXACT_F32_CONVERSION` — which pins only the shape's least *normal*, leaving
`E8M7`'s subnormals at 2<sup>&minus;133</sup> inside `float`'s subnormal range
where a caller's `MXCSR` decides the answer.  That gate excludes bf16, which is
most of the reason anyone would want the route.  `BF<32>` had no such gate to
pass, and that used to be the whole of the difference between the two cases;
what the readings above establish is that clearing it would not have been
enough either, because the rounding mode is a second control register and no
gate on a *shape* can close it.

One cost finishes the case against a general route.  `*` and `/` still favour
the integer engine at those shapes, so the result would be routing per operator
*and* per shape in a header that otherwise has one engine.  Under Clang the
narrow shapes are a wash at 0.89x to 1.11x besides, so the win on offer is a few
wide-exponent shapes under one compiler — bought with a second engine and a
control register nobody in the header owns.

## Why the headline fell without the arithmetic changing

Those figures used to read 52 of 56 at 1.195x for Clang and 28 of 56 at 1.001x
for GCC.  Nothing in this section's arithmetic changed between the two
measurements.  What changed is the other side of the ratio.

The two arms of `bench_op` are not symmetric.  The integer arm is
`op(x, y).to_bits()` — the bare operator, which goes `to_parts` → integer kernel
→ `from_parts` and touches no host float at all.  The host arm is
`T{op(x.to_float(), y.to_float())}.to_bits()`, which reads both operands through
`to_float` and builds its result through `bits_from`.  Every conversion in the
comparison is in the host arm.

So when the conversions got faster — a bit-built `exp2i` in place of `std::exp2`
and `std::ldexp`, and one `bit_cast` in place of `std::signbit` plus
`std::isnan` plus `std::isinf` — only the numerator could move.  The comparator
got faster while the thing being compared stayed exactly where it was.

Measured, interleaved, against the state before those two changes:

| | operator rows | host rows | construction from `float` |
| --- | --- | --- | --- |
| Clang 22.1.8 | 1.000x | 0.873x | 0.711x |
| GCC 16.1.1 | 0.987x | 0.950x | 0.831x |

The operator column is the check, and here it is load-bearing rather than
conventional: those rows *cannot* be reached by a conversion change, so a run
where they moved outside `[0.98, 1.02]` would be a bad run rather than a smaller
version of this finding.  They held.  And 1.195x × 0.873x = 1.043x against the
1.044x measured directly, which is the two accounts agreeing.

The lesson a headline hides: half of this ratio is not the library's arithmetic,
and it is the half a conversion change moves.  A falling ratio here is not
evidence of a slower library.  Check the operator rows before concluding
otherwise — and note that this is a different claim from the one in
[benchmarking.md](benchmarking.md) about `mul` not being a control for a
conversion change.  `mul`'s *integer* row is untouchable like every other
operator row; it is `mul`'s *host* row, and so its ratio, that such a change
moves.

Rust's 1.709x does not transfer, and the reason is not that this library is
slower.  It is that the *other* side of the ratio is faster here: `to_exact`
constructs host fields and `bit_cast`s where minifloat-rs's `to_f32` spends two
floating-point multiplies.  A ratio is a comparison, and the denominators are
different libraries.

The `BF<N>` family takes that conversion shortcut to its endpoint.  Every
`BF<10>` through `BF<32>` has `float`'s exponent field and special-value rows,
so conversion out is a left shift; below `BF<32>`, conversion in is a right
shift with a round-to-nearest-even bias, and at `BF<32>` both shifts are by
zero.  Only construction from a `float` takes this path for a `BF<N>`;
construction from a `double` still narrows through `from_parts`, and every
operator stays on the integer engine, `BF<32>` included.  A NaN still canonicalizes on construction; conversion out preserves
the stored payload as well as its sign because the whole code shifts unchanged.

The same route serves default-biased `IEEE<11, M>` through `double`: this
library admits `M` from 1 through 20, and their exponent and special-value rows
are `double`'s.  It changes no arithmetic route and does not apply to a
`BF<N>`'s `double` conversions, whose exponent field is still `float`'s.

What the two unary conversions now cost, after over before.  Interleaved
binaries, 15 alternating passes each pinned to core 2 of a Ryzen 9 7950X3D,
2026-08-23:

| | `BF20` to `float` | `BF24` to `float` | `BF20` from `float` | `BF24` from `float` |
| --- | --- | --- | --- | --- |
| GCC 11.4 | 0.179x | 0.188x | 0.344x | 0.342x |
| Clang 14 | 0.329x | 0.343x | 0.267x | 0.286x |

All four clear their compiler's unchanged unary control band, 0.751x–1.507x
under GCC and 0.853x–1.338x under Clang; the controls' geomeans are 1.005x and
0.998x.  The exhaustive `BF<16>` encoder sweep checks every `float` bit pattern
against the generic path, while the existing code sweeps referee decoding.

## The 2*p* + 2 rule, and why the benchmark skips a shape

A host float may stand in for a shape only if both operands are exact in it
**and** it carries at least 2*p* + 2 digits, where *p* is the shape's own
precision.  Below that, rounding to the intermediate and then to the shape can
differ from rounding to the shape once (Figueroa 1995).  This is the whole of
`route` in `benches/arith.cpp`, and it is why `HAS_EXACT_F32_CONVERSION` alone
does not license a comparison.

Exactness is not enough.  `IEEE<2, 13>` is exact in `float` — 14 digits into
24 — yet a product of two of its significands is 28 digits wide, so it goes
through `double`.  `IEEE<11, 4>` is not exact in `float` at all and falls back
the same way.  `IEEE<12, 3>` reaches past `double` altogether and is skipped
rather than timed against a different answer.

`BF<32>` is not skipped, though for one round it was, on the opposite grounds:
not that no host float rounds like it, but that it *is* one, so both arms of the
comparison ran the same instructions.  The library computes it on the integer
engine again, and `route`'s `IS_HOST_FLOAT` disjunct is what keeps the shape
timed against a `float` — its 2*p* + 2 is 50 digits, which a `double` would
otherwise be asked for.

`route` compares `MANTISSA_DIGITS`, the *normal-range* precision, and that is
the conservative side of the comparison rather than the loose one: a subnormal
operand or result carries fewer digits, so it needs fewer intermediate ones, and
the gate demands the larger number anyway.  The side `route` does not test is
the intermediate's own precision, which decays inside *its* subnormal range,
while `HAS_EXACT_F32_CONVERSION` pins only the target's least *normal* through
`FLT_MIN_EXP <= MIN_EXP`.  It cannot bite for the shapes benched — the `float`
route forces `M` &le; 10 and `MIN_EXP` &ge; &minus;125, leaving the target's
least subnormal at 2<sup>&minus;136</sup> or above with 13 `float` bits still
below it — and the exhaustive oracle below found no disagreement either way.

The benchmark's unary table has no such gate and runs every shape, both skipped
ones included.  A conversion has no second route to be entitled to.

## The oracle shares no arithmetic with the implementation

0.1.0's correctness chain had two links — encoding a `double` rounds correctly,
and every operator matches `T{xf op yf}` — and nothing tested the join.  It also
could not outlive the host route it was written against.

The oracle in `tests/arith.cpp` computes each result exactly and converts
nothing to a float.  `check_exact_pair` builds an `Exact` — a signed numerator
as significand-times-power-of-two, over an integer denominator — for each of the
four operations: a product of significands, a signed sum aligned in an
`std::int64_t`, a quotient left as numerator and denominator and compared by
cross-multiplication in `compare_exact`.  `reference_round` in
`tests/support.hpp` then binary-searches the format's magnitude codes for that
value, reaching it only through a comparison callback, so a value no host float
can hold referees itself.  The same `reference_round` backs `reference_encode`,
which is what `tests/encode.cpp` checks encoding against.

Two details are deliberate rather than incidental:

- `exact_sum` uses an `ALIGN_CAP` of **31**, against the engine's 32, and where
  the engine drops an out-of-range addend to **0** the oracle substitutes **1**.
  Two different windows and two opposite sticky policies, and they round alike.
  An oracle that shared the engine's constants would agree with a wrong engine.
- `CheckExactWideArithmetic` samples 2<sup>16</sup> pairs, not the 2<sup>13</sup>
  its host-route sibling `CheckWideHostArithmetic` uses, and seeds its `Lcg`
  differently so the two draw different pairs.  The Rust sibling found that
  2<sup>13</sup> misses an `IEEE<2, 13>` double rounding.

Coverage, stated precisely because the loose version keeps getting repeated.
`Arith.CorrectlyRoundedSmallFormats` runs the exact oracle over every **ordered
pair of finite operands** of all 39 shapes in `test_small_types` — every
declared width through 8 bits, all four format layers, exponent widths 2 through
7 — which is exhaustive, 2<sup>16</sup> pairs at the top end.
`Arith.CorrectlyRoundedWideFormats` runs the same oracle over the 15 shapes of
`test_wide_types`, where even the narrowest ordered-pair space is
2<sup>32</sup> and out of reach, so it samples.

Pairs involving an infinity or a NaN have no exact magnitude to compare against
and return early; those are pinned by `Arith.SpecialValueLadder`, which
compares exact codes rather than going through `same_mini`, and derives the
invalid result from `reference_encode` rather than from the library.  The
host-float comparisons — `Arith.MatchesHostRoundTrip` over the 42 shapes of
`test_paired_types`, and `Arith.WideFormatsMatchHostRoundTrip` — stay in the
file on purpose: they are narrower, since they cannot referee a shape their
`double` cannot hold, but they cover the non-finite pairs the exact oracle
skips, and they are the only check that the route `benches/arith.cpp` times
computes the same answer.  `Arith.PastDoubleRange` pins the `IEEE<12, 3>`
results neither can referee.

Both of those go through a `double`, and `route` in `benches/arith.cpp` puts
`E5M10` and `E8M7` on a **`float`**, which nothing above refereed.
`Arith.EveryPairMatchesFloatRoundTrip` does, over every ordered pair of both —
2<sup>32</sup> each, striped across `std::thread`s by `find_failing_pair` in
`tests/support.hpp`.  The remaining 16-bit shapes are not in it because they
are not on that route: `IEEE<2, 13>` and `IEEE<11, 4>` are timed against a
`double`, and `IEEE<12, 3>` is skipped outright.  `Ops.EveryPairComparesLikeHost`
carries the same 2<sup>32</sup> treatment to comparison, which
`Ops.Comparison` stops at 11 bits.  Those two sweeps and the strided wide
rounding-boundary check share most of what `make check` spends its time on.

## No lookup tables

Tempting for the 8-bit shapes: 2<sup>16</sup> entries per operator, one load
instead of a rounding.  Two reasons it is not there.

It does not generalize.  A 16-bit shape needs 2<sup>32</sup> entries per
operator — 8 GB for `E5M10` addition alone — so a table would serve the 8-bit
aliases and leave `E5M10`, `E8M7`, and every 16-bit shape a user declares on the
integer path, which would have to exist anyway.

And the measurement would lie.  A microbenchmark hammers the same 64 KB in a
loop with nothing else in L2, which is the one condition under which a table
looks good; a caller doing anything else evicts it and pays a miss where the
integer route pays a shift.  A benchmark that cannot represent the failure mode
cannot be used to argue for the thing that fails that way.

## Nulls from this round, recorded on purpose

**The branchless sign flip: measured, reverted.**  minifloat-rs found that the
zero guard in its `Neg` — a format without a negative zero must not flip a zero,
because the code it would flip into is the NaN — compiled to a `setcc`, whose
partial write of a byte register carried a false dependency, and that rewriting
the guard as the top bit of `m | -m` was worth 0.773x on `FNUZ` subtraction.
The port of that trick to `operator-` and `abs` was reverted.

The premise does not hold here.  `if constexpr (!Format::HAS_NEG_ZERO) if
(!(bits_ & ABS_MASK)) return x;` compiles to a `cmove`, not a `setcc`: three
instructions under GCC (`mov`, `and`, `cmove`) and four under Clang.  The mask
version is five under both, so it is the longer sequence, and in a loop of
independent operations that is what decides.

| `FNUZ` unary, ns | before | after | |
| --- | --- | --- | --- |
| `abs`, GCC 16.1.1 | 0.206 | 0.412 | 2.000x |
| `neg`, GCC 16.1.1 | 0.388 | 0.412 | 1.062x |
| `abs`, Clang 22.1.8 | 0.394 | 0.366 | 0.929x |
| `neg`, Clang 22.1.8 | 0.414 | 0.399 | 0.964x |

Min of 15 interleaved passes each, operator rows held as control.  The
compilers disagree in sign, and a 2x regression under one of them is not bought
by 7% under the other.  The finding was about what one back end emitted for that
guard, not about the guard, and the disassembly said so before the stopwatch
did — which is the cheaper order to ask in.

**A single row of the unary table is not a result.**  This was first written
down as a resolution limit on the 0.2 ns `neg` and `abs` rows, which was the
wrong diagnosis: the mechanism is code placement, and it reaches the
nanosecond-scale rows too.  Across a change that left the `to_float` and
`to_double` closures byte-identical, those 30 rows still measured 0.801x to
1.245x under GCC and 0.701x to 1.164x under Clang, min of 20 interleaved passes.
[benchmarking.md](benchmarking.md) has the measurement and what it costs.

Applying [benchmarking.md](benchmarking.md)'s calibration test to this section's
own numbers: the construction rows ran 0.665x–0.786x under Clang against a
0.701x–1.164x band read off the byte-identical conversion closures, and
0.735x–1.083x under GCC against 0.801x–1.245x.  Both overlap, so what stands is
the geomean over 15 shapes — 0.711x and 0.831x — and no individual shape does.
`IEEE<12, 3>` at 0.24x clears every band measured here by a factor of three, and
has the symbol table behind it besides.  So does the `FNUZ` `abs` row at 2.000x,
which is also the one with a five-instruction body replacing a three-instruction
one — though that change touched the only rows short enough to calibrate
against, so it is read against the widest band on record rather than its own.

Not entitled: the `FNUZ` `neg` row at 1.06x, and the per-shape exceptions the
`bits_from` commit recorded — `E8M7` construction at 1.083x under GCC against
0.740x under Clang, and `E11M4` at 1.013x.  `3c783e3`'s commit body reports the
first as a cross-compiler disagreement on the strength of its reproducing at
1.081x and 1.083x across two runs.  Those were two runs of the same two
binaries, and layout is a property of the binary, so re-running measured it
again rather than testing it.

## Open questions

**The two conversion sites the environment reaches.**  Named in full above.
Each is on a conversion the format cannot make exactly, so the host's answer is
defensible, but nothing has decided that it *is* the answer — a shape that
overruns `double` could compute its own scale rather than borrow one.

**A 32-bit divider on 32-bit hosts.**  A narrow shape could normalize its
dividend to bit 30 instead of bit 62, fitting the numerator in 32 bits and
avoiding 64-bit division on a 32-bit host.  Nobody has measured whether that is
worth a format-dependent path, and the 64-bit hosts this is developed on would
not show it.
