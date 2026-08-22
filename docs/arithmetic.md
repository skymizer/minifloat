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
rounds.  `bits_from` rounds a host float *in*, but not separately — it
decomposes exactly and hands the triple to that same `from_parts`.

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
wrong for exactly the shapes this library exists to support.  The one exception
is the shape that needs no refereeing because it *is* a host float, and it has
its own section below.

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

The brain floats keep the split rather than settling it — except `BF<32>`,
which turned out to have no split to keep and now has no ratio either.  On a
Ryzen 9 7950X3D, 2026-08-23, under 15 alternating A/B passes pinned to core 2,
each binary taking its own minimum over 30 internal passes, and with
`host / soft` above one favouring the integer route:

| | BF20 | BF24 | BF32 | all 12 operator rows |
| --- | --- | --- | --- | --- |
| GCC 11.4 | 1.029x | 1.031x | 0.768x | 0.934x |
| Clang 14 | 0.889x | 0.857x | 0.649x | 0.791x |

`BF20` and `BF24` are timed against a `double`; `BF32` was timed against a
`float`, and its column is history — the library gives that shape the FPU now
and the benchmark reports no ratio for it.  Of the two that remain, both
compilers give addition and subtraction to the host route and multiplication to
the integer engine.  Division favours the integer route under GCC and the host
under Clang, so neither speed nor one compiler licenses a second engine.

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

## `BF<32>` is `float`, so the library hands it to the FPU

*The one shape where adopting the host route costs no correctness at all,
because there is no conversion to be correct about.*

`route`'s 2*p* + 2 rule is Figueroa's bound on *narrowing* a wide intermediate,
and `BF<32>` narrows nothing: `IEEE<8, 23>` has `float`'s precision, exponent
range and non-finite semantics, so IEEE 754 already rounds each operator once,
to exactly the digits the shape stores.  `detail::is_host_float` is that
predicate and `Minifloat::IS_HOST_FLOAT` publishes it; the comment on the
predicate says why `FN<8, 23>`, `Finite<8, 23>` and `IEEE<7, 23>` all miss it.
Under it, all four operators, `to_float` and the `float` constructor are a
`bit_cast` in, an FPU instruction, and a `bit_cast` out.
`Arith.BF32MatchesFloatArithmetic` is the referee and needed no editing;
`Arith.SpecialValueLadder` and `Encode.RandomFloatSweep` are the two that
constrain what follows.

Two debts the route still pays.  Its NaN canonicalizes, because x86 signs its
default NaN and ARM does not — the portability bug 0.1.0's host route had, and
the reason `detail::invalid` exists.  `detail::from_host_float` puts it back,
and `operator+` for this shape is a `vaddss` and a `cmova` under GCC.  And a
constant evaluation goes back to the integer engine, since C++17's `bit_cast`
is a `memcpy` and no `memcpy` is a constant expression; `detail::in_constant_expression`
is that switch, and answers `true` — integer engine, always — on a compiler
that cannot be asked.

What it is worth, ns per element, after over before.  Interleaved A/B, 15
alternating passes each pinned to core 2 of a Ryzen 9 7950X3D, 2026-08-23, load
average 7.3 falling to 1.8, both binaries carrying the previous
`benches/arith.cpp` so the shape still reports rows on either side:

| | add | sub | mul | div | `to_float` | from `float` |
| --- | --- | --- | --- | --- | --- | --- |
| GCC 11.4 | 0.131 | 0.129 | 0.344 | 0.379 | 0.179 | 0.412 |
| Clang 14 | 0.083 | 0.082 | 0.171 | 0.378 | 0.283 | 0.351 |

Every row clears its compiler's control band — 0.720x–1.404x under GCC and
0.775x–1.353x under Clang, off 213 unchanged rows — by a factor of two or more,
and `BF32`'s own `neg`, `abs` and `f64` rows, untouched code in the same binary,
hold at 1.00x.

Two earlier readings are settled by this and worth keeping.  The first is that
the shape's column read 1.194x and 0.887x until 2026-08-23, and all of that
difference was in the baseline: applying the 2*p* + 2 rule to a shape that
narrows nothing bought it a `double` and a software re-encode to emulate what
the FPU was doing exactly, costing the baseline 2.1x on addition under GCC and
1.5x under Clang, which the integer engine then collected.  Timed against a real
`float` instead, the integer engine lost addition and subtraction at 0.474x and
0.449x under GCC against 0.434x and 0.432x under Clang.  Those four numbers
reproduced to three digits on this box before the change.

The second is the doubt recorded beside them.  The integer engine appeared to
keep multiplication at 1.449x and division at 1.128x under GCC, but both wins
were against a `Minifloat<IEEE<8, 23>>{float}` that decomposed and re-encoded a
value the constructor was entitled to `bit_cast`.  With the constructor casting,
the FPU takes both: `mul` 0.344x and `div` 0.379x above.  The doubt was
justified and the wins were the baseline's.

`benches/arith.cpp` no longer reports a ratio for the shape.  Both arms would
run the same instructions, and a row that reads 1.000x by construction is a
tautology rather than a regression detector.  The unary rows stay, a conversion
still being timed, and `route` keeps its `IS_HOST_FLOAT` disjunct so that
dropping the skip cannot quietly send the shape back through a `double`.

What this does *not* license is a general `+`/`-` route for `E8M7`- or
`E11M4`-class shapes, which lose to a host float by as much.  Those need a
conversion to be correct about, and so need the DAZ-safe gate
`MIN_EXP - MANTISSA_DIGITS >= FLT_MIN_EXP` rather than
`HAS_EXACT_F32_CONVERSION` — which pins only the shape's least *normal*, leaving
`E8M7`'s subnormals at 2<sup>&minus;133</sup> inside `float`'s subnormal range
where a caller's `MXCSR` decides the answer.  That gate excludes bf16, which is
most of the reason anyone would want the route.  `BF<32>` has no such gate to
pass, and this is the whole of the difference between the two cases.

Two costs finish the case against a general route.  Adopting the host route for
a shape turns its benchmark row into 1.000x by construction, spending the
regression detector for that shape — affordable once, for the shape that has
nothing left to detect, and not as a policy.  And `*` and `/` still favour the
integer engine at those shapes, so the result would be routing per operator
*and* per shape in a header that otherwise has one engine.  Under Clang the
narrow shapes are a wash at 0.89x to 1.11x besides, so the win on offer is a
few wide-exponent shapes under one compiler.

One behaviour does change with the caller's `MXCSR`, and it is worth naming
rather than burying: `BF<32>` arithmetic now flushes subnormals when the caller
has set DAZ or FTZ, where the integer engine did not.  That is not the hazard
above — the shape's subnormals *are* `float`'s subnormals, so there is no
mismatch, only the host's own setting reaching a type that is the host's own
`float`.  A caller who wanted `float` semantics has them.

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
spends a shift and a `bit_cast` where minifloat-rs's `to_f32` spends two
floating-point multiplies.  A ratio is a comparison, and the denominators are
different libraries.

## The 2*p* + 2 rule, and why the benchmark skips two shapes

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

`BF<32>` is skipped too, and for the opposite reason: not that no host float
rounds like it, but that it *is* one, so both arms of the comparison run the
same instructions.  The two skips print different messages because they are
different facts.

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
`Arith.CorrectlyRoundedWideFormats` runs the same oracle over the 13 shapes of
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

**A 32-bit divider on 32-bit hosts.**  A narrow shape could normalize its
dividend to bit 30 instead of bit 62, fitting the numerator in 32 bits and
avoiding 64-bit division on a 32-bit host.  Nobody has measured whether that is
worth a format-dependent path, and the 64-bit hosts this is developed on would
not show it.

