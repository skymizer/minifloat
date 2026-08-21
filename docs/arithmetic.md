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

- **multiply** — two significands of at most 15 bits multiply exactly in a
  `std::uint64_t`, exponents add.  Genuinely exact.
- **add** — `detail::add_parts` aligns both addends on the lower exponent and
  sums them signed in an `std::int64_t`.  Addends more than `detail::ALIGN_CAP`
  = 46 binades apart drop the smaller one, which is under half the ULP of the
  larger and rounds straight back to it; that cap is also what keeps the aligned
  sum inside an `std::int64_t`.
- **divide** — `detail::div_parts` computes `detail::QUOTIENT_BITS` = 46
  quotient bits and folds the remainder into the lowest as a sticky bit.  The
  two 46s are unrelated: one is an alignment window, the other a quotient width,
  and the constants' own doc comments say so because the shared value invites
  the wrong conclusion.

So two of the four carry a deliberately inexact tail.  Neither tail can change a
rounding — that is what the two constants are sized for — which is why the
thesis says *exact enough to round*.

`detail::from_parts` is the only place *arithmetic* rounds, ties to even, by way
of `detail::round_to_scale`.  One rounding, so no intermediate can lose what the
format is able to hold, and a shape whose exponent range overruns `double`'s is
served as exactly as any other.  The library's one other rounding is
genuinely elsewhere: `to_double` splitting its scale into two in-range factors
once a shape's exponent leaves `double`'s, where the second multiply rounds.  `bits_from` rounds a host float *in*, but not
separately — it decomposes exactly and hands the triple to that same
`from_parts`.

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

## Why there is no hardware route left to choose

The speed is a bonus.  The reason is correctness: a host float cannot referee a
shape it cannot hold, so keeping it would have meant keeping a route that is
wrong for exactly the shapes this library exists to support.

But it is worth knowing what the bonus is, and here the answer depends on the
compiler and on the operator — it is not one number.  `benches/arith.cpp` times each operator twice over the same
operands — once as the library computes it, once the way a caller would fake it.
On an idle Ryzen 7 8700F, 2026-08-21, under the protocol in
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
give both to the host route under *both* compilers, 0.56x to 0.86x.  That is the
library's own shape rather than a back end's: a wide exponent range is the
distance `align` has to shift across, and an FPU does that in its exponent field
for free.  `E2M13` is the opposite corner and both compilers agree on it too, the other
way: it wins every one of its four operators, 1.11x to 1.87x under GCC and 1.35x
to 1.61x under Clang.  It is the one shape whose host route pays for a `double`
round trip — `route` sends it there because 2*p* + 2 is 30 digits — without the
integer route paying a wide exponent range for it.

Between those two corners the compilers part.  Clang stays near even, 0.89x to
1.11x, while GCC gives the host route every narrow shape except `FNUZ`, which it
wins at 1.09x to 1.10x.  Same source, same box; that difference is in the back
end and it has not been diagnosed.

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

The benchmark's unary table has no such gate and runs every shape, `IEEE<12, 3>`
included.  A conversion has no second route to be entitled to.

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

- `exact_sum` uses an `ALIGN_CAP` of **48**, against the engine's 46, and where
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
`Arith.CorrectlyRoundedWideFormats` runs the same oracle over the 9 shapes of
`test_wide_types`, where every ordered pair is 2<sup>32</sup> and out of reach,
so it samples.

Pairs involving an infinity or a NaN have no exact magnitude to compare against
and return early; those are pinned by `Arith.SpecialValueLadder`, which compares
exact codes rather than going through `same_mini`, and derives the invalid
result from `reference_encode` rather than from the library.  The
host-float comparisons — `Arith.MatchesHostRoundTrip` over the 42 shapes of
`test_paired_types`, and `Arith.WideFormatsMatchHostRoundTrip` — stay in the
file on purpose: they are narrower, since they cannot referee a shape their
`double` cannot hold, but they cover the non-finite pairs the exact oracle
skips, and they are the only check that the route `benches/arith.cpp` times
computes the same answer.  `Arith.PastDoubleRange` pins the `IEEE<12, 3>` results neither
can referee.

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

**Format-dependent `QUOTIENT_BITS`.**  46 is sized for the widest significand a
minifloat can have.  A narrow shape could divide in fewer bits, which would fit
a 32-bit numerator and let `div_parts` avoid 64-bit division on a 32-bit host.
Nobody has measured whether that is worth a format-dependent constant, and the
64-bit hosts this is developed on would not show it.

**The GCC addition and subtraction gap.**  At every narrow shape that is not
`FNUZ` or `E2M13`, GCC's `operator+` and `operator-` lose to the host route
where Clang's come out even — `E5M2` is 0.67x against 1.01x, from the same
source.  Whether
that is `add_parts`'s two `align` calls failing to be if-converted, the
`std::int64_t` sum, or something in `from_parts` has not been diagnosed.

The diagnosis is a disassembly comparison of one shape's `operator+` under both
compilers, and picking the shape matters: `E5M2` or `E4M3`, not `E8M7` or
`E11M4`, since those two lose under both compilers and so have no difference to
show.  `FNUZ` is the other end of the same question — GCC wins those at 1.09x to 1.10x
while losing their non-`FNUZ` neighbours, which is a large enough split within
one compiler to be a clue on its own.
