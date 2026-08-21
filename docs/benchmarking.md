# The benchmarking protocol

*A number from this repository means a min-of-N across interleaved builds on an
idle box, under both compilers, or it means nothing.*

The box these numbers come from: AMD Ryzen 7 8700F (8 cores, 16 threads),
Fedora 44, GCC 16.1.1 and Clang 22.1.8.  A ratio from a different machine, or
from one compiler where the claim is about the library, is a different claim.

## Stop the poker solver first

The development box runs a poker solver that will happily take every core.  A
measurement taken beside it is worthless, and killing it unasked is worse.

```sh
pgrep -af poker
uptime          # the one-minute average should be near zero
```

If it is running, ask before touching it.  Proceed only once the box is idle.

## Two compilers, or it did not happen

`benches/arith.cpp` is one file over a header-only library, so the binary is
whatever the compiler decided to make of it — and the two do not decide alike.
On 2026-08-21 the same source and the same box gave Clang 22 the integer route
in 37 of 56 comparisons at a geomean of 1.044x, and GCC 16 an even 23 of 56 at
0.966x.  A single-compiler number would have supported either "clear win" or
"no difference" depending on which compiler ran.

So every claim gets both.  Delete the binary between them: the `bench` target
depends on the sources and not on `CXX`, so a bare `make bench CXX=clang++`
after a GCC build reports the target as already up to date and hands you the
GCC binary again.

```sh
rm -f bench && make bench CXX=g++     && taskset -c 2 ./bench
rm -f bench && make bench CXX=clang++ && taskset -c 2 ./bench
```

Where they disagree in sign, say so and report both.  Where they agree, the
claim is about the library rather than about one back end's heuristics, and
that is the only kind of claim worth putting in a commit body.

## Interleave the builds, never run A then B

Build both sides first, stash the binaries, and only then measure — alternating
A, B, A, B for at least 15 passes each, all on one pinned core:

```sh
STASH=$(mktemp -d)

make bench && cp bench "$STASH/before"
# ... apply the change ...
make bench && cp bench "$STASH/after"

for i in $(seq 1 15); do
  taskset -c 2 "$STASH/before" > "$STASH/A-$i.txt"
  taskset -c 2 "$STASH/after"  > "$STASH/B-$i.txt"
done
```

A compile between two measurements heats the box, and a box that drifts over
fifteen minutes will hand you whichever answer the drift had at the time.
Interleaving cancels the drift instead of hoping it is not there.  `taskset`
pins both sides to the same core so neither can win by landing on a better one;
core 2 is an arbitrary choice, held constant, and it is the core the `run-bench`
target in the `Makefile` already uses.

## Take the minimum, not the mean

Noise on a benchmark is one-sided: nothing makes a loop run faster than it can,
and everything else on the machine makes it run slower.  The minimum across
passes is the least contaminated sample there is.  A mean is a statement about
the machine's other tenants.

The harness already applies this rule once, inside a run: `measure` in
`benches/arith.cpp` takes the minimum over `PASSES` passes, each of `REPEATS`
sweeps of the operand array, and the file's header comment says why.  The protocol applies it a second time, across
runs of the whole binary, because a single run cannot see the drift that
scheduling, frequency, and the other passes introduce between one binary and the
next.  Take the minimum per line across the 15 files on each side, then divide.

That the minimum really does shed a transient is measured rather than argued.
An all-core build of this repository landed in the middle of the minifloat-rs
sibling's 15-pass probe, contaminating three passes on each side.  Harvested
both ways, the headline came out 0.626x over all 15 and 0.624x with the three
dropped, and its control geomean was 1.006x either way — 0.3% on the number
under test and no movement at all on the control.  One incident is not a
guarantee, but it is better evidence than the argument above on its own.

## Keep a control route

Every sweep carries at least one row the change cannot possibly have touched.
If the control moves, the run is noise and the headline number is noise with it.

The rule that actually does the work is upstream of picking one.  **Before
choosing a control, write down what the change reaches, and check that against
everything you were going to compare it to.**  The failure mode is not choosing
a poor control; it is not noticing that the change reaches the control too, and
then reading a row that moved for a real reason as proof the run was clean.
Sometimes the answer is that there is no control available, and saying so is a
result.  A sweep reported with "no control: `from_parts` is on every path" is
worth more than one reported beside a row that was never insulated.

Answering it here means reading the two arms of `bench_op` as separate
measurements, because they share almost nothing: the `soft` column is
`op(x, y)`, the bare operator, and the `host` column is
`T{op(x.to_float(), y.to_float())}`, which is the only arm with a conversion in
it.  Three worked examples:

- **A conversion change** — `bits_from`, `to_float`, `to_double`, `to_exact`,
  `decompose`, `exp2i`.  Every one of the 56 `soft` rows is a control, since
  none of them calls any of that.  Its effect shows up in the `host` column and
  therefore in the ratio, which is why the ratio falling does not mean the
  library got slower.
- **An addition or subtraction kernel change** — `align`, `add_parts`, the sign
  flip in `add_impl`.  The `mul` and `div` `soft` rows are controls, since those
  three are the only steps the other operators do not share.
- **A change to `to_parts`, `from_parts`, or `invalid`.**  There is no operator
  control: all four operators call all three, and `from_parts` is on the inbound
  conversion path as well.  Say there was none rather than quoting `mul` and
  implying otherwise.

Quote the control you used, by column.  A `sub` row at 0.80x is reportable only
beside a `mul` row that stayed inside the noise floor — and it has to be
`mul`'s `soft` figure, not its ratio.

## The noise floor is 0.98x, and it does not cover code placement

A ratio inside `[0.98, 1.02]` is not a result.  Say so plainly rather than
reporting it as a small win — a null recorded is worth more than a null dressed
up, and [arithmetic.md](arithmetic.md) keeps a section for exactly those.

That floor bounds *timing* noise, and timing noise is not the only thing between
two builds.  A change that resizes `.text` relocates everything after it, and
identical code at a different address does not run at the same speed.

This was measured here rather than assumed.  Commit `3c783e3` touched
`bits_from` and nothing else on the conversion paths, so the `to_float` and
`to_double` bench closures came out of both builds byte-identical once branch
targets and rip-relative displacements are normalised — 14 and 12 closures, no
instruction changed — while `.text` shrank by 4002 bytes and moved them.  Timed
under the full protocol, min of 20 interleaved passes:

| identical code, both builds | rows | geomean | range |
| --- | --- | --- | --- |
| GCC 16.1.1 | 30 | 0.996x | 0.801x – 1.245x |
| Clang 22.1.8 | 30 | 0.984x | 0.701x – 1.164x |

Eleven of the thirty GCC rows fall outside `[0.98, 1.02]`, on code that did not
change a byte.  The minifloat-rs sibling measured the same effect independently
and reached the same conclusion from the other end: a 1929-instruction,
byte-identical benchmark body moved 1.090x purely on relocation.

**Interleaving cannot help.**  Placement is a property of the binary, not of the
run, so more passes converge on the wrong number rather than away from it.
Reproducible across passes and reproducible across *builds* are different
claims, and only the second one means anything here.

## Calibrate the band; do not assume it

The 0.98x floor is a constant standing in for something that is not constant.
Measured on this box, on rows whose code did not change:

| calibration rows | duration | band |
| --- | --- | --- |
| operator rows, sign-flip build pair, GCC | 2.3 – 5.2 ns | 0.971x – 1.025x |
| operator rows, sign-flip build pair, Clang | 2.0 – 7.0 ns | 0.987x – 1.012x |
| operator rows, `bits_from` build pair, GCC | 2.2 – 5.2 ns | 0.886x – 1.093x |
| conversion rows, `bits_from` build pair, GCC | 0.7 – 1.6 ns | 0.801x – 1.245x |
| conversion rows, `bits_from` build pair, Clang | 0.7 – 1.7 ns | 0.701x – 1.164x |

Two things vary, and both matter.  Shorter rows widen the band, because a fixed
number of cycles of misalignment is a larger fraction of them.  And the *same*
rows widen from 0.971x–1.025x to 0.886x–1.093x between two build pairs, because
the second change moved far more of `.text` than the first.  A band inherited
from another change is not this change's band.

So the control rows are not a sanity check, they are a **calibration**, and the
rule that replaces the constant is:

> Read the band off the rows whose code is byte-identical, at a duration
> comparable to the effect's.  A per-row effect is reportable exactly when it
> clears that band with no overlap.  Otherwise report the aggregate, which
> layout does not bias, or count instructions.

Worked both ways.  This round's construction rows ran 0.665x–0.786x under Clang
against a 0.701x–1.164x band: overlapping, so the geomean over 15 shapes is the
claim and no single shape is.  The sibling crate's hardware rows ran
0.455x–0.806x against a 0.962x–1.090x band measured the same way: disjoint by
0.156, so there every individual row is reportable and the headline does not
rest on the aggregate at all.  Same protocol; a band three times wider on one
side than the other, and ten times wider than the 0.98x constant would have
implied; and the control says which of those you have before you write anything
down.

One limitation to state rather than paper over: a band is only calibrated where
byte-identical rows exist at that duration.  When a change touches the very rows
you would calibrate on — as the sign-flip change did, `neg` and `abs` being both
the effect and the only 0.2 ns rows — there is no same-duration control, and the
only safe reading is one that clears the widest band ever measured here.

## The two tables, and what each is for

`benches/arith.cpp` prints two.

The **ratio table** times each operator twice over the same operands, once as
the library computes it and once the way a caller would fake it through a host
float, and reports `host / soft`.  Both routes are timed in one binary over one
operand array, so a ratio here is self-contained: it survives a slow box, and it
is the only figure in this repository that can be quoted without a second run.

The **unary table** times negation, `abs`, `to_float`, `to_double`, and
construction from a `float`.  None of them has a second route — `to_float` *is*
the host route — so each row is an absolute nanoseconds-per-element figure, and
absolute figures mean nothing on their own.  A row from this table is quoted
only as a ratio between two builds measured under the interleaving above.

The unary table also runs shapes the ratio table skips.  `route` in
`benches/arith.cpp` refuses a shape no host float rounds like, which is right
for an operator comparison and wrong for a conversion: `IEEE<12, 3>` has no
opinion about which float should referee it, but it certainly has a `to_double`.

## An operator is timed only against a float that rounds like it

This is the rule behind `route` in `benches/arith.cpp`, and it is a correctness
rule, not a benchmarking nicety: below 2*p* + 2 digits in the intermediate,
rounding twice can differ from rounding once.  [arithmetic.md](arithmetic.md)
states it in full.  Do not restate it here.

## When the benchmark cannot see it

Some changes are invisible to the stopwatch by construction.  A shape the ratio
table skips still appears in the unary table, but a change that only removes a
call the compiler had already folded away leaves no time behind to measure.

For that kind, count symbols instead:

```sh
objdump -d bench | grep -cE 'call.*(exp2|ldexp|pow)'
nm -uC bench | grep -E 'exp2|ldexp'
```

A count is exact, needs no idle box, and has no noise floor.  It is the right
evidence for "this no longer calls libm" and the wrong evidence for "this is
faster".  Use the stopwatch for a change in what the code *does*, and the symbol
table for a change in *what the code links against*.

## Reference figures

Ryzen 7 8700F, `taskset -c 2`, idle box, 2026-08-21, at the head of this branch.
Ratio table geomeans over 56 comparisons; unary rows in nanoseconds per element.

| | GCC 16.1.1 | Clang 22.1.8 |
| --- | --- | --- |
| integer route wins | 23 of 56 | 37 of 56 |
| geomean | 0.966x | 1.044x |

Both fell from 28 of 56 / 1.001x and 52 of 56 / 1.195x when the conversion paths
got faster — and every conversion in this comparison is in the host arm, so what
fell was the numerator.  [arithmetic.md](arithmetic.md) has the accounting and
the rest of the numbers; the short version is that a falling ratio here is not by
itself evidence of a slower library, and the `soft` column is where to check.
