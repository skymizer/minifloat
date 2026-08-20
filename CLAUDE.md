# Working on minifloat

*The standing decisions live in `docs/`; this file is the routing table to them
and the short list of things that will bite you.*

## Where to read before you touch something

| Touching | Read first |
| --- | --- |
| any operator, `from_parts` / `to_parts`, `round_to_scale`, `ALIGN_CAP`, `QUOTIENT_BITS` | [docs/arithmetic.md](docs/arithmetic.md) |
| `tests/arith.cpp`, the exact oracle, `reference_encode` in `tests/support.hpp` | [docs/arithmetic.md](docs/arithmetic.md) |
| `benches/arith.cpp`, or any claim with a number in it | [docs/benchmarking.md](docs/benchmarking.md) |
| the format policy chain, adding a format, adding an alias | [README.md](README.md), its Design section |
| finishing a round | [CHANGELOG.md](CHANGELOG.md), its own final commit |

Those documents record decisions that have already been made and paid for.
Reversing one is fine; reversing one without reading why it was made is how the
same question gets asked a fourth time.

`README.md` is the public face — it ships to whoever installs the header, and it
is where the layering rationale lives.  The files under `docs/` are for whoever
is working on the library.  A fact belongs in exactly one of them.

## Standing rules

**Everything lives in `minifloat.hpp`.**  One installed header, no `detail/`
directory, no second translation unit.  A helper that only the tests need goes
in `tests/support.hpp` instead, where it is also independent of the library —
see the oracle section of [docs/arithmetic.md](docs/arithmetic.md).

**No new dependencies.**  The C++17 standard library and IEEE 754 `float` and
`double`, and that is the whole list.  GoogleTest is a test-only dependency and
stays one.

**`make format` before committing.**  `clang-format` is configured and CI does
not run it, so it is on you.  `.clang-tidy` is configured and CI does not run
that either.

**Add a test file to `CMakeLists.txt` by hand.**  The `Makefile` globs
`tests/*.cpp`; CMake enumerates the five files explicitly, and CI builds through
CMake.  A new test file passes locally and is silently invisible to CI until the
list is edited.

**Benchmark only on an idle box, under both compilers.**  The box runs a poker
solver that will take every core: `pgrep -af poker` first, and if it is running,
ask — never kill it unasked.  The protocol is
[docs/benchmarking.md](docs/benchmarking.md), and it is not optional:
interleaved builds, min-of-N across at least 15 alternating passes each, a
control row the change cannot touch, a 0.98x noise floor, and GCC *and* Clang,
because on the same source they disagree by 37 of 56 against 23 of 56.  `rm -f bench` between compilers; the `make` target does not depend on
`CXX`.

## The correctness gate

`make check` before anything else, and before any benchmarking.

It is exhaustive where exhaustive is affordable, which is most places: every bit
pattern of 48 declared shapes for the encoding, conversion and classification
checks, every *ordered pair* of the 42 shapes through 11 bits for comparison,
and every *ordered pair* of the 39 shapes at 8 bits and under for arithmetic.  Two gates live in there and it is worth not conflating
them:

- **the exact integer oracle** (`Arith.CorrectlyRoundedSmallFormats`) — refereed
  by cross-multiplication and a binary search over the format's own codes, with
  no float involved and no constant shared with the engine;
- **the host round-trip sweep** (`Arith.MatchesHostRoundTrip`) — narrower, but
  it covers the non-finite pairs the exact oracle skips, and it is what licenses
  `benches/arith.cpp` to time the two routes against each other.

The `Makefile`'s `test` target does not define `NDEBUG`, so the constructors'
preconditions are live under `make check` and compiled out under `make bench`.

## Commit conventions

Subjects are imperative and sentence-shaped — *Compute arithmetic on integer
significands*, not *feat: add soft-float*.  Every measured claim goes in the
commit body with the binary that produced it and the compiler it ran under, and
a null result is recorded as plainly as a win.  The changelog gets its own final
commit per round, in Keep a Changelog form.

## Voice

Documentation and comments explain the decision, not the syntax.  Prefer symbol
names over line numbers, state the thesis before the reasoning, and let a
measurement carry its own machine, compiler, and date.  A comment that says what
the next line does is worth deleting; one that says why the obvious thing was
not done is worth keeping.  Doc comments are `//!`.
