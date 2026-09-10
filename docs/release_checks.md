# Checks to run before a release

The fast tier runs on every push and takes under a minute. These do not: they
need benchmark assemblies, real model weights, and hours. They are listed here
because each one exists in response to a defect that shipped, and the failure
mode they guard against is silent — a wrong number, not an exception.

## 1. Three-route benchmark parity

```bash
python -m brainscore.validation.run_parity --resnet18 <path> --gpt2 <dir>
```

Runs every `-unified` variant through its legacy plugin, that plugin behind the
compatibility adapter, and a natively-registered model, and asserts all three
hand the metric identical arrays and produce the same score.

**Why.** "The `-unified` variants were validated bit-for-bit against legacy" was
true only of the adapter route. Nobody had compared the native one, which is how
the Pereira variants ran for months showing native models one bare sentence at a
time while adapter models got the running passage context. Both paths were
self-consistent; only comparing them exposed it.

Roughly three hours, ~10 GB resident. A 15 GB machine is OOM-killed part way
through the vision cases.

Last full run, 2026-09-08: four vision variants at exactly 0.0 activation delta
and identical scores; both Pereira variants agree to within 16 FP32 ULPs on the
native route, which is FP32 execution-path sensitivity rather than a defect (see
`docs/audits/2026-09-08-umi-review-fixes.md`).

## 2. Legacy backwards compatibility

Score registered legacy plugins through the original path and through the UMI
adapter, and require bit-identical results. Backwards compatibility is a UMI
requirement and this is what demonstrates it.

Last run, 2026-09-08: seven for seven at delta exactly 0.0 — distilgpt2, gpt2
(243 and 384), gpt2-medium, gpt2-large, gpt2-xl, gpt-neo-2.7B.

**Not yet closed:** this establishes that our two paths agree with *each other*,
not that either agrees with the published leaderboard.
`baselines/M3_RESULTS.md` still lists language production baselines as "TBD".

## 3. Published-figure reproduction

Any number on the public site should be reachable from the repository. See
`brainscore/benchmarks/induced_dyslexia/reproduction/` for the worked example:
the 32B dyslexia curve, its rerun, and the arguments that had to be recovered
from the saved artifact because the original invocation was never recorded.

## 4. Cold caches

Run the fast tier with `RESULTCACHING_DISABLE=1` and cold activation caches. A
warm cache has hidden a real defect three times here, most recently a truncation
fix that returned a byte-identical wrong score because the cache key did not
include the tokenizer configuration.

## 5. Which Pereira variant a number came from

`Pereira2018-linear` and `Pereira2018-ridge` are different benchmarks and their
numbers are not interchangeable. Ridge groups cross-validation by story; linear
splits sentences at random, which leaks within a passage. Upstream retired the
plain-linear registration for that reason in #361 (2026-05-18), leaving ridge
and linear-shuffle.

Measured 2026-09-10, same model and layer, legacy route:

| | 243 linear | 243 ridge | 384 linear | 384 ridge |
|---|---|---|---|---|
| gpt2 | 0.8713 | 0.8353 | 0.8293 | 0.6564 |
| opt-6.7b | 1.0 (clipped) | **0.8696** | 1.0 (clipped) | 0.6310 |

The ceilings differ too — 0.354 for linear against 0.134 for ridge — so the raw
correlations are on different scales. Linear's raw exceeds its ceiling for
models above roughly 2.7B and the score clips at 1.0, which is why linear
cannot rank large models. Ridge does not clip.

**The published leaderboard figure is ridge.** opt-6.7b's 0.8696 here is the
0.87 reported publicly. Any comparison against the leaderboard must use the
`-ridge` variants; `-linear` numbers are internally consistent but describe a
retired protocol.
