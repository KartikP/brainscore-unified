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
