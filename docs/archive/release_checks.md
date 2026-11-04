# Checks to run before a release

[Build tools and integrations](tool_authoring.md) | [Release qualification](production_release.md) | [Numerical policy](numerical_policy.md)

See the [release test matrix](release_test_matrix.md) for the current CI baseline, proposed GitHub Actions/Jenkins split, and routing of skipped and unselected tests.

## Required offline checks

The coordinated GitHub Actions workflow builds all four packages and runs the
offline source profile on Ubuntu and macOS. This validates core as part of the
coordinated checkout. A core-only PR does not automatically trigger this workflow;
run coordinated validation against the proposed peers before merging. Independent
core triggering and Jenkins configuration remain separate infrastructure work.

```sh
python -m brainscore.validation.workspace --root /path/to/repos --out source.json
```

The report records revisions, local changes, commands, selected and deselected
tests, unselected test files, and every skip reason. Unexpected skips, empty
runs, missing reports, failures, and timeouts fail the profile. Only the exact
test/reason pairs in `brainscore/validation/optional_checks.py` may skip. These
optional results do not qualify OpenPI/JAX, external data, or maintainer tools.
Adding an exception requires a documented prerequisite and separate validation.

The source selection lives in `brainscore/validation/workspace.py`. It includes
legacy scoring, compatibility, state changes, cache invalidation, hook cleanup,
and early-failure checks. It is not the full plugin test suite or a complete
release gate. Required scientific and environment qualifications follow below.

## 1. Three-route benchmark parity

```bash
python -m brainscore.validation.run_parity --resnet18 <path> --gpt2 <dir>
```

Runs every `-unified` variant through its legacy plugin, that plugin behind the
compatibility adapter, and a natively-registered model. Adapter activations must
be exact; scalar scores and native outputs use the explicitly selected numerical
policy. The default historical policy and the scoped CPU FP32 policy differ.
Reports identify the policy, execution environment, checkpoint hashes and failed
observations. An FP64 reference run cannot qualify default FP32 execution.

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
