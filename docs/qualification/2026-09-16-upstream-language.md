# GPT-2 matches clean upstream after a cache fix

All four fixed GPT-2/Pereira cases now match clean upstream exactly on the
recorded macOS CPU FP32 profile. Activations, input order, raw scores, ceilings,
and normalized scores agree. A separate run also matches across legacy, adapter,
and native UMI routes.

## What the comparison found

The v2 language helper always enabled the token cache. Upstream enables it only
for behavioral calls without neural recordings. That difference changed neural
activations: the initial 243-sentence ridge comparison had 160,909 unequal values,
with a maximum absolute difference of 0.000244140625.

The repair restores upstream's cache selection. Neural recording uses the full
context; behavioral-only inference retains the cache. FP32 remains unchanged.
An offline regression test independently checks full-context output. The affected
language suites pass: **49 tests**.

## Results after repair

| Pereira case | Normalized score | Upstream and three UMI routes |
| --- | ---: | --- |
| 243 sentences, linear | 0.8712620929373056 | Exact |
| 384 sentences, linear | 0.8292755863666226 | Exact |
| 243 sentences, ridge | 0.8353256254695343 | Exact |
| 384 sentences, ridge | 0.656385233398218 | Exact |

The upstream linear identifiers end in `-linear-shuffle`; the candidate calls
those same protocols `-linear`. Ridge uses the current grouped-fold protocol.
The clean-upstream workers import only core and language, with no unified package.
They disable result caches, verify checkpoint hashes and import origins, and
retain complete activation arrays. The audit compares these independently.

The three-route run selected all four language cases, so `selection_complete`
is true. It is a subset of the eight-case suite: `release_complete` remains false.
The native numerical budget was not changed; observed activation and score
differences are zero in this run.

## Scope and remaining work

This result qualifies the repaired source on one CPU profile. Earlier Linux CPU
and NVIDIA L4 results describe the previous source identities. They do not qualify
this repair. The changed helper needs fresh Linux/GPU checks and current artifact
checks before broader support can be claimed.

Source identities, environments, hashes, and detailed scores are in the
[machine-readable record](2026-09-16-upstream-language.json). Local evidence lives
under `artifacts/upstream-reference-2026-09-16/`, including the initial failure,
the corrected comparisons, full activation arrays, and the three-route report.
