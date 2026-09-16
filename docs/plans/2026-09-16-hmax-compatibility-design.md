# HMAX compatibility investigation

The release candidate is committed. This follow-up investigates the unresolved
HMAX score drift without changing benchmark definitions, model weights,
preprocessing, layer-selection policy or numerical acceptance budgets.

## Evidence and approach

The April 9 local M3 report records Python 3.11.15, Torch 2.6.0+cu124,
transformers 4.57.6, scikit-learn 1.5.2 and a T4. It lacks source commits,
selected layers, per-layer scores, patch hashes and the production environment.
Its attribution to scikit-learn/layer search remains an unverified hypothesis.

The follow-up found that HMAX already ships a fixed JSON region map: V4 uses
`c2_0` and IT uses `c2_2`. The normal model loader uses that map without
constructing LayerSelection, and the candidate wheels contain the same map.
Both mapped scores reproduce the April report to six decimal places on the
current CPU stack. Automatic selection is a fallback when no map loads, not
the current default. The original report's causal explanation is unsupported.

The chosen approach isolates fixed-layer legacy/adapter behavior first. A full
layer search mixes preprocessing, PCA training, regression and selection; it
cannot alone identify an interface regression. Simply increasing the score
budget would hide the unexplained difference. Preserve the historical scores
and record fresh evidence under explicit scope.

## Runtime fixes before scientific comparison

1. Remove the import-time process-wide HTTPS verification override.
2. Verify the declared HMAX patch-set hash before model construction. The shared
   deprecated weight-loader signature currently accepts but ignores that hash.
3. Keep activation caching disabled for HMAX without overriding a caller's
   global cache disable; restore the original environment on success or failure.
4. Remove temporary forward hooks even when layer lookup or inference fails.

These repairs affect integrity and cleanup, not the HMAX forward computation.
Tests will first reproduce the failures, then verify corrected behavior. Small
real-checkpoint probes will retain exact layer outputs and execution settings.
A small probe is not a benchmark qualification or proof of a historical cause.

## Qualification boundary

Use local CPU execution and existing staged stimuli. No additional EC2 start is
authorized by this phase. Verify the public 9.6 MB checkpoint over ordinary
certificate-verified HTTPS and retain its hash. Record candidate layers and
fixed-layer route equality before deciding the size of a full scoring run.
Qualify the declared HMAX profile against clean upstream and track older
published-reference reproduction separately. Historical logs are not a
prerequisite for testing current-upstream compatibility.

## Completed upstream control

The [master comparison](../qualification/2026-09-16-hmax-master.md) runs clean
vision master and core main through normal model loading and score_benchmark.
Both full V4/IT cases match the UMI candidate's activation values and scientific
scores exactly on the recorded macOS CPU profile. The published-score gap also
exists upstream, so it is not evidence of a UMI regression in these cases.
