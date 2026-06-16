# Topographic-alignment axis — first real-fMRI run (NSD-surface)

**EC2 g5.4xlarge, 2026-06-16.** First run of the topographic-alignment axis on real fMRI,
using `allen2022_fmri_surface` (NSD on fsaverage) as the brain target. Validates the metric +
unit-coordinate contract + benchmark + shuffle null end-to-end on real data.

## Setup

- **Brain target:** NSD fsaverage surface betas (`Allen2022_fmri_surface_train`, 412 averaged
  images), subject `subj01`, region `IT` (NSD streams ventral parcellation), reliable vertices
  (`nc_testset > 10%`). Per-vertex `(x,y,z)` attached from the fsaverage **inflated** template,
  keyed by `(hemisphere, vertex_index)` via nilearn — no per-subject processing.
- **Metric:** `topographic-alignment` (correlation-vs-distance profile r(d)) + the
  shuffle-coordinate null, via `TopographicBenchmark`.
- **Candidate:** `clip-vit-b-32` (a non-topographic model → neutral grid fallback → expected
  near/below the null; the negative control).

## Result (single hemisphere, LH — clean)

| quantity | value |
|---|---|
| reliable LH vertices | 6,128 |
| brain r(d): near (d≈0.04) | **0.377** |
| brain r(d): far (d≈0.6) | **0.014** |
| profile monotonic decay | **yes** |
| CLIP raw (r(d) vs brain) | 0.295 |
| CLIP shuffle null | 0.451 |
| **CLIP signal (raw − null)** | **−0.155** |

**Brain ventral-stream topography is textbook:** r(d) decays monotonically 0.38 → 0.01 — nearby
cortical vertices strongly correlated, smoothly falling to ~0. The metric + fsaverage coords work
on real fMRI.

**The shuffle null is load-bearing.** CLIP-on-grid gets a *spuriously positive* raw (0.295) purely
from the arbitrary grid ordering — but the null (0.451) exceeds it, so the signal is **negative**.
The null correctly rejects the positive raw as a grid artifact, not genuine topographic alignment.
Without it, one would wrongly credit a non-topographic model with topography. A positive result
(signal > 0) requires a model with *real* unit coordinates (a topographic model: Topo-Omni / TDANN).

## Notes / gotchas

- **Use a single hemisphere** (or per-hemisphere). The first run used both hemispheres with an
  RH `+200` offset, which dumped all cross-hemisphere pairs into the far-distance bins (empty
  mid-bins + a spurious bump from LH/RH homology). Single-hemisphere gives clean graded distances.
  (Both-hemisphere run: raw −0.131, null +0.373, signal −0.504 — artifact-laden, kept in git log.)
- **Inflated surface** is used so Euclidean distance approximates geodesic cortical distance
  better than the folded pial would. A true geodesic distance would be a further refinement.
- **The empirical null floor here is ~0.45** (grid-shuffle level for this brain target); a
  non-topographic model sits below it.

## Next

- Register a **topographic model** (Topo-Omni's sheet, or a TDANN) whose `process()` attaches real
  `tissue_x`/`tissue_y` — the only missing piece for a *positive* topographic score.
- Optional refinements: geodesic distance; per-region (V1→IT) profiles; multiple subjects
  (averaged per fsaverage vertex, not stacked).
