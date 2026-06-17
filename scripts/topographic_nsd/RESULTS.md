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

## Phase 1 — TDANN: the first POSITIVE result (2026-06-17)

Scored **TDANN** (SimCLR + spatial-loss ResNet-18, Margalit 2024 — `isoswap_3` checkpoint) on the
*same* NSD-surface target (subj01, IT/ventral, LH, 6,128 vertices), recording `layer4.1` (VTC-like)
and attaching TDANN's own published per-unit cortical positions as tissue coords (25,088 units).
Loaded VISSL-free via the demo's `src/model` convention (strip the `base_model.` prefix into a
torchvision ResNet-18). Script: `tdann_phase1.py`; result: `results/tdann_phase1_result.json`.

| model | raw r(d)-alignment | shuffle null | **signal (raw − null)** |
|---|---|---|---|
| CLIP ViT-B/32 (non-topographic, grid fallback) | 0.295 | 0.451 | **−0.155** |
| **TDANN (topographic, real unit positions)** | **0.751** | −0.087 | **+0.838** |

**This is the headline validation of the topographic-alignment axis on real fMRI.** A genuinely
topographic model (TDANN) clears the shuffle null by +0.84 — its units' spatial layout matches the
brain's ventral-stream organization — while a non-topographic model (CLIP) sits *below* the null.
TDANN's own r(d) profile decays from 0.506 (nearby units strongly correlated) as designed.

**Self-validating:** a +0.75 raw alignment is impossible from mismatched/wrong positions (those would
score at the null), so the checkpoint↔positions pairing is confirmed correct by the positive result.

The axis now cleanly distinguishes topographic from non-topographic models, with the shuffle null as
the discriminator — exactly what predictivity cannot do (it is permutation-invariant over units).

## Next

- **Phase 2: Topo-Omni** — vendor the `CorticalAdaptor` custom modeling, record the 304×512 vision-band
  sheet (rows 0–159 / cols 0–255) with tissue coords = sheet (row,col), score the same benchmark.
  The flagship 2026-multimodal positive result.
- Optional refinements: geodesic distance; per-region (V1→IT) profiles; multiple subjects.
