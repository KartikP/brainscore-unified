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

## Phase 2 — Topo-Omni: infrastructure works, extraction NOT yet valid (2026-06-17)

The 5B custom Topo-Omni model (`epfl-neuroai/topo-omni`, Qwen2.5-Omni-3B + `CorticalAdaptor`) loads
end-to-end **VISSL-free** (mirroring their `src/eval/extract/extract_nsd.py`: custom
`Qwen2_5OmniThinkerForConditionalGeneration` + `Qwen2_5OmniProcessor`, bf16, one A10G), produces
`out.unified_sheet` (304×512), and runs through the `TopographicBenchmark` — so the *pipeline* is proven
for a 2026 multimodal model. Script: `topo_omni_phase2.py`.

**But the result is NOT a trustworthy positive and is reported as such.** Headline signal was +0.417
(raw 0.088, null −0.329), yet **Topo-Omni's own r(d) profile is flat (~0.002, no decay)** — nothing like
TDANN's genuine 0.506→0 decay. The +0.417 is a **numerical artifact**: a flat raw profile correlated with
the brain's decaying profile gives ~0 (0.088), while the shuffle null landed spuriously negative (−0.329),
so raw−null came out positive. **My sheet extraction does not capture Topo-Omni's topographic
organization** (the paper clearly has it — Fig 13 Island Moran's I ≈ 0.5).

**Why (likely) + the fix for next session:** the paper measures topography via **selectivity maps**
(category-contrast t-values per unit) + **Island Moran's I** (spatial autocorrelation of selectivity),
NOT the raw response-correlation r(d) that worked for TDANN. The unified-sheet per-position scalar's
*raw* across-stimulus correlation appears spatially unstructured as I read it (74% of units "responsive"
but pairwise-uncorrelated). Next: either (a) replicate their `src/eval/run/run_selectivity.py` →
selectivity → Moran's I as a second topographic metric, or (b) determine why the raw-sheet r(d) is flat
(inspect what `unified_sheet[r,c]` actually represents; try the per-layer sheet tuple or a selectivity
contrast instead of raw responses). This is a focused next-session debug, not a model/infra problem.

## Headline so far

| model | r(d) genuinely decays? | signal | trustworthy? |
|---|---|---|---|
| CLIP ViT-B/32 (non-topographic) | no | −0.155 | yes — correct negative control |
| **TDANN (topographic ResNet-18)** | **yes (0.506→0)** | **+0.838** | **yes — the validated positive** |
| Topo-Omni (topographic VLM) | **no (flat ~0.002)** | +0.417 | **NO — artifact; extraction needs fixing** |

**TDANN is the validated end-to-end result.** The pipeline + metric + null are proven; the axis cleanly
separates a genuinely-topographic model (TDANN) from a non-topographic one (CLIP). Topo-Omni's *infra*
works but its sheet extraction is unresolved.

## Next

- **Fix Topo-Omni extraction** (selectivity-map + Moran's I, per the paper) — the open Phase 2 item.
- Optional refinements: geodesic distance; per-region (V1→IT) profiles; multiple subjects.
