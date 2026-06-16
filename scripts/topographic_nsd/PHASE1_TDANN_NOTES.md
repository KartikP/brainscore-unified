# Phase 1 (TDANN) — research findings + de-risked run plan

Status as of 2026-06-16: groundwork done, run not yet launched (the checkpoint↔positions
pairing needs one confirmation; won't launch a run that could silently mismatch positions).

## Confirmed

- **Model:** TDANN (Margalit 2024), a **ResNet-18** trained with a spatial-smoothness loss.
  Repo `neuroailab/TDANN`, data on **OSF project `64qv3`** (`osfclient` works: `osf -p 64qv3 fetch <remote> <local>`).
- **Canonical checkpoint:** `osfstorage/tdann_data/tdann/checkpoints/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_checkpoints/model_final_checkpoint_phase199.torch`
  (the SimCLR self-supervised + spatial-loss model; `isoswap_3` = the paper's default arrangement).
- **Per-unit positions** are per-layer `.npz` with a `coordinates` (N×2) array, N = C·H·W of the layer
  (C-major flatten). Regions: V1/V2/V4/**VTC** (VTC ≈ IT). Record **layer4.1** (VTC-like) → pair with the
  brain **IT** region of NSD-surface.
- **The clean loader is the demo's `src/` package, NOT VISSL.** The demo (`~/TDANN/demo/`) uses:
  ```python
  from src.positions import NetworkPositions
  from src.model import load_model_from_checkpoint, LAYERS
  from src.features import FeatureExtractor
  net_pos = NetworkPositions.load_from_dir(positions_dir)
  model   = load_model_from_checkpoint(weights_path)
  ```
  This avoids the VISSL dependency in `spacetorch/models/__init__.py` entirely. **Vendor `demo/src/`**
  (or import it) rather than the prefix-strip approach in `tdann_phase1.py` — it's the intended,
  de-risked path. (My `tdann_phase1.py` VISSL-strip loader is a fallback; prefer `src/`.)

## The one open question (resolve before running)

Which positions dir pairs with the `isoswap_3` checkpoint? The only published SimCLR positions dir
found is `.../positions/simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lw0/resnet18_retinotopic_init_fuzzy_swappedon_SineGrating2019_NBVER2/layer4.1.npz` — the `lw0` (loss-weight-0) in the path is
suspicious for a topographic model. **Confirm from `demo/src` defaults / the demo's intended `positions_dir`
that this is the correct arrangement for `isoswap_3`** (or find the matching positions). Using the wrong
positions silently produces a meaningless score — this is the gating check.

## De-risked run plan (next session, ~30–60 min EC2)

1. Start instance; `cd ~/TDANN/demo` (the clean `src/` is there).
2. `osf -p 64qv3 fetch` the isoswap_3 checkpoint + its positions dir (small: ResNet-18 ckpt ~45 MB,
   positions npz tiny).
3. Load via `src.model.load_model_from_checkpoint` + `src.positions.NetworkPositions.load_from_dir`;
   confirm `coordinates` count == layer4.1 unit count.
4. Extract layer4.1 features on the 412 NSD train images (`src.features.FeatureExtractor`).
5. Attach `coordinates[:, :2]` as tissue_x/tissue_y; build the NSD-surface IT/LH brain target
   (reuse `stage_and_score.py` / `tdann_phase1.py` brain-load); score through `TopographicBenchmark`.
6. **Expectation:** TDANN's spatial loss → genuine unit topography → **signal > 0** (clears the shuffle
   null), unlike CLIP-on-grid (signal −0.155). That is the first positive topographic-alignment result.

## Phase 2 (Topo-Omni) — after TDANN

Vendor the `epflneuroailab/topo-omni` custom modeling (`CorticalAdaptor`), load the 5B BF16 checkpoint
(fits one A10G), record the **vision band** of the 304×512 sheet (rows 0–159, cols 0–255), use the sheet
`(row, col)` directly as tissue coords, score the same benchmark. ~3–5 days eng + ~$10–20 EC2.
