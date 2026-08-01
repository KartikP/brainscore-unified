# EC2 runbook — post-verification cleanup (Group B)

> **RUN 2026-07-31 — ALL THREE TASKS PASSED.** Results in
> `unified/scripts/ec2_results_2026_07_31/`. 0.67 h, ~$1.09. Instance stopped.
> The env had to be rebuilt from scratch (no conda present at all) — see the note below.
> Task 5 PASS: `VisionWrapper(kind='vlm')` is score-identical to `VLMVisionWrapper`
> (IT exactly equal; V4's 1-ULP delta proved to be stale-cache-vs-fresh-extraction, not
> the facade — an original-wrapper re-extraction reproduces the migrated value exactly).
> Task 6 PASS: V-JEPA v1 Lahner visual-ROI raw **0.53285** vs 0.5329 baseline;
> `supported_modalities == ['vision']` confirms video→vision canonicalization.
> Task 7 PASS: ROI noise ceiling = **0.7305** (median sqrt(reliability), 4042 voxels);
> all six models re-scored and normalized, ranking unchanged, none above 1.0.
>
> **Env gotcha for next time:** installing `language` (or `unified`) pulls the PyPI
> `brainscore-vision` wheel, which SHADOWS the editable checkout — `brainscore_vision`
> then resolves to site-packages and the `-unified` benchmark variants silently do not
> exist. Fix: `pip uninstall -y brainscore-vision brainscore-language` then re-`pip
> install -e ./vision ./language`, and verify with `brainscore_vision.__file__`.

Two EC2-gated tasks, batched into one session because both need real weights +
scoring. Est. ~$2–4 wall, one g5.4xlarge. Both close open items from the
2026-07-30 codex recheck (`unified-model-interface-deliverables/CODEX-RECHECK-2026-07-30.md`).

## 0. Instance + env

- Reuse `i-0bdbdf83c4db9bdae` (g5.4xlarge, A10G 24 GB) — STOPPED, not terminated.
  `aws ec2 start-instances --instance-ids i-0bdbdf83c4db9bdae`.
- Env pins (rebuild only if the env is gone): `numpy<2`, `xarray==2022.3.0`,
  `scikit-learn>=1.5,<1.6`, `transformers` 4.57.x (`<5`), `importlib-metadata<5`,
  `pandas<3`. For the VLM path: `torch>=2.6` (cu124), then re-pin `pandas<3`.
- Sync code: push local `unified-model-interface-v2` for core/vision/unified, then
  `git pull` on the instance (or rsync the three repos).
- AWS creds present (S3 assembly pull for Task 6).

## Task 5 — VisionWrapper in a live plugin  (closes codex A6)

Goal: migrate ONE production registration to the `VisionWrapper` facade and prove the
score is **bit-for-bit identical**. This is what makes "no production reg uses the
facade" go away — and it must be verified against real weights, not asserted.

1. **Baseline (before any edit).** Score Qwen2.5-VL-3B:
   - `MajajHong2015public.V4-pls-unified` → expect **0.2113**
   - `MajajHong2015public.IT-pls-unified` → expect **0.3154**
   Save the exact floats.
2. **Edit** `unified/brainscore/models/qwen25_vl_3b/model.py`: replace the
   `VLMVisionWrapper(model=qwen_model.model.visual, processor=..., image_input_key=...,
   forward_kwargs_map=..., patch_count_fn=..., layer_aggregation='mean_patches',
   batch_size=4)` call with
   `VisionWrapper(model=qwen_model.model.visual, processor=..., kind='vlm', <same kwargs>)`.
   VisionWrapper forwards the extra kwargs straight to VLMVisionWrapper, so this is a
   pure delegation.
3. **Clear the stale cache** so features recompute through the facade path (the cache
   key does NOT fingerprint the wrapper class):
   `rm -rf ~/.result_caching/brainscore_vision.model_helpers.activations.core.ActivationsExtractorHelper._from_paths_stored/*qwen*`
4. **Re-score.** Assert identical to step 1 (0.2113 / 0.3154). Any delta = a bug in the
   facade forwarding; do NOT commit.
5. **Pin it.** Add the migrated Qwen score to `scripts/m3_regression_test.py` baselines
   (or a small regression test), so the facade path is guarded going forward.
6. Commit to `unified-model-interface-v2` ONLY if identical.

Lower-risk alternative if Qwen is fiddly: migrate CLIP ViT-B/32 (frame path,
`kind='frame'`) on `MajajHong2015public.IT-pls-unified` (expect ~0.374). Smaller,
faster, same identical-score proof — enough to close A6.

## Task 6 — video ≡ vision benchmark smoke  (final gate on channel unification)

Goal: confirm the `video → vision` canonicalization (core `546faa9` / `4f73e77`)
didn't break the REAL video extraction/scoring path. Unit tests use synthetic models;
this exercises real weights + real fMRI.

1. **Score V-JEPA v1 ViT-L** on `Lahner2024-fMRI-naturalistic-visualROI`. Assembly is
   ~11 GB on S3 (`s3://brainscore-storage/.../assy_Lahner2024-fMRI.nc`), pull once.
2. Expect ROI raw median r ≈ **0.5329** (post-StandardScaler-drop baseline,
   `REGION_LAYER_MAP['IT'] = backbone.blocks.16`). Deterministic (KFold rs=0), so
   confirm within ±0.005.
3. **Confirm the legacy video registration resolves under canonical vision:** in a
   REPL, load V-JEPA v1 and check `supported_modalities` reports `vision` (not
   `video`), and `check_compatibility(model, Lahner2024_video_benchmark)` passes.
4. If the score matches the baseline, channel unification is validated end-to-end.
   Record it in CLAUDE.md and flip the "video≡vision EC2 smoke" open item to DONE in
   the codex packet.

## Task 7 — Lahner noise ceiling  (unblocks putting numbers back on the site)

Goal: make Lahner's score ceiling-normalized so it is comparable to MajajHong/Pereira
instead of a raw undivided correlation. **Do this in the same session as Task 6** — that
task already pulls the 11 GB assembly, which is the only expensive part here.

1. `_split_half_reliability(n_splits=20, random_state=0)` at `benchmark.py:290` already
   computes Spearman-Brown-corrected per-voxel reliability; today it is only used to
   *select* voxels (`reliability_threshold=0.3`), never to normalize.
2. Compute it once on the visual-ROI voxel set, take the median over the masked voxels
   (matching how the score itself is summarized), and record the value.
3. Pass it as `ceiling=` to the benchmark factory (the parameter already exists,
   `benchmark.py:163`, defaulting to 1.0) and pin it as a module constant so scoring is
   reproducible without recomputing.
4. Re-score the six models (V-JEPA1 0.5329, CLIP 0.4556, V-JEPA2 0.4210, VideoMAE 0.3209,
   Qwen 0.2270, BLIP-2 0.1797 — all raw). Expect every value to rise; the **ranking must
   not change**, since it is a single positive divisor. A ranking change means a bug.
5. Sanity gate: a normalized score meaningfully above 1.0 means the ceiling is
   underestimated — do not ship it.

Then the video numbers can go back on the site next to MajajHong. Algonauts stays raw
(the challenge normalizes on its side); only its Codabench-graded number is conventional.

## Teardown

- `aws ec2 stop-instances --instance-ids i-0bdbdf83c4db9bdae`; confirm `stopped`
  (not just `stopping`). Never terminate — the EBS env is expensive to rebuild.
- Log spend to `scripts/cost_tracking/COST_LEDGER.md`.

## What this unblocks

Tasks 5 and 6 close codex A6 and the channel-unification gate.

Task 7 is what unblocks the presentation bucket. The bucket was NOT held for lack of
numbers — the numbers existed. It is held because Lahner and Algonauts report raw
undivided correlations (`ceiling=Score(1.0)`) while MajajHong and Pereira are
ceiling-normalized, so they cannot be shown on one scale. Task 7 fixes that for Lahner,
which is the only one of the two we can fix ourselves. See
`unified-model-interface-deliverables/PRESENTATION-BUCKET-2026-07-30.md` §REVISION.
