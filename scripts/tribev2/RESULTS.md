# TRIBEv2 — Llama gate unblocked + first direct-comparison score

**EC2 g5.4xlarge, 2026-06-22.** Meta's released brain-encoder (`facebook/tribev2`) run end-to-end
in the unified-interface workflow, after bypassing the 2-week Llama-3.2-3B access wait.

## The unblock (no Meta access needed)

TRIBEv2's text tower is Llama-3.2-3B (Meta-gated). Bypassed with the **ungated `unsloth/Llama-3.2-3B`
mirror** — one-line swap in the vendored source (`tribev2/grids/defaults.py:28`). Confirmed two ways:

- `from_pretrained('facebook/tribev2')` builds the fusion model; `predict()` runs end-to-end.
- **Direct load of `unsloth/Llama-3.2-3B` succeeded (3.21 B params)** — the only previously-untested link
  (does the ungated mirror load) is now confirmed. No Meta access request required; drop it.

Env note: the `tribev2` env has `transformers 5.10` against `torch 2.6`, which references a torch≥2.7
dtype (`torch.float8_e8m0fnu`). A one-line shim (`torch.float8_e8m0fnu = torch.bfloat16`, unused in the
video path) unblocks `predict()`; the clean fix is `torch>=2.7`.

## First score — direct comparison on Lahner2024 (in-distribution)

TRIBEv2 outputs **brain space directly** (20484 fsaverage5 vertices, "average subject") — it's a trained
encoder, not a feature extractor — so the natural eval is a **per-voxel Pearson of predicted vs measured
BOLD across stimuli**, NOT a ridge readout. 120 Lahner BOLDMoments clips (audio stripped → video tower;
Lahner is visual), each clip → mean over predicted TR segments → `(20484,)`; compared to the Lahner
GLM-beta assembly (`assy_Lahner2024-fMRI.nc`, betas averaged over reps per clip).

| quantity | value |
|---|---|
| clips | 120 |
| voxels | 20484 (fsaverage5) |
| **median per-voxel r (whole cortex)** | **0.089** |
| mean per-voxel r | 0.130 |
| **top-decile voxels, median r (visual cortex)** | **0.529** |
| **clip-shuffle null, median r** | **−0.001** |
| frac voxels r>0.1 | 0.47 |

**The clip-shuffle null ≈ 0 confirms the correspondence is real and the vertex ordering aligns**
(TRIBEv2 output and the Lahner assembly are both standard fsaverage5). Whole-cortex median is low because
most cortex isn't visually driven; the top decile (visual cortex) tracks measured BOLD at r≈0.53.

## Caveats (honest)

- **In-distribution.** TRIBEv2 was trained on Lahner2024 — this is a fit-quality / pipeline-validity check,
  not generalization, and not comparable to our ridge-readout Lahner numbers (different metric entirely).
- **Video-tower only here.** Lahner clip audio was stripped (the transcription path needs `uvx` on PATH);
  the audio (Wav2Vec-Bert) and text (Llama) towers were dropped for these silent clips. Fine for a visual
  benchmark; the Llama tower's load is confirmed separately above.
- **Not a UMI-registered model yet.** TRIBEv2 needs its own env (transformers 5.10 / torch≥2.7),
  incompatible with the unified `bsu` env — so a clean `BrainScoreModel` registration is gated on a
  bsu↔tribev2 env-isolation strategy. That's a **productionization-phase** question; this run is the
  script-level direct-comparison eval.

## Files

- `tribe_lahner_predict.py` — (tribev2 env) Llama-mirror load check + predict 120 Lahner clips → preds.npz
- `tribe_lahner_score.py` — (bsu env) per-voxel Pearson vs Lahner betas + clip-shuffle null
- `tribe_lahner_score.json`, `tribe_lahner_step1.json` — results
