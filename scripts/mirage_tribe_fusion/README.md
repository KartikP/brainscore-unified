# Native vs. post-hoc multimodal fusion — MIRAGE vs. TRIBEv2 (controlled, sub-01)

Controlled comparison of where multimodal fusion happens, scored through the
**same ridge readout**, same TRs, same design matrix, same held-out-clip CV — the
only variable is the feature source.

- **NATIVE (MIRAGE-style):** `Qwen3-Omni-30B-A3B-Thinking` post-fusion thinker
  hidden states (layer 24, mean-pooled) — video+audio+text interact inside one model.
- **POST-HOC (TRIBEv2-style):** `V-JEPA-2 ViT-L` (video, layer 16) + `Wav2Vec-Bert-2.0`
  (audio) + `Llama-3.2-3B` (text, layer 14), extracted separately and concatenated.

Data: Algonauts2025 CNeuroMod **sub-01**, **6,010 Friends-S1 TRs** (12 clips), 1000
Schaefer parcels. Design: per-clip lagged window (stimulus_window=5, hrf_delay=3),
KFold over clips, α-grid CV, median per-parcel Pearson r on held-out clips.

## Results (`mirage_tribe_compare.json`)

| Feature source | fusion | dim | r | extract |
|---|---|---|---|---|
| **Qwen3-Omni-30B (MIRAGE)** | native | 2048 | **0.1499** | ~68 min |
| **V-JEPA-2 + Wav2Vec-Bert + Llama (TRIBEv2)** | post-hoc concat | 5120 | **0.1459** | ~30 min |
| V-JEPA-2 video only | unimodal | 1024 | 0.1258 | ~27 min |
| Wav2Vec-Bert audio only | unimodal | 1024 | 0.0876 | ~1.6 min |
| Llama-3.2-3B text only | unimodal | 3072 | 0.0790 | ~1.2 min |

**Native fusion edges out post-hoc (+0.004) at 2.5× fewer features** — directionally
consistent with MIRAGE — but the margin is within single-subject/subset noise and
the cross-architecture bars are confounded by backbone identity.

### Within-Qwen fusion ablation (the confound-free result)

To isolate fusion from backbone identity, read the SAME Qwen3-Omni at every thinker
layer (all 49 hidden states extracted in one pass), same 2048-d readout, same CV
(`extract_qwen_alllayers.py` + `ridge_fusion_curve.py`, `qwen_fusion_curve.json`):

- **fusion OFF** (layer 0, modality tower streams before any cross-modal attention): **r = 0.1233**
- **fusion ON** (peak layer 42, fully cross-attended): **r = 0.1546**
- **fusion gain: +0.0313** (+25% relative), monotonic rise L0→L42, then a drop at L48
  (final layer specializes for next-token prediction).

Same model throughout → **the gain is fusion, with no backbone confound**. This is the
decisive evidence; the cross-architecture bars above are only suggestive. Absolute r (~0.15) is below
MIRAGE's full-set ~0.21–0.32 because of the subset, one subject, a single fixed layer
per tower (no learned layer aggregation), and a plain ridge (not a trained encoder).
Both arms share these simplifications, so the comparison is controlled; the numbers
are lower bounds. Native fusion costs ~2.3× the extraction compute and a far larger model.

## Infra lessons (EC2)

- **Qwen3-Omni-30B 4-bit MoE quantization hard-crashes at ~48% load** on a single
  48 GB GPU (silent kill, no traceback — bnb fails to quantize the 128 MoE experts,
  they load fp16 → 30B fp16 ≈ 60 GB > 48 GB). CPU-offload net does NOT save it.
  **Fix: bf16, no quantization, sharded across 4× L40S** (g6e.12xlarge, 192 GB).
- **Audio-encoder dtype mismatch** (`Input type float, bias bf16`) → wrap the forward
  in `torch.autocast('cuda', dtype=torch.bfloat16)`.
- **The "Thinking" checkpoint is thinker-only** (no Talker/Code2Wav); load the full
  `Qwen3OmniMoeForConditionalGeneration`, call `model.thinker(...)` for features.
- **`pkill -f <script>.py` in a launch command kills the launcher's own ssh shell**
  (its command line contains the string) — launch without it.
- **Llama-3.2-3B is gated**; if the HF account hasn't accepted Meta's license, use the
  ungated re-host `unsloth/Llama-3.2-3B` (identical weights).
- **Ephemeral NVMe (`/opt/dlami/nvme`) wipes on stop/start** — keep the model cache on EBS.

## Files
- `qwen_omni_smoke.py` — load + single-window forward smoke (bf16, 4-GPU).
- `extract_qwen_native.py` — native arm (per-TR thinker hidden states).
- `extract_tribe_{video,audio,text}.py` — post-hoc towers (per-TR/per-clip features).
- `ridge_compare.py` — the controlled comparison (records `scoring_sec` + `feature_dim`).
- `mirage_tribe_compare_results.json` (in `scripts/algonauts2025/`) — the scores.
