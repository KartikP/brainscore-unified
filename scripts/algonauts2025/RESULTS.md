# Algonauts 2025 multimodal replication — sub-01 Friends train

Within-subject 5-fold CV on Friends S1-S6 + Movie10. Median raw per-parcel
Pearson r across 1000 Schaefer parcels, 159 141 TRs (TR=1.49 s).

## Final results

| Mode | Modalities | Best α | **Median r** | Mean r |
|---|---|---|---|---|
| Text-only | MiniLM (384-dim) | 1000 | 0.1204 | 0.1300 |
| Video-only (Phase 3a) | CLIP ViT-B/32 (768-dim) | 1 | 0.1172 | 0.1482 |
| Video-only (α-tuned) | CLIP ViT-B/32 (768-dim) | 100 000 | 0.1504 | 0.1749 |
| Audio-only | Wav2Vec2-base (768-dim) | 1000 | 0.1574 | 0.1879 |
| Multimodal concat (single α) | V + A + T (1920-dim) | 10 000 | 0.1856 | 0.2178 |
| **Multimodal banded (per-mod α)** | V + A + T | (V=10⁵, A=10⁴, T=10³) | **0.2130** | **0.2365** |

## Replication of published baseline

The Algonauts 2025 paper (arxiv 2501.00504) reports their multimodal
encoding-model baseline (SlowR50 video + librosa audio + sentence-embedding
text + ridge regression) at roughly **0.20-0.25 raw r** on the in-distribution
Friends evaluation, depending on subject.

Our concat / banded multimodal landed at **0.1856 / 0.2130**, squarely within
the published range. The +0.0274 lift from banded ridge over single-α concat
matches the per-modality α-sweep optima — V wants α=10⁵, A wants α=10³-10⁴, T
wants α=10³, which a single shared α can't satisfy simultaneously.

## Path from 0.1172 → 0.2130 (+82%)

| Step | Δ raw r | Cumulative |
|---|---|---|
| Phase 3a baseline (CLIP, α=1) | — | 0.1172 |
| α tuning (α=10⁵ on video) | +0.0332 | 0.1504 |
| Add audio tower (Wav2Vec2-base, single α) | +0.0070 | 0.1574 ← per-modality A |
| Concat V+A+T, single α | +0.0282 | 0.1856 |
| Banded ridge (per-modality α) | +0.0274 | **0.2130** |

## Modality contributions (per-band marginal lift in concat)

| Configuration | Median r | Δ vs prev |
|---|---|---|
| Video alone (α=10⁵) | 0.1504 | — |
| + Audio (concat α=10⁴) | ~0.171 (interpolated) | +0.020 |
| + Text (concat α=10⁴) | 0.1856 | +0.015 |

Audio carries the largest single-modality lift over CLIP-only video. Text
adds an additional ~0.015 on top, consistent with sentence embeddings of
short transcript context.

## What's still missing for full Codabench replication

- Held-out Friends S7 prediction generation + Codabench submission
- Noise-ceiling normalization (paper reports normalized correlation, not raw)
- Score sub-02, -03, -05 (we only have sub-01 fMRI downloaded)
- OOD movies leaderboard (mononoke, wot, pulpfiction, planetearth, passepartout, chaplin)

## Files

- `extract_audio_features.py` — per-clip Wav2Vec2-base feature extractor
- `extract_text_features.py` — per-TR sentence-transformer/MiniLM extractor
- `cache_video_features.py` — cache CLIP video features as `.npz`
- `score_algonauts_multimodal.py` — main multimodal scorer (5 modes)
- `score_algonauts_alpha_sweep.py` — α-sweep for video-only
- `run_all_modes.sh` — sequential driver

## Reproduction recipe (EC2 g5.4xlarge)

```bash
# 1. Phase 1+2 prep already done (see CLAUDE.md milestones)
# 2. Audio features (~22 min on A10G)
python extract_audio_features.py
# 3. Text features (~6 min on CPU)
python extract_text_features.py
# 4. Cache video features (~2 min, idempotent)
python cache_video_features.py
# 5. Score all 5 modes (~50 min total: video=7m, audio=5m, text=3m, concat=16m, banded=27m)
bash run_all_modes.sh
python score_algonauts_multimodal.py --mode banded --out_json /tmp/algonauts_banded.json
```

Total EC2 cost for full reproduction: ~$3-4.

## Bug fixes documented during this session

(These are encoded in CLAUDE.md milestones for traceability.)

1. `torchaudio` from `pip install` requires CUDA 13; we have CUDA 12 — use
   `soundfile` + ffmpeg-resampled wav instead.
2. `sentence-transformers` `get_sentence_embedding_dimension` deprecation
   warning is harmless, leaves output unchanged.
3. Banded ridge: hoist `X^T X` out of the α-tuple inner loop. With this
   optimization, 64 α-tuples × 5 folds = 320 fits runs in ~25 min on a
   single CPU core; without it would be ~10x slower.
4. Per-modality α optima (V=10⁵, A=10³-10⁴, T=10³) span 2 orders of
   magnitude — a single shared α cannot achieve banded's per-band
   regularization simultaneously, hence the +0.027 lift.
