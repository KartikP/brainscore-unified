# Running TR-resolved multimodal scoring on EC2

The TR-resolved Lahner multimodal benchmark must NOT run locally. The
11.4 GB assembly download alone is enough to choke a Mac, and the
banded-ridge α-grid search across 4042-voxel visual-ROI × 127k TR
observations × 5 modes will hammer the system. Stage it on EC2.

This file documents the exact commands. Each section is a single
copy-paste-and-run unit.

## 1. Spin up the instance (or reuse the existing one)

We already have a g5.4xlarge instance from prior work
(`i-0bdbdf83c4db9bdae`, `blip2-validation`). It has:
- A10G GPU, 24 GB VRAM
- 64 GB RAM
- Persistent EBS volume at `~/brain-score-unified/`
- Conda env `brainscore-unified` with the right pins

```bash
# From your local machine:
aws ec2 start-instances --instance-ids i-0bdbdf83c4db9bdae
# Wait ~30s for the public IP to land, then SSH in
# (replace IP with whatever AWS hands you):
PUB_IP=$(aws ec2 describe-instances --instance-ids i-0bdbdf83c4db9bdae \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
ssh -i ~/.ssh/aws-blip2.pem ubuntu@$PUB_IP
```

If the instance is gone, follow `unified/scripts/ec2_setup.md` (or the
`tribe_smoke_test.sh` setup) to rebuild — Python 3.11, miniconda,
`numpy<2`, `xarray==2022.3.0`, `scikit-learn>=1.5,<1.6`,
`transformers<5.0` (or 4.57.x), `nilearn`, `scipy`, `torch>=2.6`.

## 2. Sync the codebase

The unified branch on `KartikP/brainscore-unified` is the canonical
remote. Pull the latest commit:

```bash
# On EC2:
cd ~/brain-score-unified/unified
git fetch origin unified-model-interface
git checkout unified-model-interface
git pull
# Same for core/ and (if needed) vision/, language/
cd ~/brain-score-unified/core && git pull
```

Verify the new files landed:

```bash
ls unified/brainscore/benchmarks/lahner2024/benchmark_timeresolved_multimodal.py
ls unified/brainscore/benchmarks/lahner2024/MULTIMODAL_VALIDATION_PROTOCOL.md
ls unified/brainscore/models/multimodal_av_vjepa_wav2vec2/backbone_inits.py
```

## 3. Pre-extract audio tracks (one-time)

Audio comes from the same MP4s as the video stim_set; we demux to WAV
once and cache. The output dir should live on the EBS volume so it
survives stop/start.

```bash
cd ~/brain-score-unified/unified
conda activate brainscore-unified
python -m brainscore.benchmarks.lahner2024.prepare_audio_tracks \
    --audio-dir ~/brain-score-unified/data/lahner2024_audio_16k \
    --target-rate 16000
```

Expected: ~3 min, ~96 MB output, manifest reports
`{extracted: 1007, silent_placeholder: 19}`.

## 4. Pre-download the TR-resolved assembly (one-time)

The 11.4 GB assembly comes from S3 the first time it's referenced.
Pull it once into the brainio cache:

```bash
python -c "
from brainscore.benchmarks.lahner2024.benchmark_timeresolved import \
    load_timeresolved_assembly
a = load_timeresolved_assembly()
print('assembly shape:', dict(a.sizes))
"
```

Expected: ~25-40 min download depending on EC2 region's distance from
the brainscore-storage S3 bucket. Subsequent runs are cache hits.

## 5. Run the validation protocol

```bash
# Validation harness — runs all 5 modes on auditory-ROI for the signal
# model + null variants, plus α-ablation on video_only.
python -u -m brainscore.benchmarks.lahner2024.score_timeresolved_multimodal_smoke \
    > ~/tr_smoke.log 2>&1 &
tail -F ~/tr_smoke.log
```

If the smoke result on auditory-ROI passes the validation protocol
(see `MULTIMODAL_VALIDATION_PROTOCOL.md`), proceed to the full
visual-ROI run:

```bash
# Visual-ROI variant — slower (~6× more voxels)
python -u -c "
import json, time
import brainscore
from brainscore.benchmarks.lahner2024.benchmark_timeresolved_multimodal \
    import Lahner2024BOLDMoments_timeresolved_multimodal_visualROI

t0 = time.time()
out = {}
for combo in (
    'vjepa1-wav2vec2',
    'random-vjepa1-wav2vec2',
    'vjepa1-random-wav2vec2',
    'random-vjepa1-random-wav2vec2',
):
    m = brainscore.load_model(combo)
    print(f'[{time.time()-t0:6.1f}s] {combo}', flush=True)
    out[combo] = {}
    for mode in ('concat', 'banded'):
        b = Lahner2024BOLDMoments_timeresolved_multimodal_visualROI(mode=mode)
        s = b(m)
        out[combo][mode] = float(s.attrs['raw'])
        print(f'  {mode} = {out[combo][mode]:.4f}', flush=True)
with open('/tmp/tr_visual_validation.json', 'w') as f:
    json.dump(out, f, indent=2)
" > ~/tr_visual_validation.log 2>&1 &
```

## 6. Pull results back to local

```bash
# From local machine:
scp -i ~/.ssh/aws-blip2.pem ubuntu@$PUB_IP:/tmp/lahner_tr_multimodal_smoke.json /tmp/
scp -i ~/.ssh/aws-blip2.pem ubuntu@$PUB_IP:/tmp/tr_visual_validation.json /tmp/
```

Then update the research note + CLAUDE.md with the numbers from
`/tmp/lahner_tr_multimodal_smoke.json` and `/tmp/tr_visual_validation.json`.

## 7. Stop the instance when done

```bash
aws ec2 stop-instances --instance-ids i-0bdbdf83c4db9bdae
```

EBS volume preserves: cached assembly, audio sidecar, conda env,
codebase. Stopped instance costs ~$0.10/day for the EBS storage;
running costs ~$1.20/hour.

## Estimated session costs (g5.4xlarge, US-East)

| Phase | Time | Cost (running) |
|---|---|---|
| 1. Audio pre-extract | 3 min | $0.06 |
| 2. Assembly download | 30 min | $0.60 |
| 3. Smoke (5 modes auditory-ROI) | 15 min | $0.30 |
| 4. Visual-ROI 4 combos × 2 modes | 90 min | $1.80 |
| **Total first run** | **~2.5 hrs** | **~$3.00** |
| Subsequent runs (caches warm) | 30-60 min | $0.60-1.20 |

## Model combos to score

The full validation matrix has 11 A+V combos:

| Family | Signal-Signal | Signal-Null | Null-Signal | Null-Null |
|---|---|---|---|---|
| V-JEPA v1 + Wav2Vec2 | `vjepa1-wav2vec2` | `vjepa1-random-wav2vec2` | `random-vjepa1-wav2vec2` | `random-vjepa1-random-wav2vec2` |
| CLIP ViT-B/32 + Wav2Vec2 | `clip-wav2vec2` | `clip-random-wav2vec2` | `random-clip-wav2vec2` | `random-clip-random-wav2vec2` |
| BLIP-2 ViT-G + Wav2Vec2 | `blip2-wav2vec2` | (skipped — heavy) | (skipped — heavy) | (skipped — heavy) |
| Qwen2.5-VL-3B + Wav2Vec2 | `qwen2.5-vl-wav2vec2` | (skipped — heavy) | (skipped — heavy) | (skipped — heavy) |

Plus standalone `random-wav2vec2-base` for audio-only null reference.

V-JEPA and CLIP combos cover all four signal/null permutations (cheap
re-randomization). BLIP-2 and Qwen-VL ship signal-only because their
backbones are 2.7B and 3B params respectively — the V-JEPA/CLIP nulls
already cover the validation criteria.

## What this validates

1. **Pipeline runs** end-to-end on TR-resolved data with two modalities
2. **Null floor**: random+random combo at chance
3. **Single-null asymmetry**: random-video+wav2vec2 ≈ audio_only;
   vjepa+random-wav2vec2 ≈ video_only
4. **Mode-curve coherence**: per_modality < concat < banded (or banded
   matches the better single-tower α-tuned baseline)
5. **Region × modality specificity**: visual-ROI vs auditory-ROI
   asymmetries match GLM-beta direction (audio uninformative on
   visual-ROI; small banded lift on auditory-ROI)
6. **Reproducibility**: rerun on warm cache gives identical numbers

If all 6 pass, the multimodal pipeline is trusted and we can move to
naturalistic continuous-stimulus benchmarks (Sherlock / Forrest / NNDb)
with confidence that the encoding side won't be the failure mode.
