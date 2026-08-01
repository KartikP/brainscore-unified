# EC2 runbook — two tasks on one instance start

**Task A: real cortex figures for the site** (below, ~15 min, ~$0.40)
**Task B: streaming delivery against a real decoded video** (the rest of this file)

Both need the same instance and the same rebuilt env, so start it once.

---

## Task A — replace the illustrative brain pictures with measured ones

The site's input panel currently shows textbook anatomy: model-independent, and
labelled "illustrative" for good reason. Every one of those pictures can instead show
**where a model actually predicts the brain, per voxel** — which is what Brain-Score
measures and is already sitting in `score.attrs['per_voxel_r']`.

```bash
python unified/scripts/generate_cortex_assets.py --out /tmp/cortex_assets
```

Writes `cortex_real_<input>_<hemi>_<view>.png` plus a `manifest.json`. It deliberately
does **not** overwrite the illustrative assets — look at the output first, then flip the
`cortex` field in `data.js` and update the caption, because **the figure's meaning
changes**: from "roughly where this input drives activity" to "where this model predicts
this benchmark." The caption must change with it or the page will be lying in a new way.

Covered today: `video` (V-JEPA v1 on Lahner visual-ROI) and `video + audio`
(V-JEPA1+Wav2Vec2). Both are fsaverage5 vertex-wise via `vertex_surface_map`.

**Not covered, deliberately** — recorded in the script's `CANNOT_RENDER`:
- `image` — MajajHong is **macaque V4/IT array recordings**. There is no human cortical
  surface to paint. A macaque schematic or no figure are the honest options; a human
  brain here would be a fabrication.
- `a live loop` — scored on win rate; there is no brain prediction to map.

**Status after the 2026-08-01 run — 2 of 4 remaining inputs are BLOCKED, and one of
them is a real pre-existing bug:**

- `video`, `video + audio` — **DONE**, rendered and wired into the site.
- `video + audio + text` (and therefore `audio`, `hours of continuous viewing`, which
  share the Algonauts figure) — **BLOCKED by a pre-existing bug**, not by the renderer.
  The parcel path is wired and `per_parcel_r` is now exposed, but scoring CLIP on
  `Algonauts2025-friends-sub01` dies *after* ~7 min of feature extraction with
  `AssertionError: Length of new_levels (7) must be <= self.nlevels (6)` from
  `pandas/core/indexes/multi.py:2598`, raised inside `_extract_for_modality` →
  the wrapper's `@store_xarray` path. This is the **MultiIndex-collision class already
  documented in CLAUDE.md** (`_attach_stimulus_set_meta` reassigning `stimulus_id`
  onto an assembly whose presentation dim is already a MultiIndex; the known fix is
  `reset_index('presentation')` first). It has nothing to do with cortex figures — it
  means **CLIP cannot currently be scored on Algonauts at all**, which is worth
  confirming independently before anyone quotes an Algonauts CLIP number.
  Next session: reproduce with a 2-minute smoke (a handful of TRs, not 162,688) so the
  fix can be iterated cheaply, then re-run the renderer.
- `text` (Pereira) — **UNANSWERED**. The feasibility probe crashed on my own bug
  (`or` on an xarray → ambiguous truth value); fixed in the script but not yet re-run.
  The question it asks is still the right one: do Pereira neuroids carry vertex or MNI
  coordinates? If not, it keeps a schematic and that is a finding, not a gap.
- `image` — **impossible by construction** (macaque arrays; see `CANNOT_RENDER`).

**Sanity gate:** every rendered map must show signal concentrated in plausible cortex —
visual areas for the video benchmarks. A map that lights up uniformly, or lights up
frontal cortex for a video benchmark, means the voxel-to-vertex placement is wrong
(`voxel_mask_indices` mis-ordered), not that the model is remarkable. Do not ship a
figure that fails this gate.

---

# Task B — streaming delivery against a real decoded video

The streaming substrate (core `streaming_helpers.py`) is exercised today only by
synthetic subjects and by "frames" that are filename strings. This runbook points it at
a **real decoded video with real model weights**, which is the only thing that can
falsify the design.

Est. ~0.5 h on the existing g5.4xlarge, ~$0.80. Instance `i-0bdbdf83c4db9bdae`
(STOPPED, not terminated).

## What this is actually testing

Three claims, in order of how likely they are to break:

1. **A lazy decoder stays lazy through a real codec.** `WindowedStreamSession` pulls
   from an iterator and holds at most one window. With `cv2.VideoCapture` that means
   frames are decoded on demand and peak RSS is flat in clip length. A `list(frames)`
   anywhere upstream silently defeats this and the unit tests cannot see it, because
   they feed a generator of strings.
2. **Windowed streaming reproduces the batch score.** Same model, same clip, same
   voxels — delivery must not change the number. This is the claim the whole design
   rests on and it has never been checked against a real wrapper.
3. **The real-time policies mean something on a real model.** `realtime_factor` for
   V-JEPA on a 30 fps feed is a real quantity: below 1.0 the model could run live,
   above it the `drop` / `lag` distinction starts to matter.

## Prerequisites

- Env rebuilt per `ec2_cleanup_runbook.md`. **Watch the shadowing trap**: installing
  `language`/`unified` pulls the PyPI `brainscore-vision` wheel over the editable
  checkout. Verify `brainscore_vision.__file__` points into `~/brain-score-unified`.
- `opencv-python-headless` for decoding (`pip install opencv-python-headless`).
- A real clip. Lahner BoldMoments MP4s are already on the box under the BrainIO cache;
  any 3 s 30 fps clip works. Do **not** pre-extract frames to JPEG — that is the batch
  path and it defeats the entire test.

## Step 1 — lazy decode, memory bounded

```python
import cv2, os, psutil
from brainscore_core.streaming_helpers import WindowedStreamSession, _drive_neural_session_streaming

def decode(path):                     # a generator: one frame per next(), nothing cached
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()

proc = psutil.Process(os.getpid())
for clip in (SHORT_CLIP, LONG_CLIP):      # e.g. 3 s and 60 s
    before = proc.memory_info().rss
    session = WindowedStreamSession(decode(clip), fps=30, window_ms=1000, record='IT',
                                    window_to_stimuli=frames_to_clip_row)
    _drive_neural_session_streaming(model, session)
    print(clip, session.windows_emitted, session.max_frames_held,
          (proc.memory_info().rss - before) / 1e6, 'MB')
```

**Pass:** `max_frames_held == 30` for both clips, and the RSS delta does not scale with
clip length. **Fail:** either grows with duration → something upstream is materializing
the feed; find it before trusting any streaming claim.

`frames_to_clip_row` is the caller-supplied converter — for V-JEPA it packs one window
into a single clip row; the default one-row-per-frame converter is for
frame-aggregation models.

## Step 2 — streaming score == batch score

Score one Lahner clip both ways with the same model (V-JEPA v1 ViT-L, the calibrated
one):

- batch: the existing `Lahner2024-fMRI-naturalistic-visualROI` path
- streamed: the same stimuli delivered through `WindowedStreamSession`

**Pass:** identical to float tolerance. **Expect a 1-ULP-scale difference at most** —
and if there is one, run the control from Task 5 of the cleanup runbook before blaming
the streaming path: fresh extraction vs. cached activations produced exactly that
signature last time and it was not the code under test.

**Fail:** any difference beyond that means windowing is changing what the model sees —
most likely at window boundaries, where a native-temporal model's receptive field
straddles the cut. That is a real finding and it constrains the windowing policy: it
would mean `stride_ms == window_ms` (non-overlapping) is wrong for temporal models and
the stride has to overlap by at least the model's context.

## Step 3 — real-time factor on real weights

```python
from brainscore_core.streaming_helpers import RealTimeStreamSession
inner = WindowedStreamSession(decode(CLIP), fps=30, window_ms=1000, record='IT',
                              window_to_stimuli=frames_to_clip_row)
rt = RealTimeStreamSession(inner, policy='lag')      # real clock, no injection
_drive_neural_session_streaming(model, rt)
print(rt.report())
```

Record `realtime_factor` for V-JEPA v1 (native temporal) and CLIP (frame aggregation)
on an A10G. This is a genuine number worth publishing: it says whether a brain-aligned
video model can run at the rate the world produces frames.

Then re-run with `policy='drop'` and confirm `windows_dropped > 0` exactly when
`realtime_factor > 1.0`. If a model drops windows while its factor is below 1.0, the
pacing logic is wrong.

## Step 4 — record

- Results JSON to `unified/scripts/ec2_results_<date>/streaming_video.json`.
- Cost row in `scripts/cost_tracking/COST_LEDGER.md`.
- If step 2 diverges, do **not** put a streaming number on the site: the honest claim
  becomes "streaming delivery is supported; on native-temporal models it changes the
  score at window boundaries, so batch remains the scoring path."

## Teardown

`aws ec2 stop-instances --instance-ids i-0bdbdf83c4db9bdae`; confirm `stopped`, not
`stopping`. Never terminate — the EBS env is expensive to rebuild, and it was already
lost once.
