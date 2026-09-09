# Reproducing the published dyslexia scale curve

The curve shown on the Brain-Score UMI site — a word-selective lesion dropping
lexical-decision accuracy past the 0.65 dyslexia threshold while an equally
sized random lesion does not — was produced by `scale_curve.py`, not by the
registered `Yeatman2021-induced_dyslexia` benchmark. This directory exists so
that claim can be re-run by someone who does not have our laptops.

## Command

Needs ~65 GB of VRAM for the 32B model; it was re-run on 4x A10G (g5.12xlarge),
loading across all four cards with no CPU offload. A single 48 GB card offloads
to host memory and thrashes to the point of being unusable.

```bash
RESULTCACHING_DISABLE=1 python scale_curve.py \
  --model_id Qwen/Qwen2.5-VL-32B-Instruct \
  --n_test 40 --n_localizer 60 --mask_sizes 0.0689,0.15,0.25 --seeds 2 \
  --out rerun.json
```

`--n_test 40` and the mask sizes are recovered from the published artifact
rather than from a saved invocation: the script defaults to `--n_test 50` and
`--mask_sizes 0.01,0.0689,0.15`, but the published pseudoword accuracy of 0.075
is only reachable as 3/40, and the saved keys are 0.0689/0.15/0.25.
**`--n_localizer` was never recorded**; 60 is the script default and is the one
free parameter below.

## What reproduced, 2026-09-08

Published 2026-06-03 (`published_32b_2026-06-03.json`) against the rerun
(`rerun_32b_2026-09-08.json`), both Qwen2.5-VL-32B, two seeds:

| mask | published lesion | rerun lesion | published random | rerun random |
|---|---|---|---|---|
| baseline | 0.9750 | 0.9750 | 0.9750 | 0.9750 |
| 6.89% | 0.9625 | 0.9625 | 0.9563 | 0.9563 |
| 15% | 0.9250 | 0.9125 | 0.9375 | 0.9375 |
| 25% | 0.5375 | 0.5062 | 0.8875 | 0.8875 |

The random-control arm matches to four decimals at every mask. It does not
depend on the localizer, so it is the arm that should match exactly, and it
does. The lesion arm reproduces the effect and the threshold crossing but sits
0.031 lower at the 25% mask, which is what an unrecorded `--n_localizer` buys:
a different localizer draw selects a different unit set.

Read the headline as **0.51-0.54, two seeds**, not as 0.54 exactly.

## What the number is, and is not

At the 25% mask the model still reads *real* words perfectly (1.00) in the
published run; the collapse is entirely in rejecting made-up words (0.075). The
deficit is specific and localised, which is the claim, but it is the opposite
half of the task from the one impaired in human dyslexia.

The registered `Yeatman2021-induced_dyslexia-image` benchmark is a different,
smaller protocol — one visual block of `qwen2.5-vl-3b-vwfa`, 500 units — and
gives 0.90 baseline, 0.79 lesioned, 0.95 random control: a specific deficit in
the same direction that does not cross the threshold. Both are reported on the
site; neither is presented as the other.
