# LeBel2023 — whole-cortex encoding

Predict all 20484 measured cortical vertices of one participant (UTS03) listening to 25
spoken stories. No anatomical mask is applied: every vertex is a target, including the
majority that are not language-responsive.

Source: [LeBel et al. 2023, *Scientific Data*](https://doi.org/10.1038/s41597-023-02437-z),
supplied as a per-subject pickle from the Huth-lab `encoding` package (as redistributed by
LITCoder).

## Registered identifiers

| Identifier | Features | Targets | Use |
|---|---|---|---|
| `LeBel2023-UTS03-encoding` | per word, Lanczos-resampled | all 20484 | the benchmark |
| `LeBel2023-UTS03-encoding-smoke` | per word, Lanczos-resampled | random 2000 | pipeline checks only |
| `LeBel2023-UTS03-encoding-contextwindow` | one context window per TR, last token | all 20484 | retained for comparison |

The smoke variant's number is not comparable to a full run. Full runs take ~70 s
and peak near 8 GB.

## Data

The pickle is not fetched automatically. Point `BRAINSCORE_LEBEL_PICKLE` at it, or place it
at `~/Downloads/assembly_lebel_uts03.pkl`. The first load converts it to a
`NeuroidAssembly` and caches the result under `~/.brainscore-umi-cache/lebel2023`
(override with `BRAINSCORE_LEBEL_CACHE`); later loads take under a second.

Two properties of the source drive `data.py`:

- It references `encoding.*` classes that are not installed, so it is read with an
  unpickler that substitutes permissive stand-ins.
- `tr_times` is 15 entries longer than `brain_data` for every story — the Huth-lab trim of
  5 TRs from the start and 10 from the end. So `brain_data[i]` was acquired at
  `tr_times[i + 5]`. This is asserted per story at build time.

BOLD arrives as raw uncentered intensity (mean ≈ 27000) and is standardised per vertex per
story during scoring.

## Protocol

One stimulus row per spoken word, each carrying the running 32-word context ending at
that word, so the model's representation of a row is its representation of that word in
context. About **5.9 words fall inside each 2 s sample**; their features are resampled
onto the fMRI grid with a windowed-sinc (Lanczos) filter, low-passed at the sampling
rate. Collapsing each TR to a single value instead — which the `-contextwindow` variant
does — discards the other words and costs about 20% of the score.

Model features are then copied at delays of 1–4 TRs and
concatenated, letting each vertex learn its own hemodynamic lag rather than assuming a
fixed HRF. Cross-validation holds out whole stories, because adjacent TRs within a story
are correlated enough that a random split leaks. The ridge penalty is selected per fold on
*held-out stories*, not a random row subset — a random inner split picks a penalty about an
order of magnitude too small.

Reported value: median held-out Pearson r across vertices. **Raw, not ceiling-normalised.**
Each story is heard once, so no noise ceiling is estimable. Compare against the nulls
below, not against ceiling-normalised benchmarks such as MajajHong or Pereira.

## Results

All 20484 vertices, held-out median Pearson r.

| Model | Layer | Features | median r | mean r | frac r > 0 | p99 | best vertex |
|---|---|---|---|---|---|---|---|
| GPT-2 (124M) | 11 of 12 | **per word, Lanczos** | **0.0511** | 0.0546 | 89.7% | 0.181 | 0.276 |
| GPT-2 (124M) | 11 of 12 | per TR, last token | 0.0422 | 0.0464 | 89.7% | 0.162 | 0.263 |
| Qwen3.6-27B | 52 of 64 (best) | per TR, last token | 0.0628 | 0.0678 | 92.7% | 0.210 | 0.311 |
| Qwen3.6-27B | 64 of 64 (last) | per TR, last token | 0.0455 | 0.0522 | 87.4% | 0.199 | 0.306 |

The Qwen rows predate the default change and were measured on the per-TR path; they have
not been re-run word-level, which would be expected to raise them similarly.

### Within-TR pooling

Scored on 2000 vertices with GPT-2, so comparable to each other but not to the table
above:

| pooling | median r |
|---|---|
| per-TR context, last token (the old default) | 0.0436 |
| word-level, last | 0.0445 |
| word-level, sum | 0.0478 |
| word-level, average | 0.0490 |
| **word-level, Lanczos** | **0.0525** |

The ordering `last < sum ~ average < Lanczos` reproduces the reference pipeline's, and
Lanczos beats last-token by 18% on the same word-level features.

Most of cortex is barely predicted and a minority is predicted well, which is the
distribution a whole-brain benchmark exists to show.

**Layer choice is worth as much as model scale here.** Qwen at its best layer beats GPT-2
by 49%, but Qwen at its *last* layer beats GPT-2 by only 8%. Reading a large model at its
final block — which is what a naive registration does — discards most of the advantage.

Layer sweep (2000-vertex subset, so not comparable to the table above):

| layer | 4 | 12 | 20 | 28 | 36 | 44 | 52 | 60 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| median r | 0.0225 | 0.0426 | 0.0568 | 0.0546 | 0.0567 | 0.0569 | **0.0615** | 0.0534 | 0.0489 |

Accuracy rises steeply through the first third of the network, plateaus across the middle,
and falls over the last quarter — the usual profile for language models against brain data.

Qwen3.6 features were extracted in an isolated environment, because `qwen3_5` requires a
newer transformers than this repo pins, and fed to the benchmark through the
`_model_features` seam. Reproducing that arm therefore needs an environment with
transformers >= 5 for the extraction step; the scoring half runs under the pinned
environment unchanged.

## Nulls

Run these before trusting any new result on this benchmark.

| Null | median r |
|---|---|
| GPT-2 | +0.0436 |
| Qwen3.6-27B layer 52 | +0.0615 |
| timing-shuffled, GPT-2 | −0.0055 |
| timing-shuffled, Qwen3.6-27B layer 52 | +0.0019 |
| constant features | undefined — no varying prediction |

(2000-vertex subset, so directly comparable to each other.) Both models' timing-shuffled
nulls sit at zero, so neither score comes from story identity or feature scale.

The constant null caught a real design error. An earlier version zero-padded the opening
TRs of each story where the delayed copies had no history. That padding was identical
across stories and aligned with the large BOLD response to a story starting, so a model
carrying no stimulus information scored 0.054 against GPT-2's 0.063 — the benchmark was
mostly measuring story onsets. Those TRs are now excluded rather than padded
(`_delay_within_stories` returns a `has_full_history` mask), which is regression-tested in
`tests/test_lebel2023_benchmark.py::TestDelays::test_constant_features_carry_no_signal_after_masking`.

## Known limits

- **One participant.** Nothing here separates this brain from brains in general.
- **No noise ceiling.** Needs the repeated stories from LeBel's design to become
  normalisable.
- **No anatomy.** The pickle carries no vertex-to-region mapping, so vertices are
  identified by index only. The documented LITCoder pipeline (`to_mni_lebel.py`) produces
  MNI152 volumetric output, while the vertex count here is exactly 2 × 10242 — the
  fsaverage5 surface. That discrepancy is unresolved, so no hemisphere or region coord is
  attached. Confirm the space with the data provider before adding an atlas.
- **Per-target ridge penalties do not help here.** Fitting one penalty per vertex, as
  reference pipelines do by default, scored 31% *below* a single shared penalty, and
  nesting the selection over four inner folds barely recovered it. Broken down by
  signal strength it only breaks even on the best-predicted 5% and loses elsewhere:
  selecting a penalty per target needs per-target signal-to-noise high enough to select
  on, which whole-cortex data at this SNR does not have. Available as
  `per_voxel_alpha=True`; off by default.
- **Registered models are read at their final block.** The Qwen sweep shows this costs
  roughly a quarter of the achievable score. `gpt2`'s `region_layer_map` still points at
  `h.11`; re-mapping it would improve this benchmark but would move its scores on every
  other benchmark, so it is left alone pending a deliberate layer-mapping pass.
- **Qwen3.6-27B is not registered as a model plugin.** Its features were extracted out of
  band because the repo pins `transformers<5.0`. Registering it needs that pin resolved.
