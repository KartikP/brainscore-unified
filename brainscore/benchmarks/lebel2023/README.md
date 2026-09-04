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
- `tr_times` is 15 entries longer than `brain_data` for every story, split **10 from the
  head and 5 from the tail**, so `brain_data[i]` was acquired at `tr_times[i + 10]`. The
  reference pipeline hardcodes this as `downsampled[10:-5]`. **The count does not
  determine the direction** — both splits satisfy `len(tr_times) == n_tr + 15`, which is
  all the loader can assert, and getting it backwards places features five samples early.
  That inversion made the regression fit BOLD from words up to six seconds in the future
  and roughly halved every score here before it was caught; it is now pinned by
  `tests/test_lebel2023_data.py::TestTrimDirection`.

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

| Model | Layer | median r | mean r | frac r > 0 | p99 | best vertex |
|---|---|---|---|---|---|---|
| GPT-2 (124M) | 11 of 12 | **0.1013** | 0.1134 | 95.3% | 0.360 | 0.543 |

Most of cortex is barely predicted and a minority is predicted well, which is the
distribution a whole-brain benchmark exists to show.

### How this number moved

| pipeline | median r |
|---|---|
| per-TR context, last token, inverted trim | 0.0422 |
| + word-level features, Lanczos-resampled | 0.0511 |
| + corrected 10/5 trim alignment | **0.1013** |

The alignment fix is worth about 2x on its own. Two independent checks support it rather
than an argmax over candidate splits: the reference pipeline hardcodes `[10:-5]`, and the
FIR-delay curve only rises to a 9-12 s plateau — reproducing LITcoder's Figure 3B — once
the alignment is right. At the inverted trim that curve fell monotonically, which was
previously and wrongly explained as context and delays being substitutes.

### Within-TR pooling

Scored on 2000 vertices with GPT-2, comparable to each other but not to the table above,
and measured before the alignment fix:

| pooling | median r |
|---|---|
| per-TR context, last token | 0.0436 |
| word-level, last | 0.0445 |
| word-level, sum | 0.0478 |
| word-level, average | 0.0490 |
| **word-level, Lanczos** | **0.0525** |

The ordering `last < sum ~ average < Lanczos` reproduces the reference pipeline's.

### Comparison to LITcoder

LITcoder reports ~0.21 for GPT-2 as the **mean within a LanA language mask** (their top-10%
fsaverage5 mask), against our **median over all cortex** — different quantities. On our
corrected map the best any 10% mask could achieve is 0.287, so their figure is now
comfortably reachable and no unexplained residual remains. Before the alignment fix that
ceiling was 0.145, i.e. below their number, which is what flagged a real deficit rather
than a masking difference.

> [!warning] Qwen3.6-27B figures withdrawn
> Every Qwen number previously reported here was measured at the inverted alignment and on
> the per-TR path, so all of them are wrong. They are removed rather than rescaled; the
> model needs re-scoring. Its features were extracted out of band because `transformers<5`
> blocked registration, a constraint since lifted.

## Nulls

Run these before trusting any new result on this benchmark.

| Null | median r |
|---|---|
| GPT-2 | +0.1001 |
| timing-shuffled | +0.0032 |
| constant features | undefined — no varying prediction |

(2000-vertex subset.) Re-run after the alignment fix, since a change that doubles a score
is exactly when the floor needs rechecking. The shuffled null sits at zero and 95.1% of
vertices are positive, so the increase is signal rather than an artifact.

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
- **No anatomy attached.** The pickle carries no vertex-to-region mapping, so vertices are
  identified by index only. The space itself is settled: the LITcoder paper states its
  whole-surface LeBel analyses are fsaverage5, ~22k vertices, subject UTS03 — this data.
  Adding a region coord needs an atlas projected to that surface, not a resolution of any
  ambiguity.
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
