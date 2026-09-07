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
| Qwen3.6-27B | 40 of 64 | **0.1318** | 0.1432 | 96.5% | 0.414 | 0.592 |
| GPT-2 (124M) | 9 of 12 | 0.1095 | 0.1217 | 96.0% | 0.372 | 0.552 |

These are the scores at each model's registered layer, which is what `score()` returns.
Qwen leads by 20% at the median and separates further in the tail (p99 0.414 against
0.372), where a language-responsive vertex would sit.

Qwen figures published here before the alignment fix were measured on the inverted time
axis and the per-TR path; they were discarded rather than rescaled, and this row is a
fresh run.

Most of cortex is barely predicted and a minority is predicted well, which is the
distribution a whole-brain benchmark exists to show.

### Depth

Both models swept on all 20484 vertices, every layer recorded in one pass
(`start_recording([...])`) and scored from the shared recording.

| Qwen3.6-27B | 0 | 5 | 10 | 15 | 20 | 25 | **30** | 35 | 40 | 45 | 50 | 55 | 60 | 63 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| median r | .063 | .082 | .097 | .116 | .123 | .126 | **.132** | .129 | .132 | .129 | .131 | .126 | .126 | .117 |

| GPT-2 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | **7** | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| median r | .082 | .083 | .086 | .088 | .098 | .101 | .104 | **.112** | .108 | .110 | .106 | .101 |

Qwen was swept at stride 5 rather than 4 or 8: it places a full-attention block every 4th
position, so a stride sharing a factor with 4 samples one kind of block and reports that
subpopulation's curve as the model's.

Both rise to a plateau over the middle-to-late stack and fall over the last fifth. Read
three ways:

| comparison | Qwen | GPT-2 | Qwen lead |
|---|---|---|---|
| registered layer | 0.1318 (L40) | 0.1095 (h.9) | +20% |
| best of sweep | 0.1320 (L30) | 0.1124 (h.7) | +17% |
| **plateau mean** | **0.1306** | **0.1069** | **+22%** |

**Quote +22%.** Best-of-sweep flatters whichever model happens to spike: GPT-2's best sits
1.39 sd above its own plateau against Qwen's 1.09, so picking maxima rewards its noisier
curve. The three now agree to within a few points, which they did not before: GPT-2 used
to be registered at its *last* block, 0.011 past its peak, which inflated the
registered-layer row to +30%. It has since been re-mapped to `h.9` — chosen on
Pereira2018, not here, so this benchmark stays a held-out report of that choice.

Sweeping gained Qwen **+0.0003** over its inherited layer. The layer was carried over from
a sweep run at the inverted alignment, so it was a guess; the plateau is broad enough
(0.1306 ± 0.0013 across L30-L50) that the guess cost nothing. `layers.40` stays.

### How this number moved

| pipeline | median r |
|---|---|
| per-TR context, last token, inverted trim | 0.0422 |
| + word-level features, Lanczos-resampled | 0.0511 |
| + corrected 10/5 trim alignment | **0.1013** |

(GPT-2 at `h.11`, its registered layer at the time; it is `h.9` now, which is a separate
+0.008. Holding the layer fixed is what makes the rows comparable.)

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

### Language-network mask

`language_mask.py` builds the LanA language-network mask on this surface, so the
reference pipeline's statistic — a **mean within the mask** — can be reported next to this
benchmark's **median over all cortex**. The atlas is an external Fedorenko-lab resource
([osf.io/kzwbh](https://osf.io/kzwbh/)); download its *FS Atlas* archive and point
`BRAINSCORE_LANA_ATLAS` at the directory. Without it the mask tests skip and nothing else
changes.

The mask is applied to the existing whole-cortex fit rather than refitting on the masked
vertices. This benchmark selects one shared ridge penalty across its targets, so refitting
on a subset would move that penalty too, and the masked and unmasked numbers would stop
describing the same model.

| Model | cortex median | cortex mean | LanA median | **LanA mean** |
|---|---|---|---|---|
| Qwen3.6-27B (L40) | 0.1318 | 0.1432 | 0.2297 | **0.2345** |
| GPT-2 (h.9) | 0.1095 | 0.1217 | 0.1972 | **0.2035** |

The mask lifts both models by about 1.7x, which is the check that it is the right set of
vertices: an equal-sized random mask scores 0.1430 ± 0.0018 against Qwen's 0.2345, putting
LanA 50 sd above chance.

Two assumptions in building it are silent if wrong, so both are tested rather than
asserted. The atlas ships at fsaverage7 and this benchmark is fsaverage5, taken as the
leading 10242 vertices — valid because FreeSurfer's icosahedra are hierarchical, confirmed
by the fact that those vertices are spaced 2.056 +/- 0.114 deg apart, a regular ico5 mesh,
where a random subset of the same size gives 1.05 +/- 0.485. And the mask assumes both the
atlas and this data order the hemispheres left-first: swapping them costs 0.103 (56 sd),
so the ordering is confirmed rather than a homotopy coincidence. The mask is also 74%
left-hemisphere, as a left-lateralised language network should be.

### Comparison to LITcoder

With the mask in place the comparison is finally like-for-like. LITcoder reports ~0.21 for
GPT-2 as a mean within LanA; **we get 0.2035, within 3%.**

All four protocol rows that once separated the two numbers are now matched — mask,
statistic, within-TR pooling, and FIR delays. Extending the delays from 2-8 s to 2-12 s to
cover their 9-12 s plateau moved the masked mean by -0.0002 (whole-cortex median 0.1013 to
0.1038, both at `h.11`), so the delay row, which was the leading suspect, is not the
explanation.

The residual 8% is smaller than the one parameter still unknown. Their mask file is
bring-your-own, and the top-10% convention here is read off its *filename*, not a stated
threshold. The masked mean is strongly sensitive to that choice:

| top fraction | 2% | 5% | **10%** | 15% | 20% |
|---|---|---|---|---|---|
| GPT-2 LanA mean | 0.274 | 0.236 | **0.204** | 0.187 | 0.174 |

Their 0.21 sits between top-5% and top-10%. The remaining 3% is well inside that spread,
so no conclusion is drawn from it. Note that the layer move closed most of the earlier 8%
gap without being aimed at it: `h.9` was picked on Pereira2018 before this number was
recomputed.

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
- **Registered layers are chosen elsewhere, on purpose.** GPT-2 was registered at its
  last block and has been re-mapped to `h.9`, chosen by sweeping Pereira2018 rather than
  this benchmark — picking the layer that maximises a benchmark and then reporting that
  benchmark is best-of-sweep. `h.9` is not this benchmark's best block (`h.7` is, at
  0.1124), which is the sign the two are independent. Qwen's `layers.40` is confirmed
  on-plateau and needs no change.
- **Qwen3.6-27B needs four accelerators.** It is registered and scores through the normal
  `process()` path, but at bf16 it is 54 GB and was run on 4x A10G. `device_map='auto'`
  left to itself packs each card to the brim and then OOMs on activations, so the plugin
  reserves 22% of each device (`WEIGHT_FRACTION`).
