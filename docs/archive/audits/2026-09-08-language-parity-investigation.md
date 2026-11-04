# Residual GPT-2 activation parity investigation

Decision: **(c), unresolved on the real checkpoint/data**. A specific cached versus
full-prefix numerical mechanism is demonstrated on a tiny CPU model. Its ability
to explain the reported trained-model maximum, and its score impact, are not yet
verified. Neither production extraction nor the harness tolerance was changed.

## 1. Mechanism and evidence

### Confirmed by reading the checkout and installed implementation

- `brainscore/validation/benchmark_parity.py:143` loads fresh local models in eval
  mode. The native condition uses `batch_size=1`, `max_length=1024`, and
  `transformer.h.11`. The legacy condition hooks the same block.
- `TextWrapper.get_activations` uses `padding=True`, meaning padding to the longest
  member of the current batch, not padding to `max_length`. With a single input,
  ordinary GPT-2 tokenization adds no batch padding. There is no comparison of a
  single-example run against an unequal-length padded batch in this harness.
- The default aggregation is `last_token`. Its mask sum is an integer operation
  used to choose an index. No activation mean, padding-zero multiplication, or
  floating-point pooling reduction is executed here.
- Both tokenization calls leave `add_special_tokens` at its default. Neither route
  explicitly adds or suppresses BOS/EOS. Both passage paths use the shared
  `brainscore_core.text.prepare_context`. Actual token prefix stability on the
  staged tokenizer and Pereira passages remains to be measured.
- The native hook and legacy `_tensor_to_numpy` both export FP32. Neither wrapper
  introduces an FP16/BF16 conversion in this path. FP32 exported arrays do not
  prove FP32 inference: a lower-precision model would also be exported as FP32.
  The local loader uses the default torch dtype when no dtype is supplied. The
  GPU environment's parameter/hook dtypes and TF32 setting have not been inspected.
- `HuggingfaceSubject.digest_text` feeds a whole initial sentence, then the newly
  added tokens for each subsequent sentence with `past_key_values`. It is
  sentence-chunk incremental computation, not necessarily token-at-a-time decoding.
  TextWrapper recomputes the entire running passage prefix each time.
- Installed Transformers 4.57.6 derives GPT-2 positions from cache length. For
  SDPA, an all-visible full prefix can omit the explicit mask and use
  `is_causal=True`; a multi-token cached chunk normally needs an explicit
  lower-right-aligned rectangular causal mask. Query/key matrix shapes and kernel
  inputs therefore differ even with equivalent causal visibility. Eager attention
  also changes matrix shapes when processing chunks.
- The hook is the residual output of the final GPT-2 block, **before** the separate
  `transformer.ln_f`. Its scale is not constrained to that of the normalized output.
- Native dispatch passes the text wrapper's assembly through in this single-region
  route; no extra arithmetic reduction is applied by `BrainScoreModel` dispatch.
- There is an unverified truncation confound: native sets `truncation_side='left'`.
  The harness supplies the legacy tokenizer, so the legacy constructor's
  `AutoTokenizer(..., truncation_side='left')` branch is bypassed. Legacy instead
  retains the staged tokenizer's setting and uses its `model_max_length`.
- If cache sliding actually occurs, retained cached states were computed with older
  context and absolute positions. That is not mathematically identical to a fresh
  left-cropped forward with positions reset to zero. The legacy comment asserting
  mathematical equivalence is valid only under additional conditions: no eviction,
  stable prefix tokenization, identical positions/masking, causal model in eval mode.

### Confirmed by running small CPU experiments

Command:

```sh
RESULTCACHING_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python \
  -m brainscore.validation.language_parity_diagnostics --tiny-cpu \
  --output /tmp/umi-parity-investigation/tiny-final
```

Random GPT-2: 4 blocks, width 64, 4 heads, 128 positions, seed 17, one CPU thread.
Synthetic tokenizer, cumulative lengths 13, 30, 41, 60. No pretrained weights,
downloads, real benchmark data, or real benchmark scoring. Torch 2.13.0,
Transformers 4.57.6, NumPy 1.26.4.

| Attention / dtype | Maximum last-block difference, cached vs full |
| --- | ---: |
| eager / FP32 | 1.4901161193847656e-08 |
| SDPA / FP32 | 1.862645149230957e-08 |
| eager / FP64 | 2.7755575615628914e-17 |
| SDPA / FP64 | 2.0816681711721685e-17 |

In all four experiments:

- Actual incremental token IDs concatenated to exactly the native full-prefix IDs.
- Actual embedding positions matched, all external mask entries were one, and
  internal causal visibility was checked edge by edge. No truncation or sliding.
- Direct full-prefix replay matched native hook tensors bit for bit; direct cached
  replay matched legacy hook tensors bit for bit. This isolates the demonstrated
  difference to the underlying model execution rather than assembly packaging.
- FP64 comparisons used hook tensors before either wrapper's FP32 export.
- FP32 eager's first differing sampled last-token hook was block 0 attention;
  FP32 SDPA's was block 1 attention. These are the first sampled hook differences,
  not a claim to have identified an individual CUDA instruction.
- SDPA used no explicit internal mask on native calls. Cached continuation masks
  were `[1, 1, 17, 30]`, `[1, 1, 11, 41]`, and `[1, 1, 19, 60]`.

Measured details are in `2026-09-08-language-parity-cpu.json` alongside this report.
The corresponding general limitations of shape-dependent floating-point execution
are documented by [PyTorch](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html).
The experiment, rather than that general statement alone, is the evidence here.

### Inference and outstanding real-data checks

Cached/full-prefix shape and mask differences are a concrete candidate mechanism
for the real failure. The experiment does **not** reproduce `2^-13` on trained
GPT-2, establish GPU kernel selection, rule out real tokenizer/position differences,
or certify either implementation beyond the context window. Those require the
command in section 3. Exact vision parity is consistent with the vision routes
sharing the same extractor and execution shapes; it does not settle the differently
shaped language computation.

## 2. Why exactly 2^-13 can repeat

For normal FP32 numbers in the binade `2^e <= |x| < 2^(e+1)`, the representable
spacing is `2^(e-23)`. Machine epsilon `2^-23` is the spacing at **one**, not an
absolute bound for arbitrary activations or a composed neural network.

I ran `np.spacing(np.float32(1024))` and `np.spacing(np.float32(1536))`; both return
exactly `0.0001220703125 = 2^-13`. A test comparing `1536.0` with its next FP32
neighbor produces exactly the reported magnitude with no half-precision operation.
At magnitude 512 the same delta is two FP32 spacings.

Thus one concrete explanation is that the maximum lies in a large residual-stream
coordinate: nearby cached/full results round to adjacent FP32 values around
magnitude 1024-2048. Both benchmarks use the same weights and same residual block,
so they can share that coordinate scale and the same discrete maximum step. They
are independent stimulus sets, not independent numerical formats or model scales.

**That explanation is not confirmed for the user's arrays.** A single FP16 spacing
in `[0.125, 0.25)` or BF16 spacing in `[0.015625, 0.03125)` also equals `2^-13`.
The magnitude alone cannot distinguish these possibilities. Nor is the rounded
printed string `0.00012207` by itself proof of the exact binary value.

The runner records full-precision and hexadecimal maxima, the reference/actual
values at up to eight maximal coordinates, local FP32 spacing, and the error in
those spacings. It also records the dtypes before export. A one-ULP interpretation
explains a particular observation; it is **not a bound** on all GPT-2 inputs and
cannot justify setting a general tolerance to that value.

## 3. Score impact and GPU command

**Unresolved.** The `0.0022` score gap was measured at native max_length 512; the
reported activation maximum was measured at 1024. They cannot be causally equated.
If all actual tokenized prefixes fit within 512 and native 512/1024 activations
match exactly, the max_length difference has no extraction effect in that run.
If some exceed 512, the earlier run used different context. The runner measures
both possibilities directly; neither has been assumed.

The harness already computes the native raw and normalized scores at lines 77-78,
before the assertion at line 91. The failure suppresses their reporting. This is
why a runner that preserves those values is needed, rather than a tolerance change.

Pereira uses unregularized `LinearRegression`, ten deterministic ShuffleSplit
splits (random_state 1), per-neuroid Pearson correlation, median aggregation over
neuroids, and aggregation across splits. Least squares depends on the pseudoinverse
of the centered activation matrix. An elementwise activation bound alone gives no
useful uniform prediction bound without singular-value/rank information. In
particular, small singular values or a numerical rank change can amplify a small
perturbation. This is a risk to measure, not a diagnosis of these data.

The normalized score divides raw score by the benchmark ceiling and clips to
`[0, 1]`. Away from clipping, `delta_normalized = delta_raw / ceiling`. Both scores
must therefore be checked; clipping can hide raw disagreement. No scientifically
validated score tolerance or variance estimate follows from the supplied maximum.
The retained `1e-6` score criterion is an existing regression requirement, not a new
claim about the scientific noise floor. Two repeats measure observed repeatability,
not an exhaustive estimate of run-to-run variance.

Run from the **GPU machine's unified checkout**, after copying the diagnostic script
there and setting `UMI_PARITY_GPT2` to the same staged checkpoint directory:

```sh
RUN_UMI_PARITY=1 RESULTCACHING_DISABLE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python \
  -m brainscore.validation.language_parity_diagnostics \
  --gpu --score --score-float64 --ablations --repeats 2 \
  --output /tmp/umi-language-parity
```

This command was **not run on real data here**. It evaluates only the two Pereira
benchmarks, using the harness's factory and actual benchmark implementations.
It runs legacy, adapter, native-1024 and native-512, each twice. Outputs:

- `/tmp/umi-language-parity/report.json`: every raw/normalized score and delta,
  repeats, ceiling, environment and precision flags, actual token IDs/positions,
  mask summaries, inference dtypes, untruncated lengths, and regression singular
  values/ranks. Scores are saved after each route even if activations fail parity.
- Per-route NPZ files: metric-input activations, stimulus IDs, all block last-token
  tensors, and final layer-normalization tensors.
- `float64_regression`: re-scores the same captured features with FP64 regression
  input to diagnose fitting sensitivity. This does not redefine the official score.
- Worst-prefix passage replays bypass both wrappers, using default execution,
  forced SDPA math with TF32 disabled, eager FP32 with TF32 disabled, and eager
  FP64. It checks default replay against the recorded route values and compares
  intermediate hooks. Semantic token/sliding mismatches cause an explicit skip.

An exit status of 1 means the **original** activation or score criteria failed;
the JSON/NPZ evidence is still saved. Native-512 is a diagnostic control, not a new
acceptance condition. Inspect any reported ablation error instead of treating it
as a numerical result.

Interpretation: token/position/mask mismatches require a semantic investigation
first. If actual wrapper outputs match direct replays and cached/full differences
shrink strongly under FP64 with identical inputs/visibility, that supports numerical
execution as the mechanism. The worst-coordinate values test the scale/ULP
hypothesis. Score differences and FP64-regression controls then assess sensitivity.
Even a harmless result on this dataset would not, by itself, derive a universal
activation tolerance for branch (b).

## 4. Decision

**(c): keep the failure visible pending measurement.** No native defect causing the
reported maximum has been demonstrated. Neither the maximum nor the tiny-model
experiment yields a principled general tolerance bound. A neural forward-error
bound would need relevant magnitudes, conditioning and propagation through the
weights/nonlinearities; a score bound additionally needs regression conditioning
and ceiling sensitivity. Padding-token count cannot supply such a bound here,
because the harness has no cross-example padding or masked-mean aggregation.

No tolerance was loosened: native activation and raw/normalized score checks remain
absolute `1e-6`, relative zero; adapter checks remain exact. There is no production
fix diff because a branch-(a) diagnosis would currently be speculative.

## 5. Changes and verification

Only new investigation files were added:

- `brainscore/validation/language_parity_diagnostics.py`
- `tests/test_language_parity_diagnostics.py` (four offline tests)
- This report and `2026-09-08-language-parity-cpu.json`

The original 608-test fast tier passed before changes. After adding the diagnostic
tests, the required command passed with **612 passed, 65 deselected**:

```sh
RESULTCACHING_DISABLE=1 /opt/anaconda3/envs/brainscore-unified-fresh/bin/python \
  -m pytest tests -m "unit or integration" -q -p no:cacheprovider
```

Tests cover exact spacing evidence, tiny direct replay equivalence, score persistence
on activation failure, and the observation/ablation pipeline with synthetic models
and a fake scoring metric. They do not validate the real GPU/data path.

Existing modified/untracked files in all four repositories were left unchanged.
No dependency pins or legacy plugins were modified. No commits or pushes were made.
