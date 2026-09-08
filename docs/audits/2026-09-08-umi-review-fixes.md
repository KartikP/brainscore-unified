# Unified model interface review: issue and fix audit

Date: 2026-09-08. Scope: the six review findings, implemented together across
`core`, `unified`, `vision`, and `language` on `unified-model-interface-v2`.
Changes are uncommitted; no commits or pushes were made.

## Baseline and evidence

| Repository | HEAD before these changes |
| --- | --- |
| core | `5284a10dbed30b71a1b9e794f649c63d80c47665` |
| unified | `5521cc991447f0d97972997641c201dd77cd9a37` |
| vision | `8f9f6908c347e69bf08a7d90cda7fa1bdcf123ce` |
| language | `a6ebdf68bbc5e35c1249d164dde51c5171d902ae` |

Paths below are workspace-relative; line numbers refer to this working-tree
revision. **Confirmed** means exercised with synthetic inputs or tiny randomly
initialized CPU models, not inferred from a plausible code path. **Code-derived**
marks a protocol concern whose scientific impact was not measured. No real-data
scores or pretrained-model results were generated during the original local
review. Sections 7 and 8 record subsequent GPU results supplied by the user,
separately from locally executed checks. Section 8 is the finalized parity policy.

## 1. Incompatible feature bases in Algonauts held-out prediction

**Issue, confirmed:** training and held-out extraction separately fitted SVDs,
then applied the same ridge coefficients to those different feature bases.
With cap 1, train features `[[4,0],[-4,0],[2,0],[-2,0]]`, targets equal to the
first column, and held-out features `[[1,10],[-1,10],[1,-10],[-1,-10]]`, separate
bases produced approximately `[10,10,-10,-10]`, not `[1,-1,1,-1]`.

**Fix:** a fitted `FeatureProjection` is owned by the prediction operation.
Training fits it once; held-out extraction only transforms. Width and available
neuroid identities/order must match. Training subject, split, stimulus window,
and HRF delay are checked. Features below the compression cap remain unchanged.

- Implementation: `unified/brainscore/benchmarks/algonauts2025/projection.py:6`;
  `unified/brainscore/benchmarks/algonauts2025/benchmark.py:345` and `:658`.
- Regression: `unified/tests/test_umi_review_regressions.py:50` calls the real
  `generate_predictions` path with synthetic extraction/alignment, verifies the
  corrected values and saved array. The test at `:85` covers identity and basis
  validation.

**Code-derived protocol concern:** existing CV fits SVD on all input frames
before folds. It remains explicitly transductive, with score attribute
`feature_projection_protocol='transductive_svd_all_input_frames'` at
`benchmark.py:608` and documentation in the benchmark README. This does not
implement fold-local, inductive CV or establish that past scores were inflated.
Historical CV numbers were deliberately not silently redefined.

## 2. Lesions addressed reduced-array positions instead of original units

**Issue, confirmed:** a `CompositeSelector` exposing original units `(4,1)`
could select reduced position 0 and lesion original unit 0 rather than 4.
Random controls likewise sampled `range(2)` instead of the recorded population.
This changes the intervention, not just its annotation.

**Fix:** composite recording attaches original per-layer `unit_index` before
subsetting. Functional selection maps rankings through those indices. Resolved
selections retain `unit_population`; `RandomSelection(population=...)` samples
that population. Induced Dyslexia uses this for both single- and multi-layer
localizers. Invalid/ambiguous subset addresses are rejected. Full-layer selection
keeps its existing positional convention.

- Implementation: `core/brainscore_core/recording.py:235`;
  `core/brainscore_core/selection.py:133`, `:160`, and `:213`;
  `unified/brainscore/benchmarks/induced_dyslexia/benchmark.py:106`.
- Regression: `unified/tests/test_umi_review_regressions.py:203` verifies one-
  and two-layer selection, population-matched controls, actual PyTorch hooks
  zeroing unit 4 while preserving unit 0, and hook removal on reset.
  `core/tests/test_unit_selection.py:275` and `:280` exercise rejection paths.
- Compatibility: the new random population argument is optional. Induced
  Dyslexia is now benchmark version 2; version 1 and 2 scores are not equivalent
  for subset-localized models.

## 3. Text context depended silently on which wrapper was used

**Issue, confirmed:** native TextWrapper treated table rows independently,
whereas LanguageModelAdapter passed the whole list to legacy `digest_text`,
whose later parts see earlier parts. Identical tiny GPT-2 weights therefore
produced different second-row activations for `['the cat', 'the dog']`.

**Fix:** table rows are independent unless `context_id` explicitly groups
ordered passage parts. Shared core helpers define grouping and legacy-compatible
English context joining. Native TextWrapper expands the declared prefixes;
the permanent adapter invokes legacy digestion once per group and restores
original row order, including interleaved groups. Raw legacy `digest_text`
retains its passage semantics; native `BrainScoreModel.digest_text` declares
that passage context too. Pereira's unified benchmark now supplies the context
column explicitly and is version 2.

- Implementation: `core/brainscore_core/text.py:20` and `:39`;
  `unified/brainscore/model_helpers/text_wrapper.py:213`;
  `language/brainscore_language/compat/unified_adapter.py:81`;
  `language/brainscore_language/benchmarks/pereira2018/unified.py:33` and `:60`.
- Regression: `unified/tests/test_umi_review_regressions.py:131` compares real
  native/legacy tiny-CPU-model activations for independent rows, one passage,
  and interleaved passages. Tests at `:148`, `:296`, and `:324` verify raw-list
  compatibility, Pereira's input construction, and cache-key isolation.
- Cache behavior: contextual native activations include a `context-v1` suffix
  and a hash of expanded text. Pereira passage identifiers are also versioned.
  No existing caches were deleted, and tests did not depend on them.
- Explicit limits: context belongs to one `process` call, not hidden state
  spanning streaming windows. `per_token` grouping is rejected because adding
  prefix tokens would otherwise invalidate supplied word timestamps; those
  callers must supply complete text explicitly. Long-context pretrained-model
  equivalence was not tested. Mean-token pooling retains its native aggregation
  semantics; the cross-wrapper numerical parity tests use last-token recording.

## 4. Shared layer names collapsed separate modality towers

**Issue, confirmed:** vision IT and text language-system regions both mapped
to `block` could drop the text output and label vision units as both regions.
Routing by layer string alone lost the tower identity. The CLIP registration
uses shared relative layer names, so this is not merely a synthetic topology.

**Fix:** filter requested regions by declared modality before selecting layers;
tag output regions within that modality. Composite extraction follows the same
rule. Single-target recording chooses its declared tower; cross-modality targets
require `multi_modality=True` and all requested input modalities. CLIP declares
its region-to-modality map explicitly.

- Implementation: `core/brainscore_core/recording.py:119`;
  `core/brainscore_core/capabilities/neural.py:39`;
  `core/brainscore_core/dispatch.py:198`;
  `unified/brainscore/models/clip_vit_b_32/model.py:86`.
- Regression: `unified/tests/test_umi_review_regressions.py:175` uses separate
  towers returning 1 and 9 under the same layer name, for both full-layer and
  composite mappings. It checks values, region labels, single-target routing,
  and missing-input/dispatch errors. No CLIP weights were loaded.

## 5. Reset and exception cleanup left evaluation state behind

**Issue, confirmed:** composite recording survived reset and could break a
subsequent behavioral readout. Legacy language measurement fields survived
adapter reset. PolicyWrapper history survived model reset. Induced Dyslexia
could leave installed lesions active if reading or later hook installation
raised. These make results depend on previous evaluations or failures.

**Fix:** recorder reset clears composite mode and time bins; temporary localizer
probes restore the complete previous configuration, including idle/raw-layer
states, in `finally`. Behavioral readout temporarily disables and then restores
composite mode. BrainScoreModel resets each distinct capability/extractor
provider exposing `reset`, including bound methods; it attempts all providers
and reports failures. The language adapter clears actual supported legacy
measurement fields, and the vision adapter forwards to reset methods added to
its standard commitment/neural/behavioral helpers. Both lesion conditions use
`try/finally`, enclosing installation as well as reading.

- Implementation: `core/brainscore_core/recording.py:101`;
  `core/brainscore_core/perturbation.py:54`;
  `core/brainscore_core/behavioral.py:144`;
  `core/brainscore_core/brainscore_model.py:463`;
  `language/brainscore_language/compat/unified_adapter.py:159`;
  `vision/brainscore_vision/compat/unified_adapter.py:77` and the reset methods
  in `vision/brainscore_vision/model_helpers/brain_transformation/`;
  `unified/brainscore/benchmarks/induced_dyslexia/benchmark.py:194`.
- Regressions: `unified/tests/test_umi_review_regressions.py:163`, `:235`,
  `:251`, `:260`, and `:346`; `unified/tests/test_induced_dyslexia.py:74`
  and `:88`; `core/tests/test_unit_selection.py:252`.
- Limit: custom providers/legacy plugins with additional private state must
  implement their own reset method; generic reset cannot discover arbitrary
  plugin-specific state. Weights and region mappings are retained.

## 6. Language adaptation discarded presentation identity

**Issue, confirmed:** legacy output omitted input `stimulus_id` and metadata;
LeBel's feature aligner then raised `KeyError`. Pereira's benchmark-specific ID
repair concealed the adapter contract failure.

**Fix:** the table adapter checks presentation counts, validates and reorders
available `part_number`, checks available legacy text coordinates, restores all
input presentation columns, and preserves output attributes. Conflicting
non-presentation coordinates are rejected. Pereira no longer patches IDs after
extraction. Raw non-table digestion remains a compatibility path.

- Implementation: `language/brainscore_language/compat/unified_adapter.py:81`;
  `language/brainscore_language/benchmarks/pereira2018/unified.py:57`.
- Regression: `unified/tests/test_umi_review_regressions.py:131` checks metadata
  and the real LeBel alignment helper with reordered IDs. Tests at
  `language/tests/test_unified_adapter.py:319` and `:336` cover invalid counts,
  duplicate part numbers, wrong text, reordered output, and attribute retention.
- Limit: where a plugin supplies neither part numbers nor text coordinates,
  correspondence still relies on its documented one-row-per-input ordering.

## Validation and preserved boundaries

All test commands used the designated interpreter, disabled result caching,
and enabled Hugging Face offline mode. The final runs passed:

| Suite | Result |
| --- | --- |
| unified unit/integration tier | 583 passed, 59 deselected |
| affected core interfaces, selection, dispatch, and streaming | 206 passed |
| language adapter and context preprocessing | 30 passed |
| vision adapter | 20 passed |

Reproduce from the corresponding repository directory:

```sh
export RESULTCACHING_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# unified/
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest tests -m "unit or integration" -q -p no:cacheprovider

# core/
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest tests/test_model_interface.py tests/test_unit_selection.py tests/test_capabilities.py tests/test_model_interface_public_api.py tests/test_channel_compatibility.py tests/test_streaming_helpers.py tests/test_dispatch_cache.py tests/test_streaming_behavior.py tests/test_streaming_perception.py tests/test_streaming_realtime.py tests/test_streaming_windowed.py -q -p no:cacheprovider

# language/
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest tests/test_unified_adapter.py tests/test_model_helpers/test_preprocessing.py -q -p no:cacheprovider

# vision/
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest tests/test_unified_adapter.py -q -p no:cacheprovider
```

`python -m brainscore.doctor`, using the same interpreter, reported dependencies
within bounds: xarray 2022.3.0, NumPy 1.26.4, sklearn 1.5.1, Transformers 4.57.6.
The absent user-supplied Algonauts root is expected and irrelevant to these
synthetic tests. `git diff --check` passed in all four repositories.

The unified tests emitted the existing synthetic empty-mask warnings plus
xarray/pandas and sklearn deprecation warnings exercised by the new tests;
these were not suppressed. No dependency pins were changed. No benchmark or
model plugin was removed, and both permanent adapters remain supported.
Existing full-layer selections, uncompressed feature values, legacy raw text
digestion, and standard streaming/dispatch behavior remain covered by passing
tests. Real-data benchmark scoring, model downloads, GPU runs, and numerical
revalidation of historical published scores remain outside this verification.

Land these changes across all four repositories together: the adapter and
wrapper changes depend on the new core text and selection helpers. The updated
public contract is in [the UMI API reference](../umi_api_reference.md); the
implementation decisions are in [the design record](../plans/2026-09-08-umi-review-fixes-design.md).

## 7. GPT-2 parity follow-up: scalar-only stage (superseded by section 8)

**Verdict (c): a deterministic route-dependent score discrepancy with an
unconfirmed cause.** FP32 execution differences between cached chunks and full
prefixes remain a supported hypothesis, but the supplied scalar scores do not
establish inherent numerical path dependence or scientific indifference. The
score impact is now measured; the cause remains open. No tolerance was loosened.

### GPU measurements supplied by the user

The user ran the existing diagnostic with result caching disabled and GPT-2
recording at `transformer.h.11`. All values below are supplied measurements, not
locally rerun scores. Raw NPZ arrays and the GPU `report.json` have not been
inspected in this session. Repeats are listed explicitly rather than inferred
from rounded summaries.

Pereira2018.243sentences-linear-unified:

| Route / repeat | Ceiling-normalized score | Raw score |
| --- | ---: | ---: |
| legacy_0 | 0.8711741877922449 | 0.30821208826023067 |
| adapter_0 | 0.8711741877922449 | 0.30821208826023067 |
| native_1024_0 | 0.8723817692825759 | 0.3086393176456909 |
| native_512_0 | 0.8723817692825759 | 0.3086393176456909 |
| legacy_1 | 0.8711741877922449 | 0.30821208826023067 |
| adapter_1 | 0.8711741877922449 | 0.30821208826023067 |
| native_1024_1 | 0.8723817692825759 | 0.3086393176456909 |
| native_512_1 | 0.8723817692825759 | 0.3086393176456909 |

Pereira2018.384sentences-linear-unified:

| Route / repeat | Ceiling-normalized score | Raw score |
| --- | ---: | ---: |
| legacy_0 | 0.8280753625165158 | 0.30095362369641643 |
| adapter_0 | 0.8280753625165158 | 0.30095362369641643 |
| native_1024_0 | 0.8277164923953866 | 0.30082319684364545 |
| native_512_0 | 0.8277164923953866 | 0.30082319684364545 |
| legacy_1 | 0.8280753625165158 | 0.30095362369641643 |
| adapter_1 | 0.8280753625165158 | 0.30095362369641643 |
| native_1024_1 | 0.8277164923953866 | 0.30082319684364545 |
| native_512_1 | 0.8277164923953866 | 0.30082319684364545 |

Subtracting the printed decimal values gives:

| Benchmark | Native minus legacy, normalized | Native minus legacy, raw |
| --- | ---: | ---: |
| Pereira-243 | +0.0012075814903310 | +0.00042722938546023 |
| Pereira-384 | -0.0003588701211292 | -0.00013042685277098 |

These subtractions were calculated locally with `decimal.Decimal`. Adapter
equals legacy exactly; native 512 equals native 1024 exactly; repeat 1 equals
repeat 0 for each route, as reported by the user. This establishes reproducible
differences in the published score, not just in internal activations. It rules
out the 512-versus-1024 setting as a cause in this run. The earlier, separately
reported 0.0022 score gap is not substituted for these new measurements.

The user also confirmed exact three-route vision parity, including activation
maxima of 0.0:

| Unified vision benchmark | Legacy = adapter = native score | Max activation delta |
| --- | ---: | ---: |
| MajajHong2015.V4-pls-unified | 0.512458 | 0.0 |
| MajajHong2015.IT-pls-unified | 0.389788 | 0.0 |
| MajajHong2015public.V4-pls-unified | 0.505271 | 0.0 |
| MajajHong2015public.IT-pls-unified | 0.387708 | 0.0 |

### What the evidence establishes and does not establish

The current code uses native batch size 1, dynamic padding and last-token
selection. Masked-mean pooling and unequal-length batch padding are not the
operations being compared. Legacy feeds new sentence tokens through a KV cache;
native recomputes full passage prefixes. The stored tiny-CPU experiment shows
cached/full last-block differences of approximately 1.49e-8 (eager FP32) and
1.86e-8 (SDPA FP32), falling to approximately 2.78e-17 and 2.08e-17 in FP64.
Direct replays matched their respective wrappers exactly. Those results support
the numerical-execution hypothesis but are not measurements of this trained
checkpoint's GPU residual.

The proposed implication from the new scores needs qualification:

- Bit-identical repeats rule out observed run-to-run variability in those
  repeats. They do not rule out a deterministic defect.
- Native 512/1024 equality rules out an effect of that length setting in these
  runs. It does not compare legacy versus native token IDs, positions or masks.
  A masking or tokenization error present at both lengths would survive this
  control.
- Opposite signs of score changes on two datasets show that the direction is
  dataset-dependent. A deterministic implementation error can improve one
  correlation score and worsen another. A sign flip is not a diagnostic test
  for floating-point accumulation order.
- Exact adapter parity establishes the compatibility contract for the observed
  runs. The native computation differs from that contract's underlying legacy
  computation, so neither the adapter result nor exact vision parity settles
  the native discrepancy's cause.

No new worst-coordinate magnitude, maximum activation hexadecimal value,
GPU parameter/hook dtype, FP32 ULP count, ablation result, or FP64-regression
score is claimed here. Those values cannot be recovered from scalar scores.
In particular, the earlier hypothesis that 2^-13 is a single FP32 step at a
large residual coordinate remains unverified for the real arrays.

Pereira's current code uses unregularized `LinearRegression`, fixed-seed
cross-validation, correlation and aggregation. Numerical differences in a
poorly conditioned regression can affect scores deterministically; the
regression singular values and FP64 controls are needed to evaluate that
explanation. The code divides by the ceiling and clips to [0, 1], so both raw
and normalized comparisons remain necessary. The printed scores are inside
the clipping limits. No scientific-indifference threshold was supplied or
established by these two deterministic differences.

### Acceptance policy and validator change

Adapter activation, raw-score and normalized-score checks retain `atol=0`,
`rtol=0` on all six variants, independent of native tolerance arguments.
Native activation and score checks retain their existing absolute `1e-6`
defaults and `rtol=0`. The strict native checks therefore continue to reject
these user-reported score gaps.

`brainscore/validation/benchmark_parity.py` now collects native activation
and score assertion failures and reports the normalized and raw score failures
before activation failures. Neither check is replaced or skipped. Adapter
failures still raise immediately. The assertion comment records the measured
deltas and explains why they have not been reclassified as an error bound or
a scientific-indifference threshold.

Choosing a limit merely large enough to admit +0.00120758 would fit the current
observation, not derive a numerical bound. Calling it a scientific threshold
would require an explicit application-level criterion or supporting evidence.
Moreover, any positive absolute tolerance accepts sufficiently small
single-sign drift. It cannot guarantee detection of a sign trend or growth
that remains within the bound; a separate, justified directional or historical
test would be required. No unsupported guarantee of that kind is made.

### Evidence to retrieve without rerunning benchmarks

The user, who has GPU-box access, can run this standard-library-only command
there and paste the output. It reads the existing JSON, not weights or neural
data, and needs no new inference or scoring:

```sh
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path('/tmp/umi-language-parity/report.json').read_text())
summary = {'environment': report.get('environment'), 'cases': {}}
for name, case in report['cases'].items():
    audit = case.get('token_audit', [])
    result = {
        'token_audit_rows': len(audit),
        'failed_token_audit_rows': [row for row in audit if any(v is False for v in row.values())],
        'activation_comparison': case.get('comparisons', {}).get('legacy_0_vs_native_1024_0'),
        'routes': {}, 'ablations': {},
    }
    for route in ('legacy_0', 'native_1024_0'):
        run = case['runs'][route]
        result['routes'][route] = {key: run.get(key) for key in (
            'parameter_dtypes', 'attention_implementation', 'prefixes_over_512',
            'prefixes_over_1024', 'truncation_side', 'hook_vs_metric_input')}
        result['routes'][route]['hook_dtypes'] = sorted({
            dtype for call in run.get('calls', []) for dtype in call.get('hook_dtypes', {}).values()})
        fit64 = run.get('float64_regression', {})
        result['routes'][route]['float64_regression'] = {key: fit64.get(key) for key in ('score', 'raw')}
    for mode, ablation in case.get('ablations', {}).items():
        if not isinstance(ablation, dict):
            result['ablations'][mode] = ablation
            continue
        result['ablations'][mode] = {key: ablation.get(key) for key in (
            'error', 'baseline_cached_reproduced', 'baseline_full_reproduced')}
        result['ablations'][mode]['last_block'] = ablation.get('layers', {}).get('transformer.h.11')
    summary['cases'][name] = result
print(json.dumps(summary, indent=2))
PY
```

Missing fields, empty audits and recorded ablation errors are missing evidence,
not successful checks. The decisive next comparison is whether tokens, positions
and causal visibility match; whether direct default replays reproduce each
route; and whether the inference discrepancy shrinks under FP64. Separately,
FP64 **regression** controls test the sensitivity of fitting already-extracted
features. That is a different intervention from FP64 **model inference**.

### Local verification of this follow-up

Before this change the current fast tier passed with 612 tests (608 originally
plus four diagnostic tests from the prior session). The numeric-policy tests
added here cover adapter exactness for all six variants and all three checks,
reporting both supplied native score gaps alongside synthetic activation
failures, native score drift with identical activations, and activation drift
with identical scores. The artificial activation values in those tests are
explicitly not presented as measurements of the GPU arrays.

Only the parity validator, its existing test file, and this tracked audit were
edited for this follow-up. No new report files, inference changes, dependency
changes, commits or pushes. Final local verification:

| Check | Result |
| --- | --- |
| Affected parity and diagnostic tests | 35 passed |
| Full fast/offline tier | 635 passed, 65 deselected |
| `git diff --check` | Passed |

The final count is 27 above the stated 608 baseline: four tests were already
present from the previous diagnostic session and 23 parameterized test cases
were added in this follow-up. Commands run from `unified`, with caching disabled
and Hugging Face offline mode enabled:

```sh
RESULTCACHING_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest \
  tests/test_benchmark_parity.py tests/test_language_parity_diagnostics.py \
  -q -p no:cacheprovider

RESULTCACHING_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest tests \
  -m "unit or integration" -q -p no:cacheprovider
```

No real benchmark scoring, GPU execution, model downloads or network data access
was performed locally. Hash comparison against the start-of-follow-up snapshot
confirmed that all pre-existing files outside the three scoped edits were
unchanged in all four repositories.

## 8. Final GPT-2/Pereira parity policy after GPU token and precision diagnostics

**Verdict (b) for the measured setup:** the cached-chunk versus full-prefix
activation discrepancy is consistent with FP32 precision and accumulation-order
sensitivity. The evidence now includes matching token/position/mask audits and
FP64 convergence on the real checkpoint, beyond the scalar sign pattern used
in section 7. Changing FP32 attention implementations changes the error but
does not eliminate it; this is not exclusively a default-kernel problem.
Kernel choice can still affect the size of the numerical difference.

The user supplied these results from the re-run saved at
`~/umi-language-parity`; the previous `/tmp` directory was lost on reboot.
No real GPU arrays or benchmark scores were recomputed locally during this
finalization. Bounds below are explicit regression-policy budgets calibrated
against these observations, not a theorem covering arbitrary GPT-2 inputs or
a measurement of scientific uncertainty.

### Reported environment and controls

Torch 2.6.0+cu124; Transformers 4.57.6; NumPy 1.26.4; sklearn 1.5.2; CUDA 12.4;
NVIDIA A10G; matmul precision `highest`; matmul TF32 disabled; cuDNN TF32 enabled;
default dtype FP32; autocast disabled.

- Pereira-243: 243 audited rows, zero false rows.
- Pereira-384: 384 audited rows, zero false rows.
- Every row passes prefix-token equality, no cache sliding, both external
  masks all ones, both causal-mask checks, native zero-based positions and
  matching legacy positions. This rules out the tested token/position/mask
  discrepancies for these passages.
- Native 512 versus 1024: activation max delta 0.0 and raw/normalized score
  deltas 0.0 on both benchmarks. Every route's repeatability and score-repeat
  ranges are 0.0. Adapter versus legacy activation max delta is 0.0 for both.
- Hooked-layer maximum reference magnitudes: 351.91 and 335.72, respectively.
- The score observations and exact vision results remain those in section 7:
  normalized native-minus-legacy deltas +0.0012075814903310 and
  -0.0003588701211292, summarized by the user as +0.0012 and -0.0004.

Native 512/1024 equality is a separate control, not an additional adapter or
vision variant. Adapter exactness is supported by the two language comparisons
and the four vision comparisons, and remains required on all six variants.

### Supplied precision ablations and checked arithmetic

These are worst **absolute-error** coordinates among 73 sampled layers per
variant. They are not measurements of the maximum ULP ratio over all elements.
Reference values below are the user's rounded values. "FP32 ULPs" in the FP64
rows means division by FP32 spacing, not a count of FP64 representable steps.

| Case | Variant | Worst layer | Max absolute delta | Reference (rounded) | Delta / local FP32 spacing |
| --- | --- | --- | ---: | ---: | ---: |
| 243 | default | transformer.h.11 | 0.0001220703125 | 273.237 | 4.0 |
| 243 | ieee_sdpa_math | transformer.h.11.attn | 7.927417755126953e-05 | 1.619 | 665.0 |
| 243 | ieee_eager | transformer.h.11 | 8.392333984375e-05 | 99.0666 | 11.0 |
| 243 | float64_eager | transformer.h.11.attn | 2.5579538487363607e-13 | -135.177 | approximately 1.68e-08 |
| 384 | default | transformer.h.11 | 0.0001220703125 | 278.512 | 4.0 |
| 384 | ieee_sdpa_math | transformer.h.11 | 9.1552734375e-05 | 125.896 | 12.0 |
| 384 | ieee_eager | transformer.h.11.attn | 6.103515625e-05 | -134.127 | 4.0 |
| 384 | float64_eager | transformer.h.11 | 3.126388037344441e-13 | 150.916 | approximately 2.05e-08 |

The supplied hexadecimal values for the 243 variants are, respectively,
`0x1.0p-13`, `0x1.4c8p-14`, `0x1.6p-14`, and `0x1.2p-42`. The 384 default
also reports `0x1.0p-13`. Both FP64 variants pass all original absolute 1e-6
layer checks; the FP32 variants do not.

Locally verified with Python/NumPy: normal FP32 spacing in the binade
`[2^8, 2^9)` is `2^(8-23) = 2^-15 = 0.000030517578125`. Therefore the two
default results satisfy exactly `2^-13 = 4 * 2^-15 = 0.0001220703125`.
The 665, 11, 12 and 4 ratios in the other FP32 rows also check arithmetically.
Both default references lie in the same binade, explaining the repeated
discrete step without invoking half precision. The earlier single-ULP-at-1024
hypothesis is superseded: these reported coordinates are four ULPs near 273-278.
Their scale is consistent with large residual coordinates, but the summary
does not identify a shared neuron or establish a particular "rogue dimension."

Two qualifications to the requested analysis are explicit:

- The sampled local-ULP maximum is **at least 665**, not 12. Moreover, the
  maximum-absolute-error coordinate need not maximize the ULP ratio.
- 1.619 is not numerically near zero. Its small scale relative to the residual
  stream makes local-output-relative error a poor sole measure of accumulated
  or cancellation error. The magnitude floor is an explicit policy adjustment,
  not a proof that any large local-ULP error is harmless.

No FP64-regression result was supplied. The diagnostic stores that optional
field under `runs[route]['float64_regression']`; its absence from the list of
top-level case keys does not establish that it is missing inside the runs.
FP64 model-inference convergence does not measure FP64 regression sensitivity.

### Final assertions

**Adapter:** preserve `atol=0`, `rtol=0` for activations, normalized scores and
raw scores on every variant. Native tolerance arguments cannot weaken it.
Native vision retains its existing absolute 1e-6 activation and score bounds;
the following policy is restricted to the two fixed Pereira cases.

**Native language activations:** for every reference element `x`, define

```text
magnitude = max(abs(float32(x)), 64)
spacing = 2^(floor(log2(magnitude)) - 23)
allowed_error = 16 * spacing
require abs(float64(native) - float64(reference)) <= allowed_error
```

Sixteen is a four-fraction-bit error budget, retaining roughly 19 fraction bits
relative to the effective reference scale, and the smallest power of two above
the reported 12 ULPs. The floor 64 is the smallest binary scale for which this
16-ULP budget includes the reported 665-local-ULP example: its effective ratio
is exactly 10.390625. This selection is empirical calibration under the user's
requested policy, not a derived forward-error bound. No elements are dropped.
Sixteen effective ULPs covers all supplied FP32 example coordinates.

The resulting limit is 0.0001220703125 at magnitudes below 64, and
0.00048828125 in `[256, 512)`. Thus this is a magnitude-floored ULP policy,
with a fixed absolute floor at small coordinates. It is not uniformly tighter
than a flat 0.00012207 tolerance; its purpose is scale-aware activation checking
paired with independent score and historical-bias checks. Finiteness and shape
are checked, and each element's own limit is enforced, not the array's largest
allowed error. Tests accept 16 effective ULPs and reject 17.

**Native scores:** require absolute normalized-score delta at most **0.002**.
This is an engineering agreement budget of **0.2 percentage points on the
unit score range**, not an inferred statistical noise floor. It admits both
signed observations and is less than twice the largest observed normalized
delta; 0.01 would be over eight times that observation. Raw-score delta must
be at most `0.002 * legacy_ceiling`, and ceilings must match. Raw and published
scores therefore share the same normalized units, preventing clipping from
hiding raw-score drift.

**Joint score bias and growth:** both benchmarks must be evaluated together.
For both published deltas and ceiling-scaled raw deltas, reject two nonzero
departures with the same sign. Also enforce frozen per-benchmark magnitude
envelopes **0.00125 for 243** and **0.00045 for 384**. These are the upper
rounding endpoints for the user's four-decimal magnitudes 0.0012/0.0004, allowing
half one final reported digit (0.00005), rather than an unrestricted multiplier
of the error. They conservatively trigger review if this fixture's discrepancy
grows beyond the reported precision, even while staying under 0.002.

Synthetic checks demonstrate that same-sign deltas of +0.0001/+0.0001 or
-0.0001/-0.0001 fail. A monotone outward sequence from (+0.0012, -0.0004),
through (+0.00124, -0.00044), to (+0.0013, -0.0005) fails at the final point
although every component remains under 0.002. Exact parity remains valid.
Growth within the rounding envelopes, or a defect that leaves checked scores
and activations within every budget, is not guaranteed detectable. No finite
tolerance can provide that universal guarantee. These sign/history checks are
conservative regression sentinels for this fixed fixture, not causal diagnoses.

### Wiring, retained diagnostics and verification

`brainscore/validation/benchmark_parity.py` owns the shared ULP, score and paired
drift policies. The full measured rationale is in comments beside those helpers.
Native failures still report normalized/raw score discrepancies along with
activation discrepancies. The slow suite now evaluates both Pereira benchmarks
inside one test and invokes the paired guard; the four vision tests remain
separate. A fast test calls that slow entry point with synthetic reports to
verify the joint guard is actually connected, without loading data or models.

`language_parity_diagnostics.py` uses the same native acceptance policy and
paired score guard. It retains old absolute-1e-6 statistics separately under
`strict_absolute_check`; direct ablation comparisons also retain their strict
absolute diagnostics to make precision convergence visible. Those diagnostic
statistics do not override the new suite acceptance policy.

No new report file was created. No production model/wrapper changes, GPU runs,
benchmark rescoring, downloads, commits or pushes were performed locally.
Final local verification, with result caching disabled, the pytest cache provider
disabled, and Hugging Face offline mode enabled:

| Check | Result |
| --- | --- |
| Affected parity and diagnostic tests | 70 passed, 34 warnings |
| Full fast/offline tier | 670 passed, 64 deselected, 65 warnings |
| `git diff --check` | Passed |

The fast count is **62 above the original 608 baseline**, and **35 above the
635 tests present before this finalization**. The slow collection has one fewer
test because the two Pereira cases now execute together; all six benchmark
variants are still covered. Commands run from `unified`:

```sh
RESULTCACHING_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest \
  tests/test_benchmark_parity.py tests/test_language_parity_diagnostics.py \
  -q -p no:cacheprovider --tb=short

RESULTCACHING_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest tests \
  -m "unit or integration" -q -p no:cacheprovider --tb=short
```

The real-data slow suite was not rerun locally. Its final paired Pereira entry
point, for the GPU box with `UMI_PARITY_GPT2` already set to the staged checkpoint,
is:

```sh
RUN_UMI_PARITY=1 RESULTCACHING_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m pytest \
  tests/test_benchmark_parity_slow.py::test_pereira_243_and_384_score_parity \
  -m slow -q -p no:cacheprovider -rs -s
```
