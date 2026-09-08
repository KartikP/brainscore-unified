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
scores or pretrained-model results were generated.

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
