# Extraction provenance, route validation, and induced dyslexia

2026-09-08. Changes remain uncommitted in all four repositories on
`unified-model-interface-v2`. No dependency pins or legacy registrations were
removed or changed. No real benchmark scores, model weights, or large assemblies
were computed/downloaded. One requested core-suite command attempted an external
AWS setup and failed; that is reported separately below.

## Implemented, with local evidence

### Activation cache provenance

`core/brainscore_core/extraction_cache.py:129` builds a versioned SHA-256 key from
canonical configuration. It preserves mapping order independence, distinguishes
sequence types, describes callable code/defaults/closures, and avoids object
addresses. Model structure, tensor layout/dtype/device, precision and version
metadata, and installed forward/pre-forward hooks participate
(`core/brainscore_core/extraction_cache.py:145`). Custom providers can expose
`cache_config()`; unsupported/cyclic state warns and bypasses storage instead of
falling back to an unsafe name-only key (`:136`).

The shared mechanism is used by:

| Extractor | Configuration declaration / lookup |
| --- | --- |
| TextWrapper | `unified/brainscore/model_helpers/text_wrapper.py:247` |
| VLMVisionWrapper | `unified/brainscore/model_helpers/vlm_vision_wrapper.py:146` |
| VideoWrapper | `unified/brainscore/model_helpers/video_wrapper.py:296` |
| AudioWrapper | `unified/brainscore/model_helpers/audio_wrapper.py:244` |
| Legacy ActivationsExtractorHelper | `vision/brainscore_vision/model_helpers/activations/core.py:145` |

Text fingerprints include the actual contextualized texts. The manual
`-context-v1-<hash>` suffix and Pereira's additional context suffix are retired;
context expansion itself is unchanged. Video's constructor no longer appends a
chunking suffix; its complete temporal configuration is fingerprinted, including
non-integer window/stride values. Audio had no analogous suffix.

The legacy vision extractor had the same exposure. Its key now includes
preprocessing, batch/stimulus hooks, batch size and microsaccade settings; channel
separation remains intact. LayerPCA declares its input configuration and component
count, fingerprints its fitted-PCA disk lookup, and refreshes its in-memory basis
when configuration changes (`vision/brainscore_vision/model_helpers/activations/pca.py:21`).
Lazy basis initialization and hook handle numbers do not change the key. No
ImageNet data or PCA fitting was run.

Unversioned entries deliberately miss once. They are neither read as activations
nor deleted. The storage logger reports an exact unversioned entry when found;
other misses explain that unversioned entries cannot be reused, covering the
retired suffix layouts too (`core/brainscore_core/extraction_cache.py:200`). Example
observed in the temporary-cache test: `Skipping unversioned activation cache;
recomputing with configuration fingerprint: ...`.

Evidence is executable:

- `unified/tests/test_extraction_fingerprints.py:68`: real temporary disk cache,
  extraction-call counts, hits for identical configurations and new instances,
  misses after a change, reuse after restoring configuration, and merging only
  missing layers, across all four native wrappers.
- `:98`: every constructor extraction option must appear in the declaration;
  each option and model dtype changes the fingerprint.
- `:114`: a seeded unversioned entry is ignored, logged, and remains byte-for-byte
  unchanged on disk.
- `:135`: tiny CPU GPT-2 cached output equals fresh direct extraction exactly;
  tokenizer runtime state does not create a false miss; max length, truncation
  direction and passage grouping change the key and (where asserted) activations.
- `:166`, `:207`, `:240`: legacy helper reuse, preprocessing/dtype changes,
  PCA initialization/configuration, and perturbation hook/reset behavior.
- `core/tests/test_extraction_cache.py:25`: independent Python processes with
  different hash seeds produce identical fingerprints.

Tests kept `RESULTCACHING_DISABLE=1`. Only the sentinel `cache-fixture` identifier
was enabled via a scoped monkeypatch, with storage redirected to pytest's fresh
temporary directory, to test actual result_caching reads/writes. Production cache
entries were never enabled for these tests.

### Both model routes for all six benchmarks

`unified/brainscore/validation/benchmark_parity.py:23` lists all six cases;
`:37` runs separately computed legacy/legacy, adapter/unified, and native/unified
conditions. It rejects an adapter substituted for the native model and compares
stimulus order, metric-input activations, target order, raw score and normalized
score. Adapter comparisons are exact; native comparisons have explicit absolute
tolerances (default 1e-6, zero relative tolerance), not a claim of bit-for-bit
identity. Recording layers are fixed to avoid confounding layer search.

All six registered factories were exercised locally with tiny CPU models,
synthetic stimuli/assemblies and a test metric. No biological predictivity metric
was run in this test. A negative control removes native passage context and
requires the harness to fail (`unified/tests/test_benchmark_parity.py:126`).

| Unified variant | Offline registered-route test | Real data/weights this session |
| --- | --- | --- |
| MajajHong2015.V4-pls-unified | Passed | Not run; staged |
| MajajHong2015.IT-pls-unified | Passed | Not run; staged |
| MajajHong2015public.V4-pls-unified | Passed | Not run; staged |
| MajajHong2015public.IT-pls-unified | Passed | Not run; staged |
| Pereira2018.243sentences-linear-unified | Passed | Not rerun; staged |
| Pereira2018.384sentences-linear-unified | Passed | Not run; staged |

The earlier Pereira-243 measurement supplied in the task is historical user
context, not a measurement made here. The harness contains no expected published
scores. Existing claims in the two benchmark modules now distinguish adapter
parity from native validation.

### Registered induced-dyslexia path

The new registration is `qwen2.5-vl-3b-vwfa`
(`unified/brainscore/models/qwen25_vl_3b/__init__.py:12`). Its declared VWFA search
population is visual `blocks.28` (`model.py:42`), the existing IT readout site.
This is an explicit operational choice, not an independently validated anatomical
mapping or a recovered detail of the missing historical harness. The benchmark
localizes real-versus-pseudo selective units within that population.

The callback is constructed against `qwen_model.model.visual` (`model.py:153`),
which is both the extraction root for `blocks.28` and the visual module used by
full-model generation. The explicit region/modality map routes VWFA to vision.
Registration owns this root and PyTorch-specific capability; the benchmark owns
the localizer, 500-unit lesion, seed-0 matched random control, and score.

`unified/tests/test_registered_dyslexia.py:98` calls the actual registered model
and benchmark loaders, replacing only checkpoint/processor/data-loading
boundaries and shrinking the stimulus split. The real generation wrapper,
localization, ablation, controls and reset execute against tiny CPU modules and
synthetic images. It verifies that lesions reach generation and recorded units,
cleanup restores output, and the original registration remains unchanged (`:126`).
The toy population is smaller than 500, so this test does not validate scientific
lesion selectivity or predict the real score.

## Commands actually run and observed results

All pytest runs used `/opt/anaconda3/envs/brainscore-unified-fresh/bin/python`,
`RESULTCACHING_DISABLE=1`, a new `RESULTCACHING_HOME` from `mktemp -d`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `-p no:cacheprovider`.
The commands below are relative to the indicated repository.

| Repository / pytest arguments | Observed output |
| --- | --- |
| unified: `tests -m 'unit or integration' -q` | **608 passed, 65 deselected, 58 warnings in 32.41s** |
| core: `tests -q --ignore=tests/test_plugin_management` | **642 passed, 3 skipped, 10 errors in 20.82s**; AWS Secrets Manager connection failures in submission endpoint setup |
| core: `tests -q --ignore=tests/test_plugin_management --ignore=tests/test_submission/test_endpoints.py` | **639 passed, 3 skipped, 30 warnings in 18.61s** (includes four new serialization tests; also excludes passing external-service-file tests) |
| core: `tests/test_extraction_cache.py -q` after final serializer refinements | **4 passed in 3.41s** |
| unified: `tests/test_extraction_fingerprints.py -q` after final model-metadata refinement | **15 passed in 6.49s** |
| unified: `tests/test_benchmark_parity.py -q` | **8 passed, 27 warnings in 18.68s** |
| unified: `tests/test_registered_dyslexia.py -q` | **2 passed in 5.92s** |
| vision: `tests/test_activation_cache_key.py tests/test_unified_adapter.py -q` | **22 passed in 0.17s** |
| language: `tests/test_unified_adapter.py -q` | **29 passed in 1.25s** |
| unified: `tests/test_benchmark_parity_slow.py -m slow -q -rs` without opt-in | **6 skipped in 1.03s**, before any benchmark/model loading |

The initial unified run was interrupted during unrestricted MagicMock traversal:
1 failed, 148 passed, 59 deselected; exit 130. Serialization now has a depth bound
and uses declared attributes rather than dynamically manufactured mock methods.
Old cache-propagation test stubs were updated for the new argument. An intermediate
run had 604 passing tests and a registry-count assertion failure (28 versus the
new 29); the assertion was updated, and the full run above passed. The initial
new cache-test collection also revealed an incorrect test-module import, fixed
before the passing runs.

The anticipated missing `pytest_check` / `requests_mock` collection failures did
not occur in the observed core run. The external-service failure is from
`core/tests/test_submission/test_endpoints.py:26`, not cache code. No dependencies
were installed to make tests pass. Existing xarray/sklearn warnings and the
synthetic empty-mask warnings were left alone.

`python -m brainscore.data` was run read-only: **2/3 user-supplied assets present;
algonauts2025-root missing**. This is not an inventory of downloaded MajajHong or
Pereira artifacts. `git diff --check` passed in all four repositories.

## Code-based conclusions and work not executed

Checkpoint values are not hashed or copied. An identifier/backbone revision must
continue to identify immutable weights; media paths must identify immutable
files. Custom providers must include hidden activation-affecting state in
`cache_config()`. Opaque state bypasses caching. Implementation/version/device
changes can conservatively cause misses even when their outputs happen to agree.
This mechanism does not certify equivalence of arbitrary Python programs or
repair higher-level historical score caches. Scores should be re-evaluated with
caching disabled when establishing scientific correctness.

Pereira's context expansion, tokenizer left truncation, aggregation and extraction
math were left intact. Cached/cold equality and cross-route synthetic tests support
that preservation; no claim is made that real Pereira scores were remeasured.
Legacy vision channel separation and existing xarray layer merging were sound
and retained. Induced dyslexia's existing whole-run cache disabling and try/finally
hook cleanup were retained for third-party candidates as well as built-in models.
No legacy model/benchmark plugins were removed or deprecated.

On a prepared machine, stage local torchvision ResNet18 weights and a local
Hugging Face GPT-2 model/tokenizer directory, plus all six benchmark stimuli,
assemblies and ceilings (private MajajHong access is required for its private
variants). Then run from unified, **not on this laptop**:

```sh
RESULTCACHING_DISABLE=1 RESULTCACHING_HOME="$(mktemp -d /tmp/umi-parity.XXXXXX)" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RUN_UMI_PARITY=1 \
UMI_PARITY_RESNET18=/absolute/path/resnet18-state-dict.pth \
UMI_PARITY_GPT2=/absolute/path/gpt2 \
/opt/anaconda3/envs/brainscore-unified-fresh/bin/python -m pytest \
  tests/test_benchmark_parity_slow.py -m slow -q -s -p no:cacheprovider
```

The checkpoint loaders are local-only; benchmark loaders can still request
missing data, so stage those first. Each successful case prints measured JSON
and writes it to its pytest temporary directory. A mismatch fails rather than
being reclassified as a skip. Other matched models can be supplied through the
factory argument to `validate_case`; Qwen's known sliding-cache residual should
be investigated explicitly rather than concealed with an arbitrary tolerance.

For the registered lesion path, with real Qwen weights and ROAR data staged and
sufficient hardware, run with `RESULTCACHING_DISABLE=1`:

```python
import brainscore
model = brainscore.load_model('qwen2.5-vl-3b-vwfa')
score = brainscore.load_benchmark('Yeatman2021-induced_dyslexia-image')(model)
print(float(score), score.attrs)
```

That real run was **not executed**. Confirming the public 0.54/0.89 figures also
requires establishing that the original search layer(s), checkpoint/processor,
localizer, lesion size, random seed and stimulus split match this declared
protocol. The new single-layer registration may produce different values; this
change does not retroactively establish provenance for the published numbers.
