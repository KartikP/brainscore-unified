# Activation caching

A model name and input path are not enough to identify a measurement. The same model name can have new weights, and the same path can contain a new image.

The shared extraction cache checks:

- Model parameters and buffers, including their values, shapes, and dtypes.
- Wrapper configuration, preprocessing, and supported hook configuration.
- Text contents, or file bytes for image, audio, and video inputs.
- The stimulus identifier, and channel where declared. Vision's `PytorchWrapper` also keys by its model or backbone identifier and the source of its classes. It keeps its `backbone_id` argument, because that argument names the path of its stored activations; registrations that pass it keep their paths.

The unified text, VLM, audio, and video wrappers use an explicit key: the wrapper class, the `_VERSION` in its class body, its settings (`CACHE_FIELDS`, batch size included), the model state above, the input device, and the inputs. The identifier and the source text are not part of it, so two registrations with the same weights and settings share one entry, and an edit that keeps values keeps the key. A change that moves the values a wrapper extracts must raise `_VERSION` in its class body; `tests/test_model_helpers/test_wrapper_cache_guard.py` fails when values move under an unchanged `_VERSION`. The version is read only from the body of the wrapper's own class, never inherited and never from its module: a subclass without `_VERSION` in its own class body extracts every time and warns once, wherever it is defined; with it, it is cached under its own key, and it needs its own guard test.

Changing these inputs causes a cache miss. Identical contents can reuse entries across model instances. Entries created with older keys remain on disk but do not satisfy the new content checks.

This applies to vision's standard `PytorchWrapper` and unified's text, VLM, audio, and `VideoWrapper` extraction paths. Vision's separate `activations.temporal` extractor still has name-based caching and success-only hook cleanup. It is not covered by these fixes. Disable result caching when using that extractor with changing weights or inputs; its failure cleanup still needs repair.

## Cost and limits

Each `score()` call starts a fresh weight-hash scope. The first lookup reads the weights; subsequent lookups reuse their hashes while tensor identity, storage, version counter, shape, stride, dtype, and device remain unchanged. Normal in-place edits and storage replacement invalidate the affected hashes. Scopes are discarded on success, failure, or cancellation. Direct extraction outside a scoring scope reads the weights on every lookup.

Edits through `.data` aliases or NumPy arrays bypass PyTorch's version counter. Make these edits **between scoring runs**, not during a run. A new run reads all weights again. Tensors without version counters are always rehashed. File contents are also read on each lookup; large input collections can remain expensive.

Content transfers and noncontiguous copies are bounded to small slices rather than flattening the entire weight first. Large GPU models still require an initial device-to-host transfer. Opaque configurations, unreadable files, and unsupported tensor storage bypass extraction caching with a warning.

Do not modify weights or input files concurrently with extraction. Custom providers must expose meaningful state through `cache_config()`; arbitrary remote state cannot be inferred. These checks cover activation extraction, not every dataset, score, or provider cache in the ecosystem.

For a mutable-state experiment whose configuration cannot be fully described, disable result caching:

```sh
export RESULTCACHING_DISABLE=1
```

This also skips content hashing for these wrapper disk-cache lookups. Other providers and in-memory transformations can have their own validation. Use a fresh process when changing storage locations:

```sh
export RESULTCACHING_HOME=/path/to/writable/cache
python -m brainscore.doctor     # checks that the folder is writable
```

## Early failures

Scoring checks writable result storage before constructing either the benchmark or the model. The unified entry point also checks local assets declared for that benchmark. Data errors raised by benchmark construction therefore occur before model loading.

This does not guarantee every failure is knowable in advance. Lazy or undeclared data needs, remote availability, and model compatibility without lightweight model metadata can still require later checks. Preflight does not download data to prove access or qualify scientific correctness.

## Cache-key migration

Content-aware keys use the `v2-` prefix. Existing `v1-` and name-only entries are not reused or deleted automatically. Allow time and storage for the first recomputation. This is a cache format change; it does not change the benchmark's scoring definition.
