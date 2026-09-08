# Extraction provenance and registered validation

The accepted cache contract permits a one-time miss for entries without a
configuration fingerprint. Existing files remain in place. Going forward,
identical recorded configuration reuses an entry; changed configuration selects
a different entry. No commits or pushes are part of this work.

Use a shared serializer in core, with extraction settings declared by each
wrapper. Include implementation, input identity, processor/tokenizer settings,
model metadata, and hooks. Keep result_caching's existing per-layer merging.
Expose cache_config() for custom providers, and bypass caches when state cannot
be described safely. Manual stimulus/model suffixes and trusting old entries
were considered and rejected: neither establishes extraction provenance.

Validate each of the six unified benchmarks through legacy, adapter, and native
routes. Compare inputs to the metric as well as raw/normalized scores. Test the
harness locally with synthetic data; stage real data/weights behind an explicit
slow-test opt-in. Adapter-only comparisons cannot validate native extraction.

Register an explicit Qwen VWFA variant. Model registration owns the anatomical
search population and implementation-specific perturbation callback; benchmarks
own localization, lesion size, random control, and scoring. Preserve the original
Qwen registration. A benchmark that constructs PyTorch hooks itself would assume
an implementation/root that the Subject interface does not promise.

Measured results, limitations, commands, and runnable examples are recorded in
[the audit](../audits/2026-09-08-extraction-provenance-and-validation.md).
