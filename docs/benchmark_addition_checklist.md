# Benchmark addition checklist

Every new benchmark must leave enough evidence that it is scientifically
calibrated, reproducible, and compatible with the unified interface. Treat this
as the pre-merge checklist for benchmark PRs and as the runbook for any EC2-only
validation work.

## Required evidence

| Area | Requirement | Evidence to attach |
| --- | --- | --- |
| Interface contract | Drive candidates only through `start_recording`, `start_task`, and `process`; never call legacy `look_at` or `digest_text` from new benchmark code. | Unit or integration test that runs a minimal compliant model through the benchmark path. |
| Data boundary | Resolve data through a registered data plugin via `load_stimulus_set` and `load_dataset`; keep S3 keys, one-time packaging, and run-once experiments outside the importable benchmark package. | Data registry test and a benchmark import test that does not touch S3 at import time. |
| User-supplied inputs | If any input cannot be redistributed — licensed stimuli, data under an agreement, a third-party atlas — declare it in the manifest in `brainscore/data/local.py` and read it with `local.path(...)`, so a missing file prints how to obtain and convert it. See [local_data.md](local_data.md). | A test that skips rather than fails when the asset is absent, and passes when it is present. |
| Assembly contract | Document expected dimensions, coordinates, and units. Naturalistic benchmarks must state how `stimulus_id`, `time_bin_start_ms`, `time_bin_end_ms`, modality tags, and neuroid metadata align with the target assembly. | Small fixture test covering dims, coord names, coord order, and dtype-sensitive values. |
| Null floor | Score a chance or randomized baseline. Neural predictivity raw correlations should be near zero unless the benchmark is explicitly testing a different floor. | Raw score, expected tolerance, and command or script used to produce it. |
| Ceiling | Define the ceiling source and whether reported scores are raw, ceiled, normalized, or both. | Ceiling value, computation path, and score attrs showing raw and ceiled values. |
| Modality ablations | For multimodal benchmarks, score each modality alone, naive fusion when applicable, and the intended fusion method. Include signal/null permutations when the backbones make that practical. | Table of modality modes, null permutations, raw scores, and pass/fail criteria. |
| Timing assumptions | For temporal data, document frame/audio/text sampling, wrapper windowing, HRF or GLM assumptions, cross-validation grouping, and any alignment offsets. | Timing diagram or table plus a test that fails on mismatched `time_bin` axes or missing timing coords. |
| Scoring implementation | Reuse shared scoring helpers for ridge, banded ridge, cross-validation, and Pearson calculations instead of duplicating local math. | Code reference to the shared helper and a parity test when replacing an existing scoring path. |
| Determinism | Rerun the same score with caches cleared or isolated. Any stochastic component must be seeded or excluded from the benchmark path. | Two raw scores and the cache/seed settings used. |
| Pipeline attrs | Scores must carry enough attrs for downstream docs and dashboards to identify pipeline, modality mode, ROI, timing variant, and raw score. | Assertion over `Score.attrs` in the benchmark test. |
| Regression baseline | If the benchmark changes an existing scored path, compare against `baselines/baselines.json` with `unified/tests/test_regression_baselines.py`. | Local reference-test output and EC2 full-regression result for heavy paths. |
| Heavy compute | Large downloads, full benchmark scoring, and model sweeps run on EC2 only. Stop the instance and log cost after the run. | EC2 command log, score output, instance stop confirmation, and cost ledger entry. |
| Docs status | Update public capability/status docs when the benchmark adds a new modality, domain, or support level. | Link to the docs/site diff and the benchmark identifier. |

## Lahner multimodal validation matrix

The Lahner A+V work is the reference validation pattern for new multimodal
benchmarks. It caught an apparent multimodal gain that was actually alpha tuning
on the video tower, so new multimodal benchmarks should reproduce this level of
evidence before their scores are treated as stable.

| Check | Pass criterion |
| --- | --- |
| Null floor | Chance and double-null controls land near zero on each ROI or target subset. |
| Single-null asymmetry | A null video plus signal audio model matches the signal model's `audio_only` score; signal video plus null audio matches `video_only`. |
| Mode-curve coherence | Naive fusion behaves as expected for the target signal balance, and the intended fusion method matches or beats the better alpha-tuned single modality within tolerance. |
| Region by modality specificity | ROI or target-subset changes alter the dominant modality in the scientifically expected direction. |
| Reproducibility | Two cold-cache runs produce the same raw score to the documented precision. |

Keep the benchmark-local protocol for runnable details when a benchmark has
special data preparation, but promote the reusable acceptance criteria here.
For Lahner, the runnable protocol remains in
[MULTIMODAL_VALIDATION_PROTOCOL.md](../brainscore/benchmarks/lahner2024/MULTIMODAL_VALIDATION_PROTOCOL.md),
and the EC2 matrix lives in
[EC2_RUN_INSTRUCTIONS.md](../brainscore/benchmarks/lahner2024/EC2_RUN_INSTRUCTIONS.md).

## Minimal pytest template

Use this as a starting point for benchmark tests. Replace identifiers and
fixtures with benchmark-specific values.

```python
import brainscore
from brainscore import load_dataset, load_stimulus_set


def test_data_plugin_resolves_without_benchmark_import_side_effects():
    stimulus_set = load_stimulus_set("your-stimulus-set")
    assembly = load_dataset("your-assembly")
    assert "stimulus_id" in stimulus_set.columns
    assert "presentation" in assembly.dims


def test_score_attrs_document_pipeline(minimal_candidate):
    score = brainscore.score(minimal_candidate, "your-benchmark-id")
    assert "raw" in score.attrs
    assert score.attrs["pipeline"]
    assert score.attrs["modality_mode"]
```
