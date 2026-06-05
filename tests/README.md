# Unified test suite — tiers

Tests are organized into tiers via pytest markers (registered in
`unified/pyproject.toml`; the same set is registered in `core`, `vision`,
`language`). `tests/conftest.py` auto-marks any untiered test as `unit`, so the
tiers are meaningful without hand-marking every file.

| marker | meaning | runs in CI? |
|---|---|---|
| `unit` | fast, offline, isolated — no real model weights, no network, no S3 (default) | yes (every commit) |
| `integration` | end-to-end across components (e.g. `process()` → benchmark → `score()`), still offline with synthetic models/fixtures | yes |
| `slow` | needs real model downloads / heavy data / GPU | on demand only |
| `private_access` | needs S3 / private resources (usually combined with `slow`) | on demand only |

## Running

```bash
pytest -m unit                    # fast inner loop (every commit)
pytest -m "unit or integration"   # full offline CI tier (no big models)  — ~265 tests, ~7s
pytest -m slow                    # real-model + regression tier (AWS creds + downloads)
pytest -m "slow and not private_access"   # slow but no S3
```

## What's marked `slow`

- `test_text_wrapper.py` — loads real CLIP / GPT-2 weights
- `test_vlm_vision_wrapper.py` — loads real Qwen2.5-VL-3B
- `test_roar_benchmark.py` — loads CLIP + the ROAR benchmark data
- `test_regression_baselines.py::test_score_matches_baseline` — scores real
  models against `baselines/baselines.json` (also `private_access`)

Everything else is `unit` (offline; wrappers tested with tiny CPU models,
benchmark-structure checks, message/multi-agent, layer-mapping, percept/activation
windows, visualization, harnesses). The two `score()`-orchestration e2e tests are
`integration`.

> Eventual CI target: Jenkins on ephemeral EC2 (runs `-m "unit or integration"`
> per change, `-m slow` on demand). Wiring is deferred; the markers are
> location-independent and ready now.
