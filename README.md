# brainscore (unified model interface)

The unified Brain-Score package: register a model once and score it across every
compatible benchmark domain (vision, language, audio, video, multimodal,
perturbation, embodied) through a single `process()`-based interface.

## Layout

- `brainscore/model_helpers/` — wrappers (model harnesses): image, text,
  VLM-vision, video, audio, tensor, and the stateful `PolicyWrapper` for
  closed-loop embodied dispatch.
- `brainscore/harnesses/` — environment and subject harnesses (robotics
  environment harness; human-harness scaffold).
- `brainscore/models/` — registered models (CLIP, Qwen-VL, BLIP-2, V-JEPA,
  null controls, and others).
- `brainscore/benchmarks/` — registered benchmarks (ROAR, MajajHong-unified,
  Pereira-unified, Lahner2024, Algonauts2025, and others).
- `brainscore/metrics/` — scoring metrics.

## Install

Editable install alongside the domain repositories, which provide the
model-extraction and scoring stack:

```bash
pip install -e core -e vision -e language -e unified
```

Requires Python 3.11 (the data stack pins to 3.11; see the repository for the
full environment and version constraints).

## Usage

A model is a `Subject` (formerly `UnifiedModel`). The everyday surface:

```python
from brainscore import load_model, load_benchmark

model = load_model("clip-vit-b-32")
score = load_benchmark("MajajHong2015public.IT-pls-unified")(model)
```

The full specification lives in the unified-model-interface design documents.

## Capability Status Matrix

Status labels separate real numeric validation from local structural coverage.
`EC2-only` means the full score needs large data, model weights, API calls, or
benchmark compute that must not run on the local Mac. `Demo-only` means the
interface is exercised, but the result is not a brain-alignment claim.

| Capability | Example | Status | Evidence |
| --- | --- | --- | --- |
| Vision neural encoding | MajajHong2015 V4/IT | validated | Baseline manifest plus slow replay test preserve anchored leaderboard scores. |
| Language neural encoding | Pereira2018 | validated | Same regression-baseline path as vision; normal CI checks manifest shape, slow tier re-scores. |
| Behavioral lexical decision | ROAR / Yeatman2021 | validated | Registered image/text variants, shared split, ceiling, and CLIP-above-chance scoring are tested. |
| Video neural encoding | Lahner2024 BOLDMoments | EC2-only | Local tests verify registration, temporal preprocessing, and V-JEPA wiring; full scoring is EC2-only. |
| Audio wrapper | Wav2Vec2-style features | structurally-tested | Construction, stimulus columns, aggregation, truncation, and cache keys are unit-tested without weights. |
| Audio+video fMRI | Lahner2024 multimodal | EC2-only | Registry, mode validation, null controls, and modality tagging are local; real forward/data runs are EC2-only. |
| Movie multimodal fMRI | Algonauts2025 / CNeuroMod | EC2-only | Twelve benchmark entries and scoring kernels are guarded locally; the data assembly is about 100 GB on EC2. |
| State-change perturbation | `process(StateChange)` | structurally-tested | Hook install, indexed ablation, concurrent handles, and reset are tested on a toy torch model. |
| Embodied action | GridGame `action_fn` | demo-only | Floor, ceiling, deterministic boards, and registration are tested; the game is an interface demo. |
| Closed-weight API behavior | OpenRouter / OpenAI-compatible | structurally-tested | Provider registration, lazy model construction, parsing, image payloads, and cache behavior are mocked locally. |
