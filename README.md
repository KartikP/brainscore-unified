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
