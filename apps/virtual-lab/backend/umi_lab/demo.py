"""A real, deliberately engineered tiny circuit; not a scientific benchmark."""
from __future__ import annotations

from contextlib import contextmanager

from .profile import LabProfile, Target


def create_profile() -> LabProfile:
    import numpy as np
    import torch
    from brainscore.experiments import TorchInstrumentation
    from brainscore_core.model_interface import BrainScoreModel, TaskContext
    from .sessions import FiniteBehaviorSession
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    model = torch.nn.Linear(8, 8, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(torch.eye(8))
    pairs = [(7, 3), (2, 8), (9, 4), (1, 6), (8, 2), (3, 5), (6, 1), (4, 9)]
    table = StimulusSet({
        "stimulus_id": [f"comparison-{i}" for i in range(len(pairs))],
        "sentence": [f"{a} {b}" for a, b in pairs],
        "label": ["greater" if a > b else "less" for a, b in pairs],
    })
    table.identifier = "lab-number-comparison-v1"

    def answer(stimulus_row: object, instruction: str, label_set: object) -> str:
        a, b = (int(part) for part in stimulus_row["sentence"].split())
        # Units 0 and 1 carry the comparison; the other units are distractors.
        features = torch.tensor([[max(a - b, 0), max(b - a, 0), a, b, a + b, 1, 0, 1]],
                                dtype=torch.float32)
        with torch.inference_mode():
            response = model(features)
        return "greater" if response[0, 0] >= response[0, 1] else "less"

    subject = BrainScoreModel(identifier="comparison-circuit-v1", model=model,
                              supported_modalities={"text"}, answer=answer)

    @contextmanager
    def make_session(trial: object) -> object:
        yield FiniteBehaviorSession(TaskContext(
            task_type="probabilities", instruction="Is the first number greater or less?",
            label_set=["greater", "less"], metadata={"stimulus_set": table},
        ))

    def summarize(result: object) -> dict:
        conditions = result.manifest["plan"]["protocol"]["config"]["conditions"]
        repeats = len(result.value) // len(conditions)
        rows, baseline = [], result.value[0]
        labels = list(table["label"])
        for index, condition in enumerate(conditions):
            scores = []
            for assembly in result.value[index * repeats:(index + 1) * repeats]:
                choices = list(assembly["choice"].values)
                predictions = [choices[i] for i in np.argmax(assembly.values, axis=-1)]
                scores.append(sum(a == b for a, b in zip(predictions, labels)) / len(labels))
            rows.append({"condition": condition, "accuracy": sum(scores) / len(scores),
                         "stimuli": len(labels) * repeats})
        restored = (bool(np.array_equal(baseline.values, result.value[-1].values))
                    if "restored" in conditions else None)
        return {"conditions": rows, "restored_exact": restored,
                "note": "Engineered calibration circuit. These are not Brain-Score benchmark results."}

    return LabProfile(
        subject=subject, title="Comparison circuit", task="Number comparison",
        description="An eight-unit PyTorch circuit with two signal units. No weights to download.",
        input_channels=("text",), output_channels=("behavior",),
        make_session=make_session, summarize=summarize,
        targets=(Target("population", "population", 8),),
        instrumentation=TorchInstrumentation({"population": model}),
        provenance={"kind": "engineered calibration", "weights": "8x8 identity, float32",
                    "device": "cpu", "torch": torch.__version__, "stimuli": table.identifier},
    )
