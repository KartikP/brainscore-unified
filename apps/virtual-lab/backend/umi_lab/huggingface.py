"""An optional real causal-LM profile using UMI's existing model builder."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

from .profile import LabProfile, Target


def create_profile() -> LabProfile:
    """Load a cached model once; expose one explicitly configured output site.

    UMI_LAB_MODEL: local checkpoint directory or cached Hugging Face identifier.
    UMI_LAB_TARGET: module output address (default: GPT-2 block 6 MLP projection).
    UMI_LAB_UNITS: output width (defaults to the model's hidden_size).
    UMI_LAB_DEVICE: cpu (default), mps, or cuda.
    UMI_LAB_OFFLINE: 1 (default); set 0 to permit downloading the chosen model.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.hub import cached_file
    from brainscore.experiments import TorchInstrumentation
    from brainscore.model_helpers.causal_lm import build_causal_lm_subject
    from brainscore_core.streaming import StreamEvent
    from .sessions import FiniteSession

    identifier = os.environ.get("UMI_LAB_MODEL", "gpt2")
    layer = os.environ.get("UMI_LAB_TARGET", "transformer.h.6.mlp.c_proj")
    device = os.environ.get("UMI_LAB_DEVICE", "cpu")
    revision = os.environ.get("UMI_LAB_REVISION", "main")
    offline = os.environ.get("UMI_LAB_OFFLINE", "1") != "0"
    if not offline and not Path(identifier).is_dir():
        from huggingface_hub import snapshot_download
        identifier_path = snapshot_download(identifier, revision=revision,
                                            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "pytorch_model*.bin"])
    else:
        identifier_path = identifier
    config_path = cached_file(identifier_path, "config.json", revision=revision, local_files_only=offline)
    checkpoint = Path(config_path).resolve().parent
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    module = model.get_submodule(layer)
    units = int(os.environ.get("UMI_LAB_UNITS", str(model.config.hidden_size)))
    digest = hashlib.sha256()
    weight_files = sorted([*checkpoint.glob("*.safetensors"), *checkpoint.glob("pytorch_model*.bin")])
    for weight in weight_files:
        digest.update(weight.name.encode())
        with weight.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    provenance = {"model": identifier, "checkpoint": str(checkpoint), "requested_revision": revision,
                  "weights_sha256": digest.hexdigest(), "device": device,
                  "dtype": str(next(model.parameters()).dtype), "torch": torch.__version__,
                  "task": "capital-completion-exploration-v1", "reduction": "sum"}
    subject = build_causal_lm_subject(identifier, model, tokenizer, provenance=provenance)
    prompts = ["The capital of France is", "The capital of England is",
               "The Eiffel Tower is in", "Big Ben is in",
               "The Louvre museum is in", "Buckingham Palace is in",
               "The French government meets in", "The British government meets in"]
    labels = [0, 1, 0, 1, 0, 1, 0, 1]

    @contextmanager
    def make_session(trial: object) -> object:
        # One request scores both alternatives in the same batch. No UI calls
        # enter the forward loop; tokenization and extraction remain in Python.
        yield FiniteSession([StreamEvent("conditional_logprob_request", {
            "contexts": [prompt for prompt in prompts for _ in range(2)],
            "targets": [city for _ in prompts for city in (" Paris", " London")],
            "reduction": "sum",
        }, None)], requested_output_channels=("response_trace",))

    def summarize(result: object) -> dict:
        conditions = result.manifest["plan"]["protocol"]["config"]["conditions"]
        repeats = len(result.value) // len(conditions)
        scores = []
        for events in result.value:
            payload = events[0].payload
            if payload["valid"] is not True:
                raise ValueError(f"Invalid response: {payload['parse_error']}")
            scores.append(np.asarray(payload["answer"]).reshape(len(prompts), 2))
        rows = [{"condition": condition,
                 "accuracy": float(np.mean([np.mean(np.argmax(score, axis=1) == labels)
                                             for score in scores[i * repeats:(i + 1) * repeats]])),
                 "stimuli": len(prompts) * repeats} for i, condition in enumerate(conditions)]
        restored = (bool(np.array_equal(scores[0], scores[-1])) if "restored" in conditions else None)
        return {"conditions": rows, "restored_exact": restored,
                "note": "Exploratory two-choice completion task, not a registered Brain-Score benchmark. Raw log probabilities are retained in the artifacts."}

    return LabProfile(subject=subject, title=identifier, task="Capital completion",
                      description="A causal language model, loaded once in the Python worker.",
                      input_channels=("conditional_logprob_request",), output_channels=("response_trace",),
                      make_session=make_session, summarize=summarize,
                      targets=(Target(layer, layer, units),),
                      instrumentation=TorchInstrumentation({layer: module}), provenance=provenance)
