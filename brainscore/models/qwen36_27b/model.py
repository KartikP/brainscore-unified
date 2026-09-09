"""Qwen3.6-27B registered as a BrainScoreModel — text tower.

A current-generation 27B model reached through the ordinary interface:
``start_recording`` then ``process()``, with no bespoke extraction path. This
was not possible while the repository pinned ``transformers<5`` — ``qwen3_5``
needs 5.x — which previously forced out-of-band feature extraction that bypassed
``process()`` entirely.

Architecture: ``Qwen3_5ForCausalLM.model`` is a ``Qwen3_5TextModel`` with 64
decoder layers, so recording paths are ``layers.{0..63}`` relative to it. The
vision tower is not loaded; this registration is text-only.
"""

from brainscore_core.brainscore_model import BrainScoreModel

HF_IDENTIFIER = 'Qwen/Qwen3.6-27B'

# Provisional. A layer sweep on LeBel put the useful range broadly across the
# middle of the network, and ~2/3 depth sits inside it; the sweep that produced
# a sharper estimate ran against a misaligned time axis and is not trusted.
# Refine with the layer-mapping explorer before quoting this as a committed map.
REGION_LAYER_MAP = {'language_system': 'layers.40'}


# Share of each accelerator given over to weights; the rest holds activations.
WEIGHT_FRACTION = 0.78


def _placement_budget(torch):
    """Per-device memory budget for ``device_map='auto'``, or None on CPU."""
    if not torch.cuda.is_available():
        return None
    budget = {
        index: int(torch.cuda.get_device_properties(index).total_memory
                   * WEIGHT_FRACTION)
        for index in range(torch.cuda.device_count())
    }
    budget['cpu'] = '64GiB'      # spill target if the weights still do not fit
    return budget


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'qwen3.6-27b'
    from brainscore.models._downloads import hf_preflight
    download = hf_preflight(identifier, HF_IDENTIFIER, 55.6)
    # Imported here, not at module scope: importing brainscore must not pull in
    # torch or transformers, which is asserted by tests/test_import_hygiene.py.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from brainscore.model_helpers.text_wrapper import TextWrapper

    # 54 GB at bf16, so placement is automatic. Left to itself accelerate packs
    # each device to the brim and the first forward pass then OOMs on
    # activations — the weights fit but nothing else does. Reserving a fraction
    # of each device keeps room for activations; whatever still does not fit
    # spills to host memory rather than failing outright.
    full = AutoModelForCausalLM.from_pretrained(
        HF_IDENTIFIER, dtype=torch.bfloat16, device_map='auto',
        max_memory=_placement_budget(torch), **download)
    full.eval()
    text_model = full.model                      # Qwen3_5TextModel, 64 layers

    tokenizer = AutoTokenizer.from_pretrained(HF_IDENTIFIER, **download)
    tokenizer.padding_side = 'left'              # keeps the last position real
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wrapper = TextWrapper(
        model=text_model,
        tokenizer=tokenizer,
        identifier=f'{identifier}-text',
        backbone_id='qwen3.6-27b',               # shareable across registrations
        layer_aggregation='last_token',
        max_length=128,
    )

    return BrainScoreModel(
        identifier=identifier,
        model=text_model,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={'text': wrapper},
        required_modalities={'text'},
    )
