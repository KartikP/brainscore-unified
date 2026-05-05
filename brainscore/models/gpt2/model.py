"""
GPT-2 registered as a BrainScoreModel — text-only baseline.

Pure language model with no vision encoder. This tests how much of a
behavioral benchmark like ROAR can be solved from the string alone,
without any visual word form processing.

Architecture:
  h.{0-11}  -- 12 transformer decoder layers
"""

from brainscore.perturbation import build_pytorch_ablation_fn
from brainscore_core.model_interface import BrainScoreModel


REGION_LAYER_MAP = {
    'language_system': 'h.11',
}


def get_model(identifier: str) -> BrainScoreModel:
    assert identifier == 'gpt2'

    from transformers import GPT2Model, GPT2Tokenizer
    from brainscore.model_helpers.text_wrapper import TextWrapper

    gpt2 = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # GPT-2 has no pad token; set one so batching works
    tokenizer.pad_token = tokenizer.eos_token

    text_wrapper = TextWrapper(
        model=gpt2,
        tokenizer=tokenizer,
        identifier=f'{identifier}-text',
        layer_aggregation='last_token',
        max_length=128,
    )

    return BrainScoreModel(
        identifier=identifier,
        model=gpt2,
        region_layer_map=REGION_LAYER_MAP,
        preprocessors={
            'text': text_wrapper,
        },
        required_modalities={'text'},
        # GPT-2 is not instruction-tuned; no generation_fn.
        # Behavioral tasks route through the readout path (logistic on features).
        behavioral_readout_layer='h.11',
        # Perturbation support — ablate transformer layers (e.g., 'h.10') to
        # measure their contribution to text-only ROAR or Pereira.
        state_change_fn=build_pytorch_ablation_fn(gpt2),
    )
