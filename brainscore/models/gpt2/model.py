"""
GPT-2 registered as a BrainScoreModel — text-only baseline.

Pure language model with no vision encoder. This tests how much of a
behavioral benchmark like ROAR can be solved from the string alone,
without any visual word form processing.

Architecture:
  h.{0-11}  -- 12 transformer decoder layers
"""

from brainscore_core.model_interface import BrainScoreModel


# Chosen by sweeping all 12 blocks on Pereira2018.243sentences — the mapping
# benchmark — and taking the maximum: h.9 scores 0.6007 there against h.11's
# 0.5268. Deliberately not chosen on LeBel2023, which is reported with this map
# and would make the choice circular; h.9 is not even LeBel's own best block
# (h.7 is), so the two are independent. See
# experiments/layer_mapping/gpt2_language_layer.py.
REGION_LAYER_MAP = {
    'language_system': 'h.9',
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
        # Left at the last block while the recording site moved to h.9: this
        # fits a readout rather than naming a recording site, so it warrants
        # its own criterion. Yeatman2021-lexical_decision-text is unchanged by
        # the re-map (raw 0.81 at both), confirming the two are independent.
        behavioral_readout_layer='h.11',
    )
