"""
GPT-2 registered as a BrainScoreModel — text-only baseline.

Pure language model with no vision encoder. This tests how much of a
behavioral benchmark like ROAR can be solved from the string alone,
without any visual word form processing.

Architecture:
  h.{0-11}  -- 12 transformer decoder layers
"""

from brainscore_core.model_interface import BrainScoreModel


# Confirmed by sweeping all 12 blocks on Pereira2018, the mapping benchmark:
# h.11 scores 0.873 there, ahead of h.9 (0.819) and h.7 (0.761).
#
# This was briefly re-mapped to h.9 on 2026-09-07 and is now back. That sweep
# ran while the unified Pereira variant showed native models one bare sentence
# at a time, where the legacy path had always supplied the running passage
# context; the depth curve under that bug peaked mid-stack, and it peaks at the
# last block once the two paths agree. LeBel2023 does separately prefer h.7
# (0.1124 against h.11's 0.1013) — benchmarks disagree about depth, and the
# mapping benchmark decides.
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
        # Long enough for a whole Pereira passage in context (193 tokens);
        # inputs shorter than this are unaffected, so other benchmarks do
        # not move. GPT-2's own window is 1024.
        max_length=512,
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
