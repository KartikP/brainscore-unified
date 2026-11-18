"""
GPT-2 (the 124M checkpoint) registered as a BrainScoreModel named gpt2-small -- text-only baseline.

The name gpt2 belongs to the old language plugin (brainscore_language), which
load_model reaches through its adapter.

Pure language model with no vision encoder. This tests how much of a
behavioral benchmark like ROAR can be solved from the string alone,
without any visual word form processing.

Ways to answer: the readout (a choice, probabilities and a rating, fitted on
the task's fitting stimuli) and the likelihood (the log probabilities of the
options of a verbatim choice, as EWoK asks). GPT-2 is not instruction-tuned,
so it does not write answers.

Architecture:
  h.{0-11}  -- 12 transformer decoder layers
"""

from brainscore import BrainScoreModel


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
    assert identifier == 'gpt2-small'

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from brainscore.model_helpers.causal_lm import _CausalLMParts
    from brainscore.model_helpers.text_wrapper import TextWrapper

    # The language head scores the options; its transformer is the network that is recorded.
    language_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2 = language_model.transformer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # GPT-2 has no pad token; set one so batching works
    tokenizer.pad_token = tokenizer.eos_token

    text_wrapper = TextWrapper(
        model=gpt2,
        tokenizer=tokenizer,
        identifier=f'{identifier}-text',
        # The model's own window (``n_positions`` in the hub settings of ``gpt2``), as the
        # language plugin ``gpt2`` cuts its contexts; the wrapper default (512) cuts longer ones.
        max_length=1024,
    )

    return BrainScoreModel(
        identifier=identifier,
        model=gpt2,
        region_layer_map=REGION_LAYER_MAP,
        extractor={
            'text': text_wrapper,
        },
        # GPT-2 is not instruction-tuned; no answer part.
        # Behavioral tasks route through the readout path (logistic on features).
        # Left at the last block while the recording site moved to h.9: this
        # fits a readout rather than naming a recording site, so it warrants
        # its own criterion. Yeatman2021-lexical_decision-text is unchanged by
        # the re-map (raw 0.81 at both), confirming the two are independent.
        behavioral_readout_layer='h.11',
        # The likelihood gives the log probabilities of the options of a verbatim choice.
        # GPT-2 has learned absolute positions: its cache can slide past the 1,024-token window.
        likelihood=_CausalLMParts(language_model, tokenizer, sliding_window=True).likelihood,
        # The readout derives the choice from its probabilities by argmax; the readout
        # also gives the probabilities and the rating.
        capability_config={'behavior-sessions': {'behavior:choice': ('readout', 'argmax')}},
    )
