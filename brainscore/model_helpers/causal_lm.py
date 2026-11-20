"""Register a Hugging Face causal LM as one subject that records, can be pushed, writes and scores text.

One registration declares, and answers through ``participate`` (or ``process``):

- recording: ``neural:<region>`` (``TextWrapper``; layer names accept the
  ``':input'`` site, e.g. ``'model.layers.14.mlp.down_proj:input'``), e.g. via
  ``brainscore_core.streaming_helpers.neural_response``,
- interventions: ``lesion`` / ``stimulation`` (``pytorch_intervention``; send one
  with ``brainscore_core.streaming_helpers.apply_state_change`` or the ``Intervene``
  tool, and undo only it with a reset ``StateChange`` that names its handle),
- a ``reply`` part: a written answer to the trial's text through the chat
  template, under the task's generation settings (``Text(max_tokens=...,
  temperature=..., top_p=..., seed=...)``),
- a ``likelihood`` part: the summed log probability of each option. Under a
  ``verbatim`` choice it scores ``f"{text} {option}"`` as plain text, with no
  chat template and no beginning-of-sequence token, as EWoK's ``score`` does;
  otherwise the text is a user turn of the chat template and each option is
  scored as the assistant's reply. Under a ``Segments`` task it gives the log
  probability of each part given the earlier parts.
"""
from copy import deepcopy
from typing import Any, Mapping, Optional, Sequence

from brainscore import BrainScoreModel
from brainscore_core.responses import Choice, Text
from brainscore.model_helpers.device import input_device

from brainscore.model_helpers.replies import (_Layouts, generation_settings, plain_text_scores,
                                             _scores_alone, segment_log_probabilities)
from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore.model_helpers.intervention import pytorch_intervention


class CausalLMParts:
    """The ``reply`` and ``likelihood`` parts of a text-only causal LM."""

    def __init__(self, model: Any, tokenizer: Any, chat_template_kwargs: Optional[dict] = None,
                 max_new_tokens: int = 256, sliding_window: bool = False) -> None:
        # sliding_window: the registration states that the model's cache can slide (learned
        # absolute positions), so a text longer than its window is scored as the old helper does.
        self.model, self.tokenizer = model, tokenizer
        self.sliding_window = sliding_window
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.max_new_tokens = max_new_tokens
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    @property
    def device(self):
        return input_device(self.model)

    def format(self, prompt) -> str:
        """Chat-template a trial: earlier trials as turns, then its text, ending with the assistant's turn."""
        # The task's instruction is the system turn, as a library's own client sends it.
        instruction, text = _instruction_and_text(prompt)
        messages = []
        for earlier, answered in prompt.history:
            messages += [{'role': 'user', 'content': _instruction_and_text(earlier)[1]},
                         {'role': 'assistant', 'content': answered}]
        messages.append({'role': 'user', 'content': text})
        return _chat_text(self.tokenizer, instruction, messages, self.chat_template_kwargs)

    def reply(self, prompt) -> str:
        """Greedy unless the task's Text response gives a temperature above 0."""
        import torch
        settings = generation_settings(prompt, self.max_new_tokens)
        if isinstance(prompt.response, Text) and prompt.response.seed is not None:
            torch.manual_seed(prompt.response.seed)
        enc = self.tokenizer([self.format(prompt)], return_tensors='pt',
                             add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      **settings)[:, enc['input_ids'].shape[1]:]
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    @property
    def likelihood(self) -> _Layouts:
        """The likelihood part: ``likelihood(prompt, candidates)`` scores options after the
        trial's text; ``likelihood['segment']`` scores the parts of a Segments row."""
        return _Layouts(self._continuation, self._segments)

    def _segments(self, prompt):
        return segment_log_probabilities(self.model, self.tokenizer, prompt.response.segments,
                                          sliding_window=self.sliding_window)

    def _continuation(self, prompt, candidates: Sequence[str]) -> dict:
        """The summed log probability of each candidate after the trial's text."""
        if isinstance(prompt.response, Choice) and prompt.response.verbatim:
            return plain_text_scores(self.model, self.tokenizer, prompt, candidates)
        import torch
        context = self.tokenizer(self.format(prompt), add_special_tokens=False)['input_ids']
        encoded = []
        for candidate in candidates:
            ids = self.tokenizer(candidate, add_special_tokens=False)['input_ids']
            if not ids:
                raise ValueError(f'option {candidate!r} has no tokens to score')
            encoded.append(ids)
        with torch.no_grad():
            # The prompt alone first: the pass a recording keeps.
            self.model(input_ids=torch.tensor([context], device=self.device),
                       attention_mask=torch.ones(1, len(context), dtype=torch.long, device=self.device))
        values = _scores_alone(self.model, [context + ids for ids in encoded], len(context))
        return dict(zip(candidates, values))


def _instruction_and_text(prompt):
    """``(instruction or None, the other text parts joined)`` of a text-only trial."""
    kinds = {kind for kind, _ in prompt.parts}
    if kinds - {'text'}:
        raise NotImplementedError(f'a text-only causal LM cannot be shown {sorted(kinds - {"text"})} parts')
    names = prompt.names or (None,) * len(prompt.parts)
    instruction = [content for (_, content), name in zip(prompt.parts, names) if name == 'instruction']
    text = [content for (_, content), name in zip(prompt.parts, names) if name != 'instruction']
    return ('\n\n'.join(instruction) if instruction else None), '\n\n'.join(text)


def _chat_text(tokenizer, instruction, messages, template_kwargs) -> str:
    """The chat-templated text. The instruction is the system turn; a template
    that takes no system turn gets it at the start of the first user turn."""
    from jinja2.exceptions import TemplateError
    if instruction:
        try:
            return tokenizer.apply_chat_template([{'role': 'system', 'content': instruction}, *messages],
                                                 tokenize=False, add_generation_prompt=True, **template_kwargs)
        except TemplateError:
            first = dict(messages[0], content=f"{instruction}\n\n{messages[0]['content']}")
            messages = [first, *messages[1:]]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)


def build_causal_lm_subject(identifier: str, model: Any, tokenizer: Any, *,
                            region_layer_map: Optional[Mapping[str, str]] = None,
                            provenance: Optional[Mapping[str, Any]] = None,
                            layer_aggregation: str = 'sequence',
                            chat_template_kwargs: Optional[dict] = None,
                            batch_size: int = 8) -> BrainScoreModel:
    """A ``BrainScoreModel`` for a Hugging Face causal LM: record, intervene, reply, score.

    Put model revision and settings in ``provenance``; a copy is attached to
    every recorded ``NeuroidAssembly`` (``attrs['provenance']``).
    ``region_layer_map`` names the regions that can be recorded.
    """
    parts = CausalLMParts(model, tokenizer, chat_template_kwargs)
    # A deep copy: later edits to the caller's dict do not change recorded provenance.
    provenance = {'identifier': identifier, **deepcopy(dict(provenance or {}))}
    return BrainScoreModel(
        identifier,
        region_layer_map=dict(region_layer_map or {}),
        extractor={'text': TextWrapper(model, tokenizer, identifier=identifier,
                                       layer_aggregation=layer_aggregation, batch_size=batch_size,
                                       provenance=provenance)},
        intervention=pytorch_intervention(model),
        reply=parts.reply,
        likelihood=parts.likelihood,
        # The reply gives the choice natively; the likelihood derives the probabilities.
        capability_config={'behavior-sessions': {'behavior:probabilities': ('likelihood', 'softmax')}},
    )
