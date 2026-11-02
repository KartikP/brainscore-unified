"""Register a Hugging Face causal LM as one subject that records, can be pushed, writes and scores text.

One registration declares, and answers through ``interact`` (or ``process``):

- recording: ``neural:<region>`` (``TextWrapper``; layer names accept the
  ``':input'`` site, e.g. ``'model.layers.14.mlp.down_proj:input'``), e.g. via
  ``brainscore_core.streaming_helpers.neural_response``,
- interventions: ``lesion`` / ``stimulation`` (``build_pytorch_ablation_fn``; use
  ``brainscore.instrumentation.intervene`` to apply one and undo only it),
- ``generation_request`` -> generated text, as a ``response_trace``,
- ``conditional_logprob_request`` -> log p(target | context), as a ``response_trace``.

Request keys are listed in ``response_trace.PAYLOAD_CONTRACTS`` (also in the I/O
catalog): generation takes CogGym's ``{system, messages, temperature, max_tokens}``
plus ``conversations``, ``top_p`` and ``seed``; log-probabilities take EWoK's
``{targets, contexts}`` plus ``reduction`` and ``context_format``.
"""
import json
from typing import Any, Mapping, Optional

from brainscore_core.capabilities.registry import register_capability
from brainscore_core.model_interface import BrainScoreModel

from brainscore.model_helpers.response_trace import _capability, register_trace_channels
from brainscore.model_helpers.text_wrapper import TextWrapper
from brainscore.perturbation import build_pytorch_ablation_fn


class CausalLMProvider:
    """Response-trace providers for generation and conditional log-probabilities."""

    def __init__(self, model: Any, tokenizer: Any, chat_template_kwargs: Optional[dict] = None,
                 batch_size: int = 8) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.batch_size = batch_size
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    @property
    def device(self):
        return next(self.model.parameters()).device

    def format(self, messages: list, system: Optional[str] = None) -> str:
        """Chat-template a conversation, ending with the assistant's turn."""
        if system:
            messages = [{'role': 'system', 'content': system}, *messages]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  **self.chat_template_kwargs)

    def generate(self, request: Mapping) -> dict:
        """One conversation (``messages``) or a batch (``conversations``) -> text.

        Greedy unless ``temperature > 0``. A batch returns ``texts`` (one per
        conversation) and JSON-encodes them as ``text``.
        """
        import torch
        batch = 'conversations' in request
        prompts = [self.format(m, request.get('system'))
                   for m in (request['conversations'] if batch else [request['messages']])]
        temperature = request.get('temperature') or 0.0
        kwargs = dict(max_new_tokens=request.get('max_tokens', 256), pad_token_id=self.tokenizer.pad_token_id,
                      do_sample=temperature > 0)
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=request.get('top_p', 1.0))
        if request.get('seed') is not None:
            torch.manual_seed(request['seed'])
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'  # TextWrapper's last_token assumes right padding: restore
        try:
            enc = self.tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(self.device)
        finally:
            self.tokenizer.padding_side = side
        with torch.no_grad():
            out = self.model.generate(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                                      **kwargs)[:, enc['input_ids'].shape[1]:]
        texts = [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]
        usage = {'prompt_tokens': int(enc['attention_mask'].sum()),
                 'completion_tokens': int((out != self.tokenizer.pad_token_id).sum())}
        if batch:
            return {'text': json.dumps(texts), 'texts': texts, 'reasoning': None, 'token_usage': usage}
        return {'text': texts[0], 'reasoning': None, 'token_usage': usage}

    def logprobs(self, request: Mapping) -> dict:
        """log p(target | context) per item, teacher-forced.

        ``reduction``: ``'sum'`` (default, EWoK's convention) or ``'mean'`` over
        target tokens. ``context_format``: ``'raw'`` (default) or ``'chat'``,
        which treats each context as a user message and scores the target as
        the assistant's reply. Context and target are tokenized separately and
        concatenated, so the scored tokens are exactly the target's tokens.
        """
        import torch
        targets, contexts = list(request['targets']), list(request['contexts'])
        if len(targets) != len(contexts):
            raise ValueError('conditional_logprob_request needs one context per target')
        reduction = request.get('reduction', 'sum')
        if reduction not in ('sum', 'mean'):
            raise ValueError("reduction must be 'sum' or 'mean'")
        if request.get('context_format', 'raw') == 'chat':
            contexts = [self.format([{'role': 'user', 'content': c}]) for c in contexts]
        values = []
        for start in range(0, len(targets), self.batch_size):
            pairs = [(self.tokenizer(c, add_special_tokens=False)['input_ids'],
                      self.tokenizer(t, add_special_tokens=False)['input_ids'])
                     for c, t in zip(contexts[start:start + self.batch_size], targets[start:start + self.batch_size])]
            width = max(len(c) + len(t) for c, t in pairs)
            pad = self.tokenizer.pad_token_id
            ids = torch.tensor([c + t + [pad] * (width - len(c) - len(t)) for c, t in pairs], device=self.device)
            mask = torch.tensor([[1] * (len(c) + len(t)) + [0] * (width - len(c) - len(t)) for c, t in pairs],
                                device=self.device)
            scored = torch.tensor([[0] * len(c) + [1] * len(t) + [0] * (width - len(c) - len(t)) for c, t in pairs],
                                  device=self.device)[:, 1:].float()
            with torch.no_grad():  # right padding: real tokens never attend to padding
                logp = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1].float().log_softmax(-1)
            token_lp = logp.gather(-1, ids[:, 1:, None])[..., 0] * scored
            total = token_lp.sum(1)
            values += (total / scored.sum(1) if reduction == 'mean' else total).tolist()
        return {'text': json.dumps(values, allow_nan=False), 'reasoning': None, 'token_usage': None}


def build_causal_lm_subject(identifier: str, model: Any, tokenizer: Any, *,
                            region_layer_map: Optional[Mapping[str, str]] = None,
                            provenance: Optional[Mapping[str, Any]] = None,
                            layer_aggregation: str = 'per_token',
                            chat_template_kwargs: Optional[dict] = None,
                            batch_size: int = 8) -> BrainScoreModel:
    """A ``BrainScoreModel`` for a Hugging Face causal LM: record, intervene, generate, score.

    Put model revision and settings in ``provenance``; it is attached to every
    response. ``region_layer_map`` names the regions that can be recorded.
    """
    provider = CausalLMProvider(model, tokenizer, chat_template_kwargs, batch_size)
    register_trace_channels()
    # One capability serves both request channels, so a session asking for
    # response_trace has exactly one handler; each channel keeps its own provider.
    text = _capability(('conditional_logprob_request', 'generation_request'))
    register_capability(text)
    provenance = {'identifier': identifier, **dict(provenance or {})}
    channels = {'generation_request': dict(provider=provider.generate, parse=str),
                'conditional_logprob_request': dict(provider=provider.logprobs, parse=json.loads)}
    return BrainScoreModel(
        identifier,
        region_layer_map=dict(region_layer_map or {}),
        extractor={'text': TextWrapper(model, tokenizer, identifier=identifier,
                                               layer_aggregation=layer_aggregation, batch_size=batch_size)},
        intervention=build_pytorch_ablation_fn(model),
        capability_config={text.identifier: dict(channels=channels, provenance=provenance,
                                                 streaming=False, reset=None)},
    )
