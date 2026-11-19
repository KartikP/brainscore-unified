"""``reply`` and ``likelihood`` parts for a Hugging Face chat model (text, image or video).

    reply, likelihood = chat_parts(model, processor, chat_template_kwargs={'enable_thinking': False})
    BrainScoreModel(..., reply=reply, likelihood=likelihood,
                    capability_config={'behavior-sessions': {
                        'behavior:probabilities': ('likelihood', 'softmax')}})

Both receive a ``brainscore_core.responses.Prompt``: the trial's parts in order
(instruction, stimulus parts, the response's directive). The task's instruction,
the text part named ``instruction``, is the system turn (or comes first in the
user turn when the chat template takes no system turn). The other parts are
shown as one user message in their order; images as images, clips as video with
their real frame rate. A screen in a stimulus column named ``instruction`` is an
image part and is shown like the other parts. ``reply(prompt)`` returns the model's raw reply; the task's response
parses it. Under a ``verbatim`` choice, ``likelihood`` scores
``f"{text} {candidate}"`` as plain text, with no chat template and no
beginning-of-sequence token, as EWoK's ``score`` does; ``reply`` still uses the
chat template, because a chat model cannot answer without it. A ``Text`` response carries the task's generation settings
(``max_tokens``, ``temperature``, ``top_p``, ``seed``), and the reply obeys them.
Without them the reply is greedy and at most ``max_new_tokens`` tokens long.
``likelihood(prompt, candidates)`` returns the log probability of
each candidate as the whole reply (teacher-forced, summed over its tokens). It
runs the prompt on its own before any candidate, so a recording made during
scoring is the prompt pass, as it is for a reply. The likelihood that ``chat_parts``
returns is also a mapping by layout: ``likelihood['segment']`` answers a ``Segments``
task (the log probability of each part of a text given the parts before it).
"""
from collections.abc import Mapping as _AbcMapping
from typing import Mapping, Optional, Sequence

from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.responses import Choice, Text
from brainscore.model_helpers._device import input_device


def chat_parts(model, processor, *, max_new_tokens: int = 64,
               chat_template_kwargs: Optional[Mapping] = None):
    tokenizer = getattr(processor, 'tokenizer', processor)
    template = dict(chat_template_kwargs or {})

    def inputs_for(prompt):
        return chat_inputs(model, processor, prompt, template)

    def reply(prompt) -> str:
        import torch
        inputs = inputs_for(prompt)
        settings = _generation_settings(prompt, max_new_tokens)
        if isinstance(prompt.response, Text) and prompt.response.seed is not None:
            torch.manual_seed(prompt.response.seed)
        with torch.no_grad():
            generated = model.generate(**inputs, **settings)
        return tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip()

    def segments(prompt):
        return _segment_log_probabilities(model, tokenizer, prompt.response.segments)

    def likelihood(prompt, candidates: Sequence[str]) -> dict:
        import torch
        if isinstance(prompt.response, Choice) and prompt.response.verbatim:
            return _plain_text_scores(model, tokenizer, prompt, candidates)
        inputs = inputs_for(prompt)
        prompt_ids = inputs['input_ids']
        length = prompt_ids.shape[1]
        with torch.no_grad():
            model(**inputs)     # the prompt alone first: the pass a recording keeps
        scores = {}
        for candidate in candidates:
            ids = torch.tensor([tokenizer(candidate, add_special_tokens=False)['input_ids']],
                               device=prompt_ids.device)
            if ids.shape[1] == 0:
                raise ValueError(f'option {candidate!r} has no tokens to score')
            extended = {}
            for key, value in inputs.items():
                # Every per-token input (ids, mask, token types) grows with the candidate.
                if torch.is_tensor(value) and value.dim() == 2 and value.shape == prompt_ids.shape:
                    fill = ids if key == 'input_ids' else torch.full_like(
                        ids, 1 if key == 'attention_mask' else 0)
                    extended[key] = torch.cat([value, fill.to(value.dtype)], dim=1)
                else:
                    extended[key] = value
            with torch.no_grad():
                logits = model(**extended).logits[0, length - 1:length - 1 + ids.shape[1]]
            scores[candidate] = float(logits.float().log_softmax(-1)
                                      .gather(-1, ids[0, :, None]).sum())
        return scores

    return reply, _Layouts(likelihood, segments)


def _plain_text_scores(model, tokenizer, prompt, candidates: Sequence[str]) -> dict:
    """Score each candidate after the trial's text as plain text, as EWoK's ``score`` does.

    For a ``verbatim`` task: the text is ``f"{text} {candidate}"`` with no chat
    template and no beginning-of-sequence token. The value is the summed log
    probability of the tokens after the text's own tokens; each option runs in its
    own forward pass, so its value does not depend on the other options. The text
    runs alone first, so a recording made during scoring is the text pass. An empty text
    is refused: its first token would have no prediction, and EWoK's own score
    is then undefined (NaN).
    """
    import torch
    texts = [content for kind, content in prompt.parts if kind == 'text']
    if len(texts) != len(prompt.parts) or len(texts) != 1:
        raise ExecutionNotFeasibleError('Plain-text scoring takes exactly one text part; got '
                                        f'{[kind for kind, _ in prompt.parts]}')
    if prompt.history:
        raise ExecutionNotFeasibleError(
            'Plain-text scoring scores the text alone; it cannot show earlier trials')
    text = texts[0]
    device = input_device(model)
    context = tokenizer(text, add_special_tokens=False)['input_ids']
    if not context:
        raise ValueError('The text has no tokens to condition on, so the first option token '
                         'cannot be scored; give a nonempty text')
    encoded = []
    for candidate in candidates:
        ids = tokenizer(f'{text} {candidate}', add_special_tokens=False)['input_ids']
        if len(ids) <= len(context):
            raise ValueError(f'option {candidate!r} has no tokens to score')
        encoded.append(ids)
    with torch.no_grad():
        # The text alone first: the pass a recording keeps.
        model(input_ids=torch.tensor([context], device=device),
              attention_mask=torch.ones(1, len(context), dtype=torch.long, device=device))
    return dict(zip(candidates, _scores_alone(model, encoded, len(context))))


def _scores_alone(model, sequences, start: int) -> list:
    """The summed log probability of the tokens of each sequence from position ``start`` on.

    Each sequence runs in its own forward pass (a batch of one, no padding): the log
    probability of an option is a property of the model and its context, and the shape of
    a batch of several options changes it in half precision.
    """
    import torch
    device = input_device(model)
    totals = []
    for sequence in sequences:
        ids = torch.tensor([sequence], device=device)
        # The sum of the batched layout: every position gathered, the option's positions
        # kept by a float32 mask, one sum over the row (the same order of terms as before).
        scored = torch.tensor([[0] * start + [1] * (len(sequence) - start)], device=device)[:, 1:].float()
        with torch.no_grad():
            logp = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, :-1].float().log_softmax(-1)
        totals.append(float((logp.gather(-1, ids[:, 1:, None])[..., 0] * scored).sum(1)[0]))
    return totals


class _Layouts(_AbcMapping):
    """A likelihood part by layout that is also its continuation callable.

    ``part(prompt, candidates)`` scores options as before; ``part['segment']`` answers a
    ``Segments`` task. ``BrainScoreModel(likelihood=part)`` reads both layouts.
    """

    def __init__(self, continuation, segment):
        self._parts = {'continuation': continuation, 'segment': segment}

    def __call__(self, prompt, candidates: Sequence[str]) -> dict:
        return self._parts['continuation'](prompt, candidates)

    def __getitem__(self, layout):
        return self._parts[layout]

    def __iter__(self):
        return iter(self._parts)

    def __len__(self) -> int:
        return len(self._parts)


# The join of the parts of a Segments row: brainscore_core.text.prepare_context.
_SEGMENT_JOIN = 'one space; parts of white space only dropped; no space before . , ! ? ; :'


def _segment_log_probabilities(model, tokenizer, segments: Sequence[str], sliding_window: bool = False):
    """The log probability, in bits, of each part of one text given the parts before it.

    ``(values, parameters)``: one value for each part (``None`` for a part with no scored
    token: the first part of one token) and the response record's parameters (segment
    layout, Response Format Standard 2.3). The schedule is the one of the legacy language
    helper's reading times: the context of part i is ``prepare_context(parts[:i+1])``; its
    new tokens are the tokenizer's ids of that context (the tokenizer's default special
    tokens) after the count of the context before, and every per-token input the
    tokenizer gives reaches the model. The whole schedule is checked against the model's
    window (``max_position_embeddings``) before any forward pass: the first context fits
    the window and each later part has fewer new tokens than the window. A text longer than
    the window needs ``sliding_window=True``: the registration's statement that its cache can
    slide (learned absolute positions, as GPT-2 and OPT have). Without it such a text is
    refused, because a slid cache keeps rotary keys at their old positions. The first
    context runs in full; each later part runs only its new tokens with the key-value cache,
    their positions after the cached ones, and the last logit of the run before. When the
    cache and the new tokens pass the window, the cache keeps its newest ``window - new``
    positions (``slice_kv_cache`` of the language helper). A value is minus the summed cross
    entropy of the new tokens, in the dtype of the model's logits, divided by ln 2; the first
    token of the text has no prediction and is not scored. The model runs where it is.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from brainscore_core.text import prepare_context
    from brainscore_language.model_helpers.huggingface import slice_kv_cache
    window = model.config.get_text_config().max_position_embeddings
    device = next(model.parameters()).device
    steps, count, earlier = [], 0, None
    for i in range(len(segments)):
        encoded = tokenizer(prepare_context(segments[:i + 1]), return_tensors='pt')
        total = encoded['input_ids'].shape[-1]
        # The new tokens are those after the count of the earlier text: right only when the
        # ids of the earlier text are the first ids of the longer text.
        if earlier is not None and not torch.equal(encoded['input_ids'][..., :count], earlier):
            raise ExecutionNotFeasibleError(
                f'the tokens of parts 1 to {i} are not the first tokens of parts 1 to {i + 1}, so the '
                f'new tokens of part {i + 1} cannot be found by count (the tokenizer joins pieces '
                'across parts or adds a special token after each text); use a tokenizer whose '
                'earlier tokens stay (for example add_special_tokens=False with a start token '
                'only) or score the text as one part')
        earlier = encoded['input_ids'].clone()
        if not steps and total > window:
            raise ExecutionNotFeasibleError(f'part 1 has {total} tokens, more than the window of {window}')
        if steps and total - count >= window:
            raise ExecutionNotFeasibleError(f'part {i + 1} has {total - count} new tokens, as many as '
                                            f'the window of {window} or more; it cannot be scored '
                                            'with its context')
        # Every per-token input the tokenizer gives (ids, mask, token types) reaches the model;
        # a copy of the new tokens, so the full prefix is not kept.
        steps.append({key: value[..., count:].clone() for key, value in encoded.items()})
        count = total
    if count > window and not sliding_window:
        raise ExecutionNotFeasibleError(
            f'the text has {count} tokens, more than the window of {window}, and the registration '
            'does not state that its cache can slide (sliding_window=True)')
    values, past, last = [], None, None
    for inputs in steps:
        inputs = {key: value.to(device) for key, value in inputs.items()}
        new = inputs['input_ids'][0]
        with torch.no_grad():
            if past is None:
                output = model(**inputs, use_cache=True)
                logits = output.logits
            else:
                if past.get_seq_length() + len(new) > window:
                    past = slice_kv_cache(past, window - len(new))
                start = past.get_seq_length()
                inputs['attention_mask'] = torch.ones(1, start + len(new), dtype=torch.long, device=device)
                # The positions of the new tokens follow the cached ones.
                output = model(**inputs, past_key_values=past, use_cache=True,
                               cache_position=torch.arange(start, start + len(new), device=device))
                logits = torch.cat([last.to(output.logits.device), output.logits], dim=1)
        past, last = output.past_key_values, logits[:, -1:, :]
        logits = logits[0]
        if logits.shape[0] <= 1:
            values.append(None)
            continue
        predicted = logits[-len(new) - 1:-1, :].contiguous()
        actual = new.to(predicted.device)
        if actual.shape[0] == predicted.shape[0] + 1:
            actual = actual[1:]   # the first token of the text has no prediction
        values.append(-float(F.cross_entropy(predicted, actual, reduction='sum') / np.log(2)))
    parameters = {'layout': 'segment', 'context': 'prefix', 'aggregation': 'sum', 'units': 'bits',
                  'tokenization': {'join': _SEGMENT_JOIN, 'special_tokens': 'tokenizer_default',
                                   'template': 'none', 'truncation': int(window)}}
    return values, parameters


def _generation_settings(prompt, max_new_tokens: int) -> dict:
    """The ``generate`` arguments for one trial, from the task's ``Text`` response.

    Without settings the reply is greedy and at most ``max_new_tokens`` long.
    A temperature above 0 samples, with ``top_p`` when the task gives one.
    """
    response = prompt.response
    settings = {'max_new_tokens': max_new_tokens, 'do_sample': False}
    if not isinstance(response, Text):
        return settings
    if response.max_tokens is not None:
        settings['max_new_tokens'] = response.max_tokens
    if response.temperature:
        settings.update(do_sample=True, temperature=float(response.temperature))
        if response.top_p is not None:
            settings['top_p'] = float(response.top_p)
    return settings


def chat_inputs(model, processor, prompt, template=None):
    """Processor inputs for one trial: earlier turns from ``prompt.history``, then
    the prompt's parts in order (images as images, clips as video)."""
    from PIL import Image
    images, videos, metadata = [], [], []

    def content_of(parts):
        content = []
        for kind, value in parts:
            if kind == 'text':
                content.append({'type': 'text', 'text': value})
            elif kind == 'image':
                images.append(Image.open(value).convert('RGB'))
                content.append({'type': 'image', 'image': images[-1]})
            elif kind == 'video':
                from brainscore.model_helpers.vlm_video_wrapper import clip_metadata, read_clip
                frames, fps = read_clip(value)
                videos.append(frames)
                metadata.append(clip_metadata(frames, fps))
                content.append({'type': 'video'})
            else:
                raise ExecutionNotFeasibleError(f'chat models here cannot be shown a {kind!r} part')
        return content

    # The task's instruction is the system turn, as a library's own client sends it.
    # It is the text part named 'instruction'; a screen in a column of that name
    # (present=('image:instruction', ...)) is a shown part like any other.
    # Earlier trials of the episode (memory='trial_history') become earlier turns.
    def is_instruction(part, name):
        return part[0] == 'text' and name == 'instruction'

    def shown(trial):
        names = trial.names or (None,) * len(trial.parts)
        return [part for part, name in zip(trial.parts, names) if not is_instruction(part, name)]
    names = prompt.names or (None,) * len(prompt.parts)
    instruction = [part[1] for part, name in zip(prompt.parts, names) if is_instruction(part, name)]
    messages = []
    for earlier, answered in prompt.history:
        messages.append({'role': 'user', 'content': content_of(shown(earlier))})
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': answered}]})
    messages.append({'role': 'user', 'content': content_of(shown(prompt))})
    system = ([{'role': 'system', 'content': [{'type': 'text', 'text': '\n\n'.join(instruction)}]}]
              if instruction else [])
    from jinja2.exceptions import TemplateError
    try:
        text = processor.apply_chat_template(system + messages, tokenize=False, add_generation_prompt=True,
                                             **dict(template or {}))
    except TemplateError:
        if not system:
            raise
        # A template that takes no system turn gets the instruction first in the first user turn.
        first = dict(messages[0], content=system[0]['content'] + messages[0]['content'])
        text = processor.apply_chat_template([first, *messages[1:]], tokenize=False,
                                             add_generation_prompt=True, **dict(template or {}))
    call = dict(text=[text], return_tensors='pt', padding=True)
    if images:
        call['images'] = images
    if videos:
        call.update(videos=videos, video_metadata=metadata)
    return processor(**call).to(input_device(model))
