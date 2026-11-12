"""``reply`` and ``likelihood`` parts for a Hugging Face chat model (text, image or video).

    reply, likelihood = chat_parts(model, processor, chat_template_kwargs={'enable_thinking': False})
    BrainScoreModel(..., reply=reply, likelihood=likelihood)

Both receive a ``brainscore_core.responses.Prompt``: the trial's parts in order
(instruction, stimulus parts, the response's directive). They are shown as one
user message in that order; images as images, clips as video with their real
frame rate. ``reply(prompt)`` returns the model's raw reply; the task's response
parses it. Under a ``raw_text`` choice, ``likelihood`` scores
``f"{text} {candidate}"`` as plain text, with no chat template and no
beginning-of-sequence token, as EWoK's ``score`` does; ``reply`` still uses the
chat template, because a chat model cannot answer without it. A ``Text`` response carries the task's generation settings
(``max_tokens``, ``temperature``, ``top_p``, ``seed``), and the reply obeys them.
Without them the reply is greedy and at most ``max_new_tokens`` tokens long.
``likelihood(prompt, candidates)`` returns the log probability of
each candidate as the whole reply (teacher-forced, summed over its tokens). It
runs the prompt on its own before any candidate, so a recording made during
scoring is the prompt pass, as it is for a reply.
"""
from typing import Mapping, Optional, Sequence


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
        seed = getattr(prompt.response, 'seed', None)
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            generated = model.generate(**inputs, **settings)
        return tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip()

    def likelihood(prompt, candidates: Sequence[str]) -> dict:
        import torch
        if getattr(prompt.response, 'raw_text', False):
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
                raise ValueError(f'candidate {candidate!r} has no tokens to score')
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

    return reply, likelihood


def _plain_text_scores(model, tokenizer, prompt, candidates: Sequence[str]) -> dict:
    """Score each candidate after the trial's text as plain text, as EWoK's ``score`` does.

    For a ``raw_text`` task: the text is ``f"{text} {candidate}"`` with no chat
    template and no beginning-of-sequence token. The value is the summed log
    probability of the tokens after the text's own tokens. The text runs alone
    first, so a recording made during scoring is the text pass. An empty text
    is refused: its first token would have no prediction, and EWoK's own score
    is then undefined (NaN).
    """
    import torch
    texts = [content for kind, content in prompt.parts if kind == 'text']
    if len(texts) != len(prompt.parts) or len(texts) != 1:
        raise ValueError('Plain-text scoring takes exactly one text part; got '
                         f'{[kind for kind, _ in prompt.parts]}')
    if prompt.history:
        raise ValueError('Plain-text scoring scores the text alone; it cannot show earlier trials')
    text = texts[0]
    device = next(model.parameters()).device
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
    # Right padding with any token: a real token never attends to it, and it is never scored.
    pad = next((value for value in (getattr(tokenizer, 'pad_token_id', None),
                                    getattr(tokenizer, 'eos_token_id', None)) if value is not None), 0)
    width = max(len(ids) for ids in encoded)
    ids = torch.tensor([row + [pad] * (width - len(row)) for row in encoded], device=device)
    mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in encoded], device=device)
    scored = torch.tensor([[0] * len(context) + [1] * (len(row) - len(context)) + [0] * (width - len(row))
                           for row in encoded], device=device)[:, 1:].float()
    with torch.no_grad():  # right padding: real tokens never attend to padding
        logp = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float().log_softmax(-1)
    totals = (logp.gather(-1, ids[:, 1:, None])[..., 0] * scored).sum(1).tolist()
    return dict(zip(candidates, totals))


def _generation_settings(prompt, max_new_tokens: int) -> dict:
    """The ``generate`` arguments for one trial, from the task's ``Text`` response.

    Without settings the reply is greedy and at most ``max_new_tokens`` long.
    A temperature above 0 samples, with ``top_p`` when the task gives one.
    """
    response = prompt.response
    length = getattr(response, 'max_tokens', None)
    settings = {'max_new_tokens': length if length is not None else max_new_tokens,
                'do_sample': False}
    temperature = getattr(response, 'temperature', None)
    if temperature:
        settings.update(do_sample=True, temperature=float(temperature))
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
                raise NotImplementedError(f'chat models here cannot be shown a {kind!r} part')
        return content

    # Earlier trials of the episode (memory='trial_history') become earlier turns.
    messages = []
    for earlier, answered in prompt.history:
        messages.append({'role': 'user', 'content': content_of(earlier.parts)})
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': answered}]})
    messages.append({'role': 'user', 'content': content_of(prompt.parts)})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **dict(template or {}))
    call = dict(text=[text], return_tensors='pt', padding=True)
    if images:
        call['images'] = images
    if videos:
        call.update(videos=videos, video_metadata=metadata)
    return processor(**call).to(model.device)
