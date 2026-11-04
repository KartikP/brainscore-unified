"""``reply`` and ``likelihood`` parts for a Hugging Face chat model (text, image or video).

    reply, likelihood = chat_parts(model, processor, chat_template_kwargs={'enable_thinking': False})
    BrainScoreModel(..., reply=reply, likelihood=likelihood)

Both receive a ``brainscore_core.responses.Prompt``: the trial's parts in order
(instruction, stimulus parts, the response's directive). They are shown as one
user message in that order; images as images, clips as video with their real
frame rate. ``reply(prompt)`` returns the model's raw reply; the task's response
parses it. ``likelihood(prompt, candidates)`` returns the log probability of
each candidate as the whole reply (teacher-forced, summed over its tokens).
"""
from typing import Mapping, Optional, Sequence


def chat_parts(model, processor, *, max_new_tokens: int = 64,
               chat_template_kwargs: Optional[Mapping] = None):
    tokenizer = getattr(processor, 'tokenizer', processor)
    template = dict(chat_template_kwargs or {})

    def inputs_for(prompt):
        from PIL import Image
        content, images, videos, metadata = [], [], [], []
        for kind, value in prompt.parts:
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
                raise NotImplementedError(f'chat_parts cannot show a {kind!r} part')
        text = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                             add_generation_prompt=True, **template)
        call = dict(text=[text], return_tensors='pt', padding=True)
        if images:
            call['images'] = images
        if videos:
            call.update(videos=videos, video_metadata=metadata)
        return processor(**call).to(model.device)

    def reply(prompt) -> str:
        import torch
        inputs = inputs_for(prompt)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip()

    def likelihood(prompt, candidates: Sequence[str]) -> dict:
        import torch
        inputs = inputs_for(prompt)
        prompt_ids = inputs['input_ids']
        length = prompt_ids.shape[1]
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
