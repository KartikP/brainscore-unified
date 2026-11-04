"""``reply`` and ``likelihood`` parts for a Hugging Face chat model (text or image-text).

    reply, likelihood = chat_parts(model, processor, chat_template_kwargs={'enable_thinking': False})
    BrainScoreModel(..., reply=reply, likelihood=likelihood)

``reply(stimulus_row, prompt)`` returns the model's raw reply; the task's
response parses it. ``likelihood(stimulus_row, prompt, candidates)`` returns
the log probability of each candidate as the whole reply (teacher-forced,
summed over its tokens), so a choice and its implied probabilities come from
the model's own distribution without generating.
"""
from typing import Mapping, Optional, Sequence


def chat_parts(model, processor, *, max_new_tokens: int = 64,
               chat_template_kwargs: Optional[Mapping] = None):
    from PIL import Image
    from brainscore.model_helpers.api_behavioral import IMAGE_COLUMNS, TEXT_COLUMNS, _find_column

    tokenizer = getattr(processor, 'tokenizer', processor)
    template = dict(chat_template_kwargs or {})

    def inputs_for(stimulus_row, prompt):
        image_column = _find_column(stimulus_row, IMAGE_COLUMNS)
        content, images = [], None
        if image_column:
            images = [Image.open(stimulus_row[image_column]).convert('RGB')]
            content.append({'type': 'image', 'image': images[0]})
        else:
            text_column = _find_column(stimulus_row, TEXT_COLUMNS)
            if text_column:
                prompt = f'{stimulus_row[text_column]}\n\n{prompt}'
        content.append({'type': 'text', 'text': prompt})
        text = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                             add_generation_prompt=True, **template)
        call = dict(text=[text], return_tensors='pt', padding=True)
        if images is not None:
            call['images'] = images
        return processor(**call).to(model.device)

    def reply(stimulus_row, prompt) -> str:
        import torch
        inputs = inputs_for(stimulus_row, prompt)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip()

    def likelihood(stimulus_row, prompt, candidates: Sequence[str]) -> dict:
        import torch
        inputs = inputs_for(stimulus_row, prompt)
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
