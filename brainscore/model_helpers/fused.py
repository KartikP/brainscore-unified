"""One forward pass of a chat model over all of a stimulus's parts.

    fused = fused_chat_extractor(model, processor, root=model.model.language_model,
                                 identifier='qwen2.5-vl-7b-fused')
    BrainScoreModel(..., extractor={'vision': image_wrapper, 'text': text_wrapper,
                                    ('vision', 'text'): fused})

A stimulus that brings an image and a sentence is shown to the model as one
message, and the decoder layers are recorded once. Each recording is split by
the tokens that came from each kind of input: one step per kind (``image``,
``video``, ``text``) averaged over its tokens, plus ``final``, the last position,
where the model has read everything. Output: ``(presentation, part, neuroid)``.
"""
from typing import Sequence

import numpy as np

from brainscore_core.responses import Prompt, part_columns
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore.model_helpers._extractors import LiveRecording, as_float32, value


def fused_chat_extractor(model, processor, *, root, identifier: str,
                         order: Sequence[str] = ('image', 'video', 'text'), chat_template_kwargs=None):
    return FusedChatExtractor(model, processor, root, identifier, tuple(order), chat_template_kwargs)


class FusedChatExtractor:
    def __init__(self, model, processor, root, identifier, order, template):
        self._model, self._processor, self._root = model, processor, root
        self._identifier = self.backbone_id = identifier
        self._order, self._template = order, template
        config = getattr(model, 'config', None)
        self._token_kinds = {kind: getattr(config, f'{kind}_token_id', None) for kind in ('image', 'video')}

    @property
    def identifier(self):
        return self._identifier

    def __call__(self, stimuli, layers, stimuli_identifier=None, **kwargs):
        import torch
        from brainscore.model_helpers.replies import chat_inputs
        rows = []
        for _, row in stimuli.iterrows():
            inputs = chat_inputs(self._model, self._processor, self._prompt(row), self._template)
            with self.record(layers) as recording, torch.no_grad():
                self._model(**inputs)
                taken = recording.take()
                rows.append([taken[layer] for layer in layers])
        first = rows[0]
        coords = {column: ('presentation', list(stimuli[column].values)) for column in stimuli.columns}
        coords.update(part_name=('part', list(first[0]['part_name'].values)),
                      part_index=('part', list(first[0]['part_index'].values)),
                      neuroid_id=('neuroid', [n for a in first for n in a['neuroid_id'].values]),
                      layer=('neuroid', [n for a in first for n in a['layer'].values]))
        values = np.concatenate([np.concatenate([np.asarray(a.values) for a in r], axis=2) for r in rows])
        return NeuroidAssembly(values, dims=['presentation', 'part', 'neuroid'], coords=coords)

    def record(self, layers, stimuli=None):
        """Record ``layers`` while another caller (e.g. generation) runs the model.

        ``take()`` gives ``{layer: assembly}`` for the first pass since the
        previous take (the prompt pass), split by input kind with the
        ``input_ids`` of that pass. A pass without ``input_ids``, or with more
        than one prompt, is refused before it runs.
        """
        def inputs(args, kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            if ids is None or ids.shape[0] != 1:
                raise ValueError('A fused recording splits one prompt by its input_ids; this pass '
                                 f'gives {None if ids is None else ids.shape[0]} prompts.')
            return value(ids)[0]

        def package(activity, ids, _row):
            masks = {kind: ids == token for kind, token in self._token_kinds.items()
                     if token is not None and (ids == token).any()}
            masks['text'] = ~np.any(list(masks.values()), axis=0) if masks else np.ones(len(ids), bool)
            names = [kind for kind in ('image', 'video', 'text') if kind in masks] + ['final']
            parts = {}
            for layer, hidden in activity.items():
                hidden = hidden[0]                                   # (sequence, features)
                steps = np.stack([hidden[masks[kind]].mean(axis=0) for kind in names[:-1]] + [hidden[-1]])
                parts[layer] = NeuroidAssembly(steps[None], dims=['presentation', 'part', 'neuroid'], coords={
                    'stimulus_id': ('presentation', ['?']),
                    'part_name': ('part', names), 'part_index': ('part', list(range(len(names)))),
                    'neuroid_id': ('neuroid', [f'{self._identifier}.{layer}.{i}' for i in range(steps.shape[1])]),
                    'layer': ('neuroid', [layer] * steps.shape[1])})
            return parts

        return LiveRecording(self._model, self._root, layers, as_float32, package, inputs)

    def _prompt(self, row):
        parts = []
        for kind in self._order:
            for column in part_columns(kind):
                value = row.get(column) if hasattr(row, 'get') else None
                if value is not None and not (isinstance(value, float) and value != value):
                    parts.append((kind, value))
                    break
        if not parts:
            raise ValueError(f'No image, video or text column in {list(row.index)}')
        return Prompt(tuple(parts), stimulus_id=row.get('stimulus_id'))
