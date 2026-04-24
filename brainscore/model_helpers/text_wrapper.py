"""
TextWrapper — symmetric with PytorchWrapper for text model activation extraction.

Wraps a text model (encoder or causal), handles tokenization, forward pass,
hook-based layer extraction, batching, caching (@store_xarray), and
NeuroidAssembly packaging.

Usage:
    text_wrapper = TextWrapper(
        model=clip_model.text_model,
        tokenizer=clip_tokenizer,
        identifier='clip-vit-b-32-text',
        layer_aggregation='mean_tokens',
    )
    assembly = text_wrapper(stimulus_set, layers=['encoder.layers.10'])
"""

import functools
import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Union

import numpy as np
from tqdm.auto import tqdm

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly, walk_coords
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from result_caching import store_xarray


logger = logging.getLogger(__name__)


class TextWrapper:
    """Text model wrapper symmetric with PytorchWrapper.

    Args:
        model: A PyTorch text model (e.g., CLIPTextModel, GPT2Model).
        tokenizer: A HuggingFace tokenizer.
        identifier: Model identifier for display/logging. Defaults to class name.
        backbone_id: Optional cache-key identifier. When two registrations
            share the same underlying backbone weights (e.g., two LM-adapted
            models sitting on top of the same LLaMA-3.2-3B) they can pass the
            same ``backbone_id`` so the @store_xarray cache entry is shared
            across registrations. Defaults to ``identifier`` for backwards
            compatibility — existing registrations keep the old per-model
            cache layout.
        layer_aggregation: How to reduce (batch, seq_len, hidden) to (batch, hidden).
            'last_token' for causal models, 'mean_tokens' for encoders.
        max_length: Max token length for truncation.
        batch_size: Number of texts per forward pass.
    """

    def __init__(self, model, tokenizer, identifier=None, backbone_id=None,
                 layer_aggregation='last_token', max_length=512,
                 batch_size=32):
        import torch
        self._model = model
        self._tokenizer = tokenizer
        self._layer_aggregation = layer_aggregation
        self._max_length = max_length
        self._batch_size = batch_size
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._model = self._model.to(self._device)

        self._identifier = identifier or model.__class__.__name__
        # Cache key: prefer explicit backbone_id, fall back to identifier.
        # Keeping this as a separate attribute lets BrainScoreModel swap the
        # cache-key without disturbing log/telemetry identifiers.
        self._backbone_id = backbone_id or self._identifier
        self._stimuli_identifier = None

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, value):
        self._identifier = value

    @property
    def backbone_id(self):
        return self._backbone_id

    def __call__(self, stimuli, layers, stimuli_identifier=None, **kwargs):
        """Extract activations from text stimuli.

        Args:
            stimuli: StimulusSet with 'sentence' or 'text' column, or list of strings.
            layers: List of layer name strings to extract from.
            stimuli_identifier: Identifier for caching. None to use stimulus_set.identifier.

        Returns:
            NeuroidAssembly with dims (presentation, neuroid).
        """
        if isinstance(stimuli, StimulusSet):
            return self._from_stimulus_set(stimuli, layers, stimuli_identifier)
        else:
            return self._from_texts(stimuli, layers, stimuli_identifier)

    def _from_stimulus_set(self, stimulus_set, layers, stimuli_identifier=None):
        if stimuli_identifier is None and hasattr(stimulus_set, 'identifier'):
            stimuli_identifier = stimulus_set.identifier

        texts = self._extract_texts(stimulus_set)
        activations = self._from_texts_cached(
            texts, layers, stimuli_identifier)

        activations = self._attach_stimulus_set_meta(activations, stimulus_set)
        return activations

    def _extract_texts(self, stimulus_set):
        if 'sentence' in stimulus_set.columns:
            return list(stimulus_set['sentence'].values)
        elif 'text' in stimulus_set.columns:
            return list(stimulus_set['text'].values)
        else:
            raise ValueError(
                f"No text column found in stimulus set. "
                f"Columns: {list(stimulus_set.columns)}")

    def _from_texts_cached(self, texts, layers, stimuli_identifier=None):
        if self._backbone_id and stimuli_identifier:
            return self._from_texts_stored(
                identifier=self._backbone_id,
                stimuli_identifier=stimuli_identifier,
                layers=layers,
                texts=texts,
            )
        return self._from_texts(texts, layers, stimuli_identifier)

    @store_xarray(identifier_ignore=['texts', 'layers'], combine_fields={'layers': 'layer'})
    def _from_texts_stored(self, identifier, layers, stimuli_identifier, texts):
        return self._from_texts(texts, layers, stimuli_identifier)

    def _from_texts(self, texts, layers, stimuli_identifier=None):
        if not layers:
            raise ValueError("No layers passed to retrieve activations from")

        logger.info('Running text stimuli')
        layer_activations = self._get_activations_batched(texts, layers)
        logger.info('Packaging into assembly')
        return self._package(layer_activations, texts)

    def _get_activations_batched(self, texts, layers):
        layer_activations = None
        total = len(texts)

        for batch_start in tqdm(range(0, total, self._batch_size),
                                unit_scale=self._batch_size, desc="text activations"):
            batch_end = min(batch_start + self._batch_size, total)
            batch_texts = texts[batch_start:batch_end]

            batch_activations = self.get_activations(batch_texts, layers)

            if layer_activations is None:
                layer_activations = OrderedDict()
                for layer_name, layer_output in batch_activations.items():
                    final_shape = (total,) + layer_output.shape[1:]
                    layer_activations[layer_name] = np.empty(
                        final_shape, dtype=layer_output.dtype)

            for layer_name, layer_output in batch_activations.items():
                layer_activations[layer_name][batch_start:batch_end] = layer_output

        return layer_activations

    def get_activations(self, texts, layer_names):
        """Run forward pass on a batch of texts, extract layer activations.

        Args:
            texts: List of strings.
            layer_names: List of layer name strings.

        Returns:
            OrderedDict of layer_name -> numpy array (batch, hidden_dim).
        """
        import torch

        tokens = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors='pt',
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        layer_results = OrderedDict()
        hooks = []

        for layer_name in layer_names:
            layer = self._get_layer(layer_name)
            hook = self._register_hook(layer, layer_name, layer_results)
            hooks.append(hook)

        self._model.eval()
        with torch.no_grad():
            self._model(**tokens)

        for hook in hooks:
            hook.remove()

        # Aggregate over sequence dimension
        for layer_name in layer_results:
            act = layer_results[layer_name]  # (batch, seq_len, hidden) or (batch, hidden)
            if act.ndim == 3:
                if self._layer_aggregation == 'last_token':
                    # Use attention_mask to find the actual last token per sequence
                    if 'attention_mask' in tokens:
                        lengths = tokens['attention_mask'].sum(dim=1).cpu().numpy()
                        act = np.stack([act[i, int(lengths[i]) - 1, :]
                                       for i in range(act.shape[0])])
                    else:
                        act = act[:, -1, :]
                elif self._layer_aggregation == 'mean_tokens':
                    if 'attention_mask' in tokens:
                        mask = tokens['attention_mask'].cpu().numpy()
                        mask_expanded = mask[:, :, np.newaxis].astype(act.dtype)
                        act = (act * mask_expanded).sum(axis=1) / mask.sum(
                            axis=1, keepdims=True)
                    else:
                        act = act.mean(axis=1)
            layer_results[layer_name] = act

        return layer_results

    def _get_layer(self, layer_name):
        module = self._model
        for part in layer_name.split('.'):
            module = getattr(module, part, None)
            if module is None:
                raise ValueError(
                    f"Layer '{layer_name}' not found at part '{part}'")
        return module

    def _register_hook(self, layer, layer_name, target_dict):
        def hook_fn(_module, _input, output, name=layer_name):
            if isinstance(output, (tuple, list)):
                output = output[0]
            target_dict[name] = output.cpu().data.numpy()

        return layer.register_forward_hook(hook_fn)

    def _package(self, layer_activations, texts):
        """Package layer activations into NeuroidAssembly."""
        layer_assemblies = []
        for layer_name, activations in layer_activations.items():
            # activations shape: (n_texts, hidden_dim)
            n_texts, n_features = activations.shape
            neuroid_id = [f"{self._identifier}.{layer_name}.{i}"
                          for i in range(n_features)]
            assembly = NeuroidAssembly(
                activations,
                coords={
                    'stimulus_id': ('presentation', list(range(n_texts))),
                    'neuroid_id': ('neuroid', neuroid_id),
                    'neuroid_num': ('neuroid', list(range(n_features))),
                    'model': ('neuroid', [self._identifier] * n_features),
                    'layer': ('neuroid', [layer_name] * n_features),
                },
                dims=['presentation', 'neuroid'],
            )
            layer_assemblies.append(assembly)

        if len(layer_assemblies) == 1:
            return layer_assemblies[0]

        # Merge layers along neuroid dimension
        merged = np.concatenate([a.values for a in layer_assemblies], axis=1)
        nonneuroid_coords = {
            coord: (dims, values) for coord, dims, values in walk_coords(layer_assemblies[0])
            if set(dims) != {'neuroid'}
        }
        neuroid_coords = {
            coord: [dims, values] for coord, dims, values in walk_coords(layer_assemblies[0])
            if set(dims) == {'neuroid'}
        }
        for layer_assembly in layer_assemblies[1:]:
            for coord in neuroid_coords:
                neuroid_coords[coord][1] = np.concatenate(
                    (neuroid_coords[coord][1], layer_assembly[coord].values))

        neuroid_coords = {coord: (dv[0], dv[1]) for coord, dv in neuroid_coords.items()}
        return NeuroidAssembly(
            merged,
            coords={**nonneuroid_coords, **neuroid_coords},
            dims=layer_assemblies[0].dims,
        )

    def _attach_stimulus_set_meta(self, assembly, stimulus_set):
        """Attach stimulus set metadata as coordinates on the presentation dim."""
        stimulus_ids = list(stimulus_set['stimulus_id'].values)

        # Replace the placeholder stimulus_ids with actual ones
        assembly = assembly.assign_coords(
            stimulus_id=('presentation', stimulus_ids))

        # Add all other stimulus set columns as coordinates
        for column in stimulus_set.columns:
            if column == 'stimulus_id':
                continue
            assembly = assembly.assign_coords(
                {column: ('presentation', list(stimulus_set[column].values))})

        return assembly
