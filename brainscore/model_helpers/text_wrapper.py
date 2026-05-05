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

    VALID_AGGREGATIONS = ('last_token', 'mean_tokens', 'per_token')

    def __init__(self, model, tokenizer, identifier=None, backbone_id=None,
                 layer_aggregation='last_token', max_length=512,
                 batch_size=32):
        import torch
        if layer_aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"layer_aggregation must be one of {self.VALID_AGGREGATIONS}; "
                f"got {layer_aggregation!r}."
            )
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
        if self._layer_aggregation == 'per_token':
            return self._get_per_token_activations(texts, layers)

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

    # ── per_token path: chunked, per-input forward passes ──────────────

    def _get_per_token_activations(self, texts, layers):
        """Run forward pass per text, chunking inputs longer than
        ``max_length``. Returns an OrderedDict of layer -> (n_texts,
        T_max, hidden) ndarray with NaN padding past each input's
        token count, plus a parallel ``_token_lengths`` ndarray of
        actual token counts so callers can mask.

        Inputs are processed one at a time in this mode (no
        cross-input batching) because per-input T varies and chunking
        decisions are per-input. Chunks within a single input ARE
        processed in batches of ``batch_size``.
        """
        per_input_layer_arrays = OrderedDict((layer, []) for layer in layers)
        token_lengths = np.empty(len(texts), dtype=np.int64)

        for i, text in enumerate(tqdm(
                texts, desc="text per-token", unit="text")):
            per_layer = self._extract_one_text_per_token(text, layers)
            sample_layer = next(iter(per_layer.values()))
            token_lengths[i] = sample_layer.shape[0]
            for layer_name, arr in per_layer.items():
                per_input_layer_arrays[layer_name].append(arr)

        if len(token_lengths) == 0:
            raise ValueError("per_token TextWrapper got zero texts")
        t_max = int(token_lengths.max())

        layer_outputs = OrderedDict()
        for layer_name, arrs in per_input_layer_arrays.items():
            hidden = arrs[0].shape[1]
            padded = np.full(
                (len(texts), t_max, hidden), np.nan, dtype=arrs[0].dtype)
            for i, arr in enumerate(arrs):
                padded[i, :arr.shape[0], :] = arr
            layer_outputs[layer_name] = padded

        # Stash per-input token counts so _package can attach as a
        # presentation-axis coord for benchmark code that needs to mask
        # NaN padding.
        layer_outputs['_token_lengths'] = token_lengths
        return layer_outputs

    def _extract_one_text_per_token(self, text, layers):
        """Tokenize ``text``, split into ``max_length``-sized chunks if
        needed, run a forward pass per chunk, concat per-token activations
        along the time axis. Returns OrderedDict[layer] -> (T, H) array.
        """
        # Tokenize to get the full token id sequence, no truncation.
        encoded = self._tokenizer(
            text, add_special_tokens=False, return_tensors='pt',
            truncation=False, padding=False,
        )
        input_ids = encoded['input_ids'][0]
        n_tokens = int(input_ids.shape[0])

        if n_tokens <= self._max_length:
            return self._run_per_token_chunks([text], layers)

        # Chunk: re-encode each chunk as a string so the regular forward
        # path (with the model's expected special tokens) is reused. The
        # tokenizer's decode() round-trips token ids → text reasonably for
        # standard subword tokenizers; for tokenizers with non-invertible
        # mappings, callers can override max_length to skip chunking.
        chunk_texts = []
        for start in range(0, n_tokens, self._max_length):
            end = min(start + self._max_length, n_tokens)
            chunk_text = self._tokenizer.decode(input_ids[start:end])
            chunk_texts.append(chunk_text)

        # Run chunks in batches and concat their per-token outputs.
        chunk_layer_arrays = OrderedDict((layer, []) for layer in layers)
        for batch_start in range(0, len(chunk_texts), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(chunk_texts))
            batch = chunk_texts[batch_start:batch_end]
            per_layer = self._run_per_token_chunks(batch, layers)
            for layer_name, arr in per_layer.items():
                chunk_layer_arrays[layer_name].append(arr)

        out = OrderedDict()
        for layer_name, arrs in chunk_layer_arrays.items():
            out[layer_name] = np.concatenate(arrs, axis=0)
        return out

    def _run_per_token_chunks(self, batch_texts, layer_names):
        """Single forward pass returning per-token activations concatenated
        across the input batch (so multi-chunk inputs reduce to one (T_total,
        H) array per layer regardless of how chunks were grouped into the
        forward call).
        """
        import torch

        tokens = self._tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors='pt',
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        layer_results: OrderedDict = OrderedDict()
        hooks = []
        for layer_name in layer_names:
            layer = self._get_layer(layer_name)
            hooks.append(self._register_hook(layer, layer_name, layer_results))

        self._model.eval()
        try:
            with torch.no_grad():
                self._model(**tokens)
        finally:
            for h in hooks:
                h.remove()

        # Trim each row to its real (non-padded) token count and
        # concatenate along the time axis.
        attention_mask = tokens.get('attention_mask')
        out = OrderedDict()
        for layer_name, act in layer_results.items():
            if act.ndim != 3:
                raise ValueError(
                    f"per_token mode needs (B, T, H) hook output for layer "
                    f"{layer_name!r}; got rank {act.ndim}."
                )
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).cpu().numpy()
                rows = [act[i, :int(lengths[i]), :] for i in range(act.shape[0])]
            else:
                rows = [act[i] for i in range(act.shape[0])]
            out[layer_name] = np.concatenate(rows, axis=0)
        return out

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
        """Package layer activations into NeuroidAssembly.

        Handles two output shapes:
          - 2D (n_texts, hidden_dim) for last_token / mean_tokens →
            dims=(presentation, neuroid)
          - 3D (n_texts, t_max, hidden_dim) for per_token →
            dims=(presentation, time_bin, neuroid). NaN-padded past each
            input's token_lengths[i]; an extra ``token_length``
            presentation coord is attached so callers can mask.
        """
        token_lengths = layer_activations.pop('_token_lengths', None)
        layer_assemblies = []
        for layer_name, activations in layer_activations.items():
            if activations.ndim == 3:
                layer_assemblies.append(self._pack_3d(
                    activations, layer_name, len(texts), token_lengths))
                continue
            # 2D legacy path
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

        # Multi-layer: concat along the neuroid axis. The per_token 3D
        # path uses xarray.concat (handles the time_bin axis correctly);
        # the 2D path keeps the original numpy concat for bit-for-bit BC.
        if 'time_bin' in layer_assemblies[0].dims:
            import xarray as xr
            return xr.concat(layer_assemblies, dim='neuroid')

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

    def _pack_3d(self, activations, layer_name, n_texts, token_lengths):
        """Package per-token (n_texts, t_max, hidden) into a
        (presentation, time_bin, neuroid) NeuroidAssembly.

        The wrapper's native time grid is token positions (no absolute
        ms). Benchmark code that needs ms timestamps attaches them via
        word-onset alignment as a separate coord.
        """
        _, t_max, n_features = activations.shape
        neuroid_id = [f"{self._identifier}.{layer_name}.{i}"
                      for i in range(n_features)]
        # Note: no time_bin_id coord. The wrapper's native time grid is
        # token positions (ordinal). Absolute ms are dataset-specific
        # (word-onset alignments) and the benchmark attaches them later.
        # Construction-time presentation coord is ONLY stimulus_id —
        # multiple presentation coords here would build a MultiIndex via
        # gather_indexes that conflicts with downstream assign_coords on
        # stimulus_id (e.g., from _attach_stimulus_set_meta). We attach
        # token_length AFTER construction as a non-indexed coord.
        coords = {
            'stimulus_id': ('presentation', list(range(n_texts))),
            'neuroid_id': ('neuroid', neuroid_id),
            'neuroid_num': ('neuroid', list(range(n_features))),
            'model': ('neuroid', [self._identifier] * n_features),
            'layer': ('neuroid', [layer_name] * n_features),
        }
        assembly = NeuroidAssembly(
            activations,
            coords=coords,
            dims=['presentation', 'time_bin', 'neuroid'],
        )
        if token_lengths is not None:
            assembly = assembly.assign_coords(
                token_length=('presentation', list(map(int, token_lengths))))
        return assembly

    def _attach_stimulus_set_meta(self, assembly, stimulus_set):
        """Attach stimulus set metadata as coordinates on the presentation dim.

        Resets any existing presentation MultiIndex first so re-assigning
        stimulus_id (and any other column) doesn't collide with a pre-built
        MultiIndex level. The MultiIndex was created by gather_indexes
        during NeuroidAssembly construction and only matters for the 3D
        per_token output where ``token_length`` adds a second presentation
        coord; in the 2D case the reset is a no-op.
        """
        if 'presentation' in assembly.indexes:
            try:
                assembly = assembly.reset_index('presentation')
            except (ValueError, KeyError):
                # Older xarray builds raise on reset_index of a non-MultiIndex
                # presentation dim; safe to ignore — assign_coords below will
                # handle the simple case.
                pass

        stimulus_ids = list(stimulus_set['stimulus_id'].values)
        assembly = assembly.assign_coords(
            stimulus_id=('presentation', stimulus_ids))

        for column in stimulus_set.columns:
            if column == 'stimulus_id':
                continue
            assembly = assembly.assign_coords(
                {column: ('presentation', list(stimulus_set[column].values))})

        return assembly
