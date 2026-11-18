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
        layer_aggregation='mean',
    )
    assembly = text_wrapper(stimulus_set, layers=['encoder.layers.10'])
"""

import logging
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm

from brainscore_core.assembly_builder import concat_neuroid_assemblies
from brainscore_core.extraction_cache import store_xarray
from brainscore.model_helpers._extractors import (
    CachingExtractor, LiveRecording, as_array, attention_mask, checked_aggregation,
    layer_assembly, one_dtype, run_batches,
)


logger = logging.getLogger(__name__)

def align_word_times_to_tokens(word_ids, word_onsets_ms, word_durations_ms):
    """Map per-token ``word_ids`` to per-token (start_ms, end_ms) windows.

    The principled word→token timestamp alignment for a continuous transcript:
    a fast tokenizer reports, for each token, which word it came from
    (``word_ids``; ``None`` for special tokens). Each token inherits its source
    word's ``[onset, onset+duration]`` window. When a word splits into several
    subword tokens, they share the word's window (callers can sub-divide evenly
    if needed — kept simple here). Tokens with ``word_id is None`` (BOS/EOS) get
    ``NaN`` and are bridged from the nearest real word so the axis stays
    monotonic.

    :param word_ids: length-T list; token i → word index, or ``None``.
    :param word_onsets_ms: length-W array of word onset times (ms).
    :param word_durations_ms: length-W array of word durations (ms).
    :returns: ``(start_ms, end_ms)`` float arrays of length T (NaN where a token
        maps to no word and has no neighbour to borrow from).
    """
    onsets = np.asarray(word_onsets_ms, dtype=float)
    durations = np.asarray(word_durations_ms, dtype=float)
    if onsets.shape != durations.shape:
        raise ValueError(
            f"word_onsets_ms {onsets.shape} and word_durations_ms "
            f"{durations.shape} must have the same length")
    T = len(word_ids)
    start = np.full(T, np.nan)
    end = np.full(T, np.nan)
    n_words = len(onsets)
    for i, wid in enumerate(word_ids):
        if wid is None or wid < 0 or wid >= n_words:
            continue
        start[i] = onsets[wid]
        end[i] = onsets[wid] + durations[wid]
    # bridge special-token NaNs so the timeline has no interior gaps. start:
    # forward-fill (carry the previous word's start) then back-fill (leading
    # specials borrow the first real start). end: back-fill then forward-fill
    # (trailing specials borrow the last real end).
    def _ffill(a):
        last = np.nan
        for i in range(T):
            if np.isnan(a[i]):
                a[i] = last
            else:
                last = a[i]

    def _bfill(a):
        nxt = np.nan
        for i in range(T - 1, -1, -1):
            if np.isnan(a[i]):
                a[i] = nxt
            else:
                nxt = a[i]

    _ffill(start); _bfill(start)
    _bfill(end); _ffill(end)
    return start, end



# Hugging Face states 1e20 or 1e30 as a tokenizer's ``model_max_length`` when it has no limit.
_NO_LIMIT = int(1e20)


def _model_window(model, tokenizer, identifier):
    """The most tokens the model can read: the smallest real limit among the tokenizer's
    ``model_max_length`` and ``max_position_embeddings`` of the network's text configuration
    (a text longer than the decoder's positions cannot be read). None when there is no
    tokenizer (nothing is cut)."""
    if tokenizer is None:
        return None
    config = getattr(model, 'config', None)
    config = getattr(config, 'get_text_config', lambda: config)()   # a composite model's text part
    limits = [limit for limit in (getattr(tokenizer, 'model_max_length', None),
                                  getattr(config, 'max_position_embeddings', None))
              if isinstance(limit, int) and 0 < limit < _NO_LIMIT]
    if limits:
        return min(limits)
    raise ValueError(f'{identifier}: neither the tokenizer nor the model configuration states a token '
                     'limit, so a long text cannot be cut at the model window. State max_length=<tokens> '
                     'for the TextWrapper in the registration.')


class TextWrapper(CachingExtractor):
    """Text model wrapper symmetric with PytorchWrapper.

    Args:
        model: A PyTorch text model (e.g., CLIPTextModel, GPT2Model).
        tokenizer: A HuggingFace tokenizer.
        identifier: Model identifier for display and the ``model`` coordinate.
            Defaults to the class name. It is not part of the cache key: two
            registrations with the same weights and settings share one entry.
        layer_aggregation: How to reduce (batch, seq_len, hidden) to (batch, hidden).
            'last' for causal models, 'mean' for encoders.
        max_length: The most tokens of one text; a longer text keeps its newest
            tokens. Default (None): the model's own window (``_model_window``).
        batch_size: Number of texts per forward pass.
        provenance: Optional model provenance (revision, dtype, settings). When
            given, a copy is set as ``attrs['provenance']`` on every returned
            assembly. Like every constructor setting it is part of the
            activation-cache fingerprint (``CACHE_FIELDS``), so registrations
            with different provenance never share a cache entry. Default
            (None): no attr is added.
    """

    # The version of what this class extracts: a part of the activation cache key
    # (``_extractors.cache_config``), read from this class body only. Raise it by
    # hand when a change can move the extracted values, also through code it calls.
    _VERSION = 1

    # Recording several layers in one call gives, for each layer, exactly what a
    # call for that layer alone gives, so the regions of one session share one
    # call (brainscore._model.channels).
    independent_layers = True

    VALID_AGGREGATIONS = ('last', 'mean', 'sequence')

    def __init__(self, model, tokenizer, identifier=None,
                 layer_aggregation='last', max_length=None,
                 batch_size=32, provenance=None, device=None):
        import torch
        checked_aggregation(layer_aggregation, self.VALID_AGGREGATIONS)
        self._model = model
        self._tokenizer = tokenizer
        # Truncate the OLDEST tokens, not the newest. With passage context the
        # text handed to a model is the running prefix, so it is the tail that
        # names the row being scored: right truncation would silently return a
        # representation of some earlier sentence, and `last` would read a
        # token from the middle of the passage. The legacy language path makes
        # the same choice (`truncation_side='left'` in its huggingface helper),
        # so matching it is also what keeps native and adapter scoring aligned.
        try:
            self._tokenizer.truncation_side = 'left'
        except AttributeError:                       # exotic tokenizers
            pass
        self._layer_aggregation = layer_aggregation
        self._max_length = max_length if max_length is not None else _model_window(
            model, tokenizer, identifier or model.__class__.__name__)
        self._batch_size = batch_size
        from brainscore.model_helpers._device import place_model
        self._model, self._device = place_model(self._model, device)

        self._identifier = identifier or model.__class__.__name__
        self._provenance = None if provenance is None else deepcopy(dict(provenance))

    _INPUTS = 'texts'

    def _table_inputs(self, stimulus_set):
        """The texts of a table: each row's sentence, with its passage context (``context_id``)."""
        from brainscore_core.text import contextualized_texts
        if 'context_id' in stimulus_set.columns and self._layer_aggregation == 'sequence':
            raise ValueError('context_id passage grouping requires last or mean; '
                             'sequence inputs must supply their complete text explicitly.')
        return contextualized_texts(stimulus_set)

    def _key_inputs(self, texts):
        return texts

    def _finish(self, assembly):
        if self._provenance is not None:
            assembly.attrs['provenance'] = deepcopy(self._provenance)
        return assembly

    CACHE_FIELDS = (
        '_tokenizer',
        '_layer_aggregation',
        '_max_length',
        '_batch_size',
        '_provenance',
    )

    @store_xarray(identifier_ignore=['texts', 'layers'], combine_fields={'layers': 'layer'})
    def _from_texts_stored(self, layers, stimuli_identifier, texts, extraction_fingerprint):
        return self._from_texts(texts, layers, stimuli_identifier)

    def _from_texts(self, texts, layers, stimuli_identifier=None):
        if not layers:
            raise ValueError("No layers passed to retrieve activations from")

        logger.info('Running text stimuli')
        layer_activations = self._get_activations_batched(texts, layers)
        logger.info('Packaging into assembly')
        return self._package(layer_activations, texts)

    def _get_activations_batched(self, texts, layers):
        if self._layer_aggregation == 'sequence':
            return self._get_per_token_activations(texts, layers)
        return run_batches(texts, self._batch_size, lambda batch: self.get_activations(batch, layers),
                           'text activations')

    # ── sequence path: chunked, per-input forward passes ──────────────

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
        word_ids_per_text = []

        for i, text in enumerate(tqdm(
                texts, desc="text per-token", unit="text")):
            per_layer, word_ids = self._extract_one_text_per_token(text, layers)
            sample_layer = next(iter(per_layer.values()))
            token_lengths[i] = sample_layer.shape[0]
            word_ids_per_text.append(word_ids)
            for layer_name, arr in per_layer.items():
                per_input_layer_arrays[layer_name].append(arr)

        if len(token_lengths) == 0:
            raise ValueError("sequence TextWrapper got zero texts")
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
        # NaN padding, and per-input token→word alignment for timestamps.
        layer_outputs['_token_lengths'] = token_lengths
        layer_outputs['_word_ids'] = word_ids_per_text
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
            return self._run_per_token_chunks([text], layers, want_word_ids=True)

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

        # Run chunks in batches and concat their per-token outputs. Chunking
        # re-encodes via decode(), which breaks token→word alignment, so
        # word_ids are unavailable for chunked (very long) inputs.
        chunk_layer_arrays = OrderedDict((layer, []) for layer in layers)
        for batch_start in range(0, len(chunk_texts), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(chunk_texts))
            batch = chunk_texts[batch_start:batch_end]
            per_layer, _ = self._run_per_token_chunks(batch, layers)
            for layer_name, arr in per_layer.items():
                chunk_layer_arrays[layer_name].append(arr)

        out = OrderedDict()
        for layer_name, arrs in chunk_layer_arrays.items():
            out[layer_name] = np.concatenate(arrs, axis=0)
        return out, None

    def _run_per_token_chunks(self, batch_texts, layer_names, want_word_ids=False):
        """Single forward pass returning per-token activations concatenated
        across the input batch (so multi-chunk inputs reduce to one (T_total,
        H) array per layer regardless of how chunks were grouped into the
        forward call).

        Returns ``(out, word_ids)``. ``word_ids`` is the token→word index list
        for the single-text case (``len(batch_texts) == 1`` and a fast
        tokenizer that exposes ``word_ids``), trimmed to the real token count;
        otherwise ``None``.
        """
        import torch

        encoding = self._tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors='pt',
        )
        tokens = {k: v.to(self._device) for k, v in encoding.items()}

        self._model.eval()
        with LiveRecording(self._model, self._model, layer_names, as_array) as recording, torch.no_grad():
            self._model(**tokens)
            layer_results, _ = recording.take()

        # Trim each row to its real (non-padded) token count and
        # concatenate along the time axis.
        attention_mask = tokens.get('attention_mask')
        out = OrderedDict()
        for layer_name, act in layer_results.items():
            if act.ndim != 3:
                raise ValueError(
                    f"sequence mode needs (B, T, H) hook output for layer "
                    f"{layer_name!r}; got rank {act.ndim}."
                )
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).cpu().numpy()
                rows = [act[i, :int(lengths[i]), :] for i in range(act.shape[0])]
            else:
                rows = [act[i] for i in range(act.shape[0])]
            out[layer_name] = np.concatenate(rows, axis=0)

        # Capture token→word alignment for the single-text path (the timestamp
        # source). Only fast tokenizers expose word_ids(); degrade to None.
        word_ids = None
        if want_word_ids and len(batch_texts) == 1:
            try:
                wid = encoding.word_ids(0)
            except (ValueError, AttributeError, TypeError):
                wid = None
            if wid is not None:
                if attention_mask is not None:
                    real = int(attention_mask[0].sum().item())
                    wid = list(wid)[:real]
                word_ids = list(wid)
        return out, word_ids

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

        self._model.eval()
        with LiveRecording(self._model, self._model, layer_names, as_array, self._reduce,
                           attention_mask) as recording, torch.no_grad():
            try:
                self._model(**tokens)
            except torch.OutOfMemoryError as error:
                raise torch.OutOfMemoryError(
                    f'{self._identifier}: a padded batch of {len(texts)} texts, the longest '
                    f'{tokens["input_ids"].shape[-1]} tokens (limit {self._max_length}), ran out of '
                    f'accelerator memory. State a smaller max_length (or batch_size) for the '
                    f'TextWrapper in the registration.') from error
            return recording.take()

    def _reduce(self, layer_results, mask, _row=None):
        """Reduce each ``(batch, tokens, hidden)`` output to ``(batch, hidden)`` over the
        tokens ``mask`` (the attention mask of the pass) keeps; other outputs stay.
        With no mask every position counts and the last position is the last token,
        as before, also for a right-padded batch given without its mask."""
        for layer_name in layer_results:
            act = layer_results[layer_name]
            if act.ndim == 3:
                if self._layer_aggregation == 'last':
                    if mask is not None:     # the last kept position, with left or right padding
                        last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
                        act = act[np.arange(act.shape[0]), last]
                    else:
                        act = act[:, -1, :]
                elif self._layer_aggregation == 'mean':
                    if mask is not None:
                        mask_expanded = mask[:, :, np.newaxis].astype(act.dtype)
                        act = (act * mask_expanded).sum(axis=1) / mask.sum(axis=1, keepdims=True)
                    else:
                        act = act.mean(axis=1)
            layer_results[layer_name] = act
        return layer_results

    def record(self, layers, stimuli=None):
        """Record ``layers`` while another caller runs the model (e.g. generation).

        Returns a ``LiveRecording``; its ``take()`` gives ``{layer: assembly}`` for the
        first forward pass since the previous take (of a generating model, the prompt
        pass), reduced with the ``attention_mask`` of that pass and packaged by the
        functions of ``__call__``. ``sequence`` is refused before any pass, and a
        pass of several prompts before it runs.
        """
        if self._layer_aggregation == 'sequence':
            raise NotImplementedError(
                'A recording during a task supports last or mean, not sequence.')

        def inputs(args, kwargs):
            shown = kwargs.get('input_ids')
            shown = kwargs.get('inputs_embeds', args[0] if args else None) if shown is None else shown
            if shown is not None and shown.shape[0] != 1:
                raise ValueError(f'A recording during a task keeps one prompt per pass; this pass '
                                 f'shows {shown.shape[0]}.')
            return attention_mask(args, kwargs)

        return LiveRecording(self._model, self._model, layers, as_array, lambda activity, mask, _row:
                             self._layers(self._reduce(activity, mask), [None]), inputs)

    def layer_names(self):
        """Layer paths this extractor can record, relative to the module it wraps."""
        return [name for name, _ in self._model.named_modules(remove_duplicate=False) if name]

    def _package(self, layer_activations, texts):
        """Package layer activations into NeuroidAssembly.

        Handles two output shapes:
          - 2D (n_texts, hidden_dim) for last / mean →
            dims=(presentation, neuroid)
          - 3D (n_texts, t_max, hidden_dim) for sequence →
            dims=(presentation, time_bin, neuroid). NaN-padded past each
            input's token_lengths[i]; an extra ``token_length``
            presentation coord is attached so callers can mask.
        """
        layer_assemblies = list(self._layers(layer_activations, texts).values())
        if len(layer_assemblies) == 1:
            return layer_assemblies[0]
        one_dtype(layer_activations)

        # Multi-layer: concat along the neuroid axis. The sequence 3D
        # path uses xarray.concat (handles the time_bin axis correctly);
        # the 2D path keeps the original numpy concat for bit-for-bit BC.
        if 'time_bin' in layer_assemblies[0].dims:
            return concat_neuroid_assemblies(
                layer_assemblies, strategy='xarray')
        return concat_neuroid_assemblies(layer_assemblies, strategy='manual')

    def _layers(self, layer_activations, texts):
        """One assembly per layer, in order: ``(presentation, neuroid)``, or
        ``(presentation, time_bin, neuroid)`` for sequence activations."""
        token_lengths = layer_activations.pop('_token_lengths', None)
        word_ids = layer_activations.pop('_word_ids', None)
        return OrderedDict(
            (layer, self._pack_3d(activations, layer, len(texts), token_lengths, word_ids)
             if activations.ndim == 3 else layer_assembly(self._identifier, layer, activations))
            for layer, activations in layer_activations.items())

    def _pack_3d(self, activations, layer_name, n_texts, token_lengths,
                 word_ids=None):
        """Package per-token (n_texts, t_max, hidden) into a
        (presentation, time_bin, neuroid) NeuroidAssembly.

        Attaches an ordinal ``token_position`` time grid (always) so the output
        is unambiguously time-resolved, and — when a fast tokenizer reported
        token→word alignment — a ``word_id`` coord (presentation × time_bin) plus
        the raw per-text ``word_ids`` in ``attrs`` so :meth:`attach_timestamps`
        can convert word onsets into per-token ``time_bin_start_ms`` /
        ``time_bin_end_ms``.
        """
        t_max = activations.shape[1]
        # Construction-time presentation coord is ONLY stimulus_id —
        # multiple presentation coords here would build a MultiIndex via
        # gather_indexes that conflicts with downstream assign_coords on
        # stimulus_id (e.g., from attach_stimulus_set). Extra coords are
        # attached AFTER construction as non-indexed coords.
        assembly = layer_assembly(self._identifier, layer_name, activations)
        if token_lengths is not None:
            assembly = assembly.assign_coords(
                token_length=('presentation', list(map(int, token_lengths))))
        # explicit ordinal time grid (token positions)
        assembly = assembly.assign_coords(
            token_position=('time_bin', list(range(t_max))))
        # token→word alignment, when a fast tokenizer provided it. Stored as a
        # (presentation, time_bin) int coord (-1 = special token or padding) so
        # it survives netCDF caching; attach_timestamps reconstructs from it.
        if word_ids is not None and any(w is not None for w in word_ids):
            wid_arr = np.full((n_texts, t_max), -1, dtype=int)
            for i, w in enumerate(word_ids):
                if w is None:
                    continue
                for j, v in enumerate(w[:t_max]):
                    wid_arr[i, j] = -1 if v is None else int(v)
            assembly = assembly.assign_coords(
                word_id=(('presentation', 'time_bin'), wid_arr))
        return assembly

    @staticmethod
    def attach_timestamps(assembly, word_onsets_ms, word_durations_ms):
        """Add per-token ``time_bin_start_ms`` / ``time_bin_end_ms`` coords to a
        per-token assembly by aligning word onsets to tokens via the captured
        ``word_id`` coord (see :func:`align_word_times_to_tokens`).

        For a single-presentation assembly pass flat per-word arrays; for several
        presentations pass a list of per-word arrays (one per presentation). The
        assembly must come from a ``sequence`` extraction with a fast tokenizer
        on a non-chunked input (so the ``word_id`` coord is present).
        """
        if 'word_id' not in assembly.coords:
            raise ValueError(
                "assembly carries no word_id coord; per-token timestamps require "
                "a sequence extraction with a fast tokenizer on a non-chunked "
                "input. (Raise max_length so the transcript isn't chunked, or "
                "use a fast tokenizer.)")
        n_pres = assembly.sizes['presentation']
        t_max = assembly.sizes['time_bin']
        wid = np.asarray(assembly['word_id'].values).reshape(n_pres, t_max)
        token_length = (np.asarray(assembly['token_length'].values)
                        if 'token_length' in assembly.coords
                        else np.full(n_pres, t_max, dtype=int))

        def _per_text(x):
            # single flat 1-D numeric array for a 1-presentation assembly
            if n_pres == 1 and np.ndim(np.asarray(x)[0] if len(x) else 0) == 0:
                return [np.asarray(x, dtype=float)]
            return [np.asarray(xi, dtype=float) for xi in x]

        onsets = _per_text(word_onsets_ms)
        durations = _per_text(word_durations_ms)
        start = np.full((n_pres, t_max), np.nan)
        end = np.full((n_pres, t_max), np.nan)
        for i in range(n_pres):
            L = int(token_length[i])
            # reconstruct per-token word_ids (-1 → None for special tokens)
            w = [None if v < 0 else int(v) for v in wid[i, :L]]
            if not any(v is not None for v in w):
                continue
            s, e = align_word_times_to_tokens(w, onsets[i], durations[i])
            start[i, :len(s)] = s
            end[i, :len(e)] = e
        return assembly.assign_coords(
            time_bin_start_ms=(('presentation', 'time_bin'), start),
            time_bin_end_ms=(('presentation', 'time_bin'), end))

