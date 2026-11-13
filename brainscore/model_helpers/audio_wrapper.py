"""
AudioWrapper — fifth extractor wrapper, symmetric with TextWrapper
and VideoWrapper for audio waveform inputs.

Wraps an audio backbone (Wav2Vec2, HuBERT, Wav2Vec-Bert, SeamlessM4T, Whisper,
AudioMAE, etc.), handles waveform loading + processor invocation, runs a
forward pass with hook-based layer extraction, batches across clips, caches
via ``@store_xarray``, and packages into a NeuroidAssembly that slots into the
standard Brain-Score scoring pipeline. The default ``mean_time`` aggregation
returns ``(presentation, neuroid)``; ``time_series`` additionally preserves a
``time_bin`` dimension.

Design:
- Constructor: ``AudioWrapper(model, processor, identifier, backbone_id=...)``
- ``__call__(stimuli, layers)`` — accepts a StimulusSet with ``audio_path``
  column (or a list of paths).
- Layer aggregation modes:
    ``mean_time`` (default): reduce (B, T, H) → (B, H). Matches frame-based
        extraction patterns used by TextWrapper for classification benchmarks.
    ``time_series``: keep (B, T, H), produces a temporal assembly suitable
        for naturalistic benchmarks that want per-frame alignment.
- Waveform loading uses torchaudio when available, falling back to librosa.
    The ``audio_loader`` constructor argument allows injection of a custom
    loader (for tests or non-standard file formats).
- Sampling rate: the wrapper target rate is taken from the processor's
    ``sampling_rate`` attribute; audio is resampled on load so the processor
    sees the rate it expects.
- Caching: ``backbone_id`` (default = identifier) keys the ``@store_xarray``
    cache so two registrations sharing the same audio backbone (e.g.,
    TRIBEv2 and a standalone Wav2Vec-Bert registration) reuse activations.

Long clips are split into consecutive, non-overlapping chunks of at most
``max_duration_sec`` and recombined per clip. Multi-channel audio is mixed down
to mono before the processor.
"""

import logging
from collections import OrderedDict
from typing import Callable, List, Optional

import numpy as np

from brainscore_core.assembly_builder import (
    concat_neuroid_assemblies,
    make_assembly,
    make_layer_neuroid_coords,
)
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.extraction_cache import store_xarray, file_inputs
from brainscore.model_helpers import _extractors
from brainscore.model_helpers._extractors import (
    LiveRecording, as_numpy, attach_stimulus_set, attention_mask, cache_signature, one_dtype, run_batches,
)


logger = logging.getLogger(__name__)


# ── Default audio loader ────────────────────────────────────────────

def _default_audio_loader(
    audio_path: str,
    target_sample_rate: int,
) -> np.ndarray:
    """Load a waveform, downmix to mono, resample to ``target_sample_rate``.

    Prefers torchaudio (faster; avoids librosa's numba dep when it's absent).
    Falls back to librosa. Returns a 1-D float32 numpy array in [-1, 1].
    """
    try:
        import torchaudio
        wav, sr = torchaudio.load(audio_path)  # (channels, samples)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        return wav.squeeze(0).float().numpy()
    except ImportError:
        pass
    try:
        import librosa
        wav, _ = librosa.load(audio_path, sr=target_sample_rate, mono=True)
        return wav.astype(np.float32)
    except ImportError as e:
        raise ImportError(
            "AudioWrapper default loader requires torchaudio or librosa. "
            "Install one of them, or pass a custom audio_loader to "
            "AudioWrapper(...)."
        ) from e


# ── Main wrapper ────────────────────────────────────────────────────

class AudioWrapper:
    """Wraps an audio backbone for Brain-Score activation extraction.

    Args:
        model: PyTorch audio model (has hook-compatible named children).
        processor: HuggingFace processor / feature extractor that accepts
            ``(audio=<1-D waveform>, sampling_rate=<int>)`` and returns a
            dict of input tensors. Must expose a ``sampling_rate``
            attribute (or be queryable with ``processor.feature_extractor
            .sampling_rate`` — Wav2Vec2 convention). If neither is present,
            ``target_sample_rate`` must be passed explicitly.
        identifier: Human-readable model id (used for logs + as the cache
            key when ``backbone_id`` is not set).
        backbone_id: Optional shared cache key. When multiple registrations
            use the same audio backbone weights, set them all to the same
            backbone_id so ``@store_xarray`` reuses the cached activations.
            Defaults to ``identifier``.
        target_sample_rate: Override the processor's expected sample rate.
            Usually unnecessary — read from ``processor.sampling_rate``.
        layer_aggregation: ``'mean_time'`` (default) reduces (B, T, H) →
            (B, H); ``'time_series'`` keeps the temporal axis.
        max_duration_sec: Maximum non-overlapping chunk length, in seconds.
            Longer clips are processed in consecutive chunks and recombined;
            None disables chunking.
        batch_size: Number of clips per forward pass. Default 4 — audio
            models are memory-hungry at long T.
        audio_loader: Optional callable ``(path, target_sr) -> 1-D float32
            waveform``. Defaults to torchaudio → librosa fallback.
    """

    VALID_AGGREGATIONS = ('mean_time', 'time_series')

    def __init__(
        self,
        model,
        processor,
        identifier: Optional[str] = None,
        backbone_id: Optional[str] = None,
        target_sample_rate: Optional[int] = None,
        layer_aggregation: str = 'mean_time',
        max_duration_sec: Optional[float] = 60.0,
        batch_size: int = 4,
        audio_loader: Optional[Callable[[str, int], np.ndarray]] = None,
        device: Optional[str] = None,
    ):
        import torch

        if layer_aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"layer_aggregation must be one of {self.VALID_AGGREGATIONS}, "
                f"got {layer_aggregation!r}"
            )

        self._model = model
        self._processor = processor
        self._layer_aggregation = layer_aggregation
        self._max_duration_sec = max_duration_sec
        self._batch_size = batch_size
        self._audio_loader = audio_loader or _default_audio_loader

        from brainscore.model_helpers._device import select_device
        self._device = select_device(device)
        self._model = self._model.to(self._device)

        self._target_sample_rate = (
            target_sample_rate
            or getattr(processor, 'sampling_rate', None)
            or getattr(
                getattr(processor, 'feature_extractor', None),
                'sampling_rate', None,
            )
        )
        if self._target_sample_rate is None:
            raise ValueError(
                "AudioWrapper could not determine target sampling rate from "
                "the processor. Pass target_sample_rate explicitly."
            )

        self._identifier = identifier or model.__class__.__name__
        self._backbone_id = backbone_id or self._identifier
        # Per-step duration in ms, derived from the first forward pass.
        # Used to attach time_bin_start_ms / time_bin_end_ms coords to
        # time_series outputs. Stays None until at least one batch has
        # been processed.
        self._step_ms: Optional[float] = None

    # ── Identity ────────────────────────────────────────────────────

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def backbone_id(self) -> str:
        return self._backbone_id

    @property
    def target_sample_rate(self) -> int:
        return self._target_sample_rate

    # ── Call site ───────────────────────────────────────────────────

    def __call__(
        self, stimuli, layers, stimuli_identifier: Optional[str] = None,
        **kwargs,
    ) -> NeuroidAssembly:
        """Extract audio activations.

        Args:
            stimuli: StimulusSet with an ``audio_path`` column (or
                ``audio_file_name``), or a list of audio file paths.
            layers: Layer name strings (dotted paths into ``self._model``).
            stimuli_identifier: Cache key component; falls back to
                ``stimuli.identifier``.
        """
        if isinstance(stimuli, StimulusSet):
            return self._from_stimulus_set(stimuli, layers, stimuli_identifier)
        return self._from_paths_cached(stimuli, layers, stimuli_identifier)

    def _from_stimulus_set(self, stimulus_set, layers, stimuli_identifier=None):
        if stimuli_identifier is None and hasattr(stimulus_set, 'identifier'):
            stimuli_identifier = stimulus_set.identifier
        paths = self._extract_paths(stimulus_set)
        activations = self._from_paths_cached(paths, layers, stimuli_identifier)
        return attach_stimulus_set(activations, stimulus_set)

    @staticmethod
    def _extract_paths(stimulus_set) -> List[str]:
        for col in ('audio_path', 'audio_file_name', 'audio_file', 'filename'):
            if col in stimulus_set.columns:
                return list(stimulus_set[col].values)
        raise ValueError(
            f"No audio column found in stimulus set. Columns: "
            f"{list(stimulus_set.columns)}. Expected one of: audio_path, "
            f"audio_file_name, audio_file, filename."
        )

    # ── Caching boundary ────────────────────────────────────────────

    CACHE_FIELDS = (
        '_processor',
        '_layer_aggregation',
        '_max_duration_sec',
        '_batch_size',
        '_audio_loader',
        '_target_sample_rate',
    )

    def cache_config(self):
        return _extractors.cache_config(self, self.CACHE_FIELDS)

    def _from_paths_cached(self, paths, layers, stimuli_identifier=None):
        signature = cache_signature(self, '_from_paths_stored', file_inputs(paths), stimuli_identifier)
        if signature is None:
            return self._from_paths(paths, layers, stimuli_identifier)
        return self._from_paths_stored(identifier=self._backbone_id, stimuli_identifier=stimuli_identifier,
                                       layers=layers, paths=paths, extraction_fingerprint=signature)

    @store_xarray(identifier_ignore=['paths', 'layers'],
                  combine_fields={'layers': 'layer'})
    def _from_paths_stored(self, identifier, layers, stimuli_identifier, paths,
                           extraction_fingerprint):
        return self._from_paths(paths, layers, stimuli_identifier)

    def _from_paths(self, paths, layers, stimuli_identifier=None):
        if not layers:
            raise ValueError("AudioWrapper requires at least one layer")
        logger.info(f"Running {len(paths)} audio clips through "
                    f"{self._identifier}")
        waveforms = [self._load_waveform(p) for p in paths]

        # Chunk waveforms longer than max_duration_sec so movie-length
        # audio doesn't get silently truncated. Short clips become a
        # single chunk; long clips become N chunks. We track which clip
        # each chunk belongs to so we can recombine per-clip after the
        # forward pass.
        chunks: List[np.ndarray] = []
        chunk_to_clip: List[int] = []
        for clip_idx, wav in enumerate(waveforms):
            for c in self._chunk_waveform(wav):
                chunks.append(c)
                chunk_to_clip.append(clip_idx)

        chunk_activations = self._get_activations_batched(chunks, layers)
        layer_activations = self._recombine_chunks(
            chunk_activations, chunk_to_clip, n_clips=len(waveforms))
        return self._package(layer_activations, paths)

    # ── Waveform loading ────────────────────────────────────────────

    def _load_waveform(self, audio_path: str) -> np.ndarray:
        """Load and resample a waveform. Long clips are NOT truncated here
        — chunking happens later in ``_chunk_waveform`` so the per-clip
        activations cover the full duration."""
        return self._audio_loader(audio_path, self._target_sample_rate)

    def _chunk_waveform(self, wav: np.ndarray) -> List[np.ndarray]:
        """Split a waveform into chunks of at most ``max_duration_sec``.

        - Short clips return a single-element list (no copy).
        - Long clips are split into N consecutive non-overlapping chunks.
          The last chunk may be shorter than ``max_duration_sec``.
        - When ``max_duration_sec`` is None, no chunking happens.

        Replaces the old ``_load_waveform`` truncation: M12-full needs
        full-duration features for movie-length audio, not silent
        cropping at the 60s mark.
        """
        if self._max_duration_sec is None:
            return [wav]
        max_samples = int(self._max_duration_sec * self._target_sample_rate)
        if len(wav) <= max_samples:
            return [wav]
        chunks = []
        for start in range(0, len(wav), max_samples):
            chunks.append(wav[start:start + max_samples])
        return chunks

    def _recombine_chunks(
        self,
        chunk_activations: OrderedDict,
        chunk_to_clip: List[int],
        n_clips: int,
    ) -> OrderedDict:
        """Recombine per-chunk activations back to per-clip arrays.

        - ``mean_time`` (and pooled-output (B, H)): per-clip mean across
          the clip's chunks. (Each chunk already had its time axis
          reduced via attention-masked mean; combining means across
          chunks of similar length is a close enough approximation.)
        - ``time_series``: per-clip concat along the time axis. Returns
          a (n_clips, T_max, H) array NaN-padded for shorter clips so
          downstream packaging gets a regular tensor.

        This collapses the chunk axis but preserves per-clip ordering
        of time. M12-full naturalistic benchmarks consume the result
        directly via temporal_bin alignment to the brain's TR grid.
        """
        # Bucket chunk indices by clip
        clip_chunks: List[List[int]] = [[] for _ in range(n_clips)]
        for chunk_idx, clip_idx in enumerate(chunk_to_clip):
            clip_chunks[clip_idx].append(chunk_idx)

        out: OrderedDict = OrderedDict()
        for layer_name, chunk_arr in chunk_activations.items():
            if chunk_arr.ndim == 2:  # (n_chunks, H) — pooled or mean_time
                out[layer_name] = self._recombine_2d(
                    chunk_arr, clip_chunks, n_clips)
            elif chunk_arr.ndim == 3:  # (n_chunks, T, H) — time_series
                out[layer_name] = self._recombine_3d(
                    chunk_arr, clip_chunks, n_clips)
            else:
                raise ValueError(
                    f"Unexpected chunk activation rank {chunk_arr.ndim} "
                    f"for layer {layer_name!r}; expected 2 or 3."
                )
        return out

    @staticmethod
    def _recombine_2d(chunk_arr, clip_chunks, n_clips):
        """Per-clip mean across chunks."""
        n_features = chunk_arr.shape[1]
        out = np.empty((n_clips, n_features), dtype=chunk_arr.dtype)
        for clip_idx, chunk_indices in enumerate(clip_chunks):
            out[clip_idx] = chunk_arr[chunk_indices].mean(axis=0)
        return out

    @staticmethod
    def _recombine_3d(chunk_arr, clip_chunks, n_clips):
        """Per-clip concat along time, NaN-padded to a common T_max."""
        n_features = chunk_arr.shape[2]
        clip_T = [chunk_arr[idxs].shape[0] * chunk_arr.shape[1]
                  for idxs in clip_chunks]
        # n_chunks_for_clip * chunk_T = total per-clip T
        # Each chunk contributes the same per-chunk T (batched together
        # in the same forward pass), so this is exact.
        t_max = max(clip_T) if clip_T else 0
        out = np.full((n_clips, t_max, n_features), np.nan,
                      dtype=chunk_arr.dtype)
        for clip_idx, chunk_indices in enumerate(clip_chunks):
            concatted = np.concatenate(
                [chunk_arr[i] for i in chunk_indices], axis=0)
            out[clip_idx, :concatted.shape[0], :] = concatted
        return out

    # ── Batched forward pass ────────────────────────────────────────

    def _get_activations_batched(self, waveforms, layers):
        """Batch forward-pass over waveforms; return OrderedDict of per-layer
        arrays shaped (n_clips, ...) depending on layer_aggregation.

        Pre-pads all waveforms to a uniform length so T_hook is constant
        across batches. Without this, two batches with slightly different
        max-input-length (e.g., 48112-sample vs 48128-sample clips, both
        nominally 3 s) produce different output T (150 vs 151 timesteps)
        and the cross-batch buffer slot assignment breaks. The processor
        pads internally anyway; we just lift that to a global pad so the
        per-batch shape is fixed.
        """
        if waveforms:
            max_len = max(len(w) for w in waveforms)
            waveforms = [
                w if len(w) == max_len
                else np.pad(w, (0, max_len - len(w)), mode='constant')
                for w in waveforms
            ]
        return run_batches(waveforms, self._batch_size, lambda batch: self._run_one_batch(batch, layers),
                           'audio activations')

    def _run_one_batch(self, batch_waveforms, layers):
        import torch

        # HF audio processors expect a list of 1-D waveforms (or a numpy
        # stack of equal length). They pad internally to the batch max.
        processed = self._processor(
            batch_waveforms,
            sampling_rate=self._target_sample_rate,
            return_tensors='pt',
            padding=True,
        )
        processed = {k: (v.to(self._device) if torch.is_tensor(v) else v)
                     for k, v in processed.items()}

        self._model.eval()
        with LiveRecording(self._model, self._model, layers, as_numpy, inputs=attention_mask) as recording, \
                torch.no_grad():
            self._model(**processed)
            layer_results, mask = recording.take()

        self._measure_step(layer_results, max(len(w) for w in batch_waveforms))
        return self._aggregate(layer_results, mask)

    def _measure_step(self, layer_results, input_samples):
        """Keep the duration of one output step, measured on the first pass.

        The step is the length of the input the model saw (``input_samples``,
        after padding) over the length of the first time-resolved layer output.
        It is stable across passes when the model has a uniform convolution
        stride (Wav2Vec2, Wav2Vec-Bert, HuBERT), so the first measurement is
        kept. It gives ``time_bin_start_ms`` and ``time_bin_end_ms`` of
        ``time_series`` outputs.
        """
        if self._step_ms is not None:
            return
        for act in layer_results.values():
            if act.ndim == 3 and act.shape[1] > 0:
                self._step_ms = input_samples / self._target_sample_rate * 1000.0 / act.shape[1]
                return

    def _aggregate(self, layer_results, attention_mask):
        """Apply ``layer_aggregation`` to raw hook outputs.

        Hook outputs are assumed to be ``(B, T, H)``; some audio models emit
        ``(B, H)`` from later layers (pooled output) — those pass through
        unchanged regardless of aggregation mode.
        """
        out: OrderedDict = OrderedDict()
        for layer_name, act in layer_results.items():
            if act.ndim == 2:  # (B, H) — no time axis to reduce
                out[layer_name] = act
                continue
            if act.ndim != 3:
                raise ValueError(
                    f"Unexpected activation rank {act.ndim} for layer "
                    f"{layer_name!r}; expected (B, T, H) or (B, H)."
                )
            if self._layer_aggregation == 'time_series':
                out[layer_name] = act  # keep (B, T, H)
                continue
            # mean_time
            if attention_mask is not None:
                # attention_mask is over audio input; hook output may have
                # already downsampled T by the model's conv stride. If T_out
                # differs from the mask's T_in, we fall back to unweighted
                # mean.
                mask_np = attention_mask
                if mask_np.shape[1] == act.shape[1]:
                    mask_expanded = mask_np[:, :, None].astype(act.dtype)
                    act = (act * mask_expanded).sum(axis=1) / np.clip(
                        mask_np.sum(axis=1, keepdims=True), 1, None)
                else:
                    act = act.mean(axis=1)
            else:
                act = act.mean(axis=1)
            out[layer_name] = act
        return out

    def record(self, layers, stimuli=None):
        """Record ``layers`` while another caller runs the model (an answer or a step).

        Returns a ``LiveRecording``; its ``take(row)`` gives ``{layer: assembly}``
        for the first forward pass since the previous take, aggregated with the
        attention mask of that pass and packaged by the functions of ``__call__``;
        the step of ``time_series`` outputs is measured from the clip of ``row``
        (else ``input_values``) when no pass measured it before. ``__call__`` pads
        the clips of ``stimuli`` to the longest and cuts a clip longer than
        ``max_duration_sec`` into chunks, so clips of other lengths or of several
        chunks are refused before any pass. A pass of several clips or longer than
        ``max_duration_sec``, and a later pass of the presentation with other
        input, are refused before they run.
        """
        lengths = {}
        for path in (self._extract_paths(stimuli) if stimuli is not None else []):
            chunks = self._chunk_waveform(self._load_waveform(path))
            if len(chunks) > 1:
                raise ValueError(f'{path} is longer than max_duration_sec={self._max_duration_sec} s; '
                                 f'{self._identifier!r} records it in chunks, one pass per clip cannot.')
            lengths[path] = len(chunks[0])
        if len(set(lengths.values())) > 1:
            raise ValueError(f'The clips have different lengths ({sorted(set(lengths.values()))} samples); '
                             f'{self._identifier!r} pads each to the longest, one pass per clip cannot.')
        longest = self._max_duration_sec and int(self._max_duration_sec * self._target_sample_rate)

        def inputs(args, kwargs):
            given = [kwargs[key] for key in ('input_values', 'input_features') if kwargs.get(key) is not None]
            wave = kwargs.get('input_values')
            if len(given) != 1 or given[0].shape[0] != 1 or (wave is not None and longest and wave.shape[-1] > longest):
                raise ValueError(f'A recording during a task keeps one clip of at most max_duration_sec='
                                 f'{self._max_duration_sec} s per pass, as input_values or input_features; this '
                                 f'pass shows {({k: tuple(v.shape) for k, v in kwargs.items() if hasattr(v, "shape")})}.')
            return attention_mask(args, kwargs), None if wave is None else wave.shape[-1]

        def package(activity, read, row):
            path = self._extract_paths(row.to_frame().T)[0] if row is not None else '?'
            if lengths.get(path, read[1]) is not None:
                self._measure_step(activity, lengths.get(path, read[1]))
            return self._layers(self._aggregate(activity, read[0]), [path])

        return LiveRecording(self._model, self._model, layers, as_numpy, package, inputs, one_input=True)

    def layer_names(self):
        """Layer paths this extractor can record, relative to the module it wraps."""
        return [name for name, _ in self._model.named_modules(remove_duplicate=False) if name]

    # ── Packaging ───────────────────────────────────────────────────

    def _package(self, layer_activations, paths) -> NeuroidAssembly:
        """Assemble layer outputs into a NeuroidAssembly.

        ``mean_time`` → dims = (presentation, neuroid)
        ``time_series`` → dims = (presentation, time_bin, neuroid)
        """
        layer_assemblies = list(self._layers(layer_activations, paths).values())
        if len(layer_assemblies) == 1:
            return layer_assemblies[0]
        one_dtype(layer_activations)
        # Multi-layer: concatenate along the neuroid axis (matches
        # TextWrapper / VideoWrapper convention when callers pass multiple
        # layers at once).
        return concat_neuroid_assemblies(layer_assemblies, strategy='xarray')

    def _layers(self, layer_activations, paths) -> OrderedDict:
        """One assembly per layer, in order: ``(presentation, neuroid)`` for ``(n_clips, hidden)``,
        ``(presentation, time_bin, neuroid)`` for ``(n_clips, T, hidden)``."""
        layers = OrderedDict()
        for layer_name, activations in layer_activations.items():
            if activations.ndim not in (2, 3):
                raise ValueError(
                    f"Packaging expected rank-2 or -3 activations for "
                    f"{layer_name!r}; got rank {activations.ndim}.")
            timed = activations.ndim == 3
            coords = {'stimulus_id': ('presentation', list(range(len(paths)))),  # provisional
                      'stimulus_path': ('presentation', list(paths))}
            if timed:
                coords['time_bin_id'] = ('time_bin', list(range(activations.shape[1])))
            coords.update(make_layer_neuroid_coords(self._identifier, layer_name, activations.shape[-1]))
            # With the duration of one model step (measured on the first pass), the
            # time_bin axis carries absolute ms boundaries, so a benchmark aligns it to
            # the brain's TR grid without the model's downsampling factor.
            if timed and self._step_ms is not None:
                starts = np.arange(activations.shape[1], dtype=np.float64) * self._step_ms
                coords['time_bin_start_ms'] = ('time_bin', starts)
                coords['time_bin_end_ms'] = ('time_bin', starts + self._step_ms)
            layers[layer_name] = make_assembly(activations, coords=coords, dims=[
                'presentation', *(['time_bin'] if timed else []), 'neuroid'])
        return layers


# Recording several layers in one call gives, for each layer, exactly what a
# call for that layer alone gives, so the regions of one session share one call
# (brainscore._model.channels). Set outside the class body: the class source is
# part of the activation cache key, and this line must not change that key.
AudioWrapper.independent_layers = True
