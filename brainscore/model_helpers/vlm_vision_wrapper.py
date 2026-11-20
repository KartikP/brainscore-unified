"""
VLMVisionWrapper — extractor for VLM vision encoders with
flattened-patch layouts.

Symmetric with PytorchWrapper and TextWrapper. Handles VLMs whose vision
encoder expects pixel_values of shape (total_patches, patch_dim) with
patch-to-image grouping metadata (e.g., Qwen's image_grid_thw). This layout
is incompatible with PytorchWrapper's assumption of (batch, C, H, W).

Usage (Qwen2.5-VL):
    wrapper = VLMVisionWrapper(
        model=qwen_model.model.visual,
        processor=qwen_processor,
        identifier='qwen2.5-vl-3b-vision',
    )
    assembly = wrapper(stimulus_set, layers=['blocks.28'])
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from brainscore_core.assembly_builder import concat_neuroid_assemblies, make_layer_neuroid_coords
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.extraction_cache import store_xarray
from brainscore.model_helpers.extractors import (
    CachingExtractor, LiveRecording, as_float32, catalog_columns, catalog_paths, checked_aggregation,
    layer_assembly, run_batches,
)


logger = logging.getLogger(__name__)

class VLMVisionWrapper(CachingExtractor):
    """Extractor for VLM vision encoders with flattened-patch layouts.

    The layout is the Qwen-VL one: the encoder takes the processor's
    ``pixel_values``; when the processor also gives ``image_grid_thw``, it is
    passed as ``grid_thw`` and segments the flattened patches per image
    (``t * h * w`` patches each). Without ``image_grid_thw`` each output row is
    one image. Another layout is a subclass that overrides ``_forward`` and
    ``_patch_counts`` (with its own ``_VERSION``).

    Args:
        model: The vision sub-module of a VLM (e.g., qwen_model.model.visual).
        processor: A HuggingFace AutoProcessor (or equivalent) whose output
            contains the input tensor and any auxiliary grouping metadata.
        identifier: Human-readable model identifier (the ``model``
            coordinate); not part of the cache key.
        layer_aggregation: How to reduce per-image patches to one vector.
            'mean' (default), 'first' (CLS-style), or 'sequence' (every patch kept,
            flattened into the features).
        batch_size: Number of images per forward pass.
    """

    # The version of what this class extracts: a part of the activation cache key
    # (``extractors.cache_config``), read from this class body only. Raise it by
    # hand when a change can move the extracted values, also through code it calls.
    _VERSION = 1

    # Recording several layers in one call gives, for each layer, exactly what a
    # call for that layer alone gives, so the regions of one session share one
    # call (brainscore._model.channels).
    independent_layers = True

    VALID_AGGREGATIONS = ('mean', 'first', 'sequence')

    def __init__(self, model, processor, identifier: Optional[str] = None,
                 layer_aggregation: str = 'mean',
                 batch_size: int = 4, device: Optional[str] = None):
        import torch
        self._model = model
        self._processor = processor
        self._layer_aggregation = checked_aggregation(layer_aggregation, self.VALID_AGGREGATIONS)
        self._batch_size = batch_size
        from brainscore.model_helpers.device import place_model
        self._model, self._device = place_model(self._model, device)

        self._identifier = identifier or model.__class__.__name__

    def _table_inputs(self, stimulus_set) -> List[str]:
        return self._extract_paths(stimulus_set)

    def _columns(self, assembly, stimulus_set):
        return self._attach_stimulus_set_meta(assembly, stimulus_set)

    def _extract_paths(self, stimulus_set) -> List[str]:
        if hasattr(stimulus_set, 'stimulus_paths'):
            stimulus_ids = list(stimulus_set['stimulus_id'].values)
            return [str(stimulus_set.get_stimulus(sid)) for sid in stimulus_ids]
        paths = catalog_paths(stimulus_set, 'vision:image')
        if paths is None:
            raise ValueError(f"No image column found in stimulus set: expected one of "
                             f"{catalog_columns('vision:image')} (or stimulus_paths); "
                             f"got {list(stimulus_set.columns)}.")
        return paths

    CACHE_FIELDS = (
        '_processor',
        '_layer_aggregation',
        '_batch_size',
    )

    @store_xarray(identifier_ignore=['paths', 'layers'],
                  combine_fields={'layers': 'layer'})
    def _from_paths_stored(self, layers, stimuli_identifier, paths, extraction_fingerprint):
        return self._from_paths(paths, layers, stimuli_identifier)

    def _from_paths(self, paths, layers, stimuli_identifier=None):
        if not layers:
            raise ValueError("No layers passed to retrieve activations from")

        logger.info(f'Running {len(paths)} image stimuli through VLM vision encoder')
        layer_activations = self._get_activations_batched(paths, layers)
        logger.info('Packaging into assembly')
        return self._package(layer_activations, paths)

    def _get_activations_batched(self, paths, layers):
        """Process images in batches, accumulate per-image activations per layer."""
        return run_batches(paths, self._batch_size, lambda batch: self.get_activations(batch, layers),
                           'vlm vision activations')

    def get_activations(self, image_paths: List[str], layer_names: List[str]) -> OrderedDict:
        """Run forward pass on a batch of images, extract per-image layer activations.

        Args:
            image_paths: List of image file paths.
            layer_names: List of layer name strings (relative to self._model).

        Returns:
            OrderedDict of layer_name -> numpy array (n_images, features).
        """
        import torch

        images = [Image.open(p).convert('RGB') for p in image_paths]

        # Invoke the processor on the image batch. VLM processors typically
        # require a text placeholder + image list. Subclasses can override
        # _processor_call if they need a different calling convention.
        processed = self._processor_call(images)
        processed = {k: (v.to(self._device) if torch.is_tensor(v) else v)
                     for k, v in processed.items()}

        self._model.eval()
        with LiveRecording(self._model, self._model, layer_names, as_float32) as recording, torch.no_grad():
            self._forward(processed)
            layer_results, _ = recording.take()
        return self._per_image(layer_results, len(images), self._patch_counts(processed))

    def _patch_counts(self, processed) -> Optional[List[int]]:
        """Patches per image from the processor's ``image_grid_thw``; None (each output row is an
        image) without it."""
        grid = processed.get('image_grid_thw')
        return None if grid is None else [int(t * h * w) for t, h, w in grid]

    def _per_image(self, layer_results, n_images, patch_counts):
        """Segment each layer's patches into per-image activations and aggregate them."""
        for layer_name in layer_results:
            layer_results[layer_name] = self._aggregate_per_image(
                layer_results[layer_name], n_images, patch_counts)
        return layer_results

    def _processor_call(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Call the processor to produce model-ready inputs.

        Default: Qwen-style messages-based call. Override for other VLMs.
        """
        # For Qwen-style VLMs, build a messages list with the image placeholder.
        # The processor needs text input; use apply_chat_template for a minimal
        # user message containing the images.
        messages = [{"role": "user",
                     "content": [{"type": "image", "image": img} for img in images]}]
        if hasattr(self._processor, 'apply_chat_template'):
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            out = self._processor(text=[text], images=images,
                                  return_tensors='pt', padding=True)
        else:
            # Fallback: direct processor call (e.g., CLIPProcessor-style)
            out = self._processor(images=images, return_tensors='pt')
        return dict(out)

    def _forward(self, processed: Dict[str, Any]) -> None:
        """Run the vision encoder forward pass with auxiliary kwargs."""
        image_input = processed['pixel_values']

        # Match input dtype to model dtype (FP16 VLMs)
        model_dtype = next(self._model.parameters()).dtype
        if image_input.dtype != model_dtype:
            image_input = image_input.to(model_dtype)

        grid = processed.get('image_grid_thw')
        self._model(image_input, **({} if grid is None else {'grid_thw': grid}))

    def _aggregate_per_image(self, activations: np.ndarray, n_images: int,
                             patch_counts: Optional[List[int]]) -> np.ndarray:
        """Reduce patch-flattened activations to one vector per image.

        Args:
            activations: numpy array, typically (total_patches, ...) or (n_images, ...).
            n_images: Expected number of images in this batch.
            patch_counts: List of patches per image. None if activations are
                already per-image.

        Returns:
            numpy array of shape (n_images, features).
        """
        # If already per-image (n_images on dim 0), aggregate spatial dims if any
        if patch_counts is None or activations.shape[0] == n_images:
            if activations.ndim == 3:
                # (n_images, n_patches, features) → aggregate middle dim
                return self._reduce_axis(activations, axis=1)
            if activations.ndim == 2:
                return activations
            return activations.reshape(n_images, -1)

        # Patch-flattened layout: activations.shape[0] == sum(patch_counts)
        expected_total = sum(patch_counts)
        if activations.shape[0] != expected_total:
            raise ValueError(
                f"Patch-flattened activation shape mismatch: "
                f"expected {expected_total} patches (sum of {patch_counts}), "
                f"got {activations.shape[0]}")

        per_image = []
        start = 0
        for count in patch_counts:
            segment = activations[start:start + count]
            per_image.append(self._reduce_axis(segment[np.newaxis], axis=1)[0])
            start += count
        return np.stack(per_image, axis=0)

    def _reduce_axis(self, arr: np.ndarray, axis: int) -> np.ndarray:
        """Reduce one axis per self._layer_aggregation."""
        if self._layer_aggregation == 'mean':
            return arr.mean(axis=axis)
        if self._layer_aggregation == 'first':
            return np.take(arr, 0, axis=axis)
        return arr                                     # 'sequence'

    def record(self, layers, stimuli=None):
        """Record ``layers`` while another caller runs the model (e.g. generation).

        Returns a ``LiveRecording``; its ``take()`` gives ``{layer: assembly}`` for
        the first forward pass since the previous take, segmented and aggregated
        per image by the functions of ``__call__``: the ``grid_thw`` the pass
        received gives the patches of each image. One image gives ``(presentation, neuroid)``; a pass over several images (a
        trial showing a sequence) gives one step per image, ``(presentation, part,
        neuroid)``, in the order shown. A pass without ``grid_thw`` gives one output
        row per image when its input keeps images apart (more than two axes); a
        pass of flattened patches (two axes) without ``grid_thw`` is refused before
        it runs, as is a later pass of the same presentation with other images.
        """
        def inputs(args, kwargs):
            counts = self._patch_counts({'image_grid_thw': kwargs.get('grid_thw')})
            if counts is not None:
                return [int(c) for c in counts], len(counts)
            shown = args[0] if args else kwargs['pixel_values']
            if shown.ndim == 2:
                raise ValueError(f'This pass shows flattened patches {tuple(shown.shape)} without grid_thw '
                                 f'(the image_grid_thw of the processor), so the patches of each image are '
                                 f'unknown; give the encoder grid_thw.')
            return None, shown.shape[0]

        def package(activity, read, _row):
            per_image = self._per_image(activity, read[1], read[0])
            if read[1] == 1:
                return self._layers(per_image, [None])
            return {layer: self._package_parts(layer, values) for layer, values in per_image.items()}

        return LiveRecording(self._model, self._model, layers, as_float32, package, inputs, one_input=True)

    def _package_parts(self, layer, values):
        """One trial's images of ``layer`` as ``(presentation=1, part, neuroid)``, in the order shown."""
        values = values.reshape(values.shape[0], -1)
        return NeuroidAssembly(values[None], dims=['presentation', 'part', 'neuroid'], coords={
            'stimulus_id': ('presentation', ['?']), 'part_index': ('part', list(range(values.shape[0]))),
            **make_layer_neuroid_coords(self._identifier, layer, values.shape[1])})

    def layer_names(self):
        """Layer paths this extractor can record, relative to the module it wraps."""
        return [name for name, _ in self._model.named_modules(remove_duplicate=False) if name]

    def _package(self, layer_activations: OrderedDict, paths: List[str]) -> NeuroidAssembly:
        """Package layer activations into NeuroidAssembly; extra feature axes are flattened."""
        layer_assemblies = list(self._layers(layer_activations, paths).values())
        if len(layer_assemblies) == 1:
            return layer_assemblies[0]
        return concat_neuroid_assemblies(layer_assemblies, strategy='manual')

    def _layers(self, layer_activations, paths) -> OrderedDict:
        """One ``(presentation, neuroid)`` assembly per layer, in order."""
        return OrderedDict((layer, layer_assembly(self._identifier, layer, activations.reshape(len(paths), -1)))
                           for layer, activations in layer_activations.items())

    def _attach_stimulus_set_meta(self, assembly, stimulus_set) -> NeuroidAssembly:
        """Attach stimulus set metadata as coordinates on the presentation dim."""
        stimulus_ids = list(stimulus_set['stimulus_id'].values)
        assembly = assembly.assign_coords(
            stimulus_id=('presentation', stimulus_ids))
        for column in stimulus_set.columns:
            if column == 'stimulus_id':
                continue
            assembly = assembly.assign_coords(
                {column: ('presentation', list(stimulus_set[column].values))})
        return assembly

