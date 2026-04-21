"""
VLMVisionWrapper — activations_model for VLM vision encoders with
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
        forward_kwargs_map={'grid_thw': 'image_grid_thw'},
        patch_count_fn=lambda out: [int(t*h*w) for t, h, w in out['image_grid_thw']],
    )
    assembly = wrapper(stimulus_set, layers=['blocks.28'])
"""

import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly, walk_coords
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from result_caching import store_xarray


logger = logging.getLogger(__name__)


class VLMVisionWrapper:
    """Activations_model for VLM vision encoders with flattened-patch layouts.

    Args:
        model: The vision sub-module of a VLM (e.g., qwen_model.model.visual).
        processor: A HuggingFace AutoProcessor (or equivalent) whose output
            contains the input tensor and any auxiliary grouping metadata.
        identifier: Model identifier for caching.
        image_input_key: Key in processor output for the main pixel tensor.
            Defaults to 'pixel_values'.
        forward_kwargs_map: Optional dict mapping model forward-kwarg name ->
            processor output key. E.g., {'grid_thw': 'image_grid_thw'} for Qwen.
            Values are moved to device and passed by name to the model.
        patch_count_fn: Callable(processor_output) -> list of patches per image.
            Required for per-image aggregation when the model output is
            patch-flattened. E.g., for Qwen:
                lambda out: [int(t*h*w) for t, h, w in out['image_grid_thw']]
            If None, assumes hook output is already (n_images, ...) shape.
        layer_aggregation: How to reduce per-image patches to one vector.
            'mean_patches' (default), 'first_patch' (CLS-style), or 'none'.
        batch_size: Number of images per forward pass.
    """

    def __init__(self, model, processor, identifier: Optional[str] = None,
                 image_input_key: str = 'pixel_values',
                 forward_kwargs_map: Optional[Dict[str, str]] = None,
                 patch_count_fn: Optional[Callable] = None,
                 layer_aggregation: str = 'mean_patches',
                 batch_size: int = 4):
        import torch
        self._model = model
        self._processor = processor
        self._image_input_key = image_input_key
        self._forward_kwargs_map = forward_kwargs_map or {}
        self._patch_count_fn = patch_count_fn
        self._layer_aggregation = layer_aggregation
        self._batch_size = batch_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)

        self._identifier = identifier or model.__class__.__name__

    @property
    def identifier(self) -> str:
        return self._identifier

    @identifier.setter
    def identifier(self, value: str) -> None:
        self._identifier = value

    def __call__(self, stimuli, layers, stimuli_identifier=None, **kwargs):
        """Extract activations from image stimuli.

        Args:
            stimuli: StimulusSet with 'image_file_name' (or 'image_path') column,
                or list of image paths.
            layers: List of layer name strings to extract from.
            stimuli_identifier: Identifier for caching. None to use stimulus_set.identifier.

        Returns:
            NeuroidAssembly with dims (presentation, neuroid).
        """
        if isinstance(stimuli, StimulusSet):
            return self._from_stimulus_set(stimuli, layers, stimuli_identifier)
        return self._from_paths(stimuli, layers, stimuli_identifier)

    def _from_stimulus_set(self, stimulus_set, layers, stimuli_identifier=None):
        if stimuli_identifier is None and hasattr(stimulus_set, 'identifier'):
            stimuli_identifier = stimulus_set.identifier

        paths = self._extract_paths(stimulus_set)
        activations = self._from_paths_cached(paths, layers, stimuli_identifier)
        activations = self._attach_stimulus_set_meta(activations, stimulus_set)
        return activations

    def _extract_paths(self, stimulus_set) -> List[str]:
        if hasattr(stimulus_set, 'stimulus_paths'):
            stimulus_ids = list(stimulus_set['stimulus_id'].values)
            return [str(stimulus_set.get_stimulus(sid)) for sid in stimulus_ids]
        if 'image_file_name' in stimulus_set.columns:
            return list(stimulus_set['image_file_name'].values)
        if 'image_path' in stimulus_set.columns:
            return list(stimulus_set['image_path'].values)
        raise ValueError(
            f"No image column found in stimulus set. "
            f"Columns: {list(stimulus_set.columns)}")

    def _from_paths_cached(self, paths, layers, stimuli_identifier=None):
        if self._identifier and stimuli_identifier:
            return self._from_paths_stored(
                identifier=self._identifier,
                stimuli_identifier=stimuli_identifier,
                layers=layers,
                paths=paths,
            )
        return self._from_paths(paths, layers, stimuli_identifier)

    @store_xarray(identifier_ignore=['paths', 'layers'],
                  combine_fields={'layers': 'layer'})
    def _from_paths_stored(self, identifier, layers, stimuli_identifier, paths):
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
        layer_activations: Optional[OrderedDict] = None
        total = len(paths)

        for batch_start in tqdm(range(0, total, self._batch_size),
                                unit_scale=self._batch_size,
                                desc="vlm vision activations"):
            batch_end = min(batch_start + self._batch_size, total)
            batch_paths = paths[batch_start:batch_end]

            batch_activations = self.get_activations(batch_paths, layers)

            if layer_activations is None:
                layer_activations = OrderedDict()
                for layer_name, layer_output in batch_activations.items():
                    final_shape = (total,) + layer_output.shape[1:]
                    layer_activations[layer_name] = np.empty(
                        final_shape, dtype=layer_output.dtype)

            for layer_name, layer_output in batch_activations.items():
                layer_activations[layer_name][batch_start:batch_end] = layer_output

        return layer_activations

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

        layer_results: OrderedDict = OrderedDict()
        hooks = []
        for layer_name in layer_names:
            layer = self._get_layer(layer_name)
            hook = self._register_hook(layer, layer_name, layer_results)
            hooks.append(hook)

        self._model.eval()
        with torch.no_grad():
            self._forward(processed)

        for hook in hooks:
            hook.remove()

        # Segment patches into per-image activations and aggregate
        n_images = len(images)
        if self._patch_count_fn is not None:
            patch_counts = self._patch_count_fn(processed)
        else:
            patch_counts = None

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
        image_input = processed[self._image_input_key]

        # Match input dtype to model dtype (FP16 VLMs)
        model_dtype = next(self._model.parameters()).dtype
        if image_input.dtype != model_dtype:
            image_input = image_input.to(model_dtype)

        forward_kwargs = {}
        for model_arg, proc_key in self._forward_kwargs_map.items():
            if proc_key in processed:
                forward_kwargs[model_arg] = processed[proc_key]

        self._model(image_input, **forward_kwargs)

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
        if self._layer_aggregation == 'mean_patches':
            return arr.mean(axis=axis)
        if self._layer_aggregation == 'first_patch':
            return np.take(arr, 0, axis=axis)
        if self._layer_aggregation == 'none':
            return arr
        raise ValueError(f"Unknown layer_aggregation: {self._layer_aggregation}")

    def _get_layer(self, layer_name: str):
        module = self._model
        for part in layer_name.split('.'):
            module = getattr(module, part, None)
            if module is None:
                raise ValueError(
                    f"Layer '{layer_name}' not found at part '{part}'")
        return module

    def _register_hook(self, layer, layer_name: str, target_dict: OrderedDict):
        def hook_fn(_module, _input, output, name=layer_name):
            if isinstance(output, (tuple, list)):
                output = output[0]
            target_dict[name] = output.detach().cpu().float().numpy()
        return layer.register_forward_hook(hook_fn)

    def _package(self, layer_activations: OrderedDict, paths: List[str]) -> NeuroidAssembly:
        """Package layer activations into NeuroidAssembly."""
        layer_assemblies = []
        n_stimuli = len(paths)

        for layer_name, activations in layer_activations.items():
            # Flatten any remaining feature dims
            flat = activations.reshape(n_stimuli, -1)
            n_features = flat.shape[1]
            neuroid_id = [f"{self._identifier}.{layer_name}.{i}"
                          for i in range(n_features)]
            assembly = NeuroidAssembly(
                flat,
                coords={
                    'stimulus_id': ('presentation', list(range(n_stimuli))),
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

        merged = np.concatenate([a.values for a in layer_assemblies], axis=1)
        nonneuroid_coords = {
            coord: (dims, values)
            for coord, dims, values in walk_coords(layer_assemblies[0])
            if set(dims) != {'neuroid'}
        }
        neuroid_coords = {
            coord: [dims, values]
            for coord, dims, values in walk_coords(layer_assemblies[0])
            if set(dims) == {'neuroid'}
        }
        for layer_assembly in layer_assemblies[1:]:
            for coord in neuroid_coords:
                neuroid_coords[coord][1] = np.concatenate(
                    (neuroid_coords[coord][1], layer_assembly[coord].values))

        neuroid_coords = {coord: (dv[0], dv[1])
                          for coord, dv in neuroid_coords.items()}
        return NeuroidAssembly(
            merged,
            coords={**nonneuroid_coords, **neuroid_coords},
            dims=layer_assemblies[0].dims,
        )

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
