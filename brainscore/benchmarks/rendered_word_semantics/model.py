"""Connect a Hugging Face DINO encoder to the existing UMI vision wrapper."""
from typing import Any

from PIL import Image
import torch

from brainscore.model_helpers.vision_wrapper import VisionWrapper
from brainscore_core.model_interface import BrainScoreModel


class _ClassToken(torch.nn.Module):
    """Expose the final class token as a named recording point."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.summary = torch.nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.encoder(pixel_values=pixel_values)
        # No projection or normalization: match the paper's final features.
        return self.summary(output.last_hidden_state[:, 0])


def build_vision_subject(encoder: torch.nn.Module, processor: Any, *,
                         identifier: str, batch_size: int = 16) -> BrainScoreModel:
    """Wrap a DINO encoder; images enter through process(), not a text channel."""
    if encoder.config.model_type not in {'dinov2', 'dinov3_vit'}:
        raise ValueError('This experiment supports DINOv2 and DINOv3 ViT encoders')
    model = _ClassToken(encoder.eval())

    def preprocess(image_paths: list[str]) -> torch.Tensor:
        # The processor applies the checkpoint's resize and normalization.
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        return processor(images=images, return_tensors='pt')['pixel_values']

    wrapper = VisionWrapper(
        model=model,
        preprocessing=preprocess,
        identifier=identifier,
        kind='frame',
        batch_size=batch_size,
    )
    return BrainScoreModel(
        identifier=identifier,
        model=model,
        preprocessors={'vision': preprocess},
        activations_model=wrapper,
        region_layer_map={'word_representation': 'summary'},
    )
