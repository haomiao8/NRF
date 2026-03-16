from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class CLIPImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        selected_layers: Sequence[int] = (3, 6),
    ):
        super().__init__()
        self.selected_layers = tuple(selected_layers)
        if len(self.selected_layers) != 2:
            raise ValueError("Current Feature_merger expects exactly two selected CLIP layers.")

        try:
            self.processor = CLIPImageProcessor.from_pretrained(model_name)
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        except OSError as exc:
            raise OSError(
                f"Unable to load CLIP image encoder '{model_name}'. "
                "Ensure the weights are available locally or that the environment has access to Hugging Face."
            ) from exc
        self.output_dim = int(self.vision_model.config.projection_dim)
        self.patch_dim = int(self.vision_model.config.hidden_size)

        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.vision_model.eval()

    def preprocess(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.ndim != 4:
            raise ValueError(f"Expected image batch of shape [B, H, W, C], got {tuple(image_tensor.shape)}")

        image_batch = image_tensor.detach().cpu().numpy()
        processed = self.processor(images=list(image_batch), return_tensors="pt")
        return processed["pixel_values"]

    def forward(self, image_tensor: torch.Tensor):
        pixel_values = self.preprocess(image_tensor).to(next(self.vision_model.parameters()).device)
        outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        early_features = []
        for layer_idx in self.selected_layers:
            hidden = hidden_states[layer_idx + 1]
            early_features.append(hidden[:, 1:, :])

        early_stack = torch.stack(early_features, dim=1)
        higher_feature = outputs.image_embeds
        return early_stack, higher_feature
