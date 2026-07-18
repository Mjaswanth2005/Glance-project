"""
Whole-image / text embeddings via CLIP.

Unlike the localizer, this operates on the *full* image. This is
deliberate: CLIP's global embedding is genuinely strong at scene-level
semantics -- "modern office", "park bench", "weekend vibe" -- because
those concepts really are diffuse, whole-image properties, not something
you can crop out. We lean on CLIP for exactly the part of the problem it's
good at, and lean on the localizer + color extractor (see
attribute_extractor.py) for the part it's bad at (compositional, per-
garment attributes).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_MODEL_NAME = "openai/clip-vit-base-patch32"


class ClipEmbedder:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        self.model.eval()

    def _normalize_features(self, feats: torch.Tensor | object) -> torch.Tensor:
        if isinstance(feats, torch.Tensor):
            return feats
        if hasattr(feats, "pooler_output") and getattr(feats, "pooler_output") is not None:
            return getattr(feats, "pooler_output")
        if hasattr(feats, "last_hidden_state"):
            return getattr(feats, "last_hidden_state")[:, 0]
        raise TypeError(f"Unsupported feature output type: {type(feats)!r}")

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = self._normalize_features(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0]

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            feats = self._normalize_features(feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def embed_text(self, text: str | list[str]) -> np.ndarray:
        texts = [text] if isinstance(text, str) else text
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        feats = self._normalize_features(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        result = feats.cpu().numpy()
        return result[0] if isinstance(text, str) else result
