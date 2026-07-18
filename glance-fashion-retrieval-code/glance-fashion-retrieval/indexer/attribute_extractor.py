"""
Turns one raw image into a structured record:

    {
        "image_id": "003d41dd...jpg",
        "garments": [
            {"type": "raincoat", "color": "yellow", "det_score": 0.31, "box": [...]},
            ...
        ],
        "environment": "urban street",
        "environment_scores": {...},
        "style": "casual weekend outfit",
        "style_scores": {...},
    }

This is the "compositional" half of the index: garments is a list, each
entry independently carrying its own type + color, which is what lets the
retriever tell "red tie, white shirt" apart from "white tie, red shirt".
`environment` / `style` are whole-image, CLIP zero-shot classifications --
diffuse scene properties rather than objects, so they don't get a box.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from indexer.color_extractor import dominant_color
from indexer.embedder import ClipEmbedder
from indexer.localizer import GarmentLocalizer
from indexer.taxonomy import ENVIRONMENT_TAGS, STYLE_TAGS


@dataclass
class GarmentRecord:
    type: str
    color: str
    det_score: float
    color_confidence: float
    box: tuple[int, int, int, int]


@dataclass
class ImageRecord:
    image_id: str
    garments: list[GarmentRecord] = field(default_factory=list)
    environment: str = ""
    environment_scores: dict[str, float] = field(default_factory=dict)
    style: str = ""
    style_scores: dict[str, float] = field(default_factory=dict)
    image_embedding: np.ndarray | None = None  # for the semantic/context fallback


class AttributeExtractor:
    def __init__(self, localizer: GarmentLocalizer | None = None, embedder: ClipEmbedder | None = None):
        # Both are expensive to construct (model downloads), so allow injection
        # for testing / sharing a single embedder across localizer+extractor.
        self.localizer = localizer or GarmentLocalizer()
        self.embedder = embedder or ClipEmbedder()

        # Precompute text embeddings for environment/style prompts once.
        self._env_text_emb = self.embedder.embed_text(ENVIRONMENT_TAGS)
        self._style_text_emb = self.embedder.embed_text(STYLE_TAGS)

    def extract(self, image: Image.Image, image_id: str) -> ImageRecord:
        record = ImageRecord(image_id=image_id)

        # --- per-garment structured attributes ---
        boxes = self.localizer.detect(image)
        for b in boxes:
            crop = self.localizer.crop(image, b)
            color_name, color_conf = dominant_color(crop)
            record.garments.append(
                GarmentRecord(
                    type=b.label,
                    color=color_name,
                    det_score=b.score,
                    color_confidence=color_conf,
                    box=b.box,
                )
            )

        # --- whole-image semantic / context signal ---
        img_emb = self.embedder.embed_image(image)
        record.image_embedding = img_emb

        env_scores = img_emb @ self._env_text_emb.T
        record.environment_scores = dict(zip(ENVIRONMENT_TAGS, env_scores.tolist()))
        record.environment = ENVIRONMENT_TAGS[int(np.argmax(env_scores))]

        style_scores = img_emb @ self._style_text_emb.T
        record.style_scores = dict(zip(STYLE_TAGS, style_scores.tolist()))
        record.style = STYLE_TAGS[int(np.argmax(style_scores))]

        return record
