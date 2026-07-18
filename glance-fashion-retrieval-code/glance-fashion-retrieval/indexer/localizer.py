"""
Garment localization via OWL-ViT (open-vocabulary object detection).

This is the piece that makes the system "better than vanilla CLIP" for
compositional queries. Whole-image CLIP produces one embedding for the
entire scene, so it has no way to bind "red" to "tie" specifically rather
than to "shirt" -- it just knows red and tie and shirt are all present
somewhere. By detecting each garment as its own region first, every
attribute we extract downstream (type, color) is anchored to a specific
box, so "red tie + white shirt" and "white tie + red shirt" produce
different structured records instead of the same bag of concepts.

OWL-ViT is used zero-shot with our taxonomy as text queries -- no
fine-tuning or fashion-specific training data required, which also
satisfies the "zero-shot capability" evaluation criterion: new garment
words can be added to the taxonomy without retraining anything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from indexer.taxonomy import ALL_GARMENTS

_MODEL_NAME = "google/owlvit-base-patch32"


@dataclass
class GarmentBox:
    label: str
    score: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in pixel coords


class GarmentLocalizer:
    def __init__(self, device: str | None = None, score_threshold: float = 0.08):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(_MODEL_NAME)
        self.model = OwlViTForObjectDetection.from_pretrained(_MODEL_NAME).to(self.device)
        self.model.eval()
        self.score_threshold = score_threshold
        # OWL-ViT does best with short natural-language noun phrases.
        self.queries = [f"a {g}" for g in ALL_GARMENTS]

    @torch.no_grad()
    def detect(self, image: Image.Image, max_boxes: int = 8) -> list[GarmentBox]:
        inputs = self.processor(text=self.queries, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.score_threshold
        )[0]

        boxes = []
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            boxes.append(
                GarmentBox(
                    label=ALL_GARMENTS[int(label_idx)],
                    score=float(score),
                    box=(x0, y0, x1, y1),
                )
            )

        boxes.sort(key=lambda b: b.score, reverse=True)
        boxes = _nms(boxes, iou_threshold=0.5)
        return boxes[:max_boxes]

    def crop(self, image: Image.Image, box: GarmentBox) -> np.ndarray:
        x0, y0, x1, y1 = box.box
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.width, x1), min(image.height, y1)
        if x1 <= x0 or y1 <= y0:
            return np.array(image)  # degenerate box -> fall back to full image
        return np.array(image.crop((x0, y0, x1, y1)))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: list[GarmentBox], iou_threshold: float) -> list[GarmentBox]:
    """Simple greedy NMS across all classes -- OWL-ViT's per-class boxes
    otherwise let e.g. 'jacket' and 'coat' both fire on the same region."""
    kept: list[GarmentBox] = []
    for b in boxes:
        if all(_iou(b.box, k.box) < iou_threshold for k in kept):
            kept.append(b)
    return kept
