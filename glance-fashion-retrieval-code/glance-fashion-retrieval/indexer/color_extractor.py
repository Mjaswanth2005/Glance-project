"""
Dominant-color extraction for a garment crop.

Why not just ask CLIP "is this red"? CLIP's color judgments are noisy for
fine-grained fashion colors (it regularly confuses maroon/red, navy/black,
teal/green) because color words are entangled with thousands of other
associations in its embedding space. A direct pixel-based method is cheaper
and more precise for this narrow sub-task, so we only lean on CLIP for the
things it's actually good at (semantics), not for the things classical CV
already solves well (color).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from indexer.taxonomy import COLOR_PALETTE

_PALETTE_NAMES = list(COLOR_PALETTE.keys())
_PALETTE_RGB = np.array(list(COLOR_PALETTE.values()), dtype=np.float32)


def _nearest_color_name(rgb: np.ndarray) -> str:
    dists = np.linalg.norm(_PALETTE_RGB - rgb, axis=1)
    return _PALETTE_NAMES[int(np.argmin(dists))]


def _skin_mask(pixels: np.ndarray) -> np.ndarray:
    """Heuristic skin-tone filter for crop cleanup."""
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    return (
        (r > 95)
        & (g > 40)
        & (b > 20)
        & ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 15)
        & (np.abs(r - g) > 15)
        & (r > g)
        & (r > b)
    )


def dominant_color(
    crop: np.ndarray,
    n_clusters: int = 3,
    ignore_extremes: bool = True,
) -> tuple[str, float]:
    """
    Args:
        crop: HxWx3 uint8 RGB array (a garment region crop, not the whole image).
        n_clusters: k-means clusters to try; we keep the largest cluster.
        ignore_extremes: drop near-white/near-black pixels before clustering.
            These are usually background, shadow, or specular highlight
            rather than the garment's actual color, and they otherwise
            dominate the cluster count on studio/runway shots.

    Returns:
        (color_name, confidence) where confidence is the fraction of
        (non-extreme) pixels belonging to the winning cluster.
    """
    pixels = crop.reshape(-1, 3).astype(np.float32)

    if len(pixels) == 0:
        return "unknown", 0.0

    # Remove likely skin pixels first when the crop still has enough
    # non-skin area to describe the garment.
    skin = _skin_mask(pixels)
    if (~skin).sum() >= max(50, int(0.35 * len(pixels))):
        pixels = pixels[~skin]

    if ignore_extremes:
        brightness = pixels.mean(axis=1)
        mask = (brightness > 15) & (brightness < 245)
        # Keep white/black garments if the crop is mostly extreme pixels;
        # otherwise drop extreme pixels that usually come from background,
        # glare, or shadow.
        if mask.sum() >= max(50, int(0.35 * len(pixels))):
            pixels = pixels[mask]

    if len(pixels) == 0:
        return "unknown", 0.0

    # Subsample for speed -- dominant color doesn't need every pixel.
    if len(pixels) > 2000:
        idx = np.random.choice(len(pixels), 2000, replace=False)
        pixels = pixels[idx]

    k = min(n_clusters, len(pixels))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(pixels)
    counts = np.bincount(km.labels_)
    winner = np.argmax(counts)
    confidence = float(counts[winner]) / len(pixels)

    name = _nearest_color_name(km.cluster_centers_[winner])
    return name, confidence


if __name__ == "__main__":
    # Quick smoke test against a plain synthetic crop.
    synthetic = np.tile(np.array([230, 126, 34], dtype=np.uint8), (40, 40, 1))
    print(dominant_color(synthetic))
