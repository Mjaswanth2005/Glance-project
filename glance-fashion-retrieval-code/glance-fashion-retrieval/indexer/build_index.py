"""
Part A entrypoint: raw image folder -> searchable index.

Usage:
    python -m indexer.build_index --images data/test --out data/index --limit 200

Produces, under --out:
    embeddings.npy   (N, D) float32 whole-image CLIP embeddings, L2-normalized
    index.faiss      FAISS index built from embeddings.npy
    attributes.db    SQLite DB of per-image structured garment/environment/style attributes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from indexer.attribute_extractor import AttributeExtractor
from indexer.embedder import ClipEmbedder
from indexer.localizer import GarmentLocalizer
from indexer.storage import build_faiss_index, init_db, save_faiss_index, save_record


def iter_images(folder: Path, limit: int | None):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if limit:
        paths = paths[:limit]
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of raw images")
    ap.add_argument("--out", required=True, help="output folder for the index")
    ap.add_argument("--limit", type=int, default=None, help="cap number of images (debugging)")
    ap.add_argument("--index-type", default="flat", choices=["flat", "ivf_pq"])
    args = ap.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = iter_images(images_dir, args.limit)
    if not paths:
        print(f"no images found in {images_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"indexing {len(paths)} images from {images_dir}")

    print("loading models (CLIP + OWL-ViT)...")
    embedder = ClipEmbedder()
    localizer = GarmentLocalizer()
    extractor = AttributeExtractor(localizer=localizer, embedder=embedder)

    conn = init_db(out_dir / "attributes.db")
    embeddings = []

    t0 = time.time()
    for i, path in enumerate(tqdm(paths, desc="extracting")):
        image = Image.open(path).convert("RGB")
        record = extractor.extract(image, image_id=path.name)
        embeddings.append(record.image_embedding)
        save_record(conn, record, faiss_idx=i)

        if i % 200 == 0:
            conn.commit()  # periodic commit so a crash doesn't lose everything

    conn.commit()
    conn.close()

    embeddings = np.stack(embeddings).astype(np.float32)
    np.save(out_dir / "embeddings.npy", embeddings)

    index = build_faiss_index(embeddings, index_type=args.index_type)
    save_faiss_index(index, out_dir / "index.faiss")

    dt = time.time() - t0
    print(f"done: {len(paths)} images in {dt:.1f}s ({dt / len(paths):.2f}s/image)")
    print(f"index written to {out_dir}")


if __name__ == "__main__":
    main()
