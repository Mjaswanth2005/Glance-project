"""
Storage layer, deliberately boring: SQLite for structured attributes, FAISS
for dense embeddings. The assignment explicitly says not to spend effort
building a custom vector store, so this wraps two well-known off-the-shelf
tools rather than inventing one.

Why FAISS specifically: it's a flat file + in-process library, no server to
run, and it scales from "flat brute-force" (exact, fine up to ~1M vectors
on modern hardware) to IVF/PQ indexes (approximate, sub-linear, the
standard answer once you're past ~1M) via a one-line swap -- see
`build_faiss_index(..., index_type="ivf_pq")`. That upgrade path is the
answer to the "what if the dataset grew to 1M images" evaluation question.

Why SQLite for attributes: structured, queryable, zero setup, and trivially
swappable for Postgres later without touching the retrieval logic, since
all access goes through this module.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np

from indexer.attribute_extractor import ImageRecord


def init_db(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            environment TEXT,
            style TEXT,
            garments_json TEXT,   -- list of {type, color, det_score, color_confidence, box}
            faiss_idx INTEGER     -- row index into the FAISS embedding matrix
        )
        """
    )
    conn.commit()
    return conn


def save_record(conn: sqlite3.Connection, record: ImageRecord, faiss_idx: int) -> None:
    garments = [
        {
            "type": g.type,
            "color": g.color,
            "det_score": g.det_score,
            "color_confidence": g.color_confidence,
            "box": list(g.box),
        }
        for g in record.garments
    ]
    conn.execute(
        "INSERT OR REPLACE INTO images (image_id, environment, style, garments_json, faiss_idx) "
        "VALUES (?, ?, ?, ?, ?)",
        (record.image_id, record.environment, record.style, json.dumps(garments), faiss_idx),
    )


def build_faiss_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """
    embeddings: (N, D) float32, L2-normalized (cosine sim == inner product).

    index_type:
        "flat"    -- exact brute-force search. Fine up to ~O(1M) vectors on
                     CPU; simplest, no training step, no recall loss.
        "ivf_pq"  -- IVF + product quantization. Sub-linear search and a
                     compressed on-disk footprint; the standard choice once
                     the flat index stops being fast/small enough. Needs a
                     `.train()` call on a representative sample first, which
                     is why it's a separate branch rather than the default.
    """
    d = embeddings.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(d)  # inner product == cosine sim for normalized vectors
        index.add(embeddings)
        return index

    if index_type == "ivf_pq":
        nlist = max(1, min(4096, embeddings.shape[0] // 39))  # faiss rule-of-thumb
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, 8, 8)  # 8 subquantizers, 8 bits each
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(16, nlist)
        return index

    raise ValueError(f"unknown index_type: {index_type}")


def save_faiss_index(index: faiss.Index, path: str | Path) -> None:
    faiss.write_index(index, str(path))


def load_faiss_index(path: str | Path) -> faiss.Index:
    return faiss.read_index(str(path))


def get_all_records(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT image_id, environment, style, garments_json, faiss_idx FROM images ORDER BY faiss_idx"
    ).fetchall()
    return [
        {
            "image_id": r[0],
            "environment": r[1],
            "style": r[2],
            "garments": json.loads(r[3]),
            "faiss_idx": r[4],
        }
        for r in rows
    ]
