"""
Part B entrypoint: natural language query -> top-k image ids.

Two-stage retrieve-then-rerank, which is also the answer to "would this
scale to 1M images":
  Stage 1 (FAISS ANN search): fast, approximate, whole-image semantic
      similarity narrows 1M -> a few hundred candidates in milliseconds.
  Stage 2 (structured rerank): the expensive, precise, compositional
      scoring in scorer.py only ever runs on that small candidate pool,
      not the full dataset -- so it scales with candidate_pool, not with
      dataset size.

On the current ~3.2k image dataset, stage 1 and "search everything" are
practically the same cost, but the architecture is written for the 1M
case from the start rather than bolted on later.
"""
from __future__ import annotations

from pathlib import Path

from indexer.embedder import ClipEmbedder
from indexer.storage import get_all_records, init_db, load_faiss_index
from retriever.query_parser import QueryParser
from retriever.scorer import score_structured


class Retriever:
    def __init__(self, index_dir: str | Path, alpha_garment: float = 0.8, alpha_vibe: float = 0.3):
        index_dir = Path(index_dir)
        self.faiss_index = load_faiss_index(index_dir / "index.faiss")
        conn = init_db(index_dir / "attributes.db")  # CREATE TABLE IF NOT EXISTS is a no-op here
        self.records = get_all_records(conn)  # ordered by faiss_idx, so records[i] <-> faiss row i
        conn.close()

        self.embedder = ClipEmbedder()
        self.parser = QueryParser()

        # Fusion weights: how much the structured/compositional score counts
        # vs. the semantic/vibe score, in the two query regimes.
        self.alpha_garment = alpha_garment  # weight on structured score when the query names garments
        self.alpha_vibe = alpha_vibe        # weight on structured (tag-bonus) score for pure vibe queries

    def search(self, query: str, top_k: int = 10, candidate_pool: int = 200) -> list[dict]:
        parsed = self.parser.parse(query)
        query_emb = self.embedder.embed_text(query).astype("float32")

        pool = min(candidate_pool, len(self.records))
        sims, idxs = self.faiss_index.search(query_emb[None, :], pool)
        sims, idxs = sims[0], idxs[0]

        alpha = self.alpha_garment if parsed.has_garment_mention else self.alpha_vibe

        results = []
        for sim, idx in zip(sims, idxs):
            if idx < 0:
                continue
            record = self.records[idx]
            structured = score_structured(parsed, record)
            # FAISS inner product on normalized vectors == cosine sim, roughly in [-1, 1];
            # clip to [0, 1] so it combines sensibly with the structured score's [0, 1] range.
            semantic = max(0.0, float(sim))
            final = alpha * structured + (1 - alpha) * semantic
            results.append(
                {
                    "image_id": record["image_id"],
                    "score": final,
                    "structured_score": structured,
                    "semantic_score": semantic,
                    "environment": record["environment"],
                    "style": record["style"],
                    "garments": record["garments"],
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]
