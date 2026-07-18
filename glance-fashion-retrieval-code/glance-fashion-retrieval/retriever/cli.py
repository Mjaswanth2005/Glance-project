"""
Usage:
    python -m retriever.cli "A red tie and a white shirt in a formal setting." --index data/index --top-k 5
"""
from __future__ import annotations

import argparse
import json

from retriever.search import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="natural language search query")
    ap.add_argument("--index", default="data/index", help="folder produced by indexer.build_index")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--candidate-pool", type=int, default=200)
    ap.add_argument("--json", action="store_true", help="print raw JSON instead of a formatted table")
    args = ap.parse_args()

    retriever = Retriever(args.index)
    results = retriever.search(args.query, top_k=args.top_k, candidate_pool=args.candidate_pool)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f'\nquery: "{args.query}"\n')
    for rank, r in enumerate(results, 1):
        garments = ", ".join(f"{g['color']} {g['type']}" for g in r["garments"]) or "(none detected)"
        print(
            f"{rank:2d}. {r['image_id']}  score={r['score']:.3f} "
            f"(structured={r['structured_score']:.2f}, semantic={r['semantic_score']:.2f})\n"
            f"      garments: {garments}\n"
            f"      environment: {r['environment']} | style: {r['style']}"
        )


if __name__ == "__main__":
    main()
