"""
Tests for the components that don't require downloading model weights,
so they run anywhere (including CI) without a GPU or internet access.
The model-dependent pieces (localizer.py, embedder.py) are exercised via
the end-to-end scripts instead -- see README "Running it" section.
"""
import numpy as np
import pytest

from indexer.color_extractor import dominant_color
from retriever.query_parser import QueryParser
from retriever.scorer import score_structured

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def test_dominant_color_basic():
    swatch = np.tile(np.array([196, 30, 30], dtype=np.uint8), (30, 30, 1))
    name, conf = dominant_color(swatch)
    assert name == "red"
    assert conf > 0.5


def test_dominant_color_keeps_white_garments():
    swatch = np.full((30, 30, 3), 245, dtype=np.uint8)
    name, conf = dominant_color(swatch)
    assert name == "white"
    assert conf > 0.5


def test_query_parser_all_eval_queries_produce_something():
    qp = QueryParser()
    for q in EVAL_QUERIES:
        parsed = qp.parse(q)
        assert parsed.pairs or parsed.environment_hits or parsed.style_hits, f"parsed nothing from: {q}"


def test_query_parser_compositional_binding():
    qp = QueryParser()
    parsed = qp.parse("A red tie and a white shirt in a formal setting.")
    pairs = {(p.garment, p.color) for p in parsed.pairs}
    assert ("tie", "red") in pairs
    assert ("shirt", "white") in pairs
    # the failure mode this system exists to avoid:
    assert ("tie", "white") not in pairs
    assert ("shirt", "red") not in pairs


def test_scorer_penalizes_swapped_colors():
    qp = QueryParser()
    parsed = qp.parse("A red tie and a white shirt in a formal setting.")

    correct = {
        "environment": "", "style": "",
        "garments": [
            {"type": "tie", "color": "red", "det_score": 0.9},
            {"type": "shirt", "color": "white", "det_score": 0.9},
        ],
    }
    swapped = {
        "environment": "", "style": "",
        "garments": [
            {"type": "tie", "color": "white", "det_score": 0.9},
            {"type": "shirt", "color": "red", "det_score": 0.9},
        ],
    }
    irrelevant = {"environment": "", "style": "", "garments": [{"type": "hoodie", "color": "black", "det_score": 0.9}]}

    s_correct = score_structured(parsed, correct)
    s_swapped = score_structured(parsed, swapped)
    s_irrelevant = score_structured(parsed, irrelevant)

    assert s_correct > s_swapped > s_irrelevant


def test_scorer_pure_vibe_query_has_no_garment_requirement():
    qp = QueryParser()
    parsed = qp.parse("Casual weekend outfit for a city walk.")
    assert not parsed.has_garment_mention

    record = {"environment": "urban street", "style": "casual weekend outfit", "garments": []}
    score = score_structured(parsed, record)
    assert score > 0  # style tag matched even with zero detected garments


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
