"""
Scores one image record against a ParsedQuery.

This is where compositionality is actually enforced: each query pair
(garment, color) is matched against the image's own list of independently-
detected (garment, color) regions, so a match requires *both* attributes to
land on the *same* detected object. A vanilla-CLIP whole-image similarity
can't express that constraint -- it would score "red tie, white shirt" and
"white tie, red shirt" identically since both just have {red, white, tie,
shirt} present somewhere in the frame.
"""
from __future__ import annotations

from retriever.query_parser import ParsedQuery

# Loose synonym groups so "coat" in a query can match "trench coat" /
# "overcoat" in the taxonomy, and vice versa, without needing an exact
# string match.
_SYNONYMS = {
    "coat": {"coat", "trench coat", "overcoat", "raincoat", "parka"},
    "jacket": {"jacket", "blazer", "suit jacket", "puffer jacket", "denim jacket", "leather jacket", "windbreaker"},
    "shirt": {"shirt", "button-down shirt", "dress shirt", "blouse", "t-shirt", "polo shirt"},
    "dress": {"dress", "sundress", "casual dress", "formal dress", "gown"},
    "pants": {"pants", "dress pants", "trousers", "jeans"},
}


def _garment_matches(query_term: str, image_term: str) -> bool:
    if query_term == image_term:
        return True
    group = _SYNONYMS.get(query_term)
    return group is not None and image_term in group


def score_structured(parsed: ParsedQuery, record: dict) -> float:
    """Returns a score in [0, 1]. `record` is one row from storage.get_all_records()."""
    garments = record["garments"]
    sub_scores = []

    for pair in parsed.pairs:
        best = 0.0
        for g in garments:
            if _garment_matches(pair.garment, g["type"]):
                if pair.color is None:
                    best = max(best, 0.85 * g["det_score"])  # type-only match
                elif g["color"] == pair.color:
                    best = max(best, 1.0 * g["det_score"])  # full match: right type AND color
                else:
                    best = max(best, 0.25 * g["det_score"])  # wrong color on the right garment
        sub_scores.append(best)

    garment_score = sum(sub_scores) / len(sub_scores) if sub_scores else None

    env_bonus = 0.15 if parsed.environment_hits and record["environment"] in parsed.environment_hits else (
        0.05 if parsed.environment_hits else 0.0
    )
    style_bonus = 0.15 if parsed.style_hits and record["style"] in parsed.style_hits else (
        0.05 if parsed.style_hits else 0.0
    )

    if garment_score is None:
        # Pure vibe/context query ("casual weekend outfit for a city walk")
        # -- no object-binding needed, structured score is just the tag bonuses.
        return min(1.0, env_bonus + style_bonus)

    return min(1.0, 0.7 * garment_score + env_bonus + style_bonus)
