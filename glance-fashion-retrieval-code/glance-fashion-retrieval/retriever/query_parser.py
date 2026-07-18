"""
Turns a free-text query into a structured form the scorer can bind
attribute-to-object, rather than treating the sentence as a bag of words.

This is a deliberately simple, dependency-free heuristic (regex + lexicon +
nearest-neighbor pairing), not a trained parser. That's a conscious
tradeoff for this assignment's scope -- see the README "future work"
section for the upgrade path (swap this module for an LLM-based parser
that emits the same ParsedQuery structure; nothing downstream changes).

Pairing logic: English adjective order puts the color right before the
noun ("red tie"), so we match every color word and every garment word by
position in the query, then greedily pair each color with its nearest
unpaired garment. Garments left without a color nearby become
"standalone" (type-only) matches -- e.g. a garment word that's actually
being used as the head of a location phrase, or a query with no color at
all ("casual weekend outfit").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from indexer.taxonomy import ALL_GARMENTS, COLOR_PALETTE, ENVIRONMENT_TAGS, STYLE_TAGS


@dataclass
class GarmentColorPair:
    garment: str
    color: str | None  # None if no color word was nearby


@dataclass
class ParsedQuery:
    raw_query: str
    pairs: list[GarmentColorPair] = field(default_factory=list)
    environment_hits: list[str] = field(default_factory=list)  # verbatim ENVIRONMENT_TAGS matches
    style_hits: list[str] = field(default_factory=list)        # verbatim STYLE_TAGS matches

    @property
    def has_garment_mention(self) -> bool:
        return len(self.pairs) > 0


def _find_spans(text: str, vocab: list[str]) -> list[tuple[int, int, str]]:
    """Find every vocab term present in text, longest-term-first so multi-word
    entries like 'button-down shirt' win over the bare 'shirt' substring."""
    hits = []
    for term in sorted(vocab, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(term)}\b", text):
            # skip if this span is already covered by a longer match
            if any(h[0] <= m.start() and m.end() <= h[1] for h in hits):
                continue
            hits.append((m.start(), m.end(), term))
    return sorted(hits)


class QueryParser:
    def __init__(self):
        self.garments = ALL_GARMENTS
        self.colors = list(COLOR_PALETTE.keys())

    def parse(self, query: str) -> ParsedQuery:
        text = query.lower()

        garment_spans = _find_spans(text, self.garments)
        color_spans = _find_spans(text, self.colors)

        pairs: list[GarmentColorPair] = []
        used_colors: set[int] = set()

        for g_start, g_end, g_term in garment_spans:
            best_color, best_dist, best_i = None, None, None
            for i, (c_start, c_end, c_term) in enumerate(color_spans):
                if i in used_colors:
                    continue
                dist = abs(c_start - g_start)
                if best_dist is None or dist < best_dist:
                    best_dist, best_color, best_i = dist, c_term, i
            # only bind if the color is plausibly modifying this garment
            # (within ~15 chars, i.e. roughly "the bright X <garment>")
            if best_color is not None and best_dist <= 15:
                used_colors.add(best_i)
                pairs.append(GarmentColorPair(garment=g_term, color=best_color))
            else:
                pairs.append(GarmentColorPair(garment=g_term, color=None))

        env_hits = [tag for tag in ENVIRONMENT_TAGS if _fuzzy_contains(text, tag)]
        style_hits = [tag for tag in STYLE_TAGS if _fuzzy_contains(text, tag)]

        return ParsedQuery(raw_query=query, pairs=pairs, environment_hits=env_hits, style_hits=style_hits)


def _fuzzy_contains(text: str, phrase: str) -> bool:
    """A phrase 'matches' if most of its content words appear in the text,
    not requiring the exact substring (queries rarely echo our tag phrasing
    verbatim). This is deliberately loose -- it only ever adds a *bonus* to
    the score, the CLIP embedding similarity carries the real semantic
    matching, so a false positive here is low-risk."""
    words = [w for w in re.findall(r"[a-z]+", phrase) if len(w) > 3]
    if not words:
        return False
    hits = sum(1 for w in words if w in text)
    return hits / len(words) >= 0.6


if __name__ == "__main__":
    qp = QueryParser()
    for q in [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting.",
    ]:
        print(q, "->", qp.parse(q))
