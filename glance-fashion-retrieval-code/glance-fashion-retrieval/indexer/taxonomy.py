"""
Shared vocabulary for the fashion retrieval system.

Keeping this in one place means the indexer (which tags images) and the
retriever (which parses queries) are always talking about the same set of
concepts. Extend these lists as you see gaps in real queries -- nothing
else in the codebase needs to change.
"""

# Garment / apparel categories. Grouped loosely by formality, which the
# retriever uses for style-inference queries ("casual weekend outfit").
GARMENT_TAXONOMY = {
    "formal": [
        "blazer", "suit jacket", "button-down shirt", "dress shirt",
        "blouse", "dress pants", "trousers", "pencil skirt", "tie",
        "waistcoat", "gown", "formal dress",
    ],
    "casual": [
        "t-shirt", "hoodie", "sweatshirt", "jeans", "shorts", "sundress",
        "sneakers", "cardigan", "sweater", "polo shirt", "casual dress",
    ],
    "outerwear": [
        "raincoat", "trench coat", "overcoat", "puffer jacket",
        "denim jacket", "leather jacket", "parka", "windbreaker", "vest",
    ],
    "other": [
        "jumpsuit", "skirt", "scarf", "hat", "bag", "shoe", "boots",
        "handbag", "sunglasses",
    ],
    # Generic head-nouns. People say "a blue shirt" far more often than
    # "a blue button-down shirt" -- without these, only the more specific
    # multi-word terms above would ever match, which silently breaks
    # exactly the compositional queries this system exists to handle.
    # Kept in their own bucket since they're formality-neutral on their own.
    "generic": [
        "shirt", "jacket", "coat", "pants", "dress",
    ],
}

# Flat list used for zero-shot classification prompts.
ALL_GARMENTS = [g for group in GARMENT_TAXONOMY.values() for g in group]

# Color palette -> we extract dominant RGB from a crop and snap it to the
# nearest name here. Keeping fashion-relevant shades (navy, maroon, olive)
# rather than only the 16 CSS basic colors improves precision noticeably.
COLOR_PALETTE = {
    "red":     (196, 30, 30),
    "maroon":  (114, 28, 36),
    "orange":  (230, 126, 34),
    "yellow":  (241, 196, 15),
    "gold":    (212, 175, 55),
    "green":   (39, 130, 60),
    "olive":   (107, 111, 40),
    "teal":    (23, 128, 128),
    "blue":    (41, 98, 189),
    "navy":    (20, 40, 90),
    "purple":  (110, 60, 160),
    "pink":    (231, 130, 170),
    "brown":   (101, 67, 33),
    "tan":     (210, 180, 140),
    "beige":   (222, 202, 172),
    "cream":   (240, 234, 214),
    "white":   (245, 245, 245),
    "black":   (25, 25, 25),
    "gray":    (128, 128, 128),
    "silver":  (192, 192, 197),
}

# Environment / location tags for the "vibe" side of a query. These map to
# CLIP text prompts, not detectors -- there's no bounding box for "office".
ENVIRONMENT_TAGS = [
    "modern office interior", "urban street", "city sidewalk",
    "public park", "park bench", "home interior", "living room",
    "studio backdrop", "runway", "cafe or restaurant", "beach",
]

# Style / vibe descriptors, also scored against the whole image via CLIP.
STYLE_TAGS = [
    "professional business attire", "formal outfit", "casual weekend outfit",
    "streetwear", "athleisure", "elegant evening wear",
]

GARMENT_TO_FORMALITY = {
    g: formality for formality, items in GARMENT_TAXONOMY.items() for g in items
}
