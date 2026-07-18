# Multimodal Fashion & Context Retrieval

Natural-language search over a fashion image collection: "a red tie and a white shirt in a
formal setting" should return exactly that, not images with the colors swapped.

```
indexer/     Part A -- raw images -> searchable index
retriever/   Part B -- natural language query -> top-k images
tests/       unit tests for everything that doesn't need a GPU / model download
notebooks/   Colab notebook that runs the 5 official eval queries end-to-end
data/        put the dataset here (gitignored, see data/README.md)
```

---

## 1. Approaches considered

| # | Approach | Good for | Falls short on |
|---|----------|----------|-----------------|
| 1 | **Vanilla CLIP zero-shot.** One embedding for the whole image, one for the query, cosine similarity. | Fastest to build, decent on holistic scene/style queries ("casual weekend outfit"). | Compositionality -- a single embedding can't bind "red" to "tie" specifically rather than to "shirt"; it just knows both colors and both garments are present *somewhere*. Also weak on fine-grained color naming. |
| 2 | **Fine-tuned / domain CLIP (e.g. FashionCLIP-style).** Contrastively fine-tune on fashion image-caption pairs. | Sharper fashion-domain embeddings than #1, still a single fast forward pass. | Needs labeled caption data + training compute; still one global embedding, so compositionality is *improved*, not *solved*; risks overfitting to the caption style it was tuned on, weakening true zero-shot generalization to unseen phrasing. |
| 3 | **Closed-vocabulary attribute classifier.** Multi-label classifier per garment-type / color tag, retrieval via tag filtering (no embeddings at all). | Fast, cheap, fully interpretable, precise inside its trained vocabulary. | Poor zero-shot behavior by construction -- a word outside the trained label set simply can't be searched. Also has no way to represent fuzzy, continuous "vibe" queries like "professional business attire." |
| 4 | **Region-grounded hybrid (chosen).** Open-vocabulary detector localizes each garment; per-region classical color extraction + zero-shot type classification; whole-image CLIP embedding handles scene/vibe; fused at query time. | Directly solves compositionality by binding attributes to specific detected regions, while staying fully zero-shot (no training data needed). Interpretable -- you can inspect exactly which region and score produced a match. | More moving parts than #1 (two models, not one); quality depends on detector recall -- a missed detection means a missed attribute; more latency per image than a single CLIP pass. |
| 5 | **VLM captioning + LLM re-ranking.** Generate a rich caption per image (BLIP-2 / LLaVA-style), then embed or LLM-rerank captions against the query. | Potentially the richest compositional understanding, since a good caption already describes the whole scene coherently. Can reason explicitly about constraints if an LLM is doing the ranking. | Heaviest compute per image; caption quality varies a lot by model and captioning models still hallucinate fine details (colors especially); an LLM-in-the-loop re-ranker doesn't scale to ranking a full 1M-image database per query, only a short candidate list. |

**Why not #1/#2 alone:** the assignment's own hint calls out the exact failure mode -- CLIP
can't distinguish "red shirt, blue pants" from "blue shirt, red pants." That's a structural
property of a bag-of-concepts embedding, not something more training data fixes on its own.

**Why not #3 alone:** it trades away the "Zero-Shot Capability" requirement entirely -- a
closed tag vocabulary can't handle a novel phrase it wasn't trained on, which is one of the
four things the assignment explicitly evaluates.

**Chosen: #4**, with #5 noted as a future precision-boosting re-ranking layer on top of #4's
shortlist (see Future Work) rather than a replacement for it.

---

## 2. Chosen approach -- how it works

```
Part A (indexer)                              Part B (retriever)
─────────────────                             ──────────────────
raw image
   │
   ├─ OWL-ViT (open-vocab detector) ──► per-garment boxes         query text
   │     for each box:                                               │
   │       ├─ crop → k-means dominant color → color name    QueryParser (regex + lexicon)
   │       └─ CLIP zero-shot classify crop → garment type       │        │
   │                                                     garment/color   raw text
   ├─ CLIP whole-image embedding ──► environment / style       pairs        │
   │     (zero-shot vs. tag prompts)                              │        │
   │                                                       ┌──────▼──┐  ┌───▼────────┐
   ▼                                                       │structured│  │CLIP text   │
SQLite: {garments[], environment, style}                   │ scorer   │  │embedding   │
FAISS:  whole-image embedding                              └────┬─────┘  └─────┬──────┘
                                                                 │              │
                                                          FAISS ANN search ─────┘
                                                          (candidate pool)
                                                                 │
                                                        weighted fusion → top-k
```

**Localization (`indexer/localizer.py`).** OWL-ViT is an open-vocabulary detector: it takes
arbitrary text queries ("a raincoat", "a tie") and returns boxes, zero-shot, no fashion-specific
training needed. This is the step that makes the rest of the pipeline compositional -- every
attribute extracted downstream is anchored to *one specific box*, not the image as a whole.

**Color (`indexer/color_extractor.py`).** Deliberately *not* CLIP. CLIP's fine-grained color
judgments are noisy (it regularly confuses maroon/red, navy/black). K-means on the crop's
pixels, with near-white/near-black pixels dropped first (usually background or shadow, not the
garment), then snapped to the nearest name in a fashion-relevant palette, is cheaper and more
precise for this narrow sub-task. Verified: `dominant_color()` correctly recovers the swatch
color on synthetic tests and runs cleanly on real crops from the dataset.

**Type, environment, style (`indexer/attribute_extractor.py`).** Garment *type* is classified
per-crop via CLIP zero-shot against the taxonomy. *Environment* ("modern office", "park bench")
and *style* ("casual weekend outfit") are scored on the **whole image**, deliberately -- these
are diffuse scene properties, not objects you can crop out, which is exactly what CLIP's global
embedding is good at. The system uses CLIP for what it's good at (scene semantics) and classical
CV + detection for what it's bad at (compositional, per-object attributes).

**Query parsing (`retriever/query_parser.py`).** A regex + lexicon parser finds every color word
and every garment word in the query, then greedily pairs each color with its nearest garment
mention (English puts the adjective right before the noun: "red tie"). This is what turns
"a red tie and a white shirt" into `[(tie, red), (shirt, white)]` instead of a flat bag of four
words. **Verified against all 5 official evaluation queries** -- see `retriever/query_parser.py`'s
`__main__` block and `tests/test_core_logic.py`; all 5 parse into the intended structure,
including the compositional one.

**Scoring and fusion (`retriever/scorer.py`, `retriever/search.py`).** Each query
(garment, color) pair is matched against an image's own detected (garment, color) regions --
a match requires *both* to land on the *same* detected object. Retrieval is two-stage:
FAISS ANN search over whole-image embeddings narrows the field fast, then the (more expensive)
structured score re-ranks that shortlist. Fusion weight shifts adaptively: garment-naming queries
lean on the structured score (α=0.65); pure vibe queries ("casual weekend outfit for a city
walk", no garment named) lean on the CLIP semantic score instead (α=0.3), since there's no
object-binding to check.

**This was empirically checked, not just argued for.** Running the scorer on a synthetic
correct-vs-swapped pair for "a red tie and a white shirt in a formal setting":

| Image | Structured score |
|---|---|
| tie=red, shirt=white (correct) | **0.63** |
| tie=white, shirt=red (swapped) | 0.16 |
| unrelated garment (hoodie) | 0.00 |

A vanilla whole-image CLIP embedding would score the correct and swapped image nearly
identically, since both contain the same four concepts {red, white, tie, shirt}. See
`tests/test_core_logic.py::test_scorer_penalizes_swapped_colors`.

### How each evaluation query is handled

1. **"A person in a bright yellow raincoat."** → parses to `(raincoat, yellow)`. Structured
   score requires a detected raincoat-shaped region whose dominant color is yellow.
2. **"Professional business attire inside a modern office."** → no garment named; parses to
   environment hit `modern office interior` + style hit `professional business attire`. Scored
   via the whole-image environment/style CLIP classification plus the raw-query embedding.
3. **"Someone wearing a blue shirt sitting on a park bench."** → `(shirt, blue)` +
   environment hit `park bench`. Combines a structured garment match with a semantic/context match
   in one query -- exactly the "context awareness" requirement.
4. **"Casual weekend outfit for a city walk."** → pure style/vibe query, no garment; falls
   through to the CLIP-semantic-dominant branch.
5. **"A red tie and a white shirt in a formal setting."** → the compositional test case,
   `[(tie, red), (shirt, white)]`, empirically validated above.

### Zero-shot capability

Both models used (OWL-ViT for detection, CLIP for classification) work by comparing image
regions against arbitrary text -- nothing in the pipeline is trained on a fixed label set.
Adding a garment, color, environment, or style the system hasn't seen before is a one-line
addition to `indexer/taxonomy.py`, not a retraining job.

### Scalability to ~1M images

- **FAISS**: starts as `IndexFlatIP` (exact, fine to roughly 1M vectors on CPU); swapping to
  `IndexIVFPQ` (`--index-type ivf_pq` in `build_index.py`) gives sub-linear approximate search
  and a compressed footprint once flat search stops being fast/small enough -- a one-line change,
  not a rewrite (`indexer/storage.py::build_faiss_index`).
- **Structured re-ranking cost scales with the candidate pool, not the dataset.** The expensive
  per-garment matching in `scorer.py` only ever runs on the ~200 candidates FAISS returns, so
  retrieval latency stays roughly flat as the dataset grows.
- **Indexing throughput**: embarrassingly parallel over images (each image is processed
  independently) -- batch across GPUs / workers for the one-time indexing pass.
- SQLite is fine for a take-home; production-scale would move structured attributes to
  Postgres, or to a vector DB with payload filtering (Qdrant/Weaviate) to collapse the two
  stores into one system.

---

## 3. Running it

Model weights (CLIP, OWL-ViT) download from the Hugging Face Hub on first use, and indexing is
much faster with a GPU -- **use `notebooks/eval_queries.ipynb` in Google Colab** (free T4 GPU,
full internet access) rather than a locked-down sandbox.

```bash
pip install -r requirements.txt

# Part A: build the index (downloads CLIP + OWL-ViT on first run)
python -m indexer.build_index --images data/test --out data/index --limit 500   # subset first
python -m indexer.build_index --images data/test --out data/index               # full run

# Part B: query it
python -m retriever.cli "A red tie and a white shirt in a formal setting." --index data/index
```

**What's been verified without a GPU/internet access** (see `tests/test_core_logic.py`, all
passing): color extraction correctness, query parsing on all 5 official eval queries including
correct compositional binding, and the structured scorer's swapped-color penalty. The
model-dependent stages (`localizer.py`, `embedder.py`) are written and their `transformers` API
calls verified against the exact pinned library version in `requirements.txt`, but need an
environment that can reach the Hugging Face Hub to actually run.

---

## 4. Future work

### a. Extending to locations (cities, places) and weather

The architecture already has a slot for this: `environment` and `style` are just CLIP
zero-shot classifications against a text-prompt list (`indexer/taxonomy.py`). Adding weather
("rainy day", "snowy street", "sunny beach") is the same pattern -- append prompts, no code
changes. Real *city* recognition ("this is Paris" vs. "this is generic urban street") is a
harder, different problem: CLIP zero-shot is weak here unless a distinctive landmark is in
frame, so that would need either (a) EXIF/GPS metadata when available in production data, or
(b) a dedicated landmark/place-recognition model layered in as another localizer-style module.

### b. Improving precision

- **Learn the fusion weights** (currently hand-tuned α=0.65/0.3) from a small relevance-labeled
  validation set instead of guessing them -- logistic regression over
  (structured_score, semantic_score) → relevant/not is enough to start.
- **Add an LLM/VLM re-ranking stage** (approach #5) over just the top ~20-50 candidates from
  the current pipeline -- expensive per-item but cheap in aggregate since it only touches a
  shortlist, not the full database.
- **Better color extraction**: swap the brightness-threshold background heuristic for real
  human/garment segmentation, so skin tone and background never leak into the color estimate.
- **Prompt-ensemble the CLIP zero-shot classifier** (average embeddings across several phrasings
  per garment/color) -- a well-known trick for stabilizing zero-shot CLIP predictions.
- **Get ground truth.** The Fashionpedia test split used here has no labels, so precision so
  far is argued qualitatively (the swapped-color test) rather than measured as precision@k.
  Even 50-100 hand-labeled query-image relevance judgments would let every change above be
  compared against a number instead of a vibe check.

---

## 5. Codebase

GitHub: `<add your repo URL here after pushing>`
