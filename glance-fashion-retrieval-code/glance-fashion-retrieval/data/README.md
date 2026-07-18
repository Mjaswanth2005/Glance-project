# Data

This folder is intentionally empty in git. To run the pipeline:

```bash
unzip test.zip -d data/          # produces data/test/*.jpg (3,200 images, Fashionpedia test split)
python -m indexer.build_index --images data/test --out data/index
python -m retriever.cli "A red tie and a white shirt in a formal setting." --index data/index
```

`data/index/` (generated) contains `embeddings.npy`, `index.faiss`, and `attributes.db` --
also gitignored, since it's fully reproducible from the raw images.
