import os
import sys
import sqlite3
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from indexer.embedder import ClipEmbedder
from indexer.storage import build_faiss_index, save_faiss_index

def main():
    db_path = PROJECT_ROOT / "data" / "index" / "attributes.db"
    test_dir = PROJECT_ROOT / "data" / "test"
    out_dir = PROJECT_ROOT / "data" / "index"

    print("Connecting to database...")
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT image_id, faiss_idx FROM images ORDER BY faiss_idx").fetchall()
    conn.close()

    print(f"Loaded {len(rows)} image records from database.")
    print("Loading CLIP embedder...")
    embedder = ClipEmbedder()

    embeddings = []
    print("Generating CLIP embeddings for all images...")
    for image_id, faiss_idx in tqdm(rows, desc="Embedding"):
        image_path = test_dir / image_id
        if not image_path.exists():
            print(f"Error: image {image_id} not found at {image_path}!")
            sys.exit(1)
        
        image = Image.open(image_path).convert("RGB")
        emb = embedder.embed_image(image)
        embeddings.append(emb)

    embeddings = np.stack(embeddings).astype(np.float32)
    
    print("Saving embeddings.npy...")
    np.save(out_dir / "embeddings.npy", embeddings)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings, index_type="flat")
    
    print("Saving index.faiss...")
    save_faiss_index(index, out_dir / "index.faiss")

    print("Done! Successfully generated embeddings and FAISS index for 201 images.")

if __name__ == "__main__":
    main()
