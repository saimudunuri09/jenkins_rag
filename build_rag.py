import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/jenkins_data.jsonl"
INDEX_DIR = "rag_store"

os.makedirs(INDEX_DIR, exist_ok=True)

def load_json_records():
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    print("Loading Jenkins JSON...")
    records = load_json_records()

    texts = [rec["text"] for rec in records]

    print("Encoding texts using HuggingFace...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")

    # Save metadata
    with open(f"{INDEX_DIR}/metadata.json", "w") as f:
        json.dump(records, f)

    print("✅ RAG index built successfully!")


if __name__ == "__main__":
    main()
