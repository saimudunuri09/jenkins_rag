import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

INDEX_DIR = "rag_store"
DATA_PATH = "data/jenkins_data.jsonl"


# ---------------- LOAD COMPONENTS ----------------
def load_system():
    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")

    with open(f"{INDEX_DIR}/metadata.json") as f:
        metadata = json.load(f)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # ---- small local LLM (offline) ----
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name)

    return index, metadata, embedder, tokenizer, llm


# ---------------- VECTOR SEARCH ----------------
def retrieve(query, index, metadata, embedder, k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    dist, ids = index.search(q_emb, k)
    return [metadata[i] for i in ids[0]]


# ---------------- LLM GENERATOR ----------------
def generate_answer(context, question, tokenizer, llm):
    prompt = f"""
You are an expert Jenkins CI/CD analyst.
Use ONLY the JSON data provided below.

JSON Data:
{context}

Question:
{question}

Your Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    output = llm.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------- MAIN ANSWER ENGINE ----------------
def answer(question, index, metadata, embedder, tokenizer, llm):

    # 1. Retrieve top JSON chunks
    retrieved = retrieve(question, index, metadata, embedder)

    # Prepare context
    context = "\n\n".join([rec["text"] for rec in retrieved])

    # 2. LLM REASONING + ANALYTICS
    final = generate_answer(context, question, tokenizer, llm)

    print("\n🔎 FINAL ANSWER (LLM + RAG):\n")
    print(final)
    print("\n--------------------------------------------------")


# ---------------- MAIN LOOP ----------------
def main():
    index, metadata, embedder, tokenizer, llm = load_system()

    print("Ask anything. Type 'exit' to quit.\n")

    while True:
        q = input("Your question → ")
        if q.lower() == "exit":
            break
        answer(q, index, metadata, embedder, tokenizer, llm)


if __name__ == "__main__":
    main()
