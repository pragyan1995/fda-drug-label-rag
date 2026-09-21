from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

all_chunks = []
for filename in ["doc1.txt", "doc2.txt", "doc3.txt"]:
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    for c in chunk_text(text):
        all_chunks.append({"source": filename, "text": c})

texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts)
for c, e in zip(all_chunks, embeddings):
    c["embedding"] = e

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

question = "What is the Great Red Spot?"
question_embedding = model.encode(question)

scored = []
for c in all_chunks:
    score = cosine_similarity(question_embedding, c["embedding"])
    scored.append((score, c))

scored.sort(key=lambda x: x[0], reverse=True)

print(f"Question: {question}\n")
for score, c in scored:
    print(f"Score: {score:.4f} | Source: {c['source']}")
    print(c["text"][:150], "...\n")