from sentence_transformers import SentenceTransformer
import numpy as np
import anthropic
from dotenv import load_dotenv
import os
load_dotenv()
claude = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    default_headers={"anthropic-workspace-id": os.getenv("ANTHROPIC_WORKSPACE_ID")}
    )

model = SentenceTransformer("all-MiniLM-L6-v2")


##def chunk_text(text, chunk_size=200):
##    words = text.split()
##    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # step forward, but re-include the overlap
    return chunks
    
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

scored = sorted(
    all_chunks,
    key=lambda c: cosine_similarity(question_embedding, c["embedding"]),
    reverse=True
)
top_chunk = scored[0]

prompt = f"""Answer the question using only the context below.

Context:
{top_chunk['text']}

Question: {question}
"""

response = claude.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    messages=[{"role": "user", "content": prompt}]
)

print("Retrieved from:", top_chunk["source"])
print("\nAnswer:\n", response.content[0].text)