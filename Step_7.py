import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv
import os
load_dotenv()
claude = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    default_headers={"anthropic-workspace-id": os.getenv("ANTHROPIC_WORKSPACE_ID")}
    )

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

all_chunks = []
for filename in ["doc1.txt", "doc2.txt", "doc3.txt"]:
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    for chunk in chunk_text(text):
        all_chunks.append({"source": filename, "text": chunk})

embeddings = model.encode([c["text"] for c in all_chunks])
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # inner product ~ cosine on normalized vectors
faiss.normalize_L2(embeddings)
index.add(embeddings)

question = "What is the Great Red Spot?"
question_embedding = model.encode([question]).astype("float32")
faiss.normalize_L2(question_embedding)

distances, indices = index.search(question_embedding, k=1)
top_chunk = all_chunks[indices[0][0]]



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



