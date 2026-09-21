from sentence_transformers import SentenceTransformer

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

print(f"Total chunks embedded: {len(all_chunks)}")
print(f"Embedding length: {len(all_chunks[0]['embedding'])}")
print(f"First 5 values: {all_chunks[0]['embedding'][:5]}")