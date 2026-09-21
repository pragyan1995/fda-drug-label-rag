import os

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

all_chunks = []

for filename in ["doc1.txt", "doc2.txt", "doc3.txt"]:
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    for c in chunks:
        all_chunks.append({"source": filename, "text": c})

print(f"Total chunks: {len(all_chunks)}")
for c in all_chunks:
    print(f"--- from {c['source']} ---")
    print(c['text'][:100], "...")
    print()