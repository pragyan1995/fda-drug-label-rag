import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic
import os
from dotenv import load_dotenv

# --- reuse your parser from before ---
import xml.etree.ElementTree as ET
import glob

NS = {"hl7": "urn:hl7-org:v3"}

def get_text_content(text_elem):
    if text_elem is None:
        return ""
    return " ".join(text_elem.itertext()).replace("\n", " ")

def extract_sections(section_elem, results):
    code_elem = section_elem.find("hl7:code", NS)
    section_name = code_elem.get("displayName") if code_elem is not None else "UNKNOWN SECTION"
    if section_name and "SPL listing data elements" not in section_name:
        text_elem = section_elem.find("hl7:text", NS)
        text_content = get_text_content(text_elem).strip()
        if text_content:
            results.append({"section": section_name, "text": text_content})
    for sub_component in section_elem.findall("hl7:component", NS):
        sub_section = sub_component.find("hl7:section", NS)
        if sub_section is not None:
            extract_sections(sub_section, results)

def parse_spl_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    results = []
    for component in root.findall(".//hl7:structuredBody/hl7:component", NS):
        section = component.find("hl7:section", NS)
        if section is not None:
            extract_sections(section, results)
    return results

def chunk_section_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks if chunks else [text]

# --- build the real chunk set ---
all_chunks = []
for filepath in glob.glob("dailymed_data/*.xml"):
    filename = os.path.basename(filepath)
    for sec in parse_spl_file(filepath):
        for chunk in chunk_section_text(sec["text"]):
            all_chunks.append({"source": filename, "section": sec["section"], "text": chunk})

print(f"Loaded {len(all_chunks)} chunks from real DailyMed data.\n")

# --- embed + index ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([c["text"] for c in all_chunks]).astype("float32")
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# --- retrieve + generate ---
load_dotenv()
claude = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    default_headers={"anthropic-workspace-id": os.getenv("ANTHROPIC_WORKSPACE_ID")}
)

question = "What should I do if I experience stomach bleeding symptoms?"
q_embedding = model.encode([question]).astype("float32")
faiss.normalize_L2(q_embedding)

distances, indices = index.search(q_embedding, k=3)  # top 3, so you can see if it's consistently the right section

print(f"Question: {question}\n")
for rank, idx in enumerate(indices[0]):
    c = all_chunks[idx]
    print(f"#{rank+1} | Score: {distances[0][rank]:.4f} | Section: {c['section']} | Source: {c['source']}")
    print(c["text"][:150], "...\n")

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

print("\nFinal answer (grounded in:", top_chunk['section'], "):")
print(response.content[0].text)