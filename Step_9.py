import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic
import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import time

# --- reuse the existing pipeline: load chunks, embed, index ---
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

all_chunks = []
for filepath in glob.glob("dailymed_data/*.xml"):
    filename = os.path.basename(filepath)
    for sec in parse_spl_file(filepath):
        for chunk in chunk_section_text(sec["text"]):
            all_chunks.append({"source": filename, "section": sec["section"], "text": chunk})

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([c["text"] for c in all_chunks]).astype("float32")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

load_dotenv()
claude = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    default_headers={"anthropic-workspace-id": os.getenv("ANTHROPIC_WORKSPACE_ID")}
)

def retrieve_and_generate(question):
    q_embedding = model.encode([question]).astype("float32")
    faiss.normalize_L2(q_embedding)
    distances, indices = index.search(q_embedding, k=1)
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
    return top_chunk["text"], response.content[0].text

# --- define the G-Eval groundedness metric ---
groundedness_metric = GEval(
    name="Groundedness",
    criteria=(
        "Determine whether the actual output is fully supported by the retrieval context. "
        "The output should not soften, omit, or alter urgency/severity language present in the context "
        "(e.g. downgrading 'seek medical help right away' to a vaguer phrase counts as a violation)."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.7,
    async_mode=False,   # <-- add this: run sequentially, not all 3 at once
    model="gpt-4o-mini", 
)

# --- run it on a few real questions ---
questions = [
    "What should I do if I experience stomach bleeding symptoms?",
    "What is the maximum dosage of ibuprofen per day?",
    "Can I take this medication while pregnant?",
]

test_cases = []
for q in questions:
    context, answer = retrieve_and_generate(q)
    test_cases.append(
        LLMTestCase(
            input=q,
            actual_output=answer,
            retrieval_context=[context],
        )
    )
    print(f"Q: {q}")
    print(f"A: {answer}\n")

##evaluate(test_cases, [groundedness_metric])

for i, tc in enumerate(test_cases):
    groundedness_metric.measure(tc)
    print(f"\nQ{i+1}: {tc.input}")
    print(f"Score: {groundedness_metric.score:.2f}")
    print(f"Reason: {groundedness_metric.reason}")
    time.sleep(2)  # small pause between calls to stay under rate limits