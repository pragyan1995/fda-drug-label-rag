# RAG From Scratch

A retrieval-augmented generation (RAG) pipeline built from first principles — no LangChain, no LlamaIndex — to understand exactly how retrieval and generation fit together before relying on a framework. Originally built on toy documents, then extended to real-world FDA drug label data (DailyMed) to test the pipeline against messy, structured documents rather than clean paragraphs.

## What it does

1. Pulls real drug label data from the **DailyMed API** (FDA Structured Product Labeling / SPL format, XML).
2. Parses each label's actual section structure (Warnings, Dosage & Administration, Indications & Usage, etc.) rather than treating the document as flat text — including nested sub-sections (e.g. "Warnings" contains "Do Not Use," "Ask a Doctor," "Stop Use").
3. Chunks each section's text with overlapping windows so ideas aren't lost at chunk boundaries, and tags every chunk with its source section.
4. Embeds each chunk using a local sentence-transformer model (`all-MiniLM-L6-v2`).
5. Indexes the embeddings in a FAISS vector index for fast similarity search.
6. Given a question, embeds it, retrieves the most relevant chunk(s) via cosine similarity, and passes the top chunk as grounded context to Claude (Anthropic API) to generate the final answer.

## Why these choices

- **Built without a framework first.** Frameworks like LangChain abstract away chunking, retrieval, and prompt assembly. Building it manually first means I understand what's actually happening underneath before relying on that abstraction.
- **Structure-aware chunking, not naive word-count splitting.** Real FDA SPL documents are HL7 XML with labeled `<section>` elements (identified by a `displayName` code, e.g. `"WARNINGS SECTION"`), often nested several levels deep. Naive word-count chunking would blindly slice across section boundaries — potentially cutting a warning in half or merging two unrelated sections into one chunk. This pipeline parses the actual section structure first, recurses into nested sub-sections, and only applies word-count/overlap chunking *within* a section — so no chunk ever mixes content from two different clinical sections. Each chunk carries its source section name as metadata.
- **Overlapping chunks.** Within a section, a configurable overlap (default 50 words per 200-word chunk) prevents a relevant idea from being split awkwardly across two chunks.
- **FAISS over ChromaDB.** ChromaDB requires a newer sqlite3 than ships with some Python/Anaconda environments, and getting it working reliably on Windows was a real environment fight with no conceptual payoff. FAISS has no such dependency, installs as a clean compiled wheel, and is a standard, production-used vector index.
- **Explicit grounding instruction in the prompt** ("answer using only the context below") to reduce the model answering from general training knowledge instead of the retrieved document — a core RAG correctness concern.
- **API key handled via `.env`**, never hardcoded, and excluded from version control via `.gitignore`.

## Evidence the structure-aware chunking matters

Query: *"What should I do if I experience stomach bleeding symptoms?"*

Top 3 retrieved chunks, ranked by cosine similarity:

| Rank | Score | Section | Source |
|------|-------|---------|--------|
| 1 | 0.7601 | OTC - STOP USE SECTION | ibuprofen label |
| 2 | 0.7294 | OTC - STOP USE SECTION | ibuprofen label (different manufacturer) |
| 3 | 0.6875 | OTC - ASK DOCTOR SECTION | ibuprofen label |

All three top matches came from the clinically correct sections — the ones that actually discuss stomach bleeding symptoms and next steps — not from unrelated sections like "Storage and Handling" or "Inactive Ingredients" that happen to share incidental vocabulary. This is the direct payoff of parsing real section boundaries instead of chunking by raw word count: retrieval stays anchored to the right part of the document.

Generated answer (grounded in the top-ranked "OTC - STOP USE SECTION" chunk):

> Based on the context, if you experience any of the following signs of stomach bleeding, you should seek attention: feel faint, have bloody or black stools, vomit blood, have stomach pain that does not get better. These are listed as warning signs that require immediate medical attention.

**Known limitation:** the retrieved source text says to "seek medical help right away" for an allergic reaction and lists stomach-bleeding signs as needing attention, but the generated answer softens this to "should seek attention" rather than preserving the more urgent original phrasing. This is a real grounding-fidelity gap — the model paraphrased rather than exactly preserving the actionable instruction. It's exactly the kind of drift an automated groundedness eval (e.g. G-Eval-style LLM-as-judge, checking whether the answer's claims are fully supported by the retrieved context) would catch systematically rather than relying on manual inspection. Not yet implemented — see below.

## Setup

```bash
python -m venv rag-env
rag-env\Scripts\activate       # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_WORKSPACE_ID=your_workspace_id_here
```

## Usage

Pull real DailyMed drug label data:
```bash
python Pull_data.py
```

Parse labels into section-tagged chunks:
```bash
python parse_dailymed.py
```

Run the full retrieval + generation pipeline:
```bash
python Step_8.py
```

## What I'd add next

- **Automated groundedness evaluation** (e.g. G-Eval / LLM-as-judge) to systematically catch cases like the softened phrasing above, instead of manual spot-checking.
- **Hybrid search** (BM25 + embedding similarity) to catch exact-term matches (drug names, dosage numbers) that pure semantic search can miss.
- **Reranking** with a cross-encoder over the top-k candidates for higher precision.
- **Retrieval evaluation** (recall@k) across a labeled set of question/section pairs.
- **Section-filtered retrieval** — letting a query restrict search to a specific section type (e.g. only search "Dosage & Administration" for dosing questions).

## Stack

Python · sentence-transformers · FAISS · Anthropic API · DailyMed SPL/XML (FDA structured product labeling)
