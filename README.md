# RAG From Scratch

A retrieval-augmented generation (RAG) pipeline built from first principles — no LangChain, no LlamaIndex — to understand exactly how retrieval and generation fit together before relying on a framework. Originally built on toy documents, then extended to real-world FDA drug label data (DailyMed) to test the pipeline against messy, structured documents rather than clean paragraphs.

## What it does

1. Pulls real drug label data from the **DailyMed API** (FDA Structured Product Labeling / SPL format, XML).
2. Parses each label's actual section structure (Warnings, Dosage & Administration, Indications & Usage, etc.) rather than treating the document as flat text — including nested sub-sections (e.g. "Warnings" contains "Do Not Use," "Ask a Doctor," "Stop Use").
3. Chunks each section's text with overlapping windows so ideas aren't lost at chunk boundaries, and tags every chunk with its source section.
4. Embeds each chunk using a local sentence-transformer model (`all-MiniLM-L6-v2`).
5. Indexes the embeddings in a FAISS vector index for fast similarity search.
6. Given a question, embeds it, retrieves the most relevant chunk(s) via cosine similarity, and passes the top chunk as grounded context to Claude (Anthropic API) to generate the final answer.
7. Scores each generated answer for groundedness using an automated G-Eval metric, judged by Claude itself — no manual eyeballing required.

## Why these choices

- **Built without a framework first.** Frameworks like LangChain abstract away chunking, retrieval, and prompt assembly. Building it manually first means I understand what's actually happening underneath before relying on that abstraction.
- **Structure-aware chunking, not naive word-count splitting.** Real FDA SPL documents are HL7 XML with labeled `<section>` elements (identified by a `displayName` code, e.g. `"WARNINGS SECTION"`), often nested several levels deep. Naive word-count chunking would blindly slice across section boundaries — potentially cutting a warning in half or merging two unrelated sections into one chunk. This pipeline parses the actual section structure first, recurses into nested sub-sections, and only applies word-count/overlap chunking *within* a section — so no chunk ever mixes content from two different clinical sections. Each chunk carries its source section name as metadata.
- **Overlapping chunks.** Within a section, a configurable overlap (default 50 words per 200-word chunk) prevents a relevant idea from being split awkwardly across two chunks.
- **FAISS over ChromaDB.** ChromaDB requires a newer sqlite3 than ships with some Python/Anaconda environments, and getting it working reliably on Windows was a real environment fight with no conceptual payoff. FAISS has no such dependency, installs as a clean compiled wheel, and is a standard, production-used vector index.
- **Explicit grounding instruction in the prompt** ("answer using only the context below") to reduce the model answering from general training knowledge instead of the retrieved document — a core RAG correctness concern.
- **Claude-as-judge for evaluation, not a second provider.** DeepEval's default G-Eval judge is OpenAI-based. Rather than adding a second paid API dependency, I wrote a custom DeepEvalBaseLLM wrapper so Claude judges its own pipeline's output — one provider, no extra account, and a concrete demonstration that the judge model isn't locked to a single vendor.
- **API key handled via `.env`**, never hardcoded, and excluded from version control via `.gitignore`.

## Evidence the structure-aware chunking matters

Query: "What should I do if I experience stomach bleeding symptoms?"

Top 3 retrieved chunks, ranked by cosine similarity:

| Rank | Score | Section | Source |
|------|-------|---------|--------|
| 1 | 0.7601 | OTC - STOP USE SECTION | ibuprofen label |
| 2 | 0.7294 | OTC - STOP USE SECTION | ibuprofen label (different manufacturer) |
| 3 | 0.6875 | OTC - ASK DOCTOR SECTION | ibuprofen label |

All three top matches came from the clinically correct sections — the ones that actually discuss stomach bleeding symptoms and next steps — not from unrelated sections like "Storage and Handling" or "Inactive Ingredients" that happen to share incidental vocabulary. This is the direct payoff of parsing real section boundaries instead of chunking by raw word count: retrieval stays anchored to the right part of the document.

## Automated groundedness evaluation

Manual spot-checking first caught a grounding issue (an answer that softened the source's urgency language), but eyeballing outputs doesn't scale and isn't reproducible. To catch this systematically, I added an automated G-Eval groundedness metric (via DeepEval) that scores whether each generated answer is fully supported by its retrieved context — using Claude itself as the judge model, via a custom DeepEvalBaseLLM wrapper, so the whole pipeline runs on one provider with no second API dependency.

Metric definition: the judge checks whether the answer omits, softens, or alters severity/urgency language present in the retrieved context — the exact failure mode observed manually.

### Results on 3 real DailyMed questions

| Question | Score | Finding |
|---|---|---|
| What should I do if I experience stomach bleeding symptoms? | 0.60 ⚠️ | Below threshold (0.7). The judge caught two concrete issues: the answer omitted several warning signs present in the source (heart/stroke symptoms, fever duration, new symptoms), and added the phrase "immediate medical attention," which wasn't literally present in the retrieved text. |
| What is the maximum dosage of ibuprofen per day? | 0.90 | Correctly reflected that the source doesn't state an exact human dose, without inventing one. |
| Can I take this medication while pregnant? | 0.90 | Correctly preserved "not known" and "talk with your healthcare provider" without adding false certainty. |

Why this matters: the first result is a genuine, reproducible failure caught by the eval, not by manual reading — and it's exactly the class of error that matters most in a healthcare context: an answer that quietly drops safety-relevant information while still sounding complete and confident. A groundedness score below threshold is a concrete, automatable signal that a response needs review before being shown to a user — this is the mechanism that would gate a real deployment, not a one-off observation.

Limitations of this eval setup: three questions is a smoke test, not a real evaluation suite — a production version would run this against dozens of question/answer pairs, track score trends over time, and gate deployment on a minimum pass rate rather than reporting scores after the fact.

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

Run the automated groundedness evaluation:
```bash
python step_9.py
```

## What I'd add next

- Expand the eval suite beyond a 3-question smoke test — dozens of labeled question/answer pairs, tracked over time, gating deployment on a minimum pass rate.
- Hybrid search (BM25 + embedding similarity) to catch exact-term matches (drug names, dosage numbers) that pure semantic search can miss.
- Reranking with a cross-encoder over the top-k candidates for higher precision.
- Retrieval evaluation (recall@k) across a labeled set of question/section pairs.
- Section-filtered retrieval — letting a query restrict search to a specific section type (e.g. only search "Dosage & Administration" for dosing questions).

## Stack

Python · sentence-transformers · FAISS · Anthropic API (generation + Claude-as-judge eval) · DeepEval (G-Eval) · DailyMed SPL/XML (FDA structured product labeling)
