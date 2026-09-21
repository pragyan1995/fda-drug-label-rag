# RAG From Scratch

A retrieval-augmented generation (RAG) pipeline built from first principles — no LangChain, no LlamaIndex — to understand exactly how retrieval and generation fit together before relying on a framework.

## What it does

1. Loads and chunks text documents (with overlapping windows so ideas aren't lost at chunk boundaries).
2. Embeds each chunk using a local sentence-transformer model (`all-MiniLM-L6-v2`).
3. Indexes the embeddings in a FAISS vector index for fast similarity search.
4. Given a question, embeds it, retrieves the most relevant chunk via cosine similarity (via normalized inner product), and passes that chunk as grounded context to Claude (Anthropic API) to generate the final answer.

## Why these choices

- **Built without a framework first.** Frameworks like LangChain abstract away chunking, retrieval, and prompt assembly. Building it manually first means I understand what's actually happening underneath before relying on that abstraction.
- **Overlapping chunks.** Non-overlapping chunks risk splitting a relevant idea across a chunk boundary, so retrieval quality suffers. A configurable overlap (default 50 words per 200-word chunk) mitigates that.
- **FAISS over ChromaDB.** ChromaDB requires a newer sqlite3 than ships with some Python/Anaconda environments, and getting it working reliably on Windows was a real environment fight with no conceptual payoff. FAISS has no such dependency, installs as a clean compiled wheel, and is a standard, production-used vector index.
- **Explicit grounding instruction in the prompt** ("answer using only the context below") to reduce the model answering from general training knowledge instead of the retrieved document — a core RAG correctness concern.
- **API key handled via `.env`**, never hardcoded, and excluded from version control via `.gitignore`.

## Setup

```bash
python -m venv rag-env
rag-env\Scripts\activate       # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:


## Usage

```bash
python step7.py
```

## What I'd add next

- **Hybrid search** (BM25 + embedding similarity) to catch exact-term matches that pure semantic search can miss.
- **Reranking** with a cross-encoder over the top-k candidates for higher precision.
- **Retrieval evaluation** (recall@k) and answer-groundedness checks (e.g. G-Eval-style LLM-as-judge) rather than eyeballing output quality.
- Swap the placeholder documents for a real corpus (FDA DailyMed drug label data) as the actual portfolio use case.

## Stack

Python · sentence-transformers · FAISS · Anthropic API