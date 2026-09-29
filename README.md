# FDA Drug Label Q&A (RAG)
Ask questions about FDA drug labels; every answer cites the source label.

**Stack:** Python, LangChain, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), Claude API

## Example
**Q:** [a real question you asked it]
**A:** [the real answer it gave, with the cited label]

## How it works
DailyMed labels → chunked → embedded → stored in ChromaDB → top matches retrieved → Claude answers and cites the source.

## Run it
pip install -r requirements.txt
Add ANTHROPIC_API_KEY to a .env file
python app.py

## Next
Retrieval evaluation set, Streamlit demo.
