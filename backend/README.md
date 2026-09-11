RAG pipeline demo

This workspace contains a minimal Retrieval-Augmented Generation (RAG)
implementation split across `services/` and `graph/` to illustrate the
concepts in a production-like layout.

Quickstart

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the API server from the backend folder using the venv interpreter:

```bash
.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

3. Alternatively, if you use VS Code, select the `.venv` Python interpreter before running.

Notes
- If you set `GROQ_API_KEY` in your environment the pipeline will call Groq.
- The embedding layer uses TF-IDF for this demo, so it runs without sentence-transformers.
- The implementation is intentionally small and educational; for production use
  you should add batching, persistence for the vector index, robust error handling,
  and privacy controls.
