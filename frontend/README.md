# BMS MedInsight Frontend

A simple professional frontend for the backend RAG service.

## How to run

1. Start the backend API:

```bash
cd "BMS/backend"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

2. Open `frontend/index.html` in your browser, or run a local static server:

```bash
cd "BMS/frontend"
python -m http.server 3000
```

Then visit `http://127.0.0.1:3000` in your browser.

## Notes

- The frontend sends requests to `http://127.0.0.1:8000/api/query`.
- Make sure the backend is running and the API is accessible.
- You can update the API URL in `frontend/app.js` if your backend runs on a different host or port.
