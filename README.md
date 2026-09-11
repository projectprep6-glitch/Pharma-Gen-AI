# BMS MedInsight - Complete Setup & Run Guide

BMS MedInsight is a pharmaceutical RAG (Retrieval-Augmented Generation) application that allows users to ask questions about Bristol Myers Squibb marketing and strategy, with answers grounded in indexed PDF content.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Complete Setup Instructions](#complete-setup-instructions)
5. [Running the Application](#running-the-application)
6. [API Endpoints](#api-endpoints)
7. [Environment Variables](#environment-variables)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project consists of two main components:

- **Backend**: Python-based REST API using FastAPI that handles document indexing, retrieval, and generative AI logic using RAG pipeline
- **Frontend**: HTML/CSS/JavaScript web interface for users to submit questions and view answers

### Key Features

- Document retrieval using TF-IDF + cosine similarity
- LLM integration via Groq API for answer generation
- Web search fallback using DuckDuckGo when PDF sources are insufficient
- Orchestration using LangGraph for clean workflow structure
- CORS-enabled API for frontend integration

---

## Prerequisites

Before running this project, ensure you have:

1. **Python 3.9+** - [Download here](https://www.python.org/)
2. **Node.js/npm** (optional, for running frontend with Node server)
3. **Git** (optional, for version control)
4. **Groq API Key** (optional, for LLM functionality) - [Get one here](https://console.groq.com/)

---

## Project Structure

```
BMS/
├── README.md                              (This file)
├── Project_Documentation_BMS_MedInsight.md (Full project documentation)
├── backend/
│   ├── main.py                           (FastAPI application entry point)
│   ├── requirements.txt                  (Python dependencies)
│   ├── README.md                         (Backend-specific documentation)
│   ├── .env                              (Environment variables - create this)
│   ├── .venv/                            (Virtual environment - auto-created)
│   ├── data/
│   │   ├── bms_analysis.pdf             (Sample PDF documents)
│   │   └── sample_docs.txt
│   ├── graph/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py              (RAG pipeline orchestration)
│   └── services/
│       ├── __init__.py
│       ├── bm25_search.py               (BM25 search service)
│       ├── chunking.py                  (Document chunking)
│       ├── compressor.py                (Result compression)
│       ├── embeddings.py                (Embedding service)
│       ├── llm.py                       (LLM integration)
│       ├── reranker.py                  (Result reranking)
│       ├── retriever.py                 (Document retriever)
│       └── vector_store.py              (Vector storage)
└── frontend/
    ├── index.html                        (Main HTML page)
    ├── app.js                            (Frontend JavaScript logic)
    ├── styles.css                        (Styling)
    └── README.md                         (Frontend-specific documentation)
```

---

## Complete Setup Instructions

### Step 1: Clone/Extract the Project

Extract the project files to your desired location:

```bash
cd "path\to\BMS"
```

### Step 2: Backend Setup

#### 2.1 Create Python Virtual Environment

Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
# On Windows (Command Prompt)
python -m venv .venv

# On Windows (PowerShell)
python -m venv .venv
```

Activate the virtual environment:

```bash
# On Windows (Command Prompt)
.venv\Scripts\activate

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` prefix in your terminal.

#### 2.2 Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

This will install:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `sentence-transformers` - Embeddings
- `langchain` - LLM framework
- `langgraph` - Workflow orchestration
- `groq` - Groq API client
- `duckduckgo_search` - Web search
- `rank-bm25` - BM25 search
- `scikit-learn` - ML utilities
- `pypdf` - PDF processing
- `python-dotenv` - Environment variables
- And more (see `requirements.txt`)

#### 2.3 Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
# In backend/ directory
touch .env  # On Linux/Mac
# or just create a .env file manually on Windows
```

Add your configuration:

```
# API Configuration
API_HOST=127.0.0.1
API_PORT=8000

# Groq LLM Configuration (Optional - required for LLM generation)
GROQ_API_KEY=your_groq_api_key_here

# Embedding Configuration
EMBEDDING_MODEL=bm25  # Options: bm25, sentence-transformer

# Document paths
PDF_DATA_PATH=./data/bms_analysis.pdf
```

### Step 3: Frontend Setup

The frontend is a simple HTML/JavaScript application and doesn't require installation. However, you can serve it with a local server.

#### Option A: Python HTTP Server (Recommended)

```bash
# Navigate to frontend directory
cd ../frontend

# Start a local server on port 3000
python -m http.server 3000
```

#### Option B: Node.js Server

If you have Node.js installed:

```bash
# In frontend directory
npx http-server -p 3000
```

---

## Running the Application

### Complete Startup Sequence

Follow these steps to run the entire application:

#### Step 1: Start the Backend API

Open a terminal and navigate to the backend directory:

```bash
cd backend

# Activate virtual environment (if not already activated)
.venv\Scripts\activate  # Windows

# Start the FastAPI server
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

You should see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

Leave this terminal running.

#### Step 2: Start the Frontend Server

Open **another terminal** and navigate to the frontend directory:

```bash
cd frontend

# Start the frontend server
python -m http.server 3000
```

You should see:

```
Serving HTTP on 127.0.0.1 port 3000 (http://127.0.0.1:3000/) ...
```

#### Step 3: Access the Application

Open your web browser and visit:

```
http://127.0.0.1:3000
```

You should see the BMS MedInsight interface. Type a question and click "Submit" to get an answer.

---

## API Endpoints

### Base URL

```
http://127.0.0.1:8000
```

### Available Endpoints

#### 1. Query Endpoint

**POST** `/api/query`

Submit a question and get an answer from the RAG pipeline.

**Request Body:**

```json
{
  "user_query": "What are the key marketing strategies of BMS?"
}
```

**Response:**

```json
{
  "answer": "Generated answer based on indexed documents...",
  "sources": [
    {
      "text": "Relevant passage from document...",
      "score": 0.95
    }
  ],
  "query": "What are the key marketing strategies of BMS?"
}
```

#### 2. Health Check Endpoint (if available)

**GET** `/`

Verifies the API is running.

---

## Environment Variables

The following environment variables can be configured in the `.env` file:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GROQ_API_KEY` | Your Groq API key for LLM functionality | No | - |
| `API_HOST` | Backend server host | No | `127.0.0.1` |
| `API_PORT` | Backend server port | No | `8000` |
| `EMBEDDING_MODEL` | Embedding model to use | No | `bm25` |
| `PDF_DATA_PATH` | Path to PDF documents | No | `./data/` |

### Getting a Groq API Key

1. Visit [https://console.groq.com/](https://console.groq.com/)
2. Sign up for an account
3. Create an API key
4. Add it to your `.env` file as `GROQ_API_KEY=<your_key>`

---

## Troubleshooting

### Issue: Virtual Environment Not Activating

**Solution:**
Make sure you're in the `backend/` directory and try:

```bash
# For PowerShell on Windows
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1

# For Command Prompt on Windows
.venv\Scripts\activate.bat
```

### Issue: "Module not found" Errors

**Solution:**
Ensure the virtual environment is activated and all dependencies are installed:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Backend and Frontend Can't Communicate

**Solution:**
1. Verify the backend is running on `http://127.0.0.1:8000`
2. Check that the frontend is trying to connect to the correct URL
3. Update the API URL in `frontend/app.js` if needed:

```javascript
const API_URL = "http://127.0.0.1:8000/api/query"; // Change this if needed
```

### Issue: "Port 8000 already in use"

**Solution:**
Use a different port:

```bash
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

Then update the frontend to connect to port 8001.

### Issue: "No module named 'groq'"

**Solution:**
Make sure you're using the correct Python interpreter:

```bash
# Check which Python is being used
.venv\Scripts\python.exe --version

# Reinstall requirements
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Issue: PDF Files Not Found

**Solution:**
Ensure PDF files are in the `backend/data/` directory and update the path in `.env` if needed.

### Issue: Slow Response Times

**Possible causes:**
- Large PDF files being processed
- Network latency to Groq API
- Insufficient system resources

**Solution:**
- Use a smaller subset of PDFs for testing
- Check your internet connection
- Ensure adequate RAM (4GB minimum recommended)

---

## Development Notes

### Adding New PDFs

Place PDF files in `backend/data/` directory and they will be automatically indexed when the backend starts.

### Customizing the RAG Pipeline

Edit `backend/graph/rag_pipeline.py` to modify the retrieval and generation logic.

### Modifying Embeddings

Change the embedding model in `backend/services/embeddings.py`.

### Frontend Customization

Modify `frontend/index.html`, `frontend/app.js`, and `frontend/styles.css` for UI changes.

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://docs.langchain.com/)
- [Groq API Documentation](https://console.groq.com/docs)
- [Project Documentation](./Project_Documentation_BMS_MedInsight.md)

---

## Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Project Documentation](./Project_Documentation_BMS_MedInsight.md)
3. Check backend logs in the terminal
4. Verify all environment variables are correctly set

---

**Last Updated:** September 2026
