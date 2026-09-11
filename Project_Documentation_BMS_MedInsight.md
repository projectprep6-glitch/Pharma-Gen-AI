# BMS MedInsight Project Documentation

## 1. Project Overview

This project is a small pharmaceutical RAG (Retrieval-Augmented Generation) application named **BMS MedInsight**. The goal is to let a user ask questions about Bristol Myers Squibb marketing and strategy, and return answers grounded in indexed PDF content.

The project has two main parts:
- `backend/`: the API, document indexing, retrieval, and generative AI logic
- `frontend/`: a simple web interface to submit questions and display answers

The key generative AI part lives in the backend and combines:
- document retrieval using TF-IDF + cosine similarity
- an LLM call via Groq when available
- a web search fallback using DuckDuckGo when PDF sources do not answer the question
- orchestration using `langgraph` to structure the flow cleanly

## 2. Folder Structure

```
BMS/
  backend/
    .env
    .venv/
    data/
      bms_analysis.pdf
      sample_docs.txt
    graph/
      __init__.py
      rag_pipeline.py
    services/
      __init__.py
      embeddings.py
      llm.py
      retriever.py
      vector_store.py
    main.py
    requirements.txt
    README.md
  frontend/
    index.html
    styles.css
    app.js
    README.md
  Project_Documentation_BMS_MedInsight.md
```

## 3. How the Project Works

### 3.1 High-level flow

1. The backend starts and loads the PDF documents.
2. The backend indexes the PDF passages into a vector store.
3. The frontend sends a question to the backend API.
4. The backend retrieves relevant passages from the index.
5. The backend uses the LLM to generate a concise answer.
6. If the PDF indexed data is not enough, the backend uses DuckDuckGo as a fallback.
7. The frontend displays the answer and the supporting passages.

### 3.2 Diagram of the end-to-end flow

```
Frontend (browser)
      |
      | POST /api/query
      v
backend/main.py
      |
      | RAGPipeline.query()
      v
backend/graph/rag_pipeline.py
      |
      | retrieve node -> services/retriever.py
      | answer node -> services/llm.py
      v
vector store & embeddings -> services/embeddings.py + services/vector_store.py
      |
      | optional web search fallback when PDF retrieval is weak
      v
Groq LLM or local fallback answer
      |
      v
backend returns JSON response
      |
      v
Frontend shows answer + retrieved passages
```

## 4. The Generative AI Part Explained

### 4.1 What is the Gen AI in this project?

The generative AI is the component that produces the final answer text for a user query. It is implemented in `backend/services/llm.py`.

This module does two main things:
- It uses the indexed PDF passages as the primary context
- It calls the Groq-hosted LLM when a valid `GROQ_API_KEY` is available

If the PDF search does not return strong results, it also uses a DuckDuckGo search fallback to get additional context.

### 4.2 How generation is guided

The prompt to the LLM is constructed with:
- a system instruction explaining that the model is a pharmaceutical market analyst
- a request to answer concisely and professionally
- the user question
- the retrieved PDF context, and optionally web search context

This is classic RAG: first retrieve relevant evidence, then ask the language model to answer using that evidence.

### 4.3 Why this is a smart design

- The retrieval step keeps the answer grounded in actual documents
- The generation step makes the answer readable and concise
- The fallback search prevents meaningless answers when the PDF does not mention the query topic

## 5. File-by-file detailed explanation

### 5.1 `backend/main.py`

This is the backend entry point.

#### `load_dotenv()`
- Loads environment variables from `.env`
- Required for `GROQ_API_KEY` if the Groq model is used

#### `FastAPI` and CORS setup
- Creates the API server
- Allows browser-based frontend calls with CORS

#### `QueryRequest` class
- Defines the input JSON shape for `/api/query`
- Expects `{ "user_query": "..." }`

#### `chunk_text(text, chunk_size=250, overlap=50)`
- Splits long text into smaller chunks
- Uses overlapping sliding windows to preserve context
- Returns a list of text chunks

#### `load_pdf_documents()`
- Reads `data/bms_analysis.pdf`
- Extracts text from every page
- Splits page text into paragraph-like chunks
- Uses `chunk_text()` to create smaller passages
- Returns a list of passages such as `Page 1: ...`

#### `load_sample_documents()`
- Reads `data/sample_docs.txt`
- Splits the sample text by blank lines
- Returns a fallback list of passages if the PDF is missing

#### `pipeline = RAGPipeline()`
- Builds the main backend pipeline object
- This object is used by the API for all queries

#### `pipeline.index(initial_docs)`
- Indexes the loaded documents into the vector store
- Runs when the app starts

#### `/api/query` route
- Receives the user query
- Calls `pipeline.query(payload.user_query, top_k=5)`
- Returns JSON with:
  - `source_used`
  - `retrieved`
  - `ai_answer`

#### `if __name__ == "__main__"` demo block
- Runs a demo query when the file is executed directly
- Not used when running with `uvicorn`

### 5.2 `backend/graph/rag_pipeline.py`

This file uses `langgraph` to define the RAG workflow.

#### `RAGState` type
- Defines the fields that travel through the graph:
  - `query`
  - `top_k`
  - optional `retrieved`
  - optional `answer`

#### `RAGPipeline.__init__()`
- Creates the service objects:
  - `EmbeddingService`
  - `VectorStore`
  - `Retriever`
  - `LLMService`
- Compiles the `langgraph` graph once

#### `RAGPipeline._build_graph()`
- Creates a `StateGraph(RAGState)`
- Adds two nodes:
  - `retrieve`
  - `answer`
- Connects them in this order:
  - `START -> retrieve -> answer`
- Compiles the graph into a runnable object

#### `RAGPipeline._retrieve_node(state, config)`
- Takes `query` and `top_k` from the state
- Calls `Retriever.retrieve(...)`
- Returns `{"retrieved": [...]}`
- This adds the retrieved passages back into the graph state

#### `RAGPipeline._answer_node(state, config)`
- Takes `query` and `retrieved` from the state
- Calls `LLMService.generate_answer(...)`
- Returns `{"answer": "..."}`

#### `RAGPipeline.index(documents)`
- Forwards documents to the retriever for indexing
- The retriever builds embeddings and stores them

#### `RAGPipeline.query(user_query, top_k=5)`
- Invokes the compiled graph with the query state
- Returns the final `retrieved` and `answer`

### 5.3 `backend/services/embeddings.py`

This file converts text into numeric vectors.

#### `EmbeddingService.__init__()`
- Creates a `TfidfVectorizer` from scikit-learn
- Uses English stop words
- Tracks whether the vectorizer is fitted

#### `embed_texts(texts)`
- Fits the TF-IDF model on the provided documents
- Converts documents into vectors
- Returns a NumPy matrix of vectors

#### `embed_query(query)`
- Transforms a single query into the same TF-IDF space
- Returns a query vector
- If the model is not fitted yet, it returns an empty vector

### 5.4 `backend/services/vector_store.py`

This file stores embeddings and finds similar passages.

#### `VectorStore.__init__(dim=None)`
- Creates storage for vectors and metadata
- `dim` is set once documents are indexed
- `self._nn` is the NearestNeighbors search object

#### `VectorStore._rebuild_index()`
- Rebuilds the local similarity index
- Uses `metric="cosine"`
- Uses up to 10 neighbors for the model

#### `VectorStore.add(embeddings, metadatas)`
- Adds vectors to the store
- Appends metadata for each document
- Rebuilds the index after new inserts

#### `VectorStore.search(query_embedding, top_k=5)`
- Finds the nearest stored vectors to the query embedding
- Converts distances to similarity scores
- Returns a list of tuples: `(score, metadata, index)`

### 5.5 `backend/services/retriever.py`

This is the retrieval glue between embeddings and the vector store.

#### `Retriever.__init__(embedding_service, vector_store)`
- Receives the embedding service and the vector store

#### `Retriever.add_documents(docs)`
- Converts raw documents into embeddings
- Stores them in the vector database
- Keeps the original text as metadata

#### `Retriever.retrieve(query, top_k=5)`
- Embeds the query
- Searches the vector store for nearest passages
- Formats results as a list of dictionaries
  - `score`
  - `text`
  - `index`

### 5.6 `backend/services/llm.py`

This file contains the generative AI logic.

#### `LLMService.__init__(model_name)`
- Reads `GROQ_API_KEY` from the environment
- If present, initializes a Groq client
- If not present, it will use fallback answers only

#### `LLMService._search_web(query, max_results=3)`
- Uses `DDGS()` to run a DuckDuckGo text search
- Returns a list of top search results
- Each result includes a title, body, and link

#### `LLMService._format_web_context(results)`
- Builds a text block from web results
- Each result becomes a section like:
  - `Web source 1: title - body (href)`

#### `LLMService.generate_answer(query, retrieved_texts)`

This is the heart of the Gen AI component.

1. It builds a `source_context` from the indexed PDF retrieval results.
2. It decides whether web search is needed:
   - if there are no retrieved passages
   - or all retrieved scores are below `0.1`
3. If needed, it calls `_search_web(query)` and adds that context.
4. It creates a professional prompt instructing the model to stay concise.
5. If Groq is configured, it sends the prompt to the Groq chat completion API.
6. If Groq is not available, it returns a fallback summary.

The fallback behavior is:
- if PDF sources exist: return the top passage text with score
- if web results exist: return a concise summary and link
- otherwise: say that no concise answer could be found

### 5.7 `frontend/index.html`

This is the browser UI.

- Has a title, hero section, and query card
- Includes a textarea for the user question
- Shows answer and retrieved passages
- Loads `styles.css` and `app.js`

### 5.8 `frontend/styles.css`

This file controls the look and feel.

- Uses dark gradient background
- Creates glass-style cards
- Styles buttons, text, and layout
- Makes the UI responsive for smaller screens

### 5.9 `frontend/app.js`

This file handles the frontend behavior.

#### `API_URL`
- Points to `http://127.0.0.1:8000/api/query`
- This is the backend endpoint used by the app

#### `setStatus(message, isError=false)`
- Updates the status line below the query
- Uses red text for errors

#### `renderAnswer(text)`
- Displays the AI-generated answer in the answer panel
- Converts newline characters to `<br />`

#### `renderRetrieved(items)`
- Displays each retrieved source item
- Shows the similarity score and passage text
- If there are no passages, it shows a friendly message

#### `runQuery()`
- Reads the user question from the textarea
- Validates that it is not empty
- Sends a POST request to the backend API
- Handles success and error cases
- Updates the UI accordingly

#### Event listeners
- The Run Query button starts the search
- Pressing Enter in the text area also runs the query

## 6. The Complete Technical Flow

### 6.1 At startup

- `backend/main.py` imports `RAGPipeline`
- It loads documents from PDF or sample fallback
- It calls `pipeline.index(initial_docs)`
- This builds the retrieval index before any user query

### 6.2 When the frontend sends a question

1. Browser sends POST `/api/query` with JSON:
   - `{ "user_query": "..." }`
2. `backend/main.py` receives the request
3. `main.py` calls `pipeline.query(user_query, top_k=5)`
4. `graph/rag_pipeline.py` runs the graph:
   - retrieve node
   - answer node
5. `Retriever.retrieve()` returns top passages
6. `LLMService.generate_answer()` creates the final response
7. Backend returns JSON with answer and retrieved passages
8. Frontend displays both

### 6.3 Detailed backend data flow

```
User query
   |
   v
main.py -> RAGPipeline.query(user_query)
   |
   v
rag_pipeline.py graph invoke
   |
   +--> retrieve node -> Retriever.retrieve(query)
   |         |
   |         +--> EmbeddingService.embed_query(query)
   |         +--> VectorStore.search(query_embedding)
   |         +--> return retrieved passages
   |
   +--> answer node -> LLMService.generate_answer(query, retrieved)
             |
             +--> if good PDF results -> Groq or fallback answer
             +--> if weak results -> DuckDuckGo search + answer
```

### 6.4 Why `langgraph` is used

`langgraph` is used in `backend/graph/rag_pipeline.py` to:
- build a clean workflow graph
- separate retrieval and answer generation clearly
- make the execution order explicit
- allow the pipeline to behave like a reusable graph

It is not the LLM itself. It is the orchestration layer.

## 7. The Role of Each Python File

| File | Role |
|------|------|
| `backend/main.py` | Backend API entrypoint and document loading | 
| `backend/graph/rag_pipeline.py` | Orchestrates RAG flow with langgraph |
| `backend/services/embeddings.py` | Creates TF-IDF embeddings |
| `backend/services/vector_store.py` | Stores vectors and searches them |
| `backend/services/retriever.py` | Connects query embedding to search |
| `backend/services/llm.py` | Generates final answer and handles fallback |

## 8. How to Explain the Project to an Interviewer

### In plain language

This project is a simple intelligent question-answering system for pharmaceutical documents.

- It indexes a PDF into a searchable memory
- It uses a vector-based search over those passages
- It sends the top passages to a language model
- It generates a concise answer based on evidence
- If the PDF does not answer the query, it performs a web search fallback

### Why it is useful

It is useful because it prevents the model from making up answers. The system tries to answer from actual indexed documents first, and it only relies on external web retrieval when the document does not cover the topic.

### What makes it a Generative AI project?

The generative AI part is the final answer creation step in `services/llm.py`. The system is not just searching text; it is also generating a short response in natural language.

## 9. Recommended Talking Points for the Interview

- "The backend is a FastAPI service that loads PDF passages at startup."
- "It uses TF-IDF embeddings and cosine search for retrieval."
- "The retrieval step happens in `services/retriever.py`."
- "The generative step is in `services/llm.py`, which calls Groq if available."
- "If the PDF does not contain the answer, it uses DuckDuckGo as a fallback source."
- "The project uses `langgraph` to structure the RAG workflow cleanly."

## 10. Quick Run Instructions

1. Start backend:
```bash
cd "BMS/backend"
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

2. Start frontend:
```bash
cd "BMS/frontend"
python -m http.server 3000
```

3. Open:
```bash
http://127.0.0.1:3000
```

## 11. Notes for the Interview

- `backend/services/embeddings.py` is a classical TF-IDF embedding method.
- `backend/services/vector_store.py` is an in-memory vector store.
- `backend/services/llm.py` is the generative AI layer.
- `langgraph` is the orchestration layer, not the actual model.
- The frontend is a simple static HTML + JavaScript UI.

## 12. Appendix: Function Summary

### `main.py`
- `chunk_text(text, chunk_size=250, overlap=50)`
- `load_pdf_documents()`
- `load_sample_documents()`
- `process_guarded_query(payload)`

### `rag_pipeline.py`
- `RAGPipeline.__init__()`
- `RAGPipeline._build_graph()`
- `RAGPipeline._retrieve_node(state, config)`
- `RAGPipeline._answer_node(state, config)`
- `RAGPipeline.index(documents)`
- `RAGPipeline.query(user_query, top_k=5)`

### `embeddings.py`
- `EmbeddingService.__init__()`
- `EmbeddingService.embed_texts(texts)`
- `EmbeddingService.embed_query(query)`

### `vector_store.py`
- `VectorStore.__init__(dim=None)`
- `VectorStore._rebuild_index()`
- `VectorStore.add(embeddings, metadatas)`
- `VectorStore.search(query_embedding, top_k=5)`

### `retriever.py`
- `Retriever.__init__(embedding_service, vector_store)`
- `Retriever.add_documents(docs)`
- `Retriever.retrieve(query, top_k=5)`

### `llm.py`
- `LLMService.__init__(model_name="llama-3.1-8b-instant")`
- `LLMService._search_web(query, max_results=3)`
- `LLMService._format_web_context(results)`
- `LLMService.generate_answer(query, retrieved_texts)`

### `frontend/app.js`
- `setStatus(message, isError=false)`
- `renderAnswer(text)`
- `renderRetrieved(items)`
- `runQuery()`

---

This document is designed to be copy-pasted into Word and used as a project report and reference. It describes the project structure, flow, and every major Python function in plain language.