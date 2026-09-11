import os
import pathlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pypdf import PdfReader

from graph.rag_pipeline import RAGPipeline
from services.chunking import SemanticChunker, ParentChildChunker
from services.embeddings import EmbeddingService

# Explicitly pull configurations from the local .env file
load_dotenv()

app = FastAPI(title="Chatbot Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    user_query: str


def chunk_text_hierarchical(text: str) -> tuple:
    """Create hierarchical chunks: parent (~2000 tokens) and children (~200 tokens).
    
    Returns: (parent_chunks, child_chunks, parent_references)
    """
    chunker = ParentChildChunker(parent_size=2000, child_size=200)
    hierarchy = chunker.create_hierarchy(text)
    
    parent_chunks = []
    child_chunks = []
    parent_refs = []
    
    for parent, child, parent_idx in hierarchy:
        parent_chunks.append(parent)
        child_chunks.append(child)
        parent_refs.append(parent)
    
    return parent_chunks, child_chunks, parent_refs


def load_pdf_documents() -> tuple:
    """Read the PDF and split it into hierarchical parent-child chunks.
    
    Returns: (child_chunks, parent_references) for indexing retriever with parent info
    """
    pdf_path = os.path.join(os.path.dirname(__file__), "data", "bms_analysis.pdf", "NLP Introduction.pdf")
    if not os.path.exists(pdf_path):
        return [], []

    all_children = []
    all_parents = []
    
    try:
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            
            # Create hierarchical chunks
            parent_chunks, child_chunks, parent_refs = chunk_text_hierarchical(text)
            
            # Add page reference
            for child, parent in zip(child_chunks, parent_refs):
                prefixed_child = f"Page {page_number}: {child}"
                all_children.append(prefixed_child)
                all_parents.append(f"Page {page_number}: {parent}")
    
    except Exception as e:
        print(f"Error reading PDF: {e}")

    return all_children, all_parents


def load_sample_documents() -> tuple:
    """Load sample documents with hierarchical chunking.
    
    Returns: (child_chunks, parent_references)
    """
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_docs.txt")
    if not os.path.exists(sample_path):
        return [], []
    
    raw = pathlib.Path(sample_path).read_text(encoding="utf-8")
    documents = [p.strip() for p in raw.split("\n\n") if p.strip()]
    
    all_children = []
    all_parents = []
    
    for doc in documents:
        parent_chunks, child_chunks, parent_refs = chunk_text_hierarchical(doc)
        all_children.extend(child_chunks)
        all_parents.extend(parent_refs)
    
    return all_children, all_parents


pipeline = RAGPipeline()
initial_children, initial_parents = load_pdf_documents()
if not initial_children:
    initial_children, initial_parents = load_sample_documents()

if initial_children:
    # Index with parent-child hierarchy
    pipeline.index_with_parents(initial_children, initial_parents)


@app.post("/api/query")
async def process_guarded_query(payload: QueryRequest):
    """Run the RAG pipeline and return retrieved documents plus the generated answer."""
    result = pipeline.query(payload.user_query, top_k=5)
    return {
        "source_used": "RAG Pipeline",
        "retrieved": result["retrieved"],
        "ai_answer": result["answer"],
    }


if __name__ == "__main__":
    # Demo runner: index the sample docs and run a single query.
    # This does not start the FastAPI server; it simply demonstrates the RAG flow.
    try:
        from graph.rag_pipeline import RAGPipeline
        import pathlib

        data_file = pathlib.Path(__file__).parent / "data" / "sample_docs.txt"
        raw = data_file.read_text(encoding="utf-8").strip()
        # Split sample docs by blank lines into logical passages
        docs = [p.strip() for p in raw.split("\n\n") if p.strip()]

        pipeline = RAGPipeline()
        print(f"Indexing {len(docs)} sample documents...")
        pipeline.index(docs)

        query = "What is Retrieval-Augmented Generation (RAG)?"
        print(f"\nRunning demo query: {query}\n")
        result = pipeline.query(query, top_k=3)

        print("Retrieved passages:")
        for idx, r in enumerate(result["retrieved"]):
            print(f"{idx+1}. (score={r['score']:.3f}) {r['text']}")

        print("\nGenerated answer:\n")
        print(result["answer"])

    except Exception as e:
        print("Demo failed:", e)
