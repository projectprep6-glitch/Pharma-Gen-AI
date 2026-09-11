"""
Advanced RAG pipeline with production optimizations.

Orchestration layers:
1. Query rewriting (performed in LLMService)
2. Retrieve: Dense + sparse (BM25) + reranking + hierarchical chunks
3. Compress: Extract relevant sentences only
4. Answer: Generate with Groq or fallback (temperature=0.0, JSON mode)
"""
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import START, StateGraph

from services.embeddings import EmbeddingService
from services.llm import LLMService
from services.retriever import Retriever
from services.vector_store import VectorStore
from services.compressor import ContextualCompressor


class RAGState(TypedDict):
    query: str
    top_k: int
    retrieved: NotRequired[list[dict]]
    compressed: NotRequired[str]
    answer: NotRequired[str]


class RAGPipeline:
    """Advanced RAG pipeline with all production optimizations."""

    def __init__(self):
        # Build the retrieval and LLM components.
        self.emb_service = EmbeddingService()
        self.store = VectorStore(dim=None)
        # Retriever with hybrid search (dense + BM25) and reranking
        self.retriever = Retriever(
            self.emb_service, 
            self.store, 
            use_bm25=True, 
            use_reranking=True
        )
        self.compressor = ContextualCompressor()
        self.llm = LLMService(confidence_threshold=0.7)

        # Build and compile the langgraph graph.
        self.graph = self._build_graph()

    def _build_graph(self):
        """Create the langgraph state graph with compression step."""
        builder = StateGraph(RAGState)

        # Add nodes
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("compress", self._compress_node)
        builder.add_node("answer", self._answer_node)

        # Connect graph: START -> retrieve -> compress -> answer
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "compress")
        builder.add_edge("compress", "answer")

        return builder.compile(name="AdvancedRAGPipeline")

    def _retrieve_node(self, state: RAGState, config):
        """Retrieve top-k passages with hybrid search and reranking."""
        return {
            "retrieved": self.retriever.retrieve(
                state["query"], top_k=state["top_k"]
            )
        }

    def _compress_node(self, state: RAGState, config):
        """Compress retrieved passages using LLM-based contextual compression."""
        retrieved = state.get("retrieved", [])
        if not retrieved:
            return {"compressed": ""}
        
        # Extract text from retrieved documents
        texts = [item.get("text", "") for item in retrieved]
        query = state["query"]
        
        # Compress each passage
        compressed_parts = []
        for text in texts:
            compressed = self.compressor.compress(query, text)
            if compressed:
                compressed_parts.append(compressed)
        
        return {"compressed": "\n---\n".join(compressed_parts)}

    def _answer_node(self, state: RAGState, config):
        """Generate answer with full optimization pipeline."""
        return {
            "answer": self.llm.generate_answer(
                state["query"], 
                state.get("retrieved", [])
            )
        }

    def index(self, documents: list):
        """Index documents without parent information."""
        self.retriever.add_documents(documents, parents=None)

    def index_with_parents(self, children_docs: list, parent_docs: list):
        """Index documents with parent-child hierarchy.
        
        children_docs: list of child chunks (what we retrieve)
        parent_docs: list of parent chunks (what we return to LLM)
        """
        if len(children_docs) != len(parent_docs):
            raise ValueError("Children and parent lists must have equal length")
        
        self.retriever.add_documents(children_docs, parents=parent_docs)

    def query(self, user_query: str, top_k: int = 5):
        """Run the complete optimized RAG flow."""
        result = self.graph.invoke({
            "query": user_query, 
            "top_k": top_k
        })
        return {
            "retrieved": result.get("retrieved", []), 
            "compressed": result.get("compressed", ""),
            "answer": result.get("answer", "")
        }
