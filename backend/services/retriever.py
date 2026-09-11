"""
Advanced retriever with hybrid search, parent-child chunks, and reranking.

This combines:
- Dense retrieval (vector embeddings)
- Sparse retrieval (BM25 keyword search)
- Cross-encoder reranking
- Parent-child chunk hierarchies
"""
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .bm25_search import BM25Retriever
from .reranker import CrossEncoderReranker
import numpy as np
from typing import List, Dict


class Retriever:
    """Advanced retriever with hybrid search and hierarchical chunks."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        use_bm25: bool = True,
        use_reranking: bool = True,
    ):
        self.emb = embedding_service
        self.store = vector_store
        self.bm25 = BM25Retriever() if use_bm25 else None
        self.reranker = CrossEncoderReranker() if use_reranking else None
        self.use_bm25 = use_bm25
        self.use_reranking = use_reranking

    def add_documents(self, docs: List[str], parents: List[str] = None):
        """Index documents with optional parent chunks.

        docs: list of strings (child chunks)
        parents: list of parent chunks (optional)
        """
        texts = list(docs)
        embeddings = self.emb.embed_texts(texts)

        # Store in vector store with parent references
        self.store.add(embeddings, texts, parents=parents)

        # Index in BM25 if enabled
        if self.bm25:
            self.bm25.index_documents(texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve with hybrid search and reranking.

        1. Dense retrieval (vector search)
        2. Sparse retrieval (BM25 keyword search) if enabled
        3. Fusion (Reciprocal Rank Fusion)
        4. Reranking if enabled
        5. Return parent chunk if available

        Returns:
            List of dicts with score, text, and optional parent_text
        """
        # Step 1: Dense retrieval
        q_emb = self.emb.embed_query(query)
        dense_results = self.store.search(q_emb, top_k=max(top_k, 10))

        # Step 2: Sparse retrieval (BM25)
        sparse_results = []
        if self.bm25:
            sparse_results = self.bm25.search(query, top_k=max(top_k, 10))

        # Step 3: Fusion using Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Step 4: Reranking
        if self.use_reranking and self.reranker:
            # Prepare candidates for reranking
            candidates = [
                (score, text, idx) for score, text, idx, parent in fused_results
            ]
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)

            # Rebuild results with parent information
            retrieved = []
            for score, text, idx in reranked:
                parent_text = None
                for orig_score, orig_text, orig_idx, orig_parent in fused_results:
                    if orig_idx == idx:
                        parent_text = orig_parent
                        break
                retrieved.append(
                    {
                        "score": float(score),
                        "text": text,
                        "index": idx,
                        "parent_text": parent_text,
                    }
                )
        else:
            # No reranking, just format results
            retrieved = [
                {
                    "score": float(score),
                    "text": text,
                    "index": idx,
                    "parent_text": parent,
                }
                for score, text, idx, parent in fused_results[:top_k]
            ]

        return retrieved

    def _reciprocal_rank_fusion(
        self, dense_results, sparse_results, k: float = 60.0
    ) -> List:
        """Fuse dense and sparse retrieval results using RRF.

        RRF score = 1 / (k + rank)
        where k is typically 60.
        """
        # Build rank dictionaries
        dense_ranks = {}
        for rank, (score, text, idx, parent) in enumerate(dense_results):
            if idx not in dense_ranks:
                dense_ranks[idx] = (1.0 / (k + rank), text, parent)

        sparse_ranks = {}
        for rank, (score, text, idx) in enumerate(sparse_results):
            if idx not in sparse_ranks:
                sparse_ranks[idx] = 1.0 / (k + rank)

        # Combine scores
        fused = {}
        for idx, (d_score, text, parent) in dense_ranks.items():
            s_score = sparse_ranks.get(idx, 0.0)
            fused[idx] = (d_score + s_score, text, parent)

        # Add sparse-only results
        for idx, s_score in sparse_ranks.items():
            if idx not in fused:
                # Get text from sparse results
                for score, text, sp_idx in sparse_results:
                    if sp_idx == idx:
                        fused[idx] = (s_score, text, None)
                        break

        # Sort by fused score
        sorted_results = sorted(fused.items(), key=lambda x: x[1][0], reverse=True)

        # Return as list of (score, text, idx, parent)
        return [
            (score, text, idx, parent) for idx, (score, text, parent) in sorted_results
        ]
