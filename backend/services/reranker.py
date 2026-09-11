"""
Cross-Encoder Reranking for retrieval refinement.

A Cross-Encoder scores the relevance of a query-document pair using
full cross-attention, rather than scoring embeddings independently.

This is computationally more expensive but much more accurate.
We use it to rerank the top-K candidates from dense retrieval.
"""
from typing import List, Tuple
import numpy as np


class CrossEncoderReranker:
    """Rerank retrieval candidates using a cross-encoder model.
    
    In production, this would use a model like:
    - sentence-transformers/bge-reranker-large
    - Cohere's Rerank API
    
    For this demo, we provide a lightweight fallback.
    """

    def __init__(self, model_name: str = "bge-reranker-base"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize the cross-encoder model.
        
        In production, this would load a real cross-encoder.
        For now, we provide a simple heuristic-based fallback.
        """
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            print(f"CrossEncoder model loading failed: {e}. Using heuristic fallback.")
            self.model = None

    def rerank(
        self, query: str, candidates: List[Tuple[float, str, int]], top_k: int = 3
    ) -> List[Tuple[float, str, int]]:
        """Rerank retrieval candidates.
        
        Args:
            query: the user query
            candidates: list of (score, text, index) tuples from dense retrieval
            top_k: number of results to return after reranking
            
        Returns:
            Reranked list of top_k candidates
        """
        if not candidates:
            return []

        if self.model:
            # Real cross-encoder reranking
            query_document_pairs = [[query, cand[1]] for cand in candidates]
            scores = self.model.predict(query_document_pairs)

            # Combine original score with cross-encoder score
            reranked = [
                (float(scores[i]), candidates[i][1], candidates[i][2])
                for i in range(len(candidates))
            ]
            reranked = sorted(reranked, key=lambda x: x[0], reverse=True)
            return reranked[:top_k]
        else:
            # Heuristic fallback: boost scores for exact keyword matches
            query_tokens = set(query.lower().split())
            reranked = []

            for score, text, idx in candidates:
                text_tokens = set(text.lower().split())
                overlap = len(query_tokens & text_tokens)
                boost = overlap * 0.05  # Bonus per matching token
                new_score = score + boost
                reranked.append((new_score, text, idx))

            reranked = sorted(reranked, key=lambda x: x[0], reverse=True)
            return reranked[:top_k]
