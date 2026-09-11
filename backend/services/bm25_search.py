"""
Sparse retrieval using BM25 (Best Matching 25).

BM25 is a keyword-based retrieval algorithm that excels at:
- Exact keyword matches
- Product names, drug names, serial numbers
- Technical terms and abbreviations

It complements dense vector search by handling exact matches better.
"""
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import numpy as np


class BM25Retriever:
    """Keyword-based sparse retrieval using BM25."""

    def __init__(self):
        self.bm25 = None
        self.documents = []  # Store original documents
        self.tokenized_docs = []

    def index_documents(self, documents: List[str]):
        """Index a list of documents with BM25.
        
        Args:
            documents: list of document strings
        """
        self.documents = documents
        # Simple tokenization: split by spaces and lowercase
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, str, int]]:
        """Search for documents using BM25.
        
        Args:
            query: the search query
            top_k: number of results to return
            
        Returns:
            List of tuples: (score, document_text, document_index)
        """
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((float(scores[idx]), self.documents[idx], int(idx)))
        
        return results
