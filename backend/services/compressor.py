"""
Contextual Compression for retrieved text.

This module compresses retrieved documents by extracting only the sentences
that directly answer the query, removing surrounding noise and filler.

This reduces token usage and improves LLM focus.
"""
from typing import List


class ContextualCompressor:
    """Compress retrieved text by extracting relevant sentences."""

    def __init__(self):
        pass

    def compress(self, query: str, text: str) -> str:
        """Extract sentences relevant to the query.
        
        Simple heuristic approach:
        - Split text into sentences
        - Score each sentence by keyword overlap with query
        - Return only high-scoring sentences
        
        Args:
            query: the user query
            text: the retrieved document text
            
        Returns:
            Compressed text containing only relevant sentences
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return text

        query_tokens = set(query.lower().split())
        scored_sentences = []

        for sent in sentences:
            sent_tokens = set(sent.lower().split())
            overlap = len(query_tokens & sent_tokens)
            score = overlap / (len(sent_tokens) + 1e-8)  # Normalize by sentence length
            scored_sentences.append((score, sent))

        # Filter sentences with non-zero overlap
        relevant_sentences = [sent for score, sent in scored_sentences if score > 0]

        if not relevant_sentences:
            # If no overlap, return original text (fallback)
            return text

        # Preserve order from original text
        compressed = " ".join(relevant_sentences)
        return compressed.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Simple approach: split by period, question mark, exclamation mark.
        """
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
