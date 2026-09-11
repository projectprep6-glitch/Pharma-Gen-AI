"""
Semantic chunking with parent-child hierarchy.

This module implements layout-aware and semantic chunking strategies:
- Semantic chunking: splits on semantic boundaries (when embedding distance exceeds a threshold)
- Parent-child chunking: creates large parent chunks (2000 tokens) and small child chunks (200 tokens)
  for hierarchical retrieval (retrieve child, pass parent to LLM).
"""
import numpy as np
from typing import List, Tuple


class SemanticChunker:
    """Create semantically coherent text chunks."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph breaks (double newlines).
        
        This respects structural boundaries in the document,
        preserving headers and section context.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs

    def chunk_by_sentences(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split text into sentence-level chunks with overlap.
        
        This is more semantically aware than character-level chunking
        because it respects sentence boundaries.
        """
        sentences = text.replace("\n", " ").split(". ")
        sentences = [s.strip() + "." if s.strip() else s for s in sentences]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def chunk_with_semantic_similarity(
        self, text: str, embedding_func, threshold: float = 0.5
    ) -> List[str]:
        """Split text based on semantic similarity between consecutive sentences.
        
        When the embedding distance between two sentences exceeds the threshold,
        we start a new chunk. This preserves semantic coherence.
        
        Args:
            text: the text to chunk
            embedding_func: a function that takes text and returns embedding vector
            threshold: cosine distance threshold for chunk boundary
            
        Returns:
            List of semantically coherent chunks
        """
        sentences = text.replace("\n", " ").split(". ")
        sentences = [s.strip() + "." if s.strip() else s for s in sentences]
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sent1 = sentences[i - 1]
            sent2 = sentences[i]
            
            emb1 = np.array(embedding_func(sent1))
            emb2 = np.array(embedding_func(sent2))
            
            # cosine similarity
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            distance = 1 - sim
            
            if distance > threshold:
                # Semantic boundary detected
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i]]
            else:
                current_chunk_sentences.append(sentences[i])
        
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        
        return chunks


class ParentChildChunker:
    """Create hierarchical parent-child chunks for retrieval.
    
    Large parent chunks (2000 tokens) preserve broad context.
    Small child chunks (200 tokens) are used for vector search.
    When a child is retrieved, the parent is passed to the LLM.
    """

    def __init__(self, parent_size: int = 2000, child_size: int = 200, overlap: int = 50):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def create_hierarchy(self, text: str) -> List[Tuple[str, str, int]]:
        """Create parent-child chunks.
        
        Returns:
            List of tuples: (parent_chunk, child_chunk, parent_index)
            
        This preserves the relationship between children and parents.
        """
        # Split into parent chunks first
        parent_chunks = self._split_by_tokens(text, self.parent_size, self.overlap)
        
        hierarchy = []
        for parent_idx, parent in enumerate(parent_chunks):
            # Split each parent into children
            children = self._split_by_tokens(parent, self.child_size, self.overlap // 2)
            for child in children:
                hierarchy.append((parent, child, parent_idx))
        
        return hierarchy

    def _split_by_tokens(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        """Simple token-based splitting (approximated by word count).
        
        Real implementation would use a tokenizer like tiktoken.
        Here we approximate: 1 token ~= 0.75 words.
        """
        words = text.split()
        max_words = int(max_tokens * 0.75)
        overlap_words = int(overlap * 0.75)
        
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + max_words]
            chunks.append(" ".join(chunk_words))
            i += max_words - overlap_words
        
        return chunks
