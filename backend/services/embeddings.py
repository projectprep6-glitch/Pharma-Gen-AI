"""
Simple embedding service using scikit-learn's TF-IDF.
This avoids the heavy sentence-transformers dependency and keeps the
RAG demo runnable in a basic Python environment.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingService:
    """Create text vector embeddings using TF-IDF.

    This service fits a TF-IDF vectorizer on indexed documents and
    transforms queries into the same feature space.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.fitted = False

    def embed_texts(self, texts):
        """Fit the TF-IDF vectorizer and return embeddings for the given texts."""
        if not texts:
            return np.zeros((0, 0), dtype=float)
        vectors = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return vectors.toarray()

    def embed_query(self, query: str):
        """Embed a single query using the fitted TF-IDF vectorizer."""
        if not self.fitted:
            return np.zeros((0,), dtype=float)
        vector = self.vectorizer.transform([query])
        return vector.toarray()[0]
