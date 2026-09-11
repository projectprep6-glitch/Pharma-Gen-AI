"""
A minimal in-memory vector store using scikit-learn's NearestNeighbors.
Enhanced to support parent-child chunk relationships for hierarchical retrieval.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Optional


class VectorStore:
    """Store vectors and associated metadata with parent-child hierarchy support.

    - `add(embeddings, metadatas, parents=None)` stores embeddings, metadata, and optional parent references.
    - `search(query_embedding, top_k)` returns nearest items with scores.
    """

    def __init__(self, dim: int = None):
        self.dim = dim
        self.vectors = np.zeros((0, dim or 0), dtype=float)
        self.metadatas = []  # list of dicts or strings
        self.parents = []  # parent chunk for each child chunk
        self._nn = None

    def _rebuild_index(self):
        if len(self.vectors) == 0:
            self._nn = None
            return
        n_neighbors = min(len(self.vectors), 10)
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self._nn.fit(self.vectors)

    def add(self, embeddings, metadatas, parents: Optional[List[str]] = None):
        """Add new vectors with optional parent chunk references.

        embeddings: np.ndarray shape (n, dim)
        metadatas: list with length n
        parents: list of parent chunks (optional) with length n
        """
        if len(embeddings) == 0:
            return
        if self.dim is None:
            self.dim = embeddings.shape[1]
            self.vectors = embeddings.copy()
        else:
            self.vectors = np.vstack([self.vectors, embeddings])
        self.metadatas.extend(metadatas)
        if parents:
            self.parents.extend(parents)
        else:
            self.parents.extend([None] * len(metadatas))
        self._rebuild_index()

    def search(self, query_embedding, top_k: int = 5) -> List[Tuple[float, str, int, Optional[str]]]:
        """Return list of (score, metadata, index, parent_chunk).

        Score is cosine similarity in [0,1].
        parent_chunk is None if not stored, or the parent chunk text if available.
        """
        if self._nn is None:
            return []
        distances, indices = self._nn.kneighbors(
            [query_embedding], n_neighbors=min(top_k, len(self.vectors))
        )
        distances = distances[0]
        indices = indices[0]
        results = []
        for dist, idx in zip(distances, indices):
            similarity = 1.0 - dist  # convert cosine distance to similarity
            parent = self.parents[idx] if idx < len(self.parents) else None
            results.append((float(similarity), self.metadatas[idx], int(idx), parent))
        return results
