"""
Cache local LRU pour les embeddings, évite de recalculer plusieurs fois le même texte.
"""

import numpy as np
from collections import OrderedDict
from typing import Optional


class LRUEmbeddingCache:
    """Cache thread‑safe (asyncio) des embeddings récents."""

    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, text: str) -> Optional[np.ndarray]:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        return None

    def set(self, text: str, embedding: np.ndarray):
        if text in self._cache:
            self._cache.move_to_end(text)
        self._cache[text] = embedding
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()