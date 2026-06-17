"""Embedding stage implementations."""

from __future__ import annotations

import hashlib
import logging

from config import settings
from ingestion.document_indexing.types import DocumentChunk, EmbeddedChunk

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder:
    """Embed chunks with a local Hugging Face embedding model."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool | None = None,
    ) -> None:
        self.model = None
        self.dimension = 384
        self._use_fallback = False

        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            self.model = HuggingFaceEmbeddings(
                model_name=model_name or settings.document_embedding_model_name,
                model_kwargs={"device": device or settings.document_embedding_device},
                encode_kwargs={
                    "normalize_embeddings": (
                        settings.document_embedding_normalize_embeddings
                        if normalize_embeddings is None
                        else normalize_embeddings
                    )
                },
            )
        except Exception as exc:
            self._use_fallback = True
            logger.warning(
                "Falling back to offline hash embeddings because the Hugging Face "
                "model could not be loaded: %s",
                exc,
            )

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        if self._use_fallback or self.model is None:
            vectors = [self._hash_embedding(chunk.text) for chunk in chunks]
        else:
            try:
                vectors = self.model.embed_documents([chunk.text for chunk in chunks])
            except Exception as exc:
                logger.warning(
                    "Falling back to offline hash embeddings because embedding failed: %s",
                    exc,
                )
                self._use_fallback = True
                vectors = [self._hash_embedding(chunk.text) for chunk in chunks]

        return [
            EmbeddedChunk(chunk=chunk, embedding=vector)
            for chunk, vector in zip(chunks, vectors)
        ]

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        if not text:
            return vector

        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                idx = digest[i] % self.dimension
                magnitude = (digest[i + 1] / 255.0) if i + 1 < len(digest) else 0.0
                vector[idx] += magnitude

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector
