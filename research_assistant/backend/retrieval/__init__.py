"""Dense retrieval: embeddings + ChromaDB vector store."""

from backend.retrieval.vector_store import BgeEmbedder, Embedder, VectorStore, get_vector_store

__all__ = ["BgeEmbedder", "Embedder", "VectorStore", "get_vector_store"]
