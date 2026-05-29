# retrieval/vector_store.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"     # silence those telemetry warnings

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_RESULTS
)


class VectorStore:
    def __init__(self):
        # Persistent client — saves to disk so you don't re-embed every time
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Use sentence-transformers for embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"✅ VectorStore ready — collection: {CHROMA_COLLECTION_NAME}")
        print(f"   Documents in store: {self.collection.count()}")

    def add_chunks(self, chunks: list[dict]):
        """Add chunks to ChromaDB. Skips duplicates by chunk_id."""

        # Filter out chunks already in the store
        existing_ids = set(self.collection.get()["ids"])
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

        if not new_chunks:
            print("⚠️  All chunks already in store. Skipping.")
            return

        self.collection.add(
            documents=[c["text"] for c in new_chunks],
            metadatas=[{
                "source": c["source"],
                "page": c["page"],
                "chunk_id": c["chunk_id"]
            } for c in new_chunks],
            ids=[c["chunk_id"] for c in new_chunks]
        )

        print(f"✅ Added {len(new_chunks)} chunks to vector store")
        print(f"   Total in store: {self.collection.count()}")

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """Retrieve top_k most relevant chunks for a query."""

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "chunk_id": meta["chunk_id"],
                "similarity": round(1 - dist, 4)  # cosine distance → similarity
            })

        return chunks

    def clear(self):
        """Wipe the collection — useful during testing."""
        self.client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        print("🗑️  Vector store cleared")