# retrieval/citation_mapper.py

import re
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from sentence_transformers import SentenceTransformer, util
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD_SUPPORTED, SIMILARITY_THRESHOLD_WEAK


class CitationMapper:
    def __init__(self):
        # Same embedding model as vector store — consistency matters
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ CitationMapper ready — using {EMBEDDING_MODEL}")

    def split_into_sentences(self, text: str) -> list[str]:
        """Split answer text into individual sentences."""
        # Split on period/exclamation/question followed by space or end
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out empty or very short fragments
        sentences = [s.strip() for s in raw if len(s.strip()) > 20]
        return sentences

    def map_citations(self, answer: str, source_chunks: list[dict]) -> list[dict]:
        """
        For each sentence in the answer:
        - compute similarity against every source chunk
        - find best matching chunk
        - assign citation label [1], [2], etc.
        - assign verdict: SUPPORTED / WEAK / UNSUPPORTED

        Returns list of sentence objects with citation info.
        """
        sentences = self.split_into_sentences(answer)

        if not sentences:
            return []

        # Encode all source chunks once (efficient)
        chunk_texts = [c["text"] for c in source_chunks]
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=True)

        cited_sentences = []

        for sentence in sentences:
            # Encode this sentence
            sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)

            # Compute similarity against all chunks
            similarities = util.cos_sim(sentence_embedding, chunk_embeddings)[0]
            similarities = similarities.tolist()

            # Find best matching chunk
            best_idx = similarities.index(max(similarities))
            best_score = round(max(similarities), 4)
            best_chunk = source_chunks[best_idx]

            # Assign verdict
            if best_score >= SIMILARITY_THRESHOLD_SUPPORTED:
                verdict = "SUPPORTED"
            elif best_score >= SIMILARITY_THRESHOLD_WEAK:
                verdict = "WEAK"
            else:
                verdict = "UNSUPPORTED"

            cited_sentences.append({
                "sentence": sentence,
                "citation_index": best_idx + 1,    # [1]-based for display
                "citation_source": best_chunk["source"],
                "citation_page": best_chunk["page"],
                "citation_chunk_id": best_chunk["chunk_id"],
                "similarity": best_score,
                "verdict": verdict
            })

        return cited_sentences

    def format_cited_answer(self, cited_sentences: list[dict]) -> str:
        """
        Format the answer with inline citations.
        Example: "Spring Boot was used for the backend.[1]"
        """
        lines = []
        for item in cited_sentences:
            if item["verdict"] in ("SUPPORTED", "WEAK"):
                lines.append(f"{item['sentence']} [{item['citation_index']}]")
            else:
                lines.append(f"{item['sentence']} [⚠️ unverified]")
        return " ".join(lines)

    def format_sources_list(self, source_chunks: list[dict]) -> str:
        """
        Format numbered source list for display.
        Example: [1] Laabh_Gupta.pdf — Page 1
        """
        lines = []
        for i, chunk in enumerate(source_chunks):
            lines.append(
                f"[{i+1}] {chunk['source']} — Page {chunk['page']} "
                f"(similarity: {chunk['similarity']})"
            )
        return "\n".join(lines)