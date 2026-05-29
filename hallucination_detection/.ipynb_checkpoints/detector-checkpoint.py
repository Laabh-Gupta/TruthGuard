# hallucination_detection/detector.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer, util
from config import (
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD_SUPPORTED,
    SIMILARITY_THRESHOLD_WEAK
)


class HallucinationDetector:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ HallucinationDetector ready")

    def analyze(self, answer: str, source_chunks: list[dict]) -> dict:
        """
        Full hallucination analysis of an answer against source chunks.

        Returns:
        {
            "sentences": [...per sentence results...],
            "grounding_score": 0.87,
            "supported_count": 4,
            "weak_count": 1,
            "unsupported_count": 1,
            "verdict": "MOSTLY GROUNDED",
            "unsupported_sentences": [...],
        }
        """
        sentences = self._split_sentences(answer)

        if not sentences:
            return self._empty_result()

        # Encode everything at once — efficient
        chunk_texts = [c["text"] for c in source_chunks]
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=True)
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)

        sentence_results = []

        for i, sentence in enumerate(sentences):
            sent_embedding = sentence_embeddings[i]

            # Score against every chunk
            similarities = util.cos_sim(sent_embedding, chunk_embeddings)[0]
            similarities = similarities.tolist()

            best_idx = similarities.index(max(similarities))
            best_score = round(max(similarities), 4)
            all_scores = [round(s, 4) for s in similarities]

            # Verdict
            if best_score >= SIMILARITY_THRESHOLD_SUPPORTED:
                verdict = "SUPPORTED"
            elif best_score >= SIMILARITY_THRESHOLD_WEAK:
                verdict = "WEAK"
            else:
                verdict = "UNSUPPORTED"

            # Best supporting evidence
            best_chunk = source_chunks[best_idx]

            sentence_results.append({
                "sentence": sentence,
                "verdict": verdict,
                "best_score": best_score,
                "all_scores": all_scores,
                "best_chunk_index": best_idx,
                "best_chunk_text": best_chunk["text"][:200],  # preview
                "best_chunk_source": best_chunk["source"],
                "best_chunk_page": best_chunk["page"],
            })

        # Aggregate scores
        total = len(sentence_results)
        supported = sum(1 for s in sentence_results if s["verdict"] == "SUPPORTED")
        weak = sum(1 for s in sentence_results if s["verdict"] == "WEAK")
        unsupported = sum(1 for s in sentence_results if s["verdict"] == "UNSUPPORTED")

        # Grounding score:
        # SUPPORTED = full credit (1.0)
        # WEAK = half credit (0.5)
        # UNSUPPORTED = no credit (0.0)
        grounding_score = round(
            (supported * 1.0 + weak * 0.5) / total, 4
        ) if total > 0 else 0.0

        # Overall verdict
        if grounding_score >= 0.80:
            overall_verdict = "HIGHLY GROUNDED"
        elif grounding_score >= 0.60:
            overall_verdict = "MOSTLY GROUNDED"
        elif grounding_score >= 0.40:
            overall_verdict = "PARTIALLY GROUNDED"
        else:
            overall_verdict = "POORLY GROUNDED"

        unsupported_sentences = [
            s for s in sentence_results if s["verdict"] == "UNSUPPORTED"
        ]

        return {
            "sentences": sentence_results,
            "grounding_score": grounding_score,
            "grounding_percent": f"{round(grounding_score * 100, 1)}%",
            "supported_count": supported,
            "weak_count": weak,
            "unsupported_count": unsupported,
            "total_sentences": total,
            "overall_verdict": overall_verdict,
            "unsupported_sentences": unsupported_sentences
        }

    def _split_sentences(self, text: str) -> list[str]:
        """
        Robust sentence splitter.
        Handles run-on sentences from LLMs by also splitting on commas
        when sentences are very long.
        """
        import re

        # Primary split on sentence endings
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = []

        for s in raw:
            s = s.strip()
            if len(s) < 20:
                continue
            # If a "sentence" is very long (LLM run-on), split further on semicolons
            if len(s) > 300:
                sub = re.split(r'(?<=;)\s+', s)
                sentences.extend([x.strip() for x in sub if len(x.strip()) > 20])
            else:
                sentences.append(s)

        return sentences

    def _empty_result(self) -> dict:
        return {
            "sentences": [],
            "grounding_score": 0.0,
            "grounding_percent": "0%",
            "supported_count": 0,
            "weak_count": 0,
            "unsupported_count": 0,
            "total_sentences": 0,
            "overall_verdict": "NO CONTENT",
            "unsupported_sentences": []
        }