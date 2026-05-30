# evaluation/confidence_scorer.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConfidenceScorer:
    """
    Combines hallucination detection scores into a
    single human-readable confidence report.
    """

    def score(self, hallucination_report: dict, retrieval_scores: list[float]) -> dict:
        """
        Takes hallucination report + retrieval similarity scores.
        Returns a unified confidence score with explanation.
        """

        grounding = hallucination_report["grounding_score"]
        supported = hallucination_report["supported_count"]
        weak = hallucination_report["weak_count"]
        unsupported = hallucination_report["unsupported_count"]
        total = hallucination_report["total_sentences"]

        # Average retrieval score — how relevant were the chunks?
        avg_retrieval = round(sum(retrieval_scores) / len(retrieval_scores), 4) \
            if retrieval_scores else 0.0

        # Penalize heavily for unsupported sentences
        unsupported_penalty = unsupported * 0.15

        # Final confidence = weighted combination
        raw_confidence = (grounding * 0.7) + (avg_retrieval * 0.3) - unsupported_penalty
        confidence = round(max(0.0, min(1.0, raw_confidence)), 4)
        confidence_pct = f"{round(confidence * 100, 1)}%"

        # Trust level
        if confidence >= 0.80:
            trust_level = "HIGH"
            trust_icon = "✅"
            recommendation = "Answer is well-supported. Safe to use."
        elif confidence >= 0.60:
            trust_level = "MEDIUM"
            trust_icon = "⚠️"
            recommendation = "Answer is mostly supported. Verify weak sentences."
        elif confidence >= 0.40:
            trust_level = "LOW"
            trust_icon = "🔶"
            recommendation = "Answer has gaps. Cross-check with source document."
        else:
            trust_level = "VERY LOW"
            trust_icon = "❌"
            recommendation = "Answer may be unreliable. Do not use without verification."

        return {
            "confidence_score": confidence,
            "confidence_pct": confidence_pct,
            "trust_level": trust_level,
            "trust_icon": trust_icon,
            "recommendation": recommendation,
            "grounding_score": grounding,
            "avg_retrieval_score": avg_retrieval,
            "unsupported_penalty": unsupported_penalty,
            "breakdown": {
                "supported_sentences": supported,
                "weak_sentences": weak,
                "unsupported_sentences": unsupported,
                "total_sentences": total
            }
        }

    def format_report(self, confidence_result: dict) -> str:
        """Pretty print the confidence report."""
        c = confidence_result
        lines = [
            "=" * 50,
            f"  TRUTHGUARD CONFIDENCE REPORT",
            "=" * 50,
            f"  Trust Level     : {c['trust_icon']} {c['trust_level']}",
            f"  Confidence Score: {c['confidence_pct']}",
            f"  Grounding Score : {round(c['grounding_score']*100, 1)}%",
            f"  Avg Retrieval   : {round(c['avg_retrieval_score']*100, 1)}%",
            "-" * 50,
            f"  Sentences",
            f"    ✅ Supported  : {c['breakdown']['supported_sentences']}",
            f"    ⚠️  Weak       : {c['breakdown']['weak_sentences']}",
            f"    ❌ Unsupported : {c['breakdown']['unsupported_sentences']}",
            "-" * 50,
            f"  💡 {c['recommendation']}",
            "=" * 50,
        ]
        return "\n".join(lines)