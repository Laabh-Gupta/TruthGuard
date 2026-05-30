# retrieval/rag_pipeline.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from retrieval.vector_store import VectorStore
from retrieval.citation_mapper import CitationMapper
from hallucination_detection.detector import HallucinationDetector
from evaluation.confidence_scorer import ConfidenceScorer


RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I cannot find this in the provided documents."
Do not make up information. Write in clear complete sentences.

Context:
{context}

Question: {question}

Answer:"""
)


class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_gpu=0
        )
        self.citation_mapper = CitationMapper()
        self.detector = HallucinationDetector()
        self.confidence_scorer = ConfidenceScorer()
        print(f"✅ RAG Pipeline ready — using {OLLAMA_MODEL}")

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Full TruthGuard pipeline:
        1. Retrieve
        2. Generate
        3. Cite
        4. Detect hallucinations
        5. Score confidence
        """

        # Step 1 — Retrieve
        retrieved_chunks = self.vector_store.retrieve(question, top_k=top_k)
        if not retrieved_chunks:
            return {"error": "No relevant documents found. Upload a PDF first."}

        # Step 2 — Generate
        context = "\n\n---\n\n".join([
            f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
            for c in retrieved_chunks
        ])
        prompt = RAG_PROMPT.format(context=context, question=question)
        answer = self.llm.invoke(prompt)

        # Step 3 — Citations
        cited_sentences = self.citation_mapper.map_citations(answer, retrieved_chunks)
        cited_answer = self.citation_mapper.format_cited_answer(cited_sentences)

        # Step 4 — Hallucination detection
        hallucination_report = self.detector.analyze(answer, retrieved_chunks)

        # Step 5 — Confidence score
        retrieval_scores = [c["similarity"] for c in retrieved_chunks]
        confidence = self.confidence_scorer.score(
            hallucination_report, retrieval_scores
        )
        confidence_report = self.confidence_scorer.format_report(confidence)

        return {
            "question": question,
            "answer": answer,
            "cited_answer": cited_answer,
            "cited_sentences": cited_sentences,
            "sources": retrieved_chunks,
            "hallucination_report": hallucination_report,
            "confidence": confidence,
            "confidence_report": confidence_report
        }