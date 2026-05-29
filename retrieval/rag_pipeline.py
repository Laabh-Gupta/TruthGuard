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


RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I cannot find this in the provided documents."
Do not make up information. Write in clear sentences.

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
        print(f"✅ RAG Pipeline ready — using {OLLAMA_MODEL}")

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Full RAG query with citation mapping:
        1. Retrieve relevant chunks
        2. Generate answer
        3. Map every sentence to a source chunk
        4. Return structured result
        """

        # Step 1 — Retrieve
        retrieved_chunks = self.vector_store.retrieve(question, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": "No relevant documents found.",
                "cited_answer": "No relevant documents found.",
                "cited_sentences": [],
                "sources": [],
                "question": question
            }

        # Step 2 — Build context
        context = "\n\n---\n\n".join([
            f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
            for c in retrieved_chunks
        ])

        # Step 3 — Generate answer
        prompt = RAG_PROMPT.format(context=context, question=question)
        answer = self.llm.invoke(prompt)

        # Step 4 — Map citations
        cited_sentences = self.citation_mapper.map_citations(answer, retrieved_chunks)
        cited_answer = self.citation_mapper.format_cited_answer(cited_sentences)
        sources_list = self.citation_mapper.format_sources_list(retrieved_chunks)

        return {
            "question": question,
            "answer": answer,                          # raw answer
            "cited_answer": cited_answer,              # answer with [1][2] citations
            "cited_sentences": cited_sentences,        # per-sentence breakdown
            "sources": retrieved_chunks,               # source chunks used
            "sources_list": sources_list               # formatted source list
        }