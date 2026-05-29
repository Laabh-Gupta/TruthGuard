# retrieval/rag_pipeline.py

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from retrieval.vector_store import VectorStore


# Prompt template — instructs the LLM to only use provided context
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I cannot find this in the provided documents."
Do not make up information.

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
            temperature=0.1  # Low temperature = more factual, less creative
            num_gpu=1  # Force CPU
        )
        print(f"✅ RAG Pipeline ready — using {OLLAMA_MODEL}")

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Full RAG query:
        1. Retrieve relevant chunks
        2. Build context string
        3. Generate answer
        4. Return answer + source chunks
        """

        # Step 1 — Retrieve
        retrieved_chunks = self.vector_store.retrieve(question, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": "No relevant documents found. Please upload a PDF first.",
                "sources": [],
                "question": question
            }

        # Step 2 — Build context
        context = "\n\n---\n\n".join([
            f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
            for c in retrieved_chunks
        ])

        # Step 3 — Generate
        prompt = RAG_PROMPT.format(context=context, question=question)
        answer = self.llm.invoke(prompt)

        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "question": question
        }