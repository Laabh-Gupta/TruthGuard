# evaluation/ragas_evaluator.py

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, EMBEDDING_MODEL, RAGAS_MODEL


class RAGASEvaluator:
    """
    Runs RAGAS evaluation metrics on RAG pipeline outputs.
    Uses local Ollama LLM + HuggingFace embeddings — no OpenAI needed.
    """

    def __init__(self):
        ragas_llm = OllamaLLM(
            model=RAGAS_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,        # change to 0 — more deterministic output
            num_gpu=1,
            format="json"         # force JSON output format
        )
        self.llm = LangchainLLMWrapper(ragas_llm)
    
        hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    
        print(f"✅ RAGAS Evaluator ready — using {RAGAS_MODEL} for evaluation")

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = ""
    ) -> dict:

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth if ground_truth else answer]
        }

        dataset = Dataset.from_dict(data)

        try:
            # Run metrics ONE AT A TIME — fixes timeout on CPU/slow LLMs
            faith_result = evaluate(
                dataset,
                metrics=[faithfulness],
                llm=self.llm,
                embeddings=self.embeddings,
                raise_exceptions=False
            )

            relevancy_result = evaluate(
                dataset,
                metrics=[answer_relevancy],
                llm=self.llm,
                embeddings=self.embeddings,
                raise_exceptions=False
            )

            precision_result = evaluate(
                dataset,
                metrics=[context_precision],
                llm=self.llm,
                embeddings=self.embeddings,
                raise_exceptions=False
            )

            return {
                "faithfulness": round(float(
                    faith_result.to_pandas()["faithfulness"].iloc[0]), 4),
                "answer_relevancy": round(float(
                    relevancy_result.to_pandas()["answer_relevancy"].iloc[0]), 4),
                "context_precision": round(float(
                    precision_result.to_pandas()["context_precision"].iloc[0]), 4),
                "ragas_available": True
            }

        except Exception as e:
            print(f"⚠️  RAGAS evaluation failed: {e}")
            return {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "ragas_available": False,
                "error": str(e)
            }

    def evaluate_dataset(self, eval_pairs: list[dict]) -> dict:
        """
        Evaluate multiple Q&A pairs and return aggregate scores.
        eval_pairs: list of {"question": ..., "answer": ..., "contexts": [...]}
        """
        results = []
        for pair in eval_pairs:
            result = self.evaluate_single(
                question=pair["question"],
                answer=pair["answer"],
                contexts=pair["contexts"]
            )
            result["question"] = pair["question"]
            results.append(result)
            print(f"  Evaluated: {pair['question'][:50]}...")

        # Aggregate
        valid = [r for r in results if r["ragas_available"]]
        if valid:
            avg_faithfulness = round(
                sum(r["faithfulness"] for r in valid) / len(valid), 4)
            avg_relevancy = round(
                sum(r["answer_relevancy"] for r in valid) / len(valid), 4)
            avg_precision = round(
                sum(r["context_precision"] for r in valid) / len(valid), 4)
        else:
            avg_faithfulness = avg_relevancy = avg_precision = None

        return {
            "individual_results": results,
            "aggregate": {
                "avg_faithfulness": avg_faithfulness,
                "avg_answer_relevancy": avg_relevancy,
                "avg_context_precision": avg_precision,
                "total_evaluated": len(results),
                "successful": len(valid)
            }
        }