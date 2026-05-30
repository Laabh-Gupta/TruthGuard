# 🛡️ TruthGuard — AI Answer Verification System

> Know what's grounded. Know what's not.

TruthGuard is a RAG-based hallucination detection system that verifies every sentence of an LLM's answer against source documents — giving you citations, grounding scores, and RAGAS evaluation metrics.

## 🎯 Why This Project

Most RAG projects stop at "Ask PDF." TruthGuard goes further:
- Every answer sentence is verified against retrieved context
- Unsupported claims are flagged as potential hallucinations
- Confidence scores quantify answer trustworthiness
- RAGAS metrics provide industry-standard evaluation

## 🏗️ Architecture

```
PDF Upload → Chunking → ChromaDB Vector Store
                                ↓
User Query → Retrieval → Phi-3 Mini (Ollama) → Answer
                                                    ↓
                               Citation Mapper ← TruthGuard
                               Hallucination Detector
                               Confidence Scorer
                               RAGAS Evaluator
```

## 📊 Evaluation Results

| Metric | Score |
|---|---|
| Faithfulness | 1.0 |
| Context Precision | 1.0 |
| Confidence Score | 62–87% (varies by query) |
| Hallucination Detection | Per-sentence with similarity scores |

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Phi-3 Mini via Ollama (local) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector DB | ChromaDB |
| RAG Framework | LangChain |
| Evaluation | RAGAS + DeepEval |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/truthguard
cd truthguard

# Create conda environment
conda create -n truthguard python=3.12.3
conda activate truthguard

# Install dependencies
pip install -r requirements.txt

# Install Ollama from ollama.com, then pull models
ollama pull phi3:mini
ollama pull gemma3:1b

# Run
streamlit run frontend/app.py
```

## 📁 Project Structure

```
truthguard/
├── ingestion/                  # PDF loading and chunking
├── retrieval/                  # Vector store, RAG pipeline, citation mapper
├── hallucination_detection/    # Per-sentence grounding analysis
├── evaluation/                 # RAGAS + confidence scoring
├── frontend/                   # Streamlit UI
├── tests/                      # Setup verification
└── config.py                   # Central configuration
```

## 💡 Key Features

- **Citation Mapping** — Every answer sentence traced to its source chunk
- **Hallucination Detection** — SUPPORTED / WEAK / UNSUPPORTED verdicts per sentence
- **Confidence Scoring** — Weighted grounding score per query
- **RAGAS Integration** — Faithfulness + Context Precision metrics
- **Fully Local** — Runs on consumer hardware, no API keys required

## 🖥️ Hardware

Built and tested on:
- Lenovo IdeaPad Gaming 3
- NVIDIA RTX 3050 4GB
- Intel i5-11450H
- 16GB RAM

No cloud compute required.

## 📄 License

MIT
