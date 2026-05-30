# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Settings ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"        # for RAG answers — better quality
RAGAS_MODEL = "gemma3:1b"         # for RAGAS evaluation — fast

# --- Embedding Settings ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384

# --- Retrieval Settings ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RESULTS = 5

# --- Vector DB ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "truthguard_docs"

# --- Hallucination Detection ---
SIMILARITY_THRESHOLD_SUPPORTED = 0.75
SIMILARITY_THRESHOLD_WEAK = 0.50

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")