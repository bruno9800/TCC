"""Configurações centralizadas do sistema RAG."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "regimentos_estatutos_resolucoes"
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

# Ensure data directories exist
MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# ── Retrieval Parameters ──────────────────────────────────────────────────────
INITIAL_TOP_K = 50        # candidatos iniciais (dense + BM25)
RERANK_TOP_K = 5          # documentos finais após reranking
BM25_WEIGHT = 0.3         # peso da busca esparsa no RRF
DENSE_WEIGHT = 0.7        # peso da busca densa no RRF

# ── Chunking Parameters ──────────────────────────────────────────────────────
MAX_CHUNK_TOKENS = 512    # limite de tokens por chunk antes de split
OVERLAP_TOKENS = 50       # overlap para chunks divididos

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "univasf_normas"
