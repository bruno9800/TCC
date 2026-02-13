"""
UNIVASF RAG API — Entrypoint FastAPI

Uso:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.chat.router import router as chat_router

app = FastAPI(
    title="UNIVASF RAG API",
    description="API de Chat Inteligente para consulta de documentos normativos da UNIVASF",
    version="1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringir para domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rotas ──────────────────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/chat", tags=["Chat"])


@app.get("/health", tags=["Infra"])
async def health_check():
    """Verifica se a API está online."""
    return {"status": "ok"}
