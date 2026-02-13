"""
Interface Streamlit — Chat RAG para Documentos Normativos da UNIVASF

Uso:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import logging

from src.retrieval.hybrid_search import HybridSearchEngine, SearchResult
from src.retrieval.reranker import rerank
from src.generation.generator import generate_answer, GenerationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuração da Página ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG UNIVASF — Assistente Normativo",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Customizado ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .source-card {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #1f77b4;
    }
    .source-card h4 {
        margin: 0 0 4px 0;
        color: #1f77b4;
    }
    .source-card p {
        margin: 0;
        font-size: 0.85em;
        color: #555;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 16px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Inicialização do Estado ────────────────────────────────────────────────────

@st.cache_resource
def load_search_engine():
    """Carrega o motor de busca uma única vez."""
    try:
        engine = HybridSearchEngine()
        return engine
    except Exception as e:
        st.error(f"Erro ao carregar motor de busca: {e}")
        st.info("Execute o pipeline ETL primeiro:\n```\npython scripts/run_etl.py\npython scripts/run_indexing.py\n```")
        return None


def initialize_session():
    """Inicializa variáveis de sessão."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = []


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    """Renderiza a barra lateral com configurações e info."""
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")

        filter_revoked = st.checkbox(
            "Filtrar documentos revogados",
            value=True,
            help="Exclui documentos marcados como revogados da busca"
        )

        top_k = st.slider(
            "Documentos finais (pós-reranking)",
            min_value=1,
            max_value=10,
            value=5,
            help="Número de documentos enviados ao LLM"
        )

        st.markdown("---")

        st.markdown("## 📊 Informações")
        engine = load_search_engine()
        if engine:
            st.metric("Chunks Indexados", len(engine.chunks))

            categories = set()
            for chunk in engine.chunks:
                categories.add(chunk.metadata.category)
            st.markdown(f"**Categorias:** {', '.join(sorted(categories))}")

        st.markdown("---")
        st.markdown(
            "### 📚 Sobre\n"
            "Sistema RAG para consulta de documentos normativos da UNIVASF.\n\n"
            "**TCC** — Geração Aumentada por Recuperação (RAG) para "
            "Automatizar a Consulta de Documentos Normativos."
        )

        return filter_revoked, top_k


# ── Pipeline de Resposta ───────────────────────────────────────────────────────

def run_rag_pipeline(
    query: str,
    engine: HybridSearchEngine,
    filter_revoked: bool = True,
    top_k: int = 5,
) -> tuple[GenerationResult, list[SearchResult]]:
    """
    Executa o pipeline RAG completo:
    1. Busca híbrida (dense + BM25 + RRF)
    2. Reranking (cross-encoder)
    3. Geração (LLM)
    """
    # 1. Busca híbrida — top-50 candidatos
    with st.spinner("🔍 Buscando nos documentos normativos..."):
        candidates = engine.search_hybrid(
            query=query,
            filter_revoked=filter_revoked,
        )

    # 2. Reranking — top-5
    with st.spinner("📊 Reordenando por relevância..."):
        top_results = rerank(query, candidates, top_k=top_k)

    # 3. Geração
    with st.spinner("💬 Gerando resposta fundamentada..."):
        result = generate_answer(query, top_results)

    return result, top_results


# ── Renderização de Fontes ─────────────────────────────────────────────────────

def render_sources(sources: list[SearchResult]):
    """Renderiza os cartões de fontes consultadas."""
    if not sources:
        return

    with st.expander("📎 Fontes Consultadas", expanded=False):
        for i, result in enumerate(sources, 1):
            meta = result.metadata
            source = meta.get("source", "Desconhecido")
            article = meta.get("article_id", "")
            category = meta.get("category", "")
            hierarchy = meta.get("hierarchy", "")
            score = result.score

            st.markdown(f"""
<div class="source-card">
    <h4>📄 {source} {f"— {article}" if article else ""}</h4>
    <p><strong>Categoria:</strong> {category}</p>
    {f'<p><strong>Hierarquia:</strong> {hierarchy}</p>' if hierarchy else ''}
    <p><strong>Score de relevância:</strong> {score:.4f}</p>
</div>
            """, unsafe_allow_html=True)

            with st.popover(f"Ver trecho #{i}"):
                st.text(result.content[:500] + ("..." if len(result.content) > 500 else ""))


# ── Interface Principal ────────────────────────────────────────────────────────

def main():
    initialize_session()

    # Header
    st.markdown(
        "<div class='main-header'>"
        "<h1>📜 Assistente Normativo UNIVASF</h1>"
        "<p>Consulte estatutos, regimentos e resoluções da universidade</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Sidebar
    filter_revoked, top_k = render_sidebar()

    # Carrega motor de busca
    engine = load_search_engine()
    if engine is None:
        st.stop()

    # Histórico de chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Mostra fontes para mensagens do assistente
            if msg["role"] == "assistant" and "sources" in msg:
                render_sources(msg["sources"])

    # Input do usuário
    user_input = st.chat_input(
        "Faça uma pergunta sobre as normas da UNIVASF..."
    )

    if user_input:
        # Exibe mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Executa pipeline RAG
        with st.chat_message("assistant"):
            try:
                result, top_results = run_rag_pipeline(
                    query=user_input,
                    engine=engine,
                    filter_revoked=filter_revoked,
                    top_k=top_k,
                )

                # Exibe resposta
                st.markdown(result.answer)

                # Exibe fontes
                render_sources(top_results)

                # Salva no histórico
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "sources": top_results,
                })

                # Métricas de uso
                st.caption(
                    f"🔢 Tokens: {result.prompt_tokens} (prompt) + "
                    f"{result.completion_tokens} (resposta) | "
                    f"📄 {len(top_results)} fontes consultadas"
                )

            except Exception as e:
                st.error(f"Erro ao processar sua pergunta: {e}")
                logger.error(f"Erro no pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()
