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
    1. Busca densa HNSW (ChromaDB)
    2. Reranking (cross-encoder)
    3. Geração (LLM)
    """
    # 1. Busca HNSW — top-50 candidatos
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

def _strip_markdown(text: str) -> str:
    """Remove marcadores Markdown e HTML do texto para exibição limpa."""
    import re
    # Remove headers markdown (# ## ###)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove bold/itálico (**texto** ou *texto*)
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    # Remove tags HTML
    text = re.sub(r"<[^>]+>", "", text)
    # Remove linhas horizontais markdown
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    # Remove espaços duplicados
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@st.cache_data
def _find_source_pdf(source_name: str) -> Path | None:
    """
    Busca o PDF original correspondente ao nome da fonte.

    Percorre recursivamente regimentos_estatutos_resolucoes/ procurando
    um PDF cujo nome contenha o source_name do chunk.
    """
    from src.config import DOCUMENTS_DIR
    import unicodedata

    def normalize(s: str) -> str:
        """Normaliza string para comparação (remove acentos, lowercase)."""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.lower().replace(" ", "").replace("_", "").replace("-", "")

    source_norm = normalize(source_name)

    for pdf_path in DOCUMENTS_DIR.rglob("*.pdf"):
        pdf_norm = normalize(pdf_path.stem)
        if source_norm in pdf_norm or pdf_norm in source_norm:
            return pdf_path

    return None


def render_sources(sources: list[SearchResult]):
    """Renderiza os cartões de fontes consultadas usando componentes nativos."""
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

            st.markdown(f"**📄 {source}** {f'— {article}' if article else ''}")

            cols = st.columns([2, 2, 1])
            with cols[0]:
                st.caption(f"Categoria: {category}")
            with cols[1]:
                if hierarchy:
                    st.caption(f"Hierarquia: {hierarchy}")
            with cols[2]:
                st.caption(f"Score: {score:.4f}")

            # Botões: ver trecho + baixar PDF
            btn_cols = st.columns([1, 1, 3])
            with btn_cols[0]:
                with st.popover(f"Ver trecho #{i}"):
                    clean_text = _strip_markdown(result.content[:600])
                    st.text(clean_text + ("..." if len(result.content) > 600 else ""))

            with btn_cols[1]:
                pdf_path = _find_source_pdf(source)
                if pdf_path and pdf_path.exists():
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Baixar PDF",
                            data=f,
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            key=f"dl_{i}_{source}",
                        )

            st.divider()


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

            # Mostra fontes e métricas para mensagens do assistente
            if msg["role"] == "assistant":
                if "sources" in msg:
                    render_sources(msg["sources"])
                if "metrics" in msg:
                    st.caption(msg["metrics"])

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

                # Métricas de uso
                metrics_text = (
                    f"🔢 Tokens: {result.prompt_tokens} (prompt) + "
                    f"{result.completion_tokens} (resposta) | "
                    f"📄 {len(top_results)} fontes consultadas"
                )
                st.caption(metrics_text)

                # Salva no histórico (incluindo métricas)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "sources": top_results,
                    "metrics": metrics_text,
                })

            except Exception as e:
                st.error(f"Erro ao processar sua pergunta: {e}")
                logger.error(f"Erro no pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()
