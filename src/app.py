"""
Interface Streamlit — Chat RAG para Documentos Normativos da UNIVASF

Consome a API FastAPI (POST /chat/) em vez de chamar o pipeline diretamente.

Uso:
    streamlit run src/app.py
"""

import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuração ───────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

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

        # Status da API
        st.markdown("## 📊 Informações")
        try:
            r = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
            if r.status_code == 200:
                st.success("🟢 API Online")
            else:
                st.error("🔴 API Offline")
        except httpx.ConnectError:
            st.error("🔴 API Offline")
            st.caption(f"URL: {API_BASE_URL}")

        st.markdown("---")
        st.markdown(
            "### 📚 Sobre\n"
            "Sistema RAG para consulta de documentos normativos da UNIVASF.\n\n"
            "**TCC** — Geração Aumentada por Recuperação (RAG) para "
            "Automatizar a Consulta de Documentos Normativos."
        )

        return filter_revoked, top_k


# ── Chamada à API ──────────────────────────────────────────────────────────────

def call_chat_api(
    message: str,
    history: list[dict],
    top_k: int = 5,
    filter_revoked: bool = True,
) -> dict | None:
    """
    Chama o endpoint POST /chat/ da API FastAPI.

    Args:
        message: Pergunta do usuário.
        history: Histórico de mensagens [{role, content}, ...].
        top_k: Documentos finais.
        filter_revoked: Filtrar revogados.

    Returns:
        Dicionário com a resposta da API ou None em caso de erro.
    """
    # Converte histórico para o formato da API (apenas role e content)
    api_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]

    payload = {
        "message": message,
        "history": api_history,
        "top_k": top_k,
        "filter_revoked": filter_revoked,
    }

    # Recupera a API Key do ambiente ou segura (hardcoded para teste se necessário)
    api_key = os.getenv("TCC_API_KEY", "")

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = httpx.post(
            f"{API_BASE_URL}/chat/",
            json=payload,
            headers=headers,
            timeout=120.0,  # Reranker pode demorar no primeiro carregamento
        )
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        st.error("❌ Não foi possível conectar à API. Verifique se está rodando.")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"❌ Erro da API: {e.response.status_code} — {e.response.text}")
        return None
    except httpx.ReadTimeout:
        st.error("⏳ A API demorou demais para responder. Tente novamente.")
        return None


# ── Renderização de Fontes ─────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove marcadores Markdown e HTML do texto para exibição limpa."""
    import re
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@st.cache_data
def _find_source_pdf(source_name: str) -> Path | None:
    """
    Busca o PDF original correspondente ao nome da fonte.
    """
    from src.config import DOCUMENTS_DIR
    import unicodedata

    def normalize(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.lower().replace(" ", "").replace("_", "").replace("-", "")

    source_norm = normalize(source_name)
    for pdf_path in DOCUMENTS_DIR.rglob("*.pdf"):
        pdf_norm = normalize(pdf_path.stem)
        if source_norm in pdf_norm or pdf_norm in source_norm:
            return pdf_path
    return None


def render_sources(sources: list[dict], message_index: int):
    """Renderiza os cartões de fontes consultadas usando componentes nativos."""
    if not sources:
        return

    with st.expander("📎 Fontes Consultadas", expanded=False):
        for i, src in enumerate(sources, 1):
            source = src.get("source", "Desconhecido")
            article = src.get("article_id", "")
            category = src.get("category", "")
            hierarchy = src.get("hierarchy", "")
            score = src.get("score", 0.0)
            snippet = src.get("snippet", "")

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
                if snippet:
                    with st.popover(f"Ver trecho #{i}"):
                        clean_text = _strip_markdown(snippet)
                        st.text(clean_text)

            with btn_cols[1]:
                pdf_path = _find_source_pdf(source)
                if pdf_path and pdf_path.exists():
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Baixar PDF",
                            data=f,
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            key=f"dl_{message_index}_{i}_{source}",
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

    # Histórico de chat
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Mostra fontes e métricas para mensagens do assistente
            if msg["role"] == "assistant":
                if "sources" in msg:
                    render_sources(msg["sources"], message_index=i)
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

        # Chama a API
        with st.chat_message("assistant"):
            with st.spinner("🤖 Consultando o assistente..."):
                result = call_chat_api(
                    message=user_input,
                    history=st.session_state.messages[:-1],  # Exclui a msg atual
                    top_k=top_k,
                    filter_revoked=filter_revoked,
                )

            if result:
                answer = result.get("answer", "Sem resposta.")
                sources = result.get("sources", [])
                tokens = result.get("tokens", {})
                used_search = result.get("used_search", False)

                # Exibe resposta
                st.markdown(answer)

                # Exibe fontes (se buscou)
                if sources:
                    render_sources(sources, message_index=len(st.session_state.messages))

                # Métricas de uso
                prompt_tokens = tokens.get("prompt", 0)
                completion_tokens = tokens.get("completion", 0)
                search_label = "🔍 Buscou" if used_search else "💬 Direto"
                metrics_text = (
                    f"🔢 Tokens: {prompt_tokens} (prompt) + "
                    f"{completion_tokens} (resposta) | "
                    f"{search_label} | "
                    f"📄 {len(sources)} fontes"
                )
                st.caption(metrics_text)

                # Salva no histórico
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "metrics": metrics_text,
                })


if __name__ == "__main__":
    main()
