"""
Agent Orchestrator — Function Calling Nativo (Fase 4)

Substitui a decisão binária manual (JSON parseado à mão em chat/service.py)
por function calling nativo da OpenAI, com um registro de Tools extensível.
Resolve dois problemas diagnosticados em PLANO_V2.md:

  - D5: o parsing manual de JSON não escalava para múltiplas ferramentas —
    function calling nativo é o mecanismo padrão da indústria para agentes
    multi-tool, e é o que permite ao agente escolher entre RagTool e
    ProfessorTool (ou combinar as duas) numa mesma pergunta.
  - D4: uma única função de orquestração (`run`), usada tanto pelo caminho
    streaming quanto não-streaming (ver src/chat/service.py) — elimina a
    duplicação que existia entre run_chat/stream_chat.

`run()` é um generator que produz eventos internos — {"type": "status"|"token"
|"done"|"error", ...} — src/chat/service.py traduz esses eventos para
ChatResponse (não-streaming) ou SSE (streaming). Nenhum consumidor externo
deve chamar `run()` diretamente fora de chat/service.py.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator

from openai import OpenAI
from sqlalchemy.orm import Session

from src.agent.tools import discipline_tool, professor_tool, rag_tool
from src.config import LLM_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

TOOLS = {
    rag_tool.NAME: rag_tool,
    professor_tool.NAME: professor_tool,
    discipline_tool.NAME: discipline_tool,
}
TOOL_SCHEMAS = [rag_tool.SCHEMA, professor_tool.SCHEMA, discipline_tool.SCHEMA]

STATUS_MESSAGES = {
    rag_tool.NAME: "Buscando nos documentos normativos...",
    professor_tool.NAME: "Consultando o corpo docente...",
    discipline_tool.NAME: "Consultando a matriz curricular...",
}

SYSTEM_PROMPT = """\
Você é o Assistente Acadêmico da UNIVASF (Universidade Federal do Vale do São Francisco).

## Ferramentas disponíveis:
- search_normative_documents: busca em estatutos, regimentos e resoluções oficiais da \
UNIVASF (inclui o PPC do curso). Use para perguntas sobre normas, regras, prazos, \
direitos ou procedimentos acadêmicos — inclusive o conteúdo narrativo do PPC (perfil \
do egresso, objetivos, infraestrutura, ementas de disciplinas).
- search_professors: consulta o corpo docente (nome, e-mail, área de atuação, Lattes, \
participação no NDE). Use para perguntas sobre professores específicos ou sobre quem \
atua em determinada área/disciplina.
- search_disciplines: consulta a matriz curricular oficial (período, carga horária, \
pré-requisitos/co-requisitos exatos). Use SEMPRE que a pergunta envolver em que \
período fica uma disciplina, quais os pré-requisitos, ou carga horária — não tente \
responder isso só com search_normative_documents, pois o texto em prosa do PPC não \
garante precisão nesses fatos exatos.

Você pode usar mais de uma ferramenta na mesma pergunta quando fizer sentido (ex: uma \
pergunta que envolve tanto uma norma quanto quem a aplica).

## Quando NÃO usar nenhuma ferramenta:
Cumprimentos, agradecimentos, perguntas sobre você mesmo, ou perguntas sobre o histórico \
da própria conversa (ex: "o que eu perguntei antes?"). Nesses casos, responda diretamente.

## Regras para respostas baseadas em ferramentas:
1. Responda APENAS com base no que as ferramentas retornaram. Se não encontrar, diga \
claramente que não encontrou.
2. NUNCA invente normas, artigos ou dados de professores.
3. Cite SEMPRE a fonte: "Segundo o Art. X da [Norma]..." para normas; nome e e-mail do \
professor para dados do corpo docente.
4. Se múltiplas fontes tratam do assunto, cite todas.
5. Seja DIRETO e CONCISO — máximo 2-3 parágrafos curtos. Não repita a pergunta.\
"""

_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Retorna instância singleton do cliente OpenAI."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _assistant_tool_call_message(message) -> dict:
    """Serializa a mensagem do assistant (com tool_calls) para reenvio na conversa."""
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ],
    }


def run(
    message: str,
    history: list[dict],
    db: Session,
    top_k: int = 5,
    filter_revoked: bool = True,
    course_id: int | None = None,
    model: str = LLM_MODEL,
) -> Generator[dict, None, None]:
    """
    Orquestra a conversa com function calling nativo.

    Gera eventos internos:
      {"type": "status", "text": ...}
      {"type": "token",  "text": ...}
      {"type": "done",   "sources": [...], "used_tools": [...], "tokens": {...}}
      {"type": "error",  "text": ...}
    """
    client = get_openai_client()
    context = {"db": db, "top_k": top_k, "filter_revoked": filter_revoked, "course_id": course_id}

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": message},
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0

    try:
        yield {"type": "status", "text": "Analisando pergunta..."}

        decision = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0.0,
            max_tokens=1024,
        )
        choice = decision.choices[0]
        if decision.usage:
            total_prompt_tokens += decision.usage.prompt_tokens
            total_completion_tokens += decision.usage.completion_tokens

        tool_calls = choice.message.tool_calls

        # ── Sem tools: a própria resposta já é a final (cumprimento, follow-up) ──
        if not tool_calls:
            answer = choice.message.content or ""
            yield {"type": "token", "text": answer}
            yield {
                "type": "done",
                "sources": [],
                "used_tools": [],
                "tokens": {"prompt": total_prompt_tokens, "completion": total_completion_tokens},
            }
            return

        # ── Com tools: executa cada uma e monta o contexto pra síntese final ──
        messages.append(_assistant_tool_call_message(choice.message))

        all_sources: list[dict] = []
        used_tools: list[str] = []

        for tool_call in tool_calls:
            name = tool_call.function.name
            used_tools.append(name)
            yield {"type": "status", "text": STATUS_MESSAGES.get(name, f"Consultando {name}...")}

            tool_module = TOOLS.get(name)
            if tool_module is None:
                result = {"summary": f"Ferramenta '{name}' não reconhecida.", "sources": []}
            else:
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                    result = tool_module.execute(arguments, context)
                except Exception as e:
                    logger.error(f"Erro ao executar tool '{name}': {e}", exc_info=True)
                    result = {"summary": f"Erro ao consultar '{name}': {e}", "sources": []}

            all_sources.extend(result.get("sources", []))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.get("summary", ""),
                }
            )

        yield {"type": "status", "text": "Gerando resposta..."}

        full_answer = ""
        with client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            stream=True,
            stream_options={"include_usage": True},
        ) as stream:
            for chunk in stream:
                if chunk.usage:
                    total_prompt_tokens += chunk.usage.prompt_tokens
                    total_completion_tokens += chunk.usage.completion_tokens
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_answer += delta
                        yield {"type": "token", "text": delta}

        yield {
            "type": "done",
            "sources": all_sources,
            "used_tools": used_tools,
            "tokens": {"prompt": total_prompt_tokens, "completion": total_completion_tokens},
        }
    except Exception as e:
        logger.error(f"Erro no orchestrator: {e}", exc_info=True)
        yield {"type": "error", "text": str(e)}
