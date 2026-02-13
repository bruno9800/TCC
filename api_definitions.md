Este é o **Plano de Ataque** para a sua próxima sessão de codificação. O foco é **infraestrutura e arquitetura**.

Não vamos tentar implementar *tudo*. Vamos focar em tirar o sistema do "modo script" e colocá-lo no "modo plataforma" (Docker + FastAPI + Auth + Banco Preparado).

---

# 🚀 Plano de Sessão Única: "Do Script ao Backend"

**Meta:** Ao final desta sessão, você terá uma API rodando em Docker, com autenticação funcionando (bloqueando emails não-UNIVASF) e o ChromaDB pronto para receber conexões, tudo preparado para receber as *Tools* no futuro.

---

## 🏗️ Passo 1: Organizar a Casa (Estrutura de Pastas)

A primeira coisa é reestruturar os arquivos para o padrão de microsserviço modular.

1. Crie a seguinte estrutura (mova seus arquivos atuais de `src/` para `src/rag_engine/` temporariamente):

```text
.
├── docker-compose.yml       <-- NOVO (Orquestrador)
├── Dockerfile               <-- NOVO (Imagem do Backend)
├── .env                     <-- Configurações
└── src/
    ├── main.py              <-- Entrypoint do FastAPI
    ├── config.py            <-- Variáveis de ambiente
    ├── database.py          <-- Conexão Postgres (SQLAlchemy)
    ├── models.py            <-- Tabelas (User + Professor)
    ├── auth/                <-- Rotas de Login/Registro
    │   └── router.py
    └── rag_engine/          <-- Seu código atual de RAG (Index/Search)

```

---

## 🐳 Passo 2: Infraestrutura (Docker Compose)

Em vez de instalar Postgres e Chroma na sua máquina, vamos subir tudo via Docker.

**Ação:** Crie o arquivo `docker-compose.yml` na raiz:
---

## 🗄️ Passo 3: Modelagem de Dados (Preparando o Terreno)

Aqui aplicamos a estratégia de "preparar agora, usar depois". Vamos criar as tabelas `User` e `Professor` já linkadas.

**Ação:** Edite `src/models.py`:

---

## 🔐 Passo 4: Autenticação (A Regra de Negócio)

Implemente a lógica que barra quem não é da faculdade.

**Ação:** Em `src/auth/router.py` (usando Pydantic para validar):

'Apenas e-mails @univasf.edu.br são permitidos.'
      


---

## 🤖 Passo 5: Conectar o Cérebro (RAG Migration)

Aqui você faz o seu código atual falar com o Docker.

**Ação:** Atualize a conexão do ChromaDB no seu código de busca (`src/rag_engine/vector_store.py`):


---
sucesso quando:

1. Você rodar `docker-compose up --build`.
2. Acessar `http://localhost:8000/docs` (Swagger UI).
3. Conseguir registrar um usuário `teste@univasf.edu.br` (sucesso).
4. Tentar registrar `teste@gmail.com` e receber **Erro 422/400**.
5. (Bônus) O ChromaDB estiver online e acessível.


Perfeito. Vamos focar agora em **conectar os pontos**.

Você já tem o "corpo" (Docker/Infra) e a "identidade" (Auth). Agora vamos dar o "cérebro" (Agente) e a "voz" (Rota de Chat).

Esta sessão é crítica porque é onde seu projeto deixa de ser um script de Python e vira uma API SaaS real.

---

# 🚀 Plano de Sessão: "Cérebro & Voz" (Agent & Chat API)

**Meta:** Criar o endpoint protegido `POST /chat` onde o usuário logado envia uma pergunta, e o sistema (via Agente) decide se consulta as normas ou responde direto.

---

## 🛠️ Passo 1: O "Crachá" (Dependência de Auth)

Antes de deixar alguém falar com o Agente (que custa dinheiro/tokens), precisamos garantir que o usuário está logado.

---

## 🧰 Passo 2: A Ferramenta (Tool Wrapper)

**Ação:** Crie `src/agent/tools.py`. Vamos encapsular sua busca atual.

```python
# Definição da lógica real (importada do seu código antigo)
from src.rag_engine.retrieval import hybrid_search_logic 

# 1. A Função que o Agente vai executar
def search_univasf_norms(query: str):
    """
    Executa a busca híbrida no ChromaDB e retorna os chunks formatados.
    """
    results = hybrid_search_logic(query)
    # Formata para string para o LLM ler
    context_str = "\n\n".join([f"[Fonte: {d.metadata['source']}] {d.page_content}" for d in results])
    return context_str if context_str else "Nenhum documento relevante encontrado."

# 2. A Definição para o OpenAI (Schema)
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_univasf_norms",
            "description": "Busca em documentos oficiais da UNIVASF (Resoluções, Estatutos, Regimentos). Use para responder dúvidas sobre regras acadêmicas, prazos, direitos e deveres.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "A pergunta específica otimizada para busca semântica (ex: 'critérios trancamento matrícula')."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

```

---

## 🧠 Passo 3: O Maestro (Lógica do Agente)

Aqui é onde o Agente decide (ReAct Loop).

**Ação:** Crie `src/agent/core.py`.

```python
from openai import OpenAI
import json
from src.agent.tools import TOOLS_SCHEMA, search_univasf_norms

client = OpenAI() # Pega a chave do .env automaticamente

SYSTEM_PROMPT = """
Você é o Assistente Acadêmico da UNIVASF.
Diretrizes:
1. Sempre que o usuário perguntar sobre normas, regras ou procedimentos, USE a ferramenta `search_univasf_norms`.
2. Baseie sua resposta APENAS no retorno da ferramenta. Cite a resolução/artigo.
3. Se a ferramenta não retornar nada, diga que não encontrou a informação oficial.
4. Para cumprimentos (Oi, Bom dia), responda cordialmente sem usar ferramentas.
"""

def run_agent_sync(user_message: str, chat_history: list):
    # 1. Monta o histórico (System + Conversa Passada + Pergunta Atual)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history + [{"role": "user", "content": user_message}]

    # 2. Primeira chamada (Pensamento)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message

    # 3. Verifica se o Agente quer usar a ferramenta
    if msg.tool_calls:
        # Adiciona a intenção do agente ao histórico
        messages.append(msg)

        for tool_call in msg.tool_calls:
            if tool_call.function.name == "search_univasf_norms":
                # Executa a função Python
                args = json.loads(tool_call.function.arguments)
                tool_result = search_univasf_norms(args["query"])

                # Adiciona o resultado da ferramenta ao histórico
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

        # 4. Segunda chamada (Resposta Final com o Contexto)
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return final_response.choices[0].message.content
    
    # Se não usou tool, retorna a resposta direta
    return msg.content

```

---

## 🗣️ Passo 4: A Rota Protegida (API)

Agora conectamos o HTTP ao Python.

**Ação:** Crie `src/chat/router.py`.

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from src.auth.utils import get_current_user
from src.agent.core import run_agent_sync

router = APIRouter()

# Schema de Entrada
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = [] # [{"role": "user", "content": "..."}]

@router.post("/ask")
async def chat_endpoint(
    request: ChatRequest, 
    current_user: dict = Depends(get_current_user) # <--- AQUI ESTÁ A PROTEÇÃO
):
    try:
        # Aqui você pode logar quem perguntou: print(f"User: {current_user['email']}")
        
        response = run_agent_sync(request.message, request.history)
        
        return {"response": response, "user": current_user['email']}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

```

---

## 🔗 Passo 5: Wiring (Conectar no Main)

Atualize seu `src/main.py` para incluir as rotas novas.

```python
from fastapi import FastAPI
from src.auth.router import router as auth_router
from src.chat.router import router as chat_router

app = FastAPI(title="UNIVASF RAG API")

app.include_router(auth_router, prefix="/auth", tags=["Auth"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])

```

---

## ✅ Checklist de Sucesso (Definition of Done)

Você saberá que terminou esta sessão quando:

1. Usar o **Postman/Insomnia** (ou Swagger UI).
2. Fizer login e pegar um **Bearer Token**.
3. Tentar acessar `POST /chat/ask` **sem** token e receber `401 Unauthorized`.
4. Tentar acessar `POST /chat/ask` **com** token, perguntar "Como tranco o curso?", e ver nos logs o sistema buscando no ChromaDB.
5. Receber a resposta final JSON com a explicação baseada na norma.

**Dica de Ouro:** Não se preocupe em persistir o histórico no Banco de Dados (`Postgres`) *nesta* sessão. Receba o histórico via JSON do frontend (stateless) para testar a lógica do Agente primeiro. Persistência de chat é a próxima etapa.