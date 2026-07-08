# API Reference — UNIVASF RAG

Base URL local: `http://localhost:8000`

Toda requisição ao endpoint `/chat/` requer o header de autenticação:
```
x-api-key: <TCC_API_KEY>
```

---

## GET /health

Verifica se a API está online.

```json
// Response 200
{ "status": "ok" }
```

---

## POST /chat/

Endpoint principal. O agente decide automaticamente quando acionar o pipeline RAG ou responder diretamente.

**Request:**

```json
{
  "message": "Como funciona o trancamento de matrícula?",
  "history": [
    { "role": "user", "content": "Oi!" },
    { "role": "assistant", "content": "Olá! Como posso ajudar?" }
  ],
  "top_k": 5,
  "filter_revoked": true,
  "course_id": 1
}
```

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `message` | string | Sim | Mensagem atual do usuário |
| `history` | array | Não | Histórico `[{role, content}]` — padrão: `[]` |
| `top_k` | int (1–10) | Não | Documentos pós-reranking — padrão: `5` |
| `filter_revoked` | bool | Não | Filtrar documentos revogados — padrão: `true` |
| `course_id` | int | Não | Escopa a busca a um curso (`GET /courses` lista os disponíveis). Documentos/professores/eventos institucionais (sem curso associado) continuam visíveis independente do escopo. Omitir busca em tudo — padrão: `null` |

**Response:**

```json
{
  "answer": "O trancamento de matrícula na UNIVASF é regulado pela...",
  "sources": [
    {
      "origin": "rag",
      "source": "Resolução 08_2015 - Normas_gerais",
      "category": "Resolução PROEN",
      "article_id": "Art. 45",
      "hierarchy": "Título IV > Capítulo II",
      "score": 0.8523,
      "snippet": "Art. 45. O trancamento de matrícula..."
    }
  ],
  "model": "gpt-4o",
  "tokens": {
    "prompt": 2048,
    "completion": 116
  },
  "used_search": true,
  "used_tools": ["search_normative_documents"]
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `answer` | string | Resposta do agente (Markdown) |
| `sources` | array | Fontes consultadas — de documentos normativos e/ou do corpo docente (vazio se `used_search: false`) |
| `model` | string | Modelo LLM usado |
| `tokens` | object | `{prompt, completion}` — consumo de tokens |
| `used_search` | bool | `true` = alguma ferramenta foi acionada (RAG e/ou corpo docente), `false` = resposta direta |
| `used_tools` | array de string | Nomes das ferramentas acionadas nesta resposta (ex: `["search_normative_documents", "search_professors"]` — o agente pode combinar as duas na mesma pergunta) |

O agente decide autonomamente, via *function calling* nativo da OpenAI, quais ferramentas usar — hoje: `search_normative_documents` (RAG sobre estatutos/regimentos/resoluções, inclui o PPC), `search_professors` (corpo docente), `search_disciplines` (matriz curricular) e `search_academic_calendar` (prazos/datas do calendário acadêmico). Todas exceto a primeira são consultas estruturadas (SQL), não RAG — fatos exatos não devem depender de recall de embedding. Cada item de `sources` tem um campo `origin` (`"rag"`, `"professor"`, `"discipline"` ou `"calendar"`) indicando de onde veio:

```json
// origin: "rag" — inclui download_url pronto para uso
{
  "origin": "rag",
  "source": "Resolução 08_2015 - Normas_gerais",
  "category": "Resolução PROEN",
  "article_id": "Art. 45",
  "hierarchy": "Título IV > Capítulo II",
  "score": 0.8523,
  "snippet": "Art. 45. O trancamento de matrícula...",
  "download_url": "/documents/download?source=Resolu%C3%A7%C3%A3o+08_2015+-+Normas_gerais"
}

// origin: "professor" — download_url vazio (não é um documento)
{
  "origin": "professor",
  "source": "Jadsonlee da Silva Sá",
  "category": "Professor (NDE)",
  "article_id": "",
  "hierarchy": "",
  "score": 0.0,
  "snippet": "jadsonlee.sa@univasf.edu.br — Sistemas Embarcados",
  "download_url": ""
}

// origin: "discipline"
{
  "origin": "discipline",
  "source": "Compiladores (COMP)",
  "category": "Matriz Curricular — 7º período",
  "snippet": "Carga horária: 60h | Pré-requisito: AED, LFA e OAC"
}

// origin: "calendar"
{
  "origin": "calendar",
  "source": "Trancamento do período ou cancelamento de disciplinas (com ônus), referente a 2026.1",
  "category": "Calendário Acadêmico — trancamento",
  "snippet": "2026-04-06 a 2026-04-10"
}
```

`course_id` (opcional, ver tabela acima) escopa `search_normative_documents`/`search_professors`/`search_disciplines`/`search_academic_calendar` a um curso — conteúdo institucional (sem curso associado, ex.: Estatuto, Regimento Geral, a maior parte do calendário acadêmico) continua visível independente do escopo. `GET /courses` lista os cursos disponíveis.

---

## POST /chat/stream

Versão streaming do `/chat/` via Server-Sent Events (SSE). Emite eventos progressivos enquanto o pipeline executa — o primeiro token da resposta chega antes do processamento completo.

**Request:** idêntico ao `POST /chat/`.

```json
{
  "message": "Como funciona o trancamento de matrícula?",
  "history": [],
  "top_k": 5,
  "filter_revoked": true
}
```

**Response:** `text/event-stream` — cada linha no formato `data: {json}\n\n`

| `type` | Campos | Descrição |
|--------|--------|-----------|
| `status` | `text` | Etapa atual do pipeline |
| `token` | `text` | Fragmento da resposta gerada |
| `done` | `sources`, `used_search`, `used_tools` | Finalização — mesmo formato de `sources`/`used_tools` do `/chat/` |
| `error` | `text` | Mensagem de erro |

```
data: {"type": "status", "text": "Analisando pergunta..."}
data: {"type": "status", "text": "Buscando nos documentos normativos..."}
data: {"type": "status", "text": "Gerando resposta..."}
data: {"type": "token", "text": "O trancamento de matrícula "}
data: {"type": "token", "text": "na UNIVASF é regulado..."}
data: {"type": "done", "sources": [...], "used_search": true, "used_tools": ["search_normative_documents"]}
```

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "x-api-key: sua-chave-aqui" \
  -N \
  -d '{"message": "Quais os critérios para trancamento de matrícula?"}'
```

---

## GET /documents/search

Busca semântica sobre os documentos normativos. Projetado para uso em campo de busca com debounce — sem HyDE, latência mínima (~200–400ms).

**Query params:**

| Param | Tipo | Padrão | Descrição |
|-------|------|--------|-----------|
| `q` | string (min 2 chars) | obrigatório | Termo ou frase de busca |
| `limit` | int (1–48) | `10` | Máximo de documentos retornados |
| `filter_revoked` | bool | `true` | Filtrar documentos revogados |

**Response** — documentos ordenados por relevância semântica:

```json
[
  {
    "source": "Resolução 08_2015 - Normas_gerais_Graduação",
    "filename": "Resolução 08_2015 - Normas_gerais_Graduação.pdf",
    "category": "Resolução PROEN",
    "score": 0.8741,
    "snippet": "Art. 45. O trancamento de matrícula poderá ser requerido...",
    "article_id": "Art. 45",
    "download_url": "/documents/download?source=Resolu%C3%A7%C3%A3o+08_2015..."
  }
]
```

**Padrão de uso no frontend** (campo de busca com debounce):

```typescript
// Chama a cada 300ms após o usuário parar de digitar
const results = q.length >= 2
  ? await fetch(`${API_BASE}/documents/search?q=${encodeURIComponent(q)}`, { headers })
  : await fetch(`${API_BASE}/documents/list`, { headers }); // fallback: lista completa
```

---

## GET /documents/download

Retorna o PDF original do documento citado na resposta do chat.

**Query param:** `source` — valor do campo `source` de qualquer item em `sources` (o `download_url` já vem URL-encoded).

```bash
GET /documents/download?source=Resolução 08_2015 - Normas_gerais_Graduação
```

**Response:** arquivo PDF com `Content-Type: application/pdf` e `Content-Disposition: attachment`.

**Fluxo típico — frontend:**

1. Usuário recebe a resposta do `/chat/` com `sources`
2. Para cada fonte, o `download_url` já está pronto
3. Concatenar com a base URL e abrir:

```typescript
// Abre o PDF em nova aba
const pdfUrl = `${API_BASE}${source.download_url}`;
window.open(pdfUrl, '_blank');

// Ou forçar download direto
const link = document.createElement('a');
link.href = pdfUrl;
link.download = source.source + '.pdf';
link.click();
```

**Erros:**

| Status | Situação |
|--------|----------|
| `200` | PDF retornado com sucesso |
| `401` | API key ausente ou inválida |
| `404` | Documento não encontrado — `source` incorreto |

---

## GET /documents/list

Lista todos os 48 PDFs disponíveis, com `download_url` pronto para cada um. Útil para montar um índice de documentos no frontend.

```bash
GET /documents/list
```

**Response:**

```json
[
  {
    "source": "estatuto-univasf",
    "filename": "estatuto-univasf.pdf",
    "category": "raiz",
    "download_url": "/documents/download?source=estatuto-univasf"
  },
  {
    "source": "Resolução 08_2015 - Normas_gerais_Graduação",
    "filename": "Resolução 08_2015 - Normas_gerais_Graduação.pdf",
    "category": "PROEN",
    "download_url": "/documents/download?source=Resolu%C3%A7%C3%A3o+08_2015+-+Normas_gerais_Gradua%C3%A7%C3%A3o"
  }
]
```

---

---

## GET /logs/queries

Retorna as últimas N interações registradas, mais recentes primeiro.

**Query params:**

| Param | Tipo | Padrão | Descrição |
|-------|------|--------|-----------|
| `limit` | int (1–1000) | `50` | Número de entradas a retornar |
| `only_search` | bool | `false` | Se `true`, retorna apenas queries que acionaram o pipeline RAG |

**Response:**

```json
[
  {
    "timestamp": "2026-04-20T14:32:01.123Z",
    "question": "Quais os critérios para trancamento de matrícula?",
    "search_query": "critérios trancamento matrícula UNIVASF",
    "used_search": true,
    "sources": [
      { "source": "Resolução 08_2015 - Normas_gerais", "score": 0.91, "article_id": "Art. 45" }
    ],
    "top_k": 5,
    "filter_revoked": true,
    "tokens_prompt": 2048,
    "tokens_completion": 116,
    "model": "gpt-4o"
  }
]
```

> Quando `used_search: false` (cumprimento, follow-up), `search_query` e `sources` ficam `null`/`[]`.

---

## GET /logs/stats

Retorna métricas agregadas de todas as interações — útil para análise do TCC.

```bash
GET /logs/stats
```

**Response:**

```json
{
  "total_queries": 142,
  "search_triggered": 128,
  "search_rate": 0.901,
  "total_tokens_prompt": 289536,
  "total_tokens_completion": 16448,
  "top_10_sources": [
    { "source": "Resolução 08_2015 - Normas_gerais_Graduação", "count": 34 },
    { "source": "estatuto-univasf", "count": 21 }
  ]
}
```

| Campo | Descrição |
|-------|-----------|
| `total_queries` | Total de interações registradas |
| `search_triggered` | Quantas acionaram o pipeline RAG |
| `search_rate` | Taxa de uso do RAG (0–1) |
| `total_tokens_prompt` | Total de tokens de entrada consumidos |
| `total_tokens_completion` | Total de tokens de saída consumidos |
| `top_10_sources` | Documentos mais citados nas respostas |

---

## Exemplos com curl

```bash
# Health check
curl http://localhost:8000/health

# Pergunta sobre normas
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -H "x-api-key: sua-chave-aqui" \
  -d '{"message": "Quais os critérios para trancamento de matrícula?"}'

# Com histórico
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -H "x-api-key: sua-chave-aqui" \
  -d '{
    "message": "E qual o prazo para solicitar?",
    "history": [
      {"role": "user", "content": "Como funciona o trancamento?"},
      {"role": "assistant", "content": "O trancamento é regulado pelo Art. 45..."}
    ]
  }'

# Download de um PDF (usando o source retornado pelo /chat/)
curl -OJ \
  -H "x-api-key: sua-chave-aqui" \
  "http://localhost:8000/documents/download?source=Resolu%C3%A7%C3%A3o+08_2015+-+Normas_gerais_Gradua%C3%A7%C3%A3o"
# -O salva o arquivo, -J usa o nome do Content-Disposition

# Listar todos os documentos disponíveis
curl http://localhost:8000/documents/list \
  -H "x-api-key: sua-chave-aqui"

# Últimas 100 queries que acionaram busca
curl "http://localhost:8000/logs/queries?limit=100&only_search=true" \
  -H "x-api-key: sua-chave-aqui"

# Estatísticas agregadas
curl http://localhost:8000/logs/stats \
  -H "x-api-key: sua-chave-aqui"
```

---

## Variáveis de Ambiente

### Backend (`.env`)

| Variável | Descrição |
|----------|-----------|
| `OPENAI_API_KEY` | Chave da API OpenAI (obrigatório) |
| `TCC_API_KEY` | Senha de acesso à API (obrigatório para produção) |
| `CHROMA_HOST` | Host do ChromaDB — `chromadb` em Docker, `localhost` local |
| `CHROMA_PORT` | Porta do ChromaDB — padrão `8000` |

### Frontend React (`.env`)

| Variável | Descrição |
|----------|-----------|
| `VITE_API_URL` | URL da API — padrão `http://localhost:8000` |
| `VITE_API_KEY` | Mesma chave do `TCC_API_KEY` |

---

## CORS

Configurado com `allow_origins=["*"]` para desenvolvimento. Para produção, restringir para o domínio do frontend em `src/main.py`.
