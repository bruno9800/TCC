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
  "filter_revoked": true
}
```

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `message` | string | Sim | Mensagem atual do usuário |
| `history` | array | Não | Histórico `[{role, content}]` — padrão: `[]` |
| `top_k` | int (1–10) | Não | Documentos pós-reranking — padrão: `5` |
| `filter_revoked` | bool | Não | Filtrar documentos revogados — padrão: `true` |

**Response:**

```json
{
  "answer": "O trancamento de matrícula na UNIVASF é regulado pela...",
  "sources": [
    {
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
  "used_search": true
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `answer` | string | Resposta do agente (Markdown) |
| `sources` | array | Fontes normativas consultadas (vazio se `used_search: false`) |
| `model` | string | Modelo LLM usado |
| `tokens` | object | `{prompt, completion}` — consumo de tokens |
| `used_search` | bool | `true` = pipeline RAG acionado, `false` = resposta direta |

Cada item de `sources` inclui o campo `download_url` pronto para uso:

```json
{
  "source": "Resolução 08_2015 - Normas_gerais",
  "category": "Resolução PROEN",
  "article_id": "Art. 45",
  "hierarchy": "Título IV > Capítulo II",
  "score": 0.8523,
  "snippet": "Art. 45. O trancamento de matrícula...",
  "download_url": "/documents/download?source=Resolu%C3%A7%C3%A3o+08_2015+-+Normas_gerais"
}
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
