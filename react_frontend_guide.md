# 🎨 Guia: Frontend React para a UNIVASF RAG API

Este documento é um **guia-prompt** para construir uma aplicação React que consome a API FastAPI do projeto.

---

## 1. API Reference

### Base URL

```
http://localhost:8000
```

### `GET /health`

Verifica se a API está online.

```json
// Response 200
{ "status": "ok" }
```

### `POST /chat/`

Endpoint principal de chat. O agente decide automaticamente quando buscar nos documentos normativos.

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
|---|---|---|---|
| `message` | string | ✅ | Mensagem atual do usuário |
| `history` | array | ❌ | Histórico de mensagens `[{role, content}]` |
| `top_k` | int (1-10) | ❌ | Documentos pós-reranking (default: 5) |
| `filter_revoked` | bool | ❌ | Filtrar documentos revogados (default: true) |

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
|---|---|---|
| `answer` | string | Resposta do agente (Markdown) |
| `sources` | array | Fontes normativas consultadas |
| `model` | string | Modelo LLM usado |
| `tokens` | object | `{prompt, completion}` — consumo de tokens |
| `used_search` | bool | `true` = buscou nos documentos, `false` = respondeu direto |

---

## 2. Arquitetura Sugerida (React)

```
src/
├── components/
│   ├── ChatWindow.tsx        # Container principal do chat
│   ├── MessageBubble.tsx     # Bolha de mensagem (user/assistant)
│   ├── SourceCard.tsx        # Card de fonte normativa
│   ├── ChatInput.tsx         # Input de texto + botão enviar
│   └── StatusIndicator.tsx   # Indicador de status da API
├── hooks/
│   └── useChat.ts            # Hook custom para lógica de chat
├── services/
│   └── api.ts                # Funções de chamada à API
├── types/
│   └── chat.ts               # Tipos TypeScript
└── App.tsx
```

---

## 3. Tipos TypeScript

```typescript
// types/chat.ts

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface SourceInfo {
  source: string;
  category: string;
  article_id: string;
  hierarchy: string;
  score: number;
  snippet: string;
}

interface TokenUsage {
  prompt: number;
  completion: number;
}

interface ChatRequest {
  message: string;
  history: ChatMessage[];
  top_k?: number;        // default: 5
  filter_revoked?: boolean; // default: true
}

interface ChatResponse {
  answer: string;
  sources: SourceInfo[];
  model: string;
  tokens: TokenUsage;
  used_search: boolean;
}

// Estado interno de cada mensagem no frontend
interface DisplayMessage extends ChatMessage {
  sources?: SourceInfo[];
  tokens?: TokenUsage;
  used_search?: boolean;
  isLoading?: boolean;
}
```

---

## 4. Service Layer

```typescript
// services/api.ts

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY || "";


export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function sendMessage(request: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
    },

    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Erro ao comunicar com a API");
  }

  return res.json();
}
```

---

## 5. Hook de Chat

```typescript
// hooks/useChat.ts

import { useState, useCallback } from "react";
import { sendMessage } from "../services/api";
import type { DisplayMessage, ChatMessage } from "../types/chat";

export function useChat() {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const send = useCallback(async (text: string) => {
    // Adiciona mensagem do usuário
    const userMsg: DisplayMessage = { role: "user", content: text };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    // Monta histórico (apenas role + content)
    const history: ChatMessage[] = messages.map(m => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const response = await sendMessage({
        message: text,
        history,
      });

      const assistantMsg: DisplayMessage = {
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        tokens: response.tokens,
        used_search: response.used_search,
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: DisplayMessage = {
        role: "assistant",
        content: "❌ Erro ao processar sua pergunta. Tente novamente.",
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [messages]);

  const clear = useCallback(() => setMessages([]), []);

  return { messages, isLoading, send, clear };
}
```

---

## 6. Funcionalidades Recomendadas

### MVP do frontend
- [x] Chat com histórico de mensagens
- [x] Indicador de "digitando..." (loading)
- [x] Exibir fontes consultadas (expandível)
- [x] Indicador de status da API (online/offline)
- [x] Resposta renderizada como Markdown

### Melhorias futuras
- [ ] Dark mode
- [ ] Exportar conversa (PDF/texto)
- [ ] Configurações (top_k, filter_revoked) via settings
- [ ] Animação de streaming (simular com typewriter)
- [ ] Salvar conversas no localStorage

---

## 7. Variáveis de Ambiente

```env
# .env (React / Vite)
# .env (React / Vite)
VITE_API_URL=http://localhost:8000
VITE_API_KEY=sua-senha-secreta

```

---

## 8. CORS

A API já está configurada com `allow_origins=["*"]`. Para produção, restringir para o domínio do frontend:

```python
# src/main.py — ajustar para produção
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://seu-dominio.com"],
    ...
)
```

---

## 9. Dica de Deploy

```
Frontend React (Vercel/Netlify)  →  API FastAPI (Docker)
                                         ↓
                                    ChromaDB (Docker)
```

O `docker-compose.yml` do projeto já orquestra API + ChromaDB. O frontend React pode ser hospedado separadamente apontando `VITE_API_URL` para a URL pública da API.
