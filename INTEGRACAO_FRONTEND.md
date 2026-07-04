# Guia de Integração — UNIVASF RAG API

Documentação de consumo da API para os dois frontends do sistema:

1. **Frontend Principal** — chat público para alunos/comunidade (autenticado por `x-api-key`).
2. **Painel Admin** — gestão de conteúdo (documentos, corpo docente, calendário, cursos), autenticado por JWT (login de `AdminUser`).

Para contratos detalhados de `/chat/*` e `/documents/*` com foco em referência, ver também [API.md](API.md) — este documento é organizado por **consumidor** (qual frontend chama o quê) em vez de por endpoint, e inclui exemplos de código prontos para usar.

Base URL local: `http://localhost:8000`. Todas as rotas retornam JSON, exceto `GET /documents/download` (binário) e `POST /chat/stream` (SSE).

---

## 1. Autenticação

O sistema tem **dois esquemas de autenticação independentes** — não se misturam:

| | Frontend Principal | Painel Admin |
|---|---|---|
| Esquema | API key estática | JWT Bearer (login) |
| Header | `x-api-key: <chave>` | `Authorization: Bearer <token>` |
| Onde obter a credencial | Configurada no `.env` do servidor, distribuída para o frontend em build/deploy | `POST /admin/auth/login` |
| Expira? | Não | Sim — 8h por padrão (`JWT_EXPIRE_MINUTES`) |
| Protege | `/chat`, `/documents`, `/logs`, `/professors`, `/academic-events`, `/courses` | `/admin/*` |

> Se o servidor não tiver `TCC_API_KEY` configurada (`.env`), as rotas públicas ficam sem autenticação — modo válido só para desenvolvimento local, nunca em produção.

### 1.1 Frontend Principal — enviando a API key

```ts
const API_KEY = import.meta.env.VITE_TCC_API_KEY; // ou process.env, conforme o bundler

async function apiFetch(path: string, init: RequestInit = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      "x-api-key": API_KEY,
      "Content-Type": "application/json",
      ...init.headers,
    },
  });
  if (!res.ok) throw await res.json().catch(() => new Error(res.statusText));
  return res.json();
}
```

### 1.2 Painel Admin — login e armazenamento do token

```ts
async function adminLogin(email: string, password: string): Promise<string> {
  const res = await fetch(`${BASE_URL}/admin/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error("Credenciais inválidas");
  const { access_token } = await res.json();
  return access_token; // guardar em memória/sessionStorage — nunca em localStorage se puder evitar (XSS)
}

async function adminFetch(path: string, token: string, init: RequestInit = {}) {
  const res = await fetch(`${BASE_URL}/admin${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      ...init.headers,
    },
  });
  if (res.status === 401) {
    // token expirado/inválido — redirecionar para tela de login
  }
  if (!res.ok) throw await res.json().catch(() => new Error(res.statusText));
  return res.status === 204 ? null : res.json();
}
```

O painel admin não tem endpoint de refresh — quando o token expira (8h), a UI deve tratar `401` redirecionando para o login novamente.

---

## 2. Frontend Principal — Chat e Consulta Pública

| Rota | Método | Uso típico |
|---|---|---|
| `/chat/` | POST | Enviar uma pergunta, receber resposta completa |
| `/chat/stream` | POST | Mesma coisa, mas com tokens chegando incrementalmente (UX de "digitando") |
| `/documents/list` | GET | Listar documentos disponíveis (tela inicial, antes de perguntar algo) |
| `/documents/search` | GET | Busca-enquanto-digita (autocomplete de documentos) |
| `/documents/download` | GET | Baixar o PDF de uma fonte citada na resposta |
| `/professors` | GET | Listar/filtrar corpo docente (fora do chat, ex: página "Professores") |
| `/academic-events` | GET | Listar/filtrar calendário (fora do chat, ex: página "Calendário") |
| `/courses` | GET | Descobrir quais `course_id` existem, para popular um seletor de curso |

### 2.1 `POST /chat/` — pergunta e resposta completa

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
|---|---|---|---|
| `message` | string | Sim | Pergunta atual do usuário |
| `history` | `{role, content}[]` | Não | Turnos anteriores da conversa — padrão `[]` |
| `top_k` | int (1–10) | Não | Nº de fontes pós-reranking — padrão `5` |
| `filter_revoked` | bool | Não | Exclui normas revogadas — padrão `true` |
| `course_id` | int \| null | Não | Escopa a busca a um curso (ver `GET /courses`). Conteúdo institucional (Estatuto, Regimento Geral, a maior parte do calendário) continua visível independente do escopo. Omitir = busca em tudo |

**Response (200):**

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
      "snippet": "Art. 45. O trancamento de matrícula...",
      "download_url": "/documents/download?source=Resolu%C3%A7%C3%A3o+08_2015+-+Normas_gerais"
    }
  ],
  "model": "gpt-4o",
  "tokens": { "prompt": 2048, "completion": 116 },
  "used_search": true,
  "used_tools": ["search_normative_documents"]
}
```

`sources[].origin` tem 4 valores possíveis — a UI deve renderizar cada um de forma diferente (um "card de fonte" genérico funciona, mas fica melhor distinguindo visualmente):

| `origin` | Tem `download_url`? | Campos relevantes |
|---|---|---|
| `"rag"` | Sim | `article_id`, `hierarchy`, `score`, `snippet` |
| `"professor"` | Não | `category` (ex: `"Professor (NDE)"`), `snippet` (e-mail + área) |
| `"discipline"` | Não | `category` (ex: `"Matriz Curricular — 7º período"`), `snippet` (carga horária + pré-requisitos) |
| `"calendar"` | Não | `category` (ex: `"Calendário Acadêmico — trancamento"`), `snippet` (datas) |

```ts
interface ChatResponse {
  answer: string;
  sources: SourceInfo[];
  model: string;
  tokens: { prompt: number; completion: number };
  used_search: boolean;
  used_tools: string[];
}

async function sendMessage(message: string, history: ChatMessage[], courseId?: number) {
  return apiFetch("/chat/", {
    method: "POST",
    body: JSON.stringify({ message, history, course_id: courseId ?? null }),
  }) as Promise<ChatResponse>;
}
```

### 2.2 `POST /chat/stream` — streaming (SSE)

Mesmo request body do `/chat/`. Resposta é `text/event-stream`, cada linha `data: {json}\n\n`. Como o `x-api-key` precisa ir num header customizado, **não dá para usar `EventSource` nativo do browser** (ele não suporta headers) — leia o stream manualmente via `fetch` + `ReadableStream`:

```ts
type ChatStreamEvent =
  | { type: "status"; text: string }
  | { type: "token"; text: string }
  | { type: "done"; sources: SourceInfo[]; used_search: boolean; used_tools: string[] }
  | { type: "error"; text: string };

async function streamMessage(
  message: string,
  history: ChatMessage[],
  courseId: number | undefined,
  onEvent: (e: ChatStreamEvent) => void,
) {
  const res = await fetch(`${BASE_URL}/chat/stream`, {
    method: "POST",
    headers: { "x-api-key": API_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, course_id: courseId ?? null }),
  });
  if (!res.ok || !res.body) throw new Error(`Erro ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? ""; // último pedaço pode estar incompleto
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      onEvent(JSON.parse(line.slice(5).trim()) as ChatStreamEvent);
    }
  }
}
```

Uso típico numa tela de chat:

```ts
let answer = "";
await streamMessage(userMessage, history, selectedCourseId, (event) => {
  switch (event.type) {
    case "status":
      setStatusLabel(event.text); // "Buscando nos documentos normativos..."
      break;
    case "token":
      answer += event.text;
      setPartialAnswer(answer); // re-renderiza incrementalmente
      break;
    case "done":
      setSources(event.sources);
      setFinalAnswer(answer);
      break;
    case "error":
      setError(event.text);
      break;
  }
});
```

### 2.3 `GET /documents/list` — listagem inicial

Sem query params. Retorna todos os documentos `status="indexed"`:

```json
[
  {
    "source": "Resolução 08_2015 - Normas_gerais",
    "filename": "resolucao-08-2015.pdf",
    "category": "Resolução PROEN",
    "download_url": "/documents/download?source=Resolu%C3%A7%C3%A3o+08_2015+-+Normas_gerais"
  }
]
```

### 2.4 `GET /documents/search?q=...` — busca-enquanto-digita

| Query param | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `q` | string (min 2) | Sim | Termo de busca |
| `limit` | int (1–48) | Não | Padrão `10` |
| `filter_revoked` | bool | Não | Padrão `true` |

Latência baixa (~200–400ms, sem HyDE) — pensado para debounce em campo de busca, não para o chat em si. Retorna um item por documento-fonte (melhor chunk), com `score`/`snippet`/`download_url`.

```ts
const results = await apiFetch(`/documents/search?q=${encodeURIComponent(query)}&limit=10`);
```

### 2.5 `GET /documents/download?source=...`

`source` é o mesmo valor de `sources[].source` retornado pelo chat — passar direto, sem normalizar (o backend já faz o encode). Resposta é o PDF binário (`Content-Type: application/pdf`):

```ts
function downloadUrl(source: string) {
  return `${BASE_URL}/documents/download?source=${encodeURIComponent(source)}`;
}
// <a href={downloadUrl(s.source)} target="_blank">Baixar PDF</a>
// x-api-key precisa ir via header — se o link for aberto direto pelo browser (não via fetch),
// ou a rota fica sem proteção de fato, ou usa um token de curta duração em query string
// (não implementado hoje: o download público atual depende do mesmo x-api-key do resto).
```

### 2.6 `GET /professors`, `GET /academic-events`, `GET /courses`

Mesma forma dos endpoints acima — `GET` simples com `x-api-key`, sem paginação (os volumes de dados são pequenos: dezenas de professores/eventos, poucos cursos).

```ts
interface DisciplineOut {
  id: number;
  course_id: number;
  name: string;
  code: string | null;
  period: number | null;            // 1–10, null = optativa
  workload: number | null;          // horas
  prerequisites_text: string | null;
}

// GET /professors?course_id=&area=&name=
interface ProfessorOut {
  id: number;
  name: string;
  email: string;
  email_secondary: string | null;
  department: string | null;
  course_id: number | null;
  area: string | null;
  lattes_url: string | null;
  personal_site_url: string | null;
  is_nde: boolean;
  nde_role: string | null;
  bio: string | null;
  created_at: string;
  disciplines: { semester_year: string; schedule_text: string | null; room: string | null; discipline: DisciplineOut }[];
  // ⚠️ Hoje sempre `[]` na prática: os professores foram semeados sem passar por
  // POST /admin/professors/{id}/disciplines. Só populado depois que o admin
  // associar disciplinas explicitamente (seção 3.2).
}

// GET /academic-events?course_id=&category=&academic_period=&date_from=&date_to=
interface AcademicEventOut {
  id: number;
  course_id: number | null;
  title: string;
  start_date: string;   // "YYYY-MM-DD"
  end_date: string | null;
  category: string | null;      // "matrícula" | "feriado" | "trancamento" | "colação" | "período_letivo" | "exames" | "planejamento" | "outro"
  legal_reference: string | null;
  campus: string | null;        // "JUA" | "PNZ" | "PAV" | "SAL" | "SBF" | "SRN" | null (todos)
  academic_period: string | null; // "2026.1"
}

// GET /courses?active_only=
interface CourseOut {
  id: number;
  code: string;
  name: string;
  active: boolean;
}
```

Uso sugerido: `GET /courses` alimenta um seletor de curso na tela de chat (`<select>` com "Todos os cursos" = omitir `course_id`); `GET /professors`/`GET /academic-events` alimentam páginas dedicadas (ex: "Corpo Docente", "Calendário Acadêmico") fora do fluxo de chat, reusando os mesmos dados que o agente consulta internamente.

---

## 3. Painel Admin — Gestão de Conteúdo

Todas as rotas abaixo são `/admin/*`, exigem `Authorization: Bearer <token>` (seção 1.2), e retornam `401` se o token faltar/expirar.

### 3.1 Documentos

| Rota | Método | Descrição |
|---|---|---|
| `/admin/documents` | POST | Upload (multipart) — dispara ETL → chunk → embed → index |
| `/admin/documents` | GET | Lista (`?course_id=&knowledge_base_id=&status_filter=`) |
| `/admin/documents/{id}` | GET | Detalhe |
| `/admin/documents/{id}` | PATCH | Atualiza metadados (título, categoria, curso, revogação) |
| `/admin/documents/{id}` | DELETE | Remove documento + vetores + arquivo |
| `/admin/documents/{id}/reindex` | POST | Força reprocessamento |

**Upload (multipart, não JSON):**

```ts
async function uploadDocument(
  file: File,
  knowledgeBaseSlug: string,
  opts: { courseId?: number; title?: string; category?: string },
  token: string,
) {
  const form = new FormData();
  form.append("file", file);
  form.append("knowledge_base_slug", knowledgeBaseSlug); // ex: "regulamentos"
  if (opts.courseId != null) form.append("course_id", String(opts.courseId));
  if (opts.title) form.append("title", opts.title);
  if (opts.category) form.append("category", opts.category);

  const res = await fetch(`${BASE_URL}/admin/documents`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` }, // NÃO define Content-Type — o browser seta o boundary do multipart
    body: form,
  });
  if (!res.ok) throw await res.json();
  return res.json(); // DocumentOut, status="processing" até a ingestão terminar
}
```

O upload é **síncrono** — a resposta só volta depois que o pipeline inteiro (ETL → chunk → embed → index) termina. Para PDFs grandes (o PPC real, 144 páginas, levou dezenas de segundos), a UI deve mostrar um spinner/loading state explícito, não assumir resposta imediata.

**`DocumentOut` (resposta de POST/GET/PATCH):**

```ts
interface DocumentOut {
  id: number;
  knowledge_base_id: number;
  course_id: number | null;
  title: string;
  filename: string;
  category: string | null;
  status: "processing" | "chunked" | "indexed" | "failed";
  version: number;
  revoked: boolean;
  revoked_reason: string | null;
  superseded_by_document_id: number | null;
  uploaded_at: string;
  indexed_at: string | null;
}
```

**PATCH** (`DocumentUpdateRequest`, todos os campos opcionais — só envia o que muda): `title`, `category`, `course_id`, `knowledge_base_id`, `revoked`, `revoked_reason`, `superseded_by_document_id`.

### 3.2 Corpo Docente

| Rota | Método | Descrição |
|---|---|---|
| `/admin/professors` | POST | Cadastra professor |
| `/admin/professors` | GET | Lista (`?course_id=&area=&name=`) |
| `/admin/professors/{id}` | GET | Detalhe |
| `/admin/professors/{id}` | PATCH | Atualiza |
| `/admin/professors/{id}` | DELETE | Remove |
| `/admin/professors/{id}/disciplines` | POST | Associa/atualiza disciplina lecionada num semestre |
| `/admin/disciplines` | POST | Cadastra disciplina |
| `/admin/disciplines` | GET | Lista (`?course_id=`) |

**`ProfessorCreateRequest`:**

```ts
interface ProfessorCreateRequest {
  name: string;
  email: string;
  email_secondary?: string | null;
  department?: string | null;
  course_id?: number | null;
  area?: string | null;
  lattes_url?: string | null;
  personal_site_url?: string | null;
  is_nde?: boolean;       // padrão false
  nde_role?: string | null; // ex: "Coordenador"
  bio?: string | null;
}
```

`ProfessorUpdateRequest` tem os mesmos campos, todos opcionais (PATCH parcial — só envia o que muda).

**`DisciplineCreateRequest`** (`course_id` é o único campo obrigatório além de `name`):

```ts
interface DisciplineCreateRequest {
  course_id: number;
  name: string;
  code?: string | null;
  period?: number | null;           // 1–10, null = optativa
  workload?: number | null;         // horas
  prerequisites_text?: string | null; // texto livre, ex: "AED, LFA e OAC"
}
```

**Associar disciplina a um professor** (idempotente — reenviar atualiza horário/sala em vez de duplicar):

```ts
await adminFetch(`/professors/${professorId}/disciplines`, token, {
  method: "POST",
  body: JSON.stringify({
    discipline_id: disciplineId,
    semester_year: "2026.1",
    schedule_text: "Seg/Qua 08h-10h",
    room: "Bloco B - Sala 12",
  }),
});
```

Não existe `PATCH`/`DELETE` para `Discipline` — cadastro/edição de disciplinas é uma operação rara, feita uma vez por matriz curricular; se necessário, apagar e recriar.

### 3.3 Calendário Acadêmico

| Rota | Método | Descrição |
|---|---|---|
| `/admin/academic-events` | POST | Cadastra evento |
| `/admin/academic-events` | GET | Lista (`?course_id=&category=&academic_period=`) |
| `/admin/academic-events/{id}` | GET | Detalhe |
| `/admin/academic-events/{id}` | PATCH | Atualiza |
| `/admin/academic-events/{id}` | DELETE | Remove |

```ts
interface AcademicEventCreateRequest {
  title: string;
  start_date: string;              // "YYYY-MM-DD"
  end_date?: string | null;
  course_id?: number | null;       // null = institucional (vale pra todos os cursos)
  category?: string | null;
  legal_reference?: string | null;
  campus?: string | null;
  academic_period?: string | null; // "2026.1"
}
```

`AcademicEventUpdateRequest` tem os mesmos campos, todos opcionais (PATCH parcial).

### 3.4 Cursos

| Rota | Método | Descrição |
|---|---|---|
| `/admin/courses` | POST | Cadastra curso |
| `/admin/courses` | GET | Lista (`?active_only=`) |
| `/admin/courses/{id}` | GET | Detalhe |

```ts
interface CourseCreateRequest {
  code: string;   // ex: "ENGCOMP", único
  name: string;
  active?: boolean; // padrão true
}
```

Sem `PATCH`/`DELETE` — cursos mudam raramente. Para "desativar" um curso sem apagar seu histórico (documentos/professores/disciplinas/eventos vinculados), o campo `active` existe no modelo mas hoje só é setável na criação; se a UI precisar editar isso depois, é um endpoint pequeno a acrescentar (fora do escopo atual).

### 3.5 Análises (opcional, para um dashboard admin)

`GET /logs/queries` e `GET /logs/queries/stats` **não** ficam sob `/admin` — são protegidos por `x-api-key` (mesmo esquema do frontend principal), não por login de admin. Se o painel admin quiser exibir métricas de uso, ele precisa da `x-api-key` além do token JWT:

```ts
// GET /logs/queries/stats  → { total_queries, search_triggered, search_rate, total_tokens_prompt, total_tokens_completion, top_10_sources }
// GET /logs/queries?limit=&only_search=  → últimas N entradas, mais recentes primeiro
```

---

## 4. Referência Rápida — Todos os Endpoints

| Rota | Método | Auth | Consumidor |
|---|---|---|---|
| `/health` | GET | Nenhuma | Ambos (health check) |
| `/chat/` | POST | `x-api-key` | Principal |
| `/chat/stream` | POST | `x-api-key` | Principal |
| `/documents/list` | GET | `x-api-key` | Principal |
| `/documents/search` | GET | `x-api-key` | Principal |
| `/documents/download` | GET | `x-api-key` | Principal |
| `/professors` | GET | `x-api-key` | Principal |
| `/academic-events` | GET | `x-api-key` | Principal |
| `/courses` | GET | `x-api-key` | Principal |
| `/logs/queries` | GET | `x-api-key` | Admin (analytics) |
| `/logs/queries/stats` | GET | `x-api-key` | Admin (analytics) |
| `/admin/auth/login` | POST | Nenhuma (é o próprio login) | Admin |
| `/admin/documents` | POST, GET | Bearer JWT | Admin |
| `/admin/documents/{id}` | GET, PATCH, DELETE | Bearer JWT | Admin |
| `/admin/documents/{id}/reindex` | POST | Bearer JWT | Admin |
| `/admin/professors` | POST, GET | Bearer JWT | Admin |
| `/admin/professors/{id}` | GET, PATCH, DELETE | Bearer JWT | Admin |
| `/admin/professors/{id}/disciplines` | POST | Bearer JWT | Admin |
| `/admin/disciplines` | POST, GET | Bearer JWT | Admin |
| `/admin/academic-events` | POST, GET | Bearer JWT | Admin |
| `/admin/academic-events/{id}` | GET, PATCH, DELETE | Bearer JWT | Admin |
| `/admin/courses` | POST, GET | Bearer JWT | Admin |
| `/admin/courses/{id}` | GET | Bearer JWT | Admin |

---

## 5. Erros Comuns

Erros seguem o formato padrão do FastAPI: `{"detail": "mensagem"}` (ou `{"detail": [...]}` para erro de validação de schema, 422).

| Status | Quando acontece | Onde tratar |
|---|---|---|
| `401` | JWT ausente/inválido/expirado (rotas `/admin/*`) | Painel admin: redirecionar para login |
| `403` | `x-api-key` ausente ou errada (rotas públicas — `src/auth.py` sempre responde 403, nunca 401, para esse esquema) | Frontend principal: raro em produção, indica config errada no build/deploy |
| `404` | Recurso não encontrado (`document_id`, `professor_id`, `event_id`, `course_id` inexistente) | Exibir mensagem específica, não genérica |
| `400` | Erro de validação de negócio (ex: `knowledge_base_slug` inexistente no upload) | Exibir `detail` da resposta diretamente — já vem em texto legível |
| `422` | Payload não bate com o schema esperado (campo obrigatório faltando, tipo errado) | Bug do frontend — não deveria acontecer em produção com os tipos acima |
| `500` | Erro interno (falha no LLM, no ChromaDB, etc.) | Mensagem genérica + retry manual — `detail` pode conter stack trace, não exibir ao usuário final |
