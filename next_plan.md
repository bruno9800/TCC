Este plano foi desenhado para ser **modular**, **simples** e focado no **MVP**, garantindo que a complexidade técnica não atrapalhe a entrega de valor (Chat + Gestão de Professores).

A grande sacada aqui é tratar o **Cronograma do Professor** não apenas como um arquivo para download, mas como **conhecimento** que o Chat pode ler e responder.

---

# 🏛️ Arquitetura do Sistema: UNIVASF RAG v2.0 (MVP)

A arquitetura é dividida em três grandes blocos lógicos: **Gestão (Admin/Auth)**, **Conhecimento (RAG/Vetorial)** e **Interação (Chat)**.

## 1. 📐 Diagrama de Alto Nível

```mermaid
graph TD
    subgraph "Frontend (React)"
        Login[Login / Cadastro]
        DashProf[Dashboard Professor]
        ChatUI[Chat Inteligente]
    end

    subgraph "Backend (FastAPI)"
        AuthAPI[🔐 Módulo Auth]
        ProfAPI[🎓 Módulo Professor]
        AgentAPI[🤖 Módulo Chat (Agent)]
    end

    subgraph "Bancos de Dados"
        Postgres[(PostgreSQL\nUsuários, Perfis, Logs)]
        Chroma[(ChromaDB\nNormas + Cronogramas)]
    end

    Login --> AuthAPI
    DashProf --> ProfAPI
    ChatUI --> AgentAPI

    AuthAPI --> Postgres
    ProfAPI --> Postgres
    ProfAPI -- "Indexa Cronograma" --> Chroma
    AgentAPI -- "Lê Contexto" --> Chroma

```
Esse projeto nao tem como fim fazer o frontend, apenas o backend, o agente e as tools.

---

## 2. 🗂️ Modelagem de Dados (Simples e Eficiente)

Usaremos **PostgreSQL** para dados estruturados e **ChromaDB** para dados não estruturados (textos).

### A. Tabela de Usuários (`users`)

Aqui aplicamos a regra de negócio do e-mail universitário.

| Coluna | Tipo | Descrição |
| --- | --- | --- |
| `id` | UUID | Identificador único. |
| `email` | VARCHAR | **Validar:** Deve terminar em `@univasf.edu.br`. |
| `password_hash` | VARCHAR | Senha criptografada (bcrypt). |
| `role` | ENUM | `student` ou `professor`. |
| `university` | VARCHAR | Default: `UNIVASF`. |
| `is_active` | BOOLEAN | Validação de e-mail. |

### B. Tabela de Perfis (`profiles`)

Dados específicos acadêmicos.

| Coluna | Tipo | Descrição |
| --- | --- | --- |
| `user_id` | UUID | FK para `users`. |
| `full_name` | VARCHAR | Nome completo. |
| `cpf` | VARCHAR | CPF (Apenas números). |
| `matricula` | VARCHAR | Matrícula SIAPE ou Discente. |
| `cohort` | VARCHAR | Turma de entrada (ex: `2019.2`). |
| `course` | VARCHAR | Curso (ex: `Engenharia de Computação`). |

### C. Tabela de Professores (`professors_data`)

Extensão para quem tem `role='professor'`.

| Coluna | Tipo | Descrição |
| --- | --- | --- |
| `user_id` | UUID | FK para `users`. |
| `department` | VARCHAR | Colegiado (ex: `Colegiado de Eng. Comp.`). |
| `subjects` | JSON | Lista de matérias (ex: `["Cálculo I", "IA"]`). |
| `schedule_text` | TEXT | **O Pulo do Gato:** O texto extraído do PDF do cronograma. |
| `last_update` | DATETIME | Data da última atualização. |

---

## 3. 🧩 Módulos do Sistema (Backend FastAPI)

A estrutura de pastas do seu projeto deve refletir essa divisão:

```text
src/
├── auth/           # Login e Cadastro
│   ├── router.py
│   └── service.py
├── professors/     # Gestão de Cronogramas
│   ├── router.py
│   └── indexer.py  <-- Mágica acontece aqui
├── chat/           # O Agente RAG
│   ├── router.py
│   ├── agent.py
│   └── tools.py
├── database/       # Conexão Postgres e Chroma
└── main.py         # Entrypoint

```

### Módulo 1: Auth & Cadastro (`src/auth`)

* **Regra de Ouro:** No endpoint de registro (`POST /register`), o sistema verifica se `email.endswith("@univasf.edu.br")`. Se não, retorna erro 403.
* **Fluxo:** O usuário se cadastra -> Recebe link de confirmação no email -> Faz login -> Recebe Token JWT.

### Módulo 2: O "Hub" do Professor (`src/professors`)

Aqui resolvemos o problema de como o professor atualiza suas informações e cronograma.

**Funcionalidade:** "Upload de Cronograma".

1. O professor acessa o painel e faz upload de um PDF (Plano de Ensino).
2. O Backend recebe o PDF.
3. **Processamento (ETL Rápido):**
* Extrai o texto do PDF.
* Salva o texto cru no PostgreSQL (`schedule_text`) para exibição simples.
* **Vetorização:** Envia esse texto para o **ChromaDB** com metadados especiais.



```python
# Exemplo de Metadado no ChromaDB para um cronograma
metadata = {
    "type": "schedule",
    "professor_id": "123-abc",
    "professor_name": "Prof. Girafales",
    "subject": "Inteligência Artificial",
    "semester": "2024.1"
}

```

*Por que isso é genial?* Porque agora o Chat pode buscar "Quando é a prova do Girafales?" e encontrar a resposta nesse documento específico.

### Módulo 4: Inteligência Dinâmica (Chat & Ferramentas)

A interface não terá campos fixos, pois as informações mudam conforme a ferramenta que o Agente escolhe usar:

*   **Configurações de Entrada:** O frontend envia `top_k` e `filter_revoked`, que o Agente repassa para as ferramentas de busca.
*   **Fontes Contextuais (Context-Specific):** 
    *   Se a `LegalTool` for usada: A resposta trará `hierarquia`, `artigo` e link para o PDF da norma.
    *   Se a `ScheduleTool` for usada: A resposta trará `professor_name`, `sala` e `horários`.
*   **Transparência Total:** Independente da ferramenta, a API sempre retornará o `score` (confiança) e os `snippets` (trechos) reais usados para gerar a resposta.
*   **Log de Auditoria:** O frontend terá acesso aos `tokens` gastos e ao modelo usado pelo Agente naquela interação específica.

### Módulo 3: O Chat Inteligente (`src/chat`)

O Agente agora terá duas ferramentas principais (`Tools`):

1. **`LegalTool` (Normas):** Busca na coleção `normas_univasf` (seus 48 PDFs originais).
* *Prompt:* "Use para dúvidas sobre leis, trancamento, direitos."


2. **`ScheduleTool` (Cronogramas):** Busca na coleção `cronogramas_professores`.
* *Prompt:* "Use quando o aluno perguntar sobre datas de provas, ementa de disciplina ou horários de um professor específico."



---

## 4. 🚀 Fluxo de Uso (User Stories)

### Cenário A: O Professor Atualizando

1. O **Prof. Severo** faz login.
2. Clica em "Meus Dados".
3. Edita: "Atendimento aos alunos: Quartas, 14h, Sala 20".
4. Clica em "Atualizar Cronograma" e sobe o PDF de "Cálculo 1".
5. O sistema processa e diz: "Cronograma indexado com sucesso!".

### Cenário B: O Aluno Perguntando

1. **Aluno:** "Quando é a primeira prova de Cálculo do Prof. Severo?"
2. **Agente (Cérebro):**
* Identifica intenção: *Dúvida sobre disciplina/professor*.
* Seleciona Tool: `ScheduleTool(query="prova cálculo Severo")`.


3. **Sistema:** Busca no ChromaDB os chunks associados ao Prof. Severo.
4. **Agente (Resposta):** "De acordo com o cronograma atualizado do Prof. Severo, a primeira prova de Cálculo 1 está marcada para o dia **15/04/2026**."

---

## 5. 🗺️ Plano de Implementação MVP

### 1: Fundação (Postgres + Auth)

1. Subir container Docker do PostgreSQL.
2. Criar tabelas (`users`, `professors`).
3. Implementar rota `/register` com validação de e-mail UNIVASF.
4. Implementar rota `/login` gerando JWT.

### 2: Módulo do Professor (Upload + Indexação)

1. Criar rota `POST /professor/schedule`.
2. Integrar a lógica de leitura de PDF (que você já tem no ETL) nesta rota.
3. Fazer com que, ao salvar, ele insira os vetores no ChromaDB com a tag `type: schedule`.

### 3: Chat Integrado (Agente)

1. Refatorar seu script de chat atual para ser uma API (`POST /chat`).
2. Configurar o Agente para decidir entre buscar nas Normas ou nos Cronogramas.
3. Testar: Cadastrar um professor fake, subir um cronograma e tentar perguntar sobre ele no chat.

---

## 6. 🐳 Infraestrutura (Docker e Deploy)

Para deixar de ser um script e virar um projeto profissional, usaremos Docker para gerenciar o Backend e os Bancos de Dados.

### A. Dockerfile (Imagem do Backend)
Crie um arquivo `Dockerfile` na raiz para empacotar o FastAPI:
---

### Conclusão: De Script para Plataforma
Com essa estrutura, seu TCC deixa de ser apenas uma "IA que lê PDF" e se torna uma **infraestrutura escalável de serviços universitários**, pronta para integrar com Web, App ou qualquer outra tecnologia.
