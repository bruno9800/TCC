# 📜 RAG UNIVASF — Assistente Normativo

Sistema **Advanced RAG** (Retrieval-Augmented Generation) para consulta inteligente de documentos normativos da UNIVASF (Universidade Federal do Vale do São Francisco).

> **TCC** — Desenvolvimento de uma Solução de Geração Aumentada por Recuperação (RAG) para Automatizar a Consulta de Documentos Normativos da UNIVASF

---

## 📐 Arquitetura

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  48 PDFs    │───▶│  ETL Pipeline │───▶│  Chunks     │
│  (Normas)   │    │  PDF → MD     │    │  JSONL      │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                              │
                                              ▼
                                     ┌────────────────┐
                                     │  ChromaDB       │
                                     │  + Embeddings   │
                                     └────────┬───────┘
                                              │
                   ┌──────────────────────────┤
                   ▼                          ▼
           ┌──────────────┐          ┌──────────────┐
           │  BM25         │          │  Dense Search │
           │  (Esparsa)    │          │  (Vetorial)   │
           └──────┬───────┘          └──────┬───────┘
                  │       Fusion (RRF)       │
                  └───────────┬─────────────┘
                              ▼
                     ┌────────────────┐
                     │  Reranker       │
                     │  Cross-Encoder  │
                     │  Top-50 → Top-5 │
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │  LLM (GPT-4o)  │
                     │  + Citações     │
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │  Streamlit UI   │
                     │  Chat + Fontes  │
                     └────────────────┘
```

---

## 🗂️ Estrutura dos Módulos

### `src/etl/` — Extração e Transformação de Dados

| Arquivo | Descrição |
|---------|-----------|
| `pdf_converter.py` | Converte PDFs para Markdown usando `pymupdf4llm`, preservando cabeçalhos, tabelas e hierarquia visual. Classifica automaticamente cada documento por categoria (Estatuto, Regimento, Resolução PROEN/PROEX/PRPPGI). |
| `revocation_filter.py` | Detecta documentos revogados via regex no nome do arquivo (ex: `_REVOGADA.pdf`) e no conteúdo. Marca cada documento com status `vigente` ou `revogado`. |

### `src/chunking/` — Segmentação Semântica

| Arquivo | Descrição |
|---------|-----------|
| `legal_chunker.py` | **Módulo central do TCC.** Implementa chunking semântico-hierárquico baseado na estrutura legal (Lei Complementar 95/1998). Cada chunk corresponde a um **Artigo completo** (caput + parágrafos + incisos), garantindo que a exceção nunca seja separada da regra. Artigos longos são divididos com herança de contexto do caput. Cada chunk é enriquecido com metadados: hierarquia (Título > Capítulo > Seção), fonte, categoria e status de vigência. |

### `src/indexing/` — Indexação Vetorial

| Arquivo | Descrição |
|---------|-----------|
| `vector_store.py` | Gera embeddings via OpenAI (`text-embedding-3-large`) e armazena no ChromaDB com persistência em disco. Suporta filtros por metadados (status, categoria, fonte). |

### `src/retrieval/` — Recuperação em Dois Estágios

| Arquivo | Descrição |
|---------|-----------|
| `hybrid_search.py` | Combina **busca densa** (vetorial via ChromaDB) com **busca esparsa** (BM25 por palavras-chave) usando **Reciprocal Rank Fusion (RRF)**. Pré-filtra documentos revogados. Retorna top-50 candidatos. |
| `reranker.py` | Aplica um modelo **Cross-Encoder** (`BAAI/bge-reranker-v2-m3`) para reordenar os 50 candidatos por relevância contextual profunda, selecionando apenas os **top-5** para envio ao LLM. |

### `src/generation/` — Geração de Respostas

| Arquivo | Descrição |
|---------|-----------|
| `generator.py` | Constrói o prompt com persona de assistente jurídico universitário, injeta os 5 documentos recuperados como contexto, e gera respostas com **citações obrigatórias** (Artigo + Norma de origem). Utiliza GPT-4o com temperatura baixa (0.1) para máxima fidelidade. |

### `src/evaluation/` — Avaliação de Qualidade

| Arquivo | Descrição |
|---------|-----------|
| `golden_dataset.json` | Dataset de validação com 15 perguntas/respostas baseadas em dúvidas reais de alunos (matrícula, trancamento, estágio, colação de grau, etc.). |
| `ragas_eval.py` | Pipeline de avaliação usando o framework **RAGAS** com 4 métricas: **Faithfulness** (fidelidade ao contexto), **Answer Relevance** (relevância da resposta), **Context Precision** (precisão do ranking) e **Context Recall** (cobertura do contexto). |

### `src/app.py` — Interface do Usuário

| Arquivo | Descrição |
|---------|-----------|
| `app.py` | Interface **Streamlit** com chat interativo, exibição de fontes consultadas (com score de relevância e trecho do documento), sidebar com filtros e estatísticas, e histórico de conversas. |

### `src/config.py` — Configurações

| Arquivo | Descrição |
|---------|-----------|
| `config.py` | Configurações centralizadas: caminhos de diretórios, chaves de API, nomes de modelos, parâmetros de retrieval (top-k, pesos RRF) e parâmetros de chunking (max tokens). |

---

## 🚀 Passo a Passo de Execução

### Pré-requisitos

- Python 3.10+
- Chave da API OpenAI

### 1. Instalar dependências

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Edite o arquivo `.env` e insira sua chave da OpenAI:

```env
OPENAI_API_KEY=sk-sua-chave-aqui
```

### 3. Executar o ETL (PDF → Markdown → Chunks)

```bash
python scripts/run_etl.py
```

Este script:
1. Descobre todos os 48 PDFs em `regimentos_estatutos_resolucoes/`
2. Converte cada PDF para Markdown preservando a estrutura
3. Analisa o status de revogação de cada documento
4. Aplica chunking semântico por Artigo com metadados
5. Salva os chunks em `data/chunks/` como arquivos JSONL

### 4. Indexar no banco vetorial

```bash
python scripts/run_indexing.py
```

Este script:
1. Carrega todos os chunks JSONL
2. Gera embeddings via OpenAI (`text-embedding-3-large`)
3. Armazena vetores + metadados no ChromaDB (persistido em `data/vectorstore/`)

### 5. Abrir a interface

```bash
streamlit run src/app.py
```

Acesse `http://localhost:8501` no navegador e faça perguntas como:
- *"Quais são os critérios para trancamento de matrícula?"*
- *"O que diz o estatuto sobre colação de grau?"*
- *"Quais as normas para dispensa de componentes curriculares?"*

### 6. (Opcional) Rodar avaliação RAGAS

```bash
python scripts/run_eval.py
```

Executa o pipeline completo para cada pergunta do golden dataset e calcula as métricas de qualidade. O relatório é salvo em `data/ragas_report_<timestamp>.json`.

---

## 📊 Métricas de Avaliação (RAGAS)

| Métrica | O que mede | Target |
|---------|-----------|--------|
| **Faithfulness** | A resposta deriva apenas dos documentos recuperados? (detecta alucinação) | > 0.8 |
| **Answer Relevance** | A resposta atende à dúvida do usuário? | > 0.8 |
| **Context Precision** | Os documentos relevantes apareceram no topo do ranking? | > 0.7 |
| **Context Recall** | O sistema encontrou toda a informação necessária? | > 0.7 |

---

## 📄 Documentos Fonte

O sistema processa **48 PDFs** organizados em:

| Diretório | Qtd | Conteúdo |
|-----------|-----|----------|
| `/` (raiz) | 2 | Estatuto e Regimento Geral da UNIVASF |
| `PROEN/` | 37 | Resoluções da Pró-Reitoria de Ensino |
| `PROEX/` | 7 | Documentos da Pró-Reitoria de Extensão |
| `PRPPGI/` | 4 | Resoluções da Pró-Reitoria de Pesquisa e Pós-Graduação |
