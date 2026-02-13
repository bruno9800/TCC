Walkthrough — Sistema RAG para Documentos Normativos da UNIVASF
O que foi implementado
Sistema Advanced RAG completo com 8 módulos, cobrindo o pipeline inteiro desde a ingestão de PDFs até a interface de chat.

Estrutura do Projeto
TCC/
├── pyproject.toml                     # dependências
├── .env.example                       # template de variáveis de ambiente
├── src/
│   ├── config.py                      # configurações centralizadas
│   ├── etl/
│   │   ├── pdf_converter.py           # PDF → Markdown (pymupdf4llm)
│   │   └── revocation_filter.py       # detecção de revogações
│   ├── chunking/
│   │   └── legal_chunker.py           # chunking semântico por artigo ⭐
│   ├── indexing/
│   │   └── vector_store.py            # ChromaDB + OpenAI embeddings
│   ├── retrieval/
│   │   ├── hybrid_search.py           # dense + BM25 + RRF fusion
│   │   └── reranker.py                # cross-encoder reranking
│   ├── generation/
│   │   └── generator.py               # prompt + LLM + citações
│   ├── evaluation/
│   │   ├── golden_dataset.json         # 15 perguntas gold
│   │   └── ragas_eval.py              # pipeline RAGAS
│   └── app.py                         # Streamlit UI
├── scripts/
│   ├── run_etl.py                     # executa ETL completa
│   ├── run_indexing.py                # indexa chunks no ChromaDB
│   └── run_eval.py                    # roda avaliação RAGAS
└── regimentos_estatutos_resolucoes/   # 48 PDFs fonte
Verificação
✅ Instalação de Dependências
Todas as dependências instaladas com sucesso no virtualenv .venv.

✅ Importação de Módulos (8/8)
Todos os módulos importam sem erros.

✅ Teste Funcional do Chunker
Dado um trecho com 3 artigos de estrutura legal:

4 chunks gerados (1 preâmbulo + 3 artigos)
Article IDs extraídos: Art. 1º, Art. 2º, Art. 3º
Hierarquia rastreada: mudança de CAPÍTULO I para CAPÍTULO II detectada
Parágrafos mantidos com seus artigos: § 1º e § 2º agrupados com Art. 1º
✅ PDFs Detectados
48 PDFs encontrados em regimentos_estatutos_resolucoes/.

Como Usar
Passo 1: Configurar API Key
bash
cp .env.example .env
# Edite .env e coloque sua OPENAI_API_KEY
Passo 2: Executar ETL (PDFs → Chunks)
bash
source .venv/bin/activate
python scripts/run_etl.py
Converte 48 PDFs → Markdown → Chunks JSONL com metadados

Passo 3: Indexar no ChromaDB
bash
python scripts/run_indexing.py
Gera embeddings OpenAI e cria índice vetorial persistente

Passo 4: Abrir Interface
bash
streamlit run src/app.py
Passo 5 (Opcional): Avaliar com RAGAS
bash
python scripts/run_eval.py
Executa 15 perguntas do golden dataset e calcula métricas (Faithfulness, Answer Relevance, Context Precision, Context Recall)