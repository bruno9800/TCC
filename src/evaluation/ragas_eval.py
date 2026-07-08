"""
Pipeline de Avaliação RAGAS

Avalia a qualidade do sistema RAG usando o framework RAGAS
com métricas de Faithfulness, Answer Relevance, Context Precision e Context Recall.

Este caminho é deliberadamente desacoplado do agente (src/agent/) — chama o
pipeline de retrieval+geração "cru" (generate_answer), medindo a qualidade
do RAG em si, sem a camada de decisão multi-tool por cima (ver EVOLUTION_V2.md,
Fase 4, "Por que src/generation e src/evaluation não foram tocados").

O LLM juiz e os embeddings do RAGAS são configurados EXPLICITAMENTE (não usam
o default do framework, que apontaria para um modelo mais caro): o juiz usa
RAGAS_JUDGE_MODEL (padrão: o mesmo LLM_MODEL da geração) e os embeddings usam
o mesmo EMBEDDING_MODEL do pipeline — controle de custo e consistência
metodológica.
"""

import json
import logging
import os
from pathlib import Path

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import EMBEDDING_MODEL, LLM_MODEL
from src.generation.generator import generate_answer
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

# Modelo usado como juiz pelo RAGAS. Separável do modelo de geração via env
# (avaliar geração barata com um juiz mais forte, se desejado), mas por padrão
# usa o mesmo LLM_MODEL para manter o custo previsível. `or` (não default do
# getenv) porque o Makefile pode passar a variável como string vazia.
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL") or LLM_MODEL

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_golden_dataset() -> list[dict]:
    """Carrega o dataset de validação."""
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    search_engine: HybridSearchEngine | None = None,
    output_path: Path | None = None,
) -> dict:
    """
    Executa a avaliação RAGAS completa.

    Pipeline por pergunta:
    1. Busca híbrida → top-50 candidatos
    2. Reranking → top-5
    3. Geração → resposta com citações
    4. Avaliação RAGAS sobre o conjunto

    Args:
        search_engine: Motor de busca híbrida (cria se None).
        output_path: Onde salvar o relatório (opcional).

    Returns:
        Dicionário com scores por métrica (médias) e detalhes por pergunta.
    """
    if search_engine is None:
        search_engine = HybridSearchEngine()

    golden = load_golden_dataset()

    # Coleta de dados para o RAGAS
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for i, item in enumerate(golden):
        query = item["question"]
        gt = item["ground_truth"]

        logger.info(f"[{i+1}/{len(golden)}] Processando: {query[:60]}...")

        # 1. Busca híbrida
        candidates = search_engine.search_hybrid(query)

        # 2. Reranking
        top_results = rerank(query, candidates)

        # 3. Geração
        generation = generate_answer(query, top_results)

        # Salva dados
        questions.append(query)
        answers.append(generation.answer)
        contexts.append([r.content for r in top_results])
        ground_truths.append(gt)

    # 4. Avaliação RAGAS — juiz e embeddings explícitos (controle de custo)
    logger.info(f"Avaliando com RAGAS (juiz: {RAGAS_JUDGE_MODEL}, embeddings: {EMBEDDING_MODEL})...")
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=RAGAS_JUDGE_MODEL, temperature=0.0))
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL))

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    results = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # EvaluationResult (ragas>=0.2) não é dict — extrai scores via DataFrame,
    # uma linha por pergunta, uma coluna por métrica. NaN (falha de avaliação
    # em uma amostra) é ignorado na média, mas contado no relatório.
    df = results.to_pandas()
    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]

    report = {
        "config": {
            "generation_model": LLM_MODEL,
            "judge_model": RAGAS_JUDGE_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "ragas_version": _ragas_version(),
        },
        "metrics": {col: round(float(df[col].mean()), 4) for col in metric_cols},
        "failed_samples": {
            col: int(df[col].isna().sum()) for col in metric_cols if df[col].isna().any()
        },
        "num_questions": len(golden),
        "details": [
            {
                "question": questions[i],
                "answer": answers[i][:200] + ("..." if len(answers[i]) > 200 else ""),
                "num_contexts": len(contexts[i]),
                "ground_truth": ground_truths[i],
                "scores": {
                    col: (round(float(df.iloc[i][col]), 4) if df.iloc[i][col] == df.iloc[i][col] else None)
                    for col in metric_cols
                },
            }
            for i in range(len(golden))
        ],
    }

    # Salva relatório
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Relatório salvo em: {output_path}")

    return report


def _ragas_version() -> str:
    import ragas

    return getattr(ragas, "__version__", "desconhecida")


def format_report(report: dict) -> str:
    """
    Formata o relatório como texto pronto para colar na monografia:
    tabela de médias por métrica + detalhamento por pergunta.
    """
    lines: list[str] = []
    cfg = report["config"]

    lines.append("=" * 72)
    lines.append("RESULTADOS DA AVALIAÇÃO RAGAS")
    lines.append("=" * 72)
    lines.append(f"Perguntas avaliadas : {report['num_questions']}")
    lines.append(f"Modelo de geração   : {cfg['generation_model']}")
    lines.append(f"Modelo juiz (RAGAS) : {cfg['judge_model']}")
    lines.append(f"Modelo de embedding : {cfg['embedding_model']}")
    lines.append(f"Versão do RAGAS     : {cfg['ragas_version']}")
    lines.append("")
    lines.append(f"{'Métrica':<22} {'Média':>8}")
    lines.append("-" * 32)
    for metric, value in report["metrics"].items():
        lines.append(f"{metric:<22} {value:>8.4f}")

    if report.get("failed_samples"):
        lines.append("")
        lines.append("Amostras com falha de avaliação (NaN, excluídas da média):")
        for metric, count in report["failed_samples"].items():
            lines.append(f"  {metric}: {count}")

    lines.append("")
    lines.append("-" * 72)
    lines.append("DETALHAMENTO POR PERGUNTA")
    lines.append("-" * 72)
    for i, item in enumerate(report["details"], start=1):
        lines.append(f"\n[{i}] {item['question']}")
        scores = "  ".join(
            f"{k}={v if v is not None else 'NaN'}" for k, v in item["scores"].items()
        )
        lines.append(f"    {scores}")

    return "\n".join(lines)
