#!/usr/bin/env python3
"""
Script para executar a avaliação RAGAS.

Uso:
    python scripts/run_eval.py
    RAGAS_JUDGE_MODEL=gpt-4o python scripts/run_eval.py   # juiz diferente da geração

Executa o pipeline completo (busca + rerank + geração) para cada pergunta
do golden dataset, calcula métricas RAGAS e imprime uma tabela de resultados
pronta para a monografia. O relatório completo (JSON, com scores por
pergunta) é salvo em data/ragas_report_<timestamp>.json.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.evaluation.ragas_eval import format_report, run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("AVALIAÇÃO RAGAS — Sistema RAG UNIVASF")
    logger.info("=" * 60)

    # Define caminho do relatório com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PROJECT_ROOT / "data" / f"ragas_report_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = run_evaluation(output_path=output_path)

    # Tabela final em stdout — pronta para copiar para o TCC
    print("\n" + format_report(report))
    print(f"\nRelatório completo salvo em: {output_path}")


if __name__ == "__main__":
    main()
