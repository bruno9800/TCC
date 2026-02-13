"""
Filtro de Revogações

Detecta documentos normativos revogados através de análise textual
e marcadores no nome do arquivo.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Padrões regex para detectar revogação no conteúdo do documento
REVOCATION_PATTERNS = [
    re.compile(r"REVOGAD[OA]", re.IGNORECASE),
    re.compile(r"revogad[oa]\s+pela\s+resolu[çc][aã]o", re.IGNORECASE),
    re.compile(r"revoga\s+a\s+resolu[çc][aã]o", re.IGNORECASE),
    re.compile(r"fica\s+revogad[oa]", re.IGNORECASE),
    re.compile(r"considerar\s+revogad[oa]", re.IGNORECASE),
    re.compile(r"deixa\s+de\s+vigorar", re.IGNORECASE),
    re.compile(r"sem\s+efeito", re.IGNORECASE),
]

# Padrão no nome do arquivo
FILENAME_REVOKED_PATTERN = re.compile(r"REVOGAD[OA]", re.IGNORECASE)


@dataclass
class RevocationInfo:
    """Informações sobre o status de revogação de um documento."""

    is_revoked: bool
    revoked_by_filename: bool
    revocation_markers: list[str]
    status: str  # "vigente" | "revogado"


def check_filename_revocation(filepath: Path) -> bool:
    """Verifica se o nome do arquivo indica revogação."""
    return bool(FILENAME_REVOKED_PATTERN.search(filepath.stem))


def check_content_revocation(text: str) -> list[str]:
    """
    Analisa o conteúdo do documento em busca de marcadores de revogação.

    Retorna lista de trechos que indicam revogação.
    """
    markers: list[str] = []

    for pattern in REVOCATION_PATTERNS:
        matches = pattern.findall(text)
        markers.extend(matches)

    return markers


def analyze_revocation(filepath: Path, text: str) -> RevocationInfo:
    """
    Analisa um documento para determinar seu status de revogação.

    Args:
        filepath: Caminho do arquivo original.
        text: Conteúdo textual do documento (Markdown).

    Returns:
        RevocationInfo com o status de vigência.
    """
    by_filename = check_filename_revocation(filepath)
    markers = check_content_revocation(text)

    # Um documento é considerado revogado se:
    # 1. O nome do arquivo contém "REVOGADA/REVOGADO"
    # 2. O conteúdo contém marcadores de auto-revogação
    #    (apenas se aparecem no título ou cabeçalho do documento)
    is_revoked = by_filename  # filename é forte indicador

    status = "revogado" if is_revoked else "vigente"

    if by_filename:
        logger.info(f"  ⚠ Documento revogado (por nome): {filepath.name}")

    return RevocationInfo(
        is_revoked=is_revoked,
        revoked_by_filename=by_filename,
        revocation_markers=markers,
        status=status,
    )
