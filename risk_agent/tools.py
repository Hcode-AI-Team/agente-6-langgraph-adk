"""Tools available to the Risk Specialist Agent.

Both tools share the same Vertex AI Vector Search index as the orchestrator's
RAG node — ensuring the specialist reasons over the same knowledge base.

Tool design guidelines (Google ADK):
- Plain Python functions with full type hints are auto-wrapped as FunctionTool.
- The docstring becomes the tool description shown to the LLM; first line is the
  summary, Args section describes parameters, Returns describes the output.
- Keep return values as plain strings so the LLM can reason over them directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable when this module is loaded by the ADK server
# (working directory may differ from the project root).
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.clients.vector_search import VectorSearchError, search_policies  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def search_credit_policies(query: str, top_k: int = 5) -> str:
    """Pesquisa na base de conhecimento de políticas de crédito do Banco BV.

    Realiza busca semântica no índice Vertex AI Vector Search compartilhado
    com o orquestrador principal. Utilize para obter trechos de política
    relevantes para a análise de risco em curso.

    Args:
        query: Consulta em linguagem natural para busca semântica nas políticas.
               Exemplos: "limites de crédito para reestruturação de dívida",
               "provisão IFRS 9 estágio 3", "covenants financeiros middle market".
        top_k: Número de trechos a retornar (padrão 5, máximo 20).

    Returns:
        Trechos de política indexados por relevância, ou mensagem de erro.
    """
    top_k = max(1, min(top_k, 20))
    logger.info("tool:search_credit_policies | query=%r top_k=%d", query, top_k)

    try:
        docs = search_policies(query, top_k=top_k)
    except VectorSearchError as exc:
        logger.error("tool:search_credit_policies | VectorSearchError: %s", exc)
        return f"[Erro na busca vetorial: {exc}]"

    if not docs:
        return "(Nenhum trecho encontrado para esta consulta — tente termos mais específicos.)"

    sections = []
    for i, doc in enumerate(docs, 1):
        relevance = max(0.0, 1.0 - doc.distance)
        sections.append(
            f"[Trecho {i}] Relevância: {relevance:.2%}\n{doc.text.strip()}"
        )

    logger.info("tool:search_credit_policies | returned %d docs", len(docs))
    return "\n\n---\n\n".join(sections)


def check_regulatory_thresholds(
    operation_type: str,
    amount_brl: float,
    client_segment: str = "middle_market",
) -> str:
    """Verifica limites regulatórios e internos aplicáveis a uma operação de crédito.

    Consulta os thresholds do Banco BV para alçadas de aprovação, provisão
    mínima (IFRS 9 / Res. BCB 4.966) e requisitos de rating interno.

    Args:
        operation_type: Tipo de operação. Valores: "capital_de_giro", "financiamento",
                        "reestruturacao", "limite_rotativo", "garantia", "outros".
        amount_brl: Valor da operação em Reais (ex: 10_000_000.0 para R$10M).
        client_segment: Segmento do cliente. Valores: "varejo", "pme",
                        "middle_market", "corporate", "large_corporate".

    Returns:
        Resumo dos limites regulatórios, alçada de aprovação exigida e
        requisitos de provisão mínima aplicáveis.
    """
    logger.info(
        "tool:check_regulatory_thresholds | op=%s amount=%.0f segment=%s",
        operation_type,
        amount_brl,
        client_segment,
    )

    # Approval authority tiers (simplified BV policy simulation).
    # In production this would query an internal API or database.
    tiers = [
        (500_000,    "Gerente de Agência"),
        (2_000_000,  "Superintendente Regional"),
        (10_000_000, "Comitê de Crédito Local"),
        (50_000_000, "Comitê de Crédito Central"),
        (float("inf"), "Diretoria de Crédito + Conselho"),
    ]
    authority = next(label for limit, label in tiers if amount_brl <= limit)

    # IFRS 9 / Res. 4.966 minimum provisioning by segment.
    provision_rates = {
        "varejo":         {"stage1": "0.5%", "stage2": "5%",  "stage3": "35%"},
        "pme":            {"stage1": "0.8%", "stage2": "8%",  "stage3": "40%"},
        "middle_market":  {"stage1": "1.0%", "stage2": "10%", "stage3": "45%"},
        "corporate":      {"stage1": "0.5%", "stage2": "7%",  "stage3": "40%"},
        "large_corporate":{"stage1": "0.3%", "stage2": "5%",  "stage3": "35%"},
    }
    rates = provision_rates.get(client_segment, provision_rates["middle_market"])

    # Additional requirements for restructuring operations.
    extra = ""
    if operation_type == "reestruturacao":
        extra = (
            "\n\nREESTRUTURAÇÃO: Classificação mínima em Estágio 2 (IFRS 9) na data "
            "da operação. Exige avaliação de viabilidade econômico-financeira e "
            "aprovação do Comitê de Reestruturação independentemente do valor."
        )

    result = (
        f"OPERAÇÃO: {operation_type.replace('_', ' ').title()}\n"
        f"VALOR: R$ {amount_brl:,.2f}\n"
        f"SEGMENTO: {client_segment.replace('_', ' ').title()}\n\n"
        f"ALÇADA DE APROVAÇÃO REQUERIDA: {authority}\n\n"
        f"PROVISÃO MÍNIMA (IFRS 9 / Res. BCB 4.966):\n"
        f"  Estágio 1 (adimplente):      {rates['stage1']}\n"
        f"  Estágio 2 (deteriorado):     {rates['stage2']}\n"
        f"  Estágio 3 (inadimplente):    {rates['stage3']}"
        f"{extra}"
    )

    logger.info("tool:check_regulatory_thresholds | authority=%s", authority)
    return result
