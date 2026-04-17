"""Node 2 — Evaluator: decide whether to escalate to the specialist risk agent.

The evaluator uses the LLM with `with_structured_output` to produce a typed
`RiskAssessment` decision.  This guarantees that the conditional edge in
graph.py always reads a validated Python object — not a raw string that might
fail to parse.

The system prompt encodes the bank's triage policy.  When credit-risk criteria
change, only this file needs updating.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from ..clients.llm import get_llm
from ..logging_config import get_logger
from ..models import RiskAssessment
from ..state import OrchestratorState

logger = get_logger(__name__)

# System-level instructions that define the evaluator's decision criteria.
# Written in Portuguese because the LLM will reason about Portuguese-language
# policy documents and must produce Portuguese-language rationales.
_SYSTEM_PROMPT = """Você é um avaliador de triagem do Banco BV.

Sua tarefa é decidir se a dúvida do gestor de conta pode ser respondida
apenas com a política interna recuperada (RAG) ou se exige a delegação
para o Agente Especialista de Risco Financeiro.

Exija análise de risco quando:
- A operação envolver crédito acima de limites padrão ou reestruturação.
- Houver exposição concentrada, garantias atípicas ou cliente em workout.
- A política interna (RAG) não cobrir os aspectos de risco da pergunta.
- Houver menção a covenants, rating interno ou provisão (IFRS 9 / Res. 4.966).

Caso contrário, o RAG é suficiente.
"""

_USER_PROMPT = """Pergunta do gestor:
{question}

Trechos recuperados da base de conhecimento (RAG):
{context}

Decida e justifique."""


def _format_context(state: OrchestratorState) -> str:
    """Render the RAG documents into a numbered list for the prompt.

    Each line shows the retrieval score and a truncated preview so the LLM
    can gauge coverage without the full text inflating the prompt length.
    """
    if not state.rag_context:
        return "(no documents retrieved)"
    lines = [
        f"[{i}] (score={doc.distance:.4f}) {doc.preview(500)}"
        for i, doc in enumerate(state.rag_context, start=1)
    ]
    return "\n".join(lines)


def node_evaluator(state: OrchestratorState) -> dict:
    """Classify whether the query requires escalation to the risk-agent node.

    Returns a partial state dict with `requires_risk_assessment` and
    `evaluator_rationale` populated.
    """
    logger.info("node_evaluator | %d docs in context", len(state.rag_context))

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("user", _USER_PROMPT)]
    )
    # with_structured_output forces the LLM to return JSON that validates
    # against RiskAssessment's schema, eliminating free-form parse errors.
    chain = prompt | get_llm().with_structured_output(RiskAssessment)

    result: RiskAssessment = chain.invoke(
        {
            "question": state.question,
            "context": _format_context(state),
        }
    )

    logger.info(
        "node_evaluator | requires_escalation=%s rationale=%s",
        result.requires_escalation,
        result.rationale,
    )
    return {
        "requires_risk_assessment": result.requires_escalation,
        "evaluator_rationale": result.rationale,
    }
