"""Node 2 — Evaluator: decide whether to escalate to the specialist risk agent.

The evaluator uses the LLM with `with_structured_output` to produce a typed
`RiskAssessment` decision.  This guarantees that the conditional edge in
graph.py always reads a validated Python object — not a raw string that might
fail to parse.

When a conversation_summary exists, it is injected into the system prompt as
a behavioral hint: the evaluator learns, over multiple turns, which question
types from THIS account manager consistently require escalation.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from ..clients.llm import get_llm
from ..logging_config import get_logger
from ..models import RiskAssessment
from ..state import OrchestratorState

logger = get_logger(__name__)

_BASE_SYSTEM_PROMPT = """Você é um avaliador de triagem do Banco BV.

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

_SESSION_HINT = """
Perfil comportamental da sessão (padrões aprendidos de interações anteriores):
{summary}
Use este perfil para calibrar o limiar de escalação com base no histórico do gestor.
"""

_USER_PROMPT = """Pergunta do gestor:
{question}

Trechos recuperados da base de conhecimento (RAG):
{context}

Decida e justifique."""


def _format_context(state: OrchestratorState) -> str:
    """Render the RAG documents into a numbered list for the prompt."""
    if not state.rag_context:
        return "(no documents retrieved)"
    lines = [
        f"[{i}] (score={doc.distance:.4f}) {doc.preview(500)}"
        for i, doc in enumerate(state.rag_context, start=1)
    ]
    return "\n".join(lines)


def node_evaluator(state: OrchestratorState) -> dict:
    """Classify whether the query requires escalation to the risk-agent node."""
    logger.info(
        "node_evaluator | %d docs in context | summary=%s",
        len(state.rag_context),
        "yes" if state.conversation_summary else "none",
    )

    system_content = _BASE_SYSTEM_PROMPT
    if state.conversation_summary:
        system_content += _SESSION_HINT.format(summary=state.conversation_summary)

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_content), ("user", _USER_PROMPT)]
    )
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
