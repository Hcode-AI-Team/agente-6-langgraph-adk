"""Node 4 — Synthesis: compile the final answer for the account manager.

Combines the policy documents retrieved by the RAG node with the specialist
assessment from the risk-agent node (when present) into a single, cited response.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from ..clients.llm import get_llm
from ..logging_config import get_logger
from ..state import OrchestratorState

logger = get_logger(__name__)

# The synthesis prompt explicitly instructs the model to cite policy chunks by
# index number and to highlight any specialist assessment in its own section.
# This makes the final answer auditable — reviewers can trace every claim back
# to a specific policy excerpt or the risk-agent's output.
_SYSTEM_PROMPT = """Você é um assistente sênior do Banco BV que responde a gestores
de conta de forma objetiva, citando a política interna e, quando houver,
o parecer do Agente Especialista de Risco.

Regras:
- Sempre cite numericamente os trechos da política utilizados, no formato [n].
- Se houver parecer do especialista, destaque-o em seção própria.
- Se o RAG não tiver cobertura suficiente, seja explícito sobre isso.
- Use português do Brasil, tom corporativo e direto.
"""

_USER_PROMPT = """Pergunta do gestor:
{question}

Política interna (RAG):
{context}

Parecer do Agente de Risco (ADK A2A):
{assessment}

Elabore a resposta final ao gestor."""


def _format_context(state: OrchestratorState) -> str:
    """Render RAG documents as a numbered list for the synthesis prompt."""
    if not state.rag_context:
        return "(no documents retrieved)"
    return "\n".join(
        f"[{i}] {doc.text}" for i, doc in enumerate(state.rag_context, start=1)
    )


def node_synthesis(state: OrchestratorState) -> dict:
    """Generate the final response by combining RAG context and risk assessment."""
    # When the evaluator decided RAG was sufficient, this will be None.
    assessment = state.risk_assessment_response or "(not requested)"
    logger.info(
        "node_synthesis | docs=%d assessment_chars=%d",
        len(state.rag_context),
        len(state.risk_assessment_response or ""),
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("user", _USER_PROMPT)]
    )
    chain = prompt | get_llm()

    response = chain.invoke(
        {
            "question": state.question,
            "context": _format_context(state),
            "assessment": assessment,
        }
    )

    # AIMessage.content is a string; fallback handles unexpected response types.
    text = getattr(response, "content", str(response))
    return {"final_response": text}
