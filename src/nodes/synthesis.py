"""Node 4 — Synthesis: compile the final answer for the account manager.

Combines the policy documents retrieved by the RAG node with the specialist
assessment from the risk-agent node (when present) into a single, cited response.

After generating the answer, the Q&A pair is appended to recent_messages so that
the summarizer node can periodically consolidate the session history.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from ..clients.llm import get_llm
from ..logging_config import get_logger
from ..state import ConversationMessage, OrchestratorState

logger = get_logger(__name__)

_BASE_SYSTEM_PROMPT = """Você é um assistente sênior do Banco BV que responde a gestores
de conta de forma objetiva, citando a política interna e, quando houver,
o parecer do Agente Especialista de Risco.

Regras:
- Sempre cite numericamente os trechos da política utilizados, no formato [n].
- Se houver parecer do especialista, destaque-o em seção própria.
- Se o RAG não tiver cobertura suficiente, seja explícito sobre isso.
- Use português do Brasil, tom corporativo e direto.
"""

_SESSION_CONTEXT_SECTION = """
Perfil comportamental da sessão (padrões aprendidos de interações anteriores):
{summary}
Adapte o formato e nível de detalhe da resposta ao perfil acima.
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
    assessment = state.risk_assessment_response or "(not requested)"
    logger.info(
        "node_synthesis | docs=%d assessment_chars=%d summary_chars=%d",
        len(state.rag_context),
        len(state.risk_assessment_response or ""),
        len(state.conversation_summary),
    )

    # Inject the procedural memory profile when available.
    system_content = _BASE_SYSTEM_PROMPT
    if state.conversation_summary:
        system_content += _SESSION_CONTEXT_SECTION.format(summary=state.conversation_summary)

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_content), ("user", _USER_PROMPT)]
    )
    chain = prompt | get_llm()

    response = chain.invoke(
        {
            "question": state.question,
            "context": _format_context(state),
            "assessment": assessment,
        }
    )

    text = getattr(response, "content", str(response))

    # Append the current Q&A to episodic memory for future summarization.
    updated_messages = state.recent_messages + [
        ConversationMessage(role="user", content=state.question),
        ConversationMessage(role="assistant", content=text),
    ]

    return {"final_response": text, "recent_messages": updated_messages}
