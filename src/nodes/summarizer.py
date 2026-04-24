"""Node 5 — Summarizer: consolidate episodic memory into procedural behavioral patterns.

Runs unconditionally after synthesis but exits immediately when the conversation
is too short to justify summarization.

Design — two-tier memory (best practice from MemGPT / LangGraph Memory patterns):

  Tier 1 — Episodic (recent_messages): verbatim Q&A pairs from the last N turns.
            Gives the LLM precise context for the current session.

  Tier 2 — Procedural (conversation_summary): distilled behavioral profile built
            by compressing older episodic memories.  Contains NOT a log of events
            but extracted PATTERNS:
              - Which question types consistently required escalation to the risk agent
              - Regulatory frameworks most referenced (IFRS 9, Res. 4.966, etc.)
              - Preferred answer format and level of detail
              - Client / sector context for the account manager
            These patterns are injected as "soft inference instructions" into the
            evaluator and synthesis prompts on future turns, transferring learned
            behavior without re-reading the full raw history.

Consolidation trigger: when len(recent_messages) >= summary_message_threshold,
older messages are distilled into conversation_summary and the list is pruned to
keep only the `summary_messages_to_keep` most recent entries.
"""

from __future__ import annotations

from ..clients.llm import get_llm
from ..config import get_settings
from ..logging_config import get_logger
from ..state import ConversationMessage, OrchestratorState

logger = get_logger(__name__)

_CONSOLIDATION_PROMPT = """Você é o módulo de memória do BV Credit-Policy Orchestrator.

Sua função é consolidar o histórico de conversas de um gestor de contas em um
**perfil comportamental compacto** que será usado como instrução de contexto nas
próximas interações — não como um log, mas como padrões aprendidos.

Extraia e preserve:
1. **Padrões de escalação**: quais tipos de operação/pergunta consistentemente
   foram roteados para o Agente Especialista de Risco (vs. resolvidos só com RAG).
2. **Frameworks regulatórios recorrentes**: menções a IFRS 9, Res. BCB 4.966,
   covenants, ratings internos, provisões, Basel III, etc.
3. **Perfil do gestor**: clientes atendidos, setores (agro, PME, middle market),
   faixas de crédito típicas, nível de detalhe preferido nas respostas.
4. **Preferências de formato**: o gestor prefere respostas com citações numeradas?
   Seções separadas? Resumos executivos? Recomendações explícitas?
5. **Lacunas identificadas**: temas onde o RAG teve cobertura insuficiente,
   indicando que futuras buscas devem usar termos mais específicos.

{previous_summary_section}

Histórico recente a consolidar:
{messages_text}

Gere o perfil comportamental atualizado (máximo 400 palavras, português corporativo):"""


def _format_messages(messages: list[ConversationMessage]) -> str:
    lines = []
    for i, msg in enumerate(messages, 1):
        prefix = "GESTOR" if msg.role == "user" else "ASSISTENTE"
        lines.append(f"[{i}] {prefix}: {msg.content[:600]}{'...' if len(msg.content) > 600 else ''}")
    return "\n".join(lines)


def node_summarizer(state: OrchestratorState) -> dict:
    """Consolidate episodic memory into procedural patterns when threshold is reached."""
    settings = get_settings()

    if len(state.recent_messages) < settings.summary_message_threshold:
        return {}

    logger.info(
        "node_summarizer | consolidating %d messages (threshold=%d)",
        len(state.recent_messages),
        settings.summary_message_threshold,
    )

    previous_section = (
        f"Perfil comportamental anterior (a ser atualizado e enriquecido):\n{state.conversation_summary}"
        if state.conversation_summary
        else "(primeira consolidação — não há perfil anterior)"
    )

    prompt = _CONSOLIDATION_PROMPT.format(
        previous_summary_section=previous_section,
        messages_text=_format_messages(state.recent_messages),
    )

    response = get_llm().invoke(prompt)
    new_summary = getattr(response, "content", str(response)).strip()

    keep = settings.summary_messages_to_keep
    pruned = state.recent_messages[-keep:] if keep > 0 else []

    logger.info(
        "node_summarizer | summary updated (%d chars), kept %d recent messages",
        len(new_summary),
        len(pruned),
    )

    return {
        "conversation_summary": new_summary,
        "recent_messages": pruned,
    }
