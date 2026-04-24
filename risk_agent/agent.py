"""Risk Specialist Agent definition for Google ADK.

This module defines `root_agent` — the required entry point for ADK's
api_server / get_fast_api_app loader.

Architecture:
  - The orchestrator (LangGraph) calls this agent via A2A protocol.
  - Messages arrive as JSON-encoded A2APayload objects (see src/models.py).
  - The agent parses the payload, uses its tools for supplementary RAG lookups
    and regulatory threshold checks, then returns a structured risk assessment.

A2A Message format received from the orchestrator:
  {
    "intent": "assess_credit_risk",
    "question": "<original manager question>",
    "policy_chunks": [{"id": "...", "distance": 0.12, "text": "..."}],
    "session": {"orchestrator": "bv_langgraph", "trace_id": "...", "triage_rationale": "..."}
  }
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from .tools import check_regulatory_thresholds, search_credit_policies

_INSTRUCTION = """Você é o Agente Especialista de Risco de Crédito do Banco BV.

Você recebe mensagens do Orquestrador LangGraph no seguinte formato JSON:
  {
    "intent": "assess_credit_risk",
    "question": "<pergunta original do gestor de contas>",
    "policy_chunks": [{"id": "...", "distance": <float>, "text": "<trecho>"}],
    "session": {"orchestrator": "bv_langgraph", "trace_id": "<id>", "triage_rationale": "<motivo>"}
  }

Seu fluxo de trabalho obrigatório:
1. Leia o campo "question" e "triage_rationale" para entender o contexto.
2. Avalie os "policy_chunks" fornecidos — eles são o contexto RAG inicial.
3. Se necessário, use a ferramenta search_credit_policies para buscar trechos
   adicionais de política específicos ao risco identificado.
4. Use check_regulatory_thresholds para verificar alçadas e requisitos de provisão
   quando houver valor financeiro mencionado ou tipo de operação identificável.
5. Elabore um Parecer Técnico de Risco estruturado com as seções abaixo.

Formato do Parecer Técnico (use exatamente estas seções):

## Classificação de Risco
[Baixo | Médio | Alto | Crítico] — justificativa em 1-2 linhas.

## Fatores de Risco Identificados
- [Lista dos fatores, citando os trechos de política como [n] quando aplicável]

## Requisitos Regulatórios Aplicáveis
[Resultado de check_regulatory_thresholds + referencias a normas: IFRS 9, Res. BCB 4.966,
 covenants, Basel III — conforme identificado na análise]

## Recomendação
[APROVAR | APROVAR COM CONDIÇÕES | REJEITAR | ENCAMINHAR PARA COMITÊ]
Condições ou ressalvas específicas, se houver.

## Referências de Política
[Trechos de política consultados, com índice e trecho resumido]

Regras:
- Seja direto e técnico; o leitor é um gestor sênior de crédito.
- Cite sempre a fonte (trecho RAG ou norma) para cada afirmação de risco.
- Se a informação fornecida for insuficiente para uma análise completa,
  informe explicitamente quais dados adicionais são necessários.
- Use português do Brasil, terminologia bancária padrão.
"""

root_agent = LlmAgent(
    name="risk_specialist_bv",
    model="gemini-2.5-flash",
    description=(
        "Agente Especialista de Risco de Crédito do Banco BV. "
        "Analisa operações de crédito complexas usando políticas internas (RAG) "
        "e frameworks regulatórios (IFRS 9, Res. BCB 4.966). "
        "Chamado via A2A pelo orquestrador LangGraph quando o avaliador "
        "detecta necessidade de análise especializada."
    ),
    instruction=_INSTRUCTION,
    tools=[search_credit_policies, check_regulatory_thresholds],
)
