# Lab 1 — Orquestrador de Agentes: LangGraph + Vertex AI RAG + Google ADK (A2A)

**Contexto (Banco BV):** fundação de um sistema de atendimento corporativo que ajuda
gestores de conta a resolver dúvidas complexas combinando **políticas internas**
(RAG no Vertex AI Vector Search) com **delegação A2A** para agentes
especialistas (Agente de Risco Financeiro no Google ADK).

## Documentação e materiais de apoio

### Guias (Markdown)

| Documento | Descrição |
|---|---|
| [**`docs/LAB_GUIADO.md`**](docs/LAB_GUIADO.md) | Lab guiado completo (~90 min): multi-agentes, Google ADK e protocolo A2A em profundidade. |
| [**`docs/SETUP.md`**](docs/SETUP.md) | Passo a passo de instalação, configuração do `.env`, autenticação GCP e primeira execução. |
| [**`docs/LANGGRAPH_NODES_EDGES.md`**](docs/LANGGRAPH_NODES_EDGES.md) | Aula didática (~1h) sobre `add_node`, `add_edge` e `add_conditional_edges` usando o código real deste projeto. |

### Dossiês interativos (HTML, abra no navegador)

| Arquivo | Descrição |
|---|---|
| [**`index.html`**](index.html) | SPA interativa "Engenharia Agêntica com LangGraph" — topologias, estado/persistência, Human-in-the-Loop e Time Travel, com gráficos comparativos (Tailwind + Chart.js). |
| [**`claude_code.html`**](claude_code.html) | Dossiê interativo "O Domínio do Claude Code (2026)" — comparativo de mercado, instalação e uso em workflows agênticos. |

---

## Arquitetura

```mermaid
flowchart LR
    Start([Pergunta do gestor]) --> RAG[rag_retrieval]
    RAG --> Avaliador[evaluator<br/>LLM decide]
    Avaliador -->|requires_risk_assessment=true| ADK[risk_agent<br/>A2A RemoteA2aAgent]
    Avaliador -->|requires_risk_assessment=false| Sintese[synthesis]
    ADK --> Sintese
    Sintese --> End([Resposta final])
```

### Fluxo

| Nó | Responsabilidade |
|---|---|
| `rag_retrieval` | Embeda a pergunta (`text-embedding-004`) e consulta o `MatchingEngineIndexEndpoint`, devolvendo os top-K trechos da política interna. |
| `evaluator` | Gemini com **saída estruturada Pydantic** decide se o RAG é suficiente ou se a operação exige análise de risco especializada. |
| `risk_agent` *(condicional)* | Orquestra uma chamada **A2A** ao Agente de Risco (ADK) via `RemoteA2aAgent`, enviando um `PayloadA2A` Pydantic com pergunta, trechos e metadados de sessão. |
| `synthesis` | Gemini compila a resposta final, citando a política interna `[n]` e, se houver, o parecer do especialista. |

---

## Destaques técnicos

- **Pydantic em toda a borda**: `OrchestratorState`, `DocumentoRAG`, `AvaliacaoRisco`, `PayloadA2A` — tipagem forte de ponta a ponta.
- **`pydantic-settings`** para configuração validada a partir do `.env` (URLs, ranges, enums).
- **Logging estruturado** por módulo (`src/logging_config.py`), nível configurável via `LOG_LEVEL`.
- **Enum `Node`** elimina strings mágicas no grafo.
- **Graceful degradation**: falha na chamada A2A vira mensagem de indisponibilidade na síntese, sem quebrar o fluxo.
- **Streaming** com `--verbose`: trilha de execução por nó.

---

## Estrutura do projeto

```
agente-6-langgraph-adk/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml              # metadados + configuração de pytest e ruff
├── requirements.txt            # dependências de runtime
├── requirements-dev.txt        # dependências de teste/lint
├── main.py                     # CLI (Rich) — entrada única
├── index.html                  # SPA "Engenharia Agêntica com LangGraph"
├── claude_code.html            # Dossiê interativo "Claude Code (2026)"
├── docs/
│   ├── LAB_GUIADO.md           # material didático (multi-agentes, ADK, A2A)
│   ├── SETUP.md                # guia de instalação e execução passo a passo
│   └── LANGGRAPH_NODES_EDGES.md # aula sobre add_node / add_edge / conditional
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic-Settings
│   ├── constants.py            # Enum Node (rag_retrieval, evaluator, ...)
│   ├── logging_config.py       # logging estruturado
│   ├── state.py                # OrchestratorState (BaseModel)
│   ├── models.py               # RiskAssessment, PayloadA2A, SessaoA2A
│   ├── graph.py                # StateGraph + conditional edges
│   ├── clients/
│   │   ├── vector_search.py    # Vector Search + embeddings
│   │   ├── adk_client.py       # RemoteA2aAgent (Agent Client)
│   │   └── llm.py              # ChatVertexAI (Gemini)
│   └── nodes/
│       ├── rag_retrieval.py
│       ├── evaluator.py
│       ├── router.py
│       ├── risk_agent.py
│       └── synthesis.py
└── tests/
    ├── unit/                   # test_config, test_models, test_nodes, test_router, test_state
    └── integration/            # test_graph (topologia do StateGraph)
```

---

## Pré-requisitos

1. **Python 3.11+**
2. **Credenciais GCP** com permissões de Vertex AI (Vector Search + Gemini).
3. **Vertex AI Vector Search** já provisionado: `Index ID`, `Index Endpoint ID`, `Deployed Index ID`.
4. **Agente de Risco** publicado no **Google ADK** com A2A habilitado (expõe `/.well-known/agent.json`).

---

## Instalação e configuração

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # edite com seus IDs e URLs
gcloud auth application-default login
```

---

## Execução

```bash
# Chamada única
python main.py --question "Posso aprovar R$ 10M de capital de giro para cliente Corporate com rating B?"

# Com trilha de execução (delta por nó)
python main.py -q "Como calcular provisão IFRS 9?" --verbose

# Modo interativo
python main.py
```

Se alguma variável obrigatória do `.env` faltar, o CLI imprime um
`ValidationError` do Pydantic indicando exatamente o campo faltante e
retorna com código 2.

> Guia detalhado de instalação e troubleshooting em [`docs/SETUP.md`](docs/SETUP.md).

---

## Testes

```bash
pip install -r requirements-dev.txt

# Toda a suíte (unit + integration)
pytest

# Apenas testes unitários (não dependem de GCP)
pytest tests/unit

# Teste de topologia do grafo
pytest tests/integration/test_graph.py
```

Configuração em [`pyproject.toml`](pyproject.toml) (pytest + ruff).

---

## O Protocolo A2A (Agent-to-Agent)

No paradigma **A2A**, o Orquestrador (LangGraph) atua como **Agent
Client** e invoca um **Agent Server** remoto (o Agente de Risco do ADK)
via o _Agent Card_ publicado em `/.well-known/agent.json`.

Neste projeto a integração é feita pelo `RemoteA2aAgent` do
`google-adk[a2a]` (ver [`src/clients/adk_client.py`](src/clients/adk_client.py)).
O payload enviado é um `PayloadA2A` Pydantic serializado em JSON:

```json
{
  "intent": "avaliar_risco_credito",
  "pergunta": "...",
  "trechos_politica": [
    {"id": "...", "distance": 0.12, "texto": "...", "metadata": {}}
  ],
  "sessao": {
    "orquestrador": "bv_langgraph",
    "trace_id": "...",
    "justificativa_triagem": "..."
  }
}
```

---

## Variáveis de ambiente

| Variável | Descrição |
|---|---|
| `GOOGLE_CLOUD_PROJECT` | ID do projeto GCP. |
| `GOOGLE_CLOUD_LOCATION` | Região (ex.: `us-central1`). |
| `GOOGLE_APPLICATION_CREDENTIALS` | Caminho opcional para JSON da service account. |
| `VECTOR_SEARCH_INDEX_NAME` | Resource name completo do índice (`projects/<num>/locations/<loc>/indexes/<id>`). |
| `VECTOR_SEARCH_INDEX_ENDPOINT_NAME` | Resource name completo do Index Endpoint (`projects/<num>/locations/<loc>/indexEndpoints/<id>`). |
| `VECTOR_SEARCH_DEPLOYED_INDEX_ID` | ID do deployed index. |
| `GOOGLE_CLOUD_STORAGE_BUCKET` | (Opcional) Bucket GCS de staging para datapoints/RAG. |
| `VERTEX_EMBEDDING_MODEL` | Modelo de embedding (default: `text-embedding-004`). |
| `VERTEX_RAG_TOP_K` | Nº de vizinhos (default: `5`). |
| `VERTEX_LLM_MODEL` | Modelo Gemini (default: `gemini-2.5-flash`). |
| `VERTEX_LLM_TEMPERATURE` | Temperatura do LLM (default: `0.2`). |
| `ADK_RISK_AGENT_CARD_URL` | URL do agent card do Agente de Risco. |
| `ADK_RISK_AGENT_TIMEOUT` | Timeout (segundos) da chamada A2A. |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`. |

---

## Próximos passos (Labs seguintes)

- Publicar o **Agente de Risco** no ADK e expô-lo via A2A (`to_a2a()`).
- Adicionar **persistência** (Checkpointer do LangGraph) para conversas longas.
- Avaliação (traces e evals) do orquestrador.
- Novos especialistas A2A (Compliance, Jurídico, Cadastro) em paralelo.

Consulte o [**Lab Guiado**](docs/LAB_GUIADO.md) para exercícios de
aprofundamento e extensões.
