# Setup & Execution Guide

Step-by-step instructions to configure, install and run the
**Banco Credit-Policy Orchestrator** from scratch.

---

## Prerequisites

Before you start, confirm you have the following:

| Requirement                               | Version    | Why                                                          |
| ----------------------------------------- | ---------- | ------------------------------------------------------------ |
| Python                                    | ≥ 3.11     | `StrEnum`, `match` statements, and improved type annotations |
| Google Cloud SDK (`gcloud`)               | any recent | Authentication                                               |
| GCP project with Vertex AI enabled        | —          | LLM + Vector Search                                          |
| Vertex AI Vector Search index (populated) | —          | RAG knowledge base                                           |
| Google ADK risk-agent deployed with A2A   | —          | Specialist escalation                                        |

---

## Step 1 — Clone and create a virtual environment

```bash
# Clone the repository (adjust URL as needed)
git clone <repo-url>
cd agente-6-langgraph-adk

# Create an isolated Python environment so project dependencies
# do not conflict with other projects on your machine.
python -m venv .venv

# Activate the environment
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
```

---

## Step 2 — Install dependencies

```bash
# Production dependencies
pip install -r requirements.txt

# Development / test dependencies (only needed for running tests)
pip install -r requirements-dev.txt
```

---

## Step 3 — Configure environment variables

The application reads **all** configuration from environment variables.
The `.env.example` file lists every variable with a description.

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env

# Windows (CMD)
copy .env.example .env
```

Open `.env` in your editor and fill in your values.

### Required variables

| Variable                            | Description                                   | Example                                              |
| ----------------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| `GOOGLE_CLOUD_PROJECT`              | Your GCP project ID                           | `bv-ai-lab`                                          |
| `VECTOR_SEARCH_INDEX_NAME`          | Full resource name of the Vector Search index | `projects/123/locations/us-east1/indexes/456`        |
| `VECTOR_SEARCH_INDEX_ENDPOINT_NAME` | Full resource name of the Index Endpoint      | `projects/123/locations/us-east1/indexEndpoints/789` |
| `VECTOR_SEARCH_DEPLOYED_INDEX_ID`   | ID of the deployed index                      | `deployed_credit_policy`                             |
| `ADK_RISK_AGENT_CARD_URL`           | Full URL to the agent's `agent.json`          | `https://agent.run.app/.well-known/agent.json`       |

### Optional variables (have sensible defaults)

| Variable                         | Default              | Description                                            |
| -------------------------------- | -------------------- | ------------------------------------------------------ |
| `GOOGLE_CLOUD_LOCATION`          | `us-central1`        | Vertex AI region                                       |
| `GOOGLE_APPLICATION_CREDENTIALS` | _(unset)_            | Path to a service-account JSON key                     |
| `GOOGLE_CLOUD_STORAGE_BUCKET`    | _(unset)_            | GCS bucket used to stage Vector Search datapoints      |
| `VERTEX_EMBEDDING_MODEL`         | `text-embedding-004` | Dense retrieval embedding model                        |
| `VERTEX_RAG_TOP_K`               | `5`                  | Number of documents retrieved per query                |
| `VERTEX_LLM_MODEL`               | `gemini-2.5-flash`   | Gemini model used by evaluator and synthesis           |
| `VERTEX_LLM_TEMPERATURE`         | `0.2`                | Sampling temperature (lower = more deterministic)      |
| `ADK_RISK_AGENT_TIMEOUT`         | `30`                 | Seconds before A2A call times out                      |
| `LOG_LEVEL`                      | `INFO`               | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Step 4 — Authenticate with Google Cloud

The application uses **Application Default Credentials (ADC)**. The simplest
way to set them up locally is:

```bash
gcloud auth application-default login
```

This opens a browser for OAuth sign-in and saves a credentials file locally.
No changes to `.env` are needed when using ADC.

**Alternative — service account key file:**
If you prefer a key file (e.g. in CI/CD), download the JSON key from the GCP
Console and set:

```
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

---

## Step 5 — Verify the configuration

Run the following command. It will validate all required variables and print
the active configuration without making any API calls:

```bash
python -c "from src.config import get_settings; s = get_settings(); print('OK', s.google_cloud_project)"
```

If a required variable is missing you will see a `ValidationError` listing
exactly which field is missing.

---

## Step 6 — Run the application

### Single question

```bash
python main.py --question "What is the working-capital credit limit for Corporate clients?"
```

### Single question with execution trace

Shows which nodes were visited and which state fields were updated:

```bash
python main.py -q "Can I approve R$50M restructuring for client X?" --verbose
```

### Interactive mode

```bash
python main.py
```

Type your question at the `Manager>` prompt. Type `exit` to quit.

---

## Step 7 — Run the tests

Tests run entirely offline — no GCP credentials are required.

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run a specific test file
pytest tests/unit/test_router.py -v

# Show test coverage (requires pytest-cov)
pip install pytest-cov
pytest --cov=src --cov-report=term-missing
```

Expected output (all green):

```
tests/unit/test_state.py        ............    12 passed
tests/unit/test_models.py       ..............  14 passed
tests/unit/test_config.py       ...........     11 passed
tests/unit/test_router.py       .....            5 passed
tests/unit/test_nodes.py        .................17 passed
tests/integration/test_graph.py .........        9 passed
```

---

## Troubleshooting

| Symptom                             | Likely cause                                           | Fix                                                  |
| ----------------------------------- | ------------------------------------------------------ | ---------------------------------------------------- |
| `ValidationError` on startup        | A required `.env` variable is missing                  | Run step 5 to see which field is missing             |
| `DefaultCredentialsError`           | GCP credentials not configured                         | Run `gcloud auth application-default login`          |
| `find_neighbors` returns empty list | Wrong `VECTOR_SEARCH_DEPLOYED_INDEX_ID` or empty index | Verify with `gcloud ai index-endpoints describe ...` |
| A2A timeout                         | Risk-agent server is down or slow                      | Raise `ADK_RISK_AGENT_TIMEOUT`; check agent logs     |
| `ModuleNotFoundError: google.adk`   | Missing A2A extras                                     | `pip install 'google-adk[a2a]'`                      |
| `with_structured_output` fails      | Model does not support tool calling                    | Use `gemini-2.5-flash` or `gemini-2.5-pro`           |

---

## Project structure reference

```
agente-6-langgraph-adk/
├── main.py                     CLI entry point
├── requirements.txt            Production dependencies
├── requirements-dev.txt        Test/dev dependencies
├── pyproject.toml              pytest + ruff configuration
├── .env.example                Configuration template
├── docs/
│   ├── SETUP.md                This file
│   └── LAB_GUIADO.md           Guided lab: multi-agent theory + code walkthrough
├── src/
│   ├── config.py               Pydantic-Settings (env vars → typed Settings)
│   ├── constants.py            Node enum, A2A app name
│   ├── logging_config.py       Centralised logging setup
│   ├── models.py               RiskAssessment, A2APayload, PolicyChunk
│   ├── state.py                OrchestratorState, RAGDocument
│   ├── graph.py                StateGraph definition + run() helper
│   ├── clients/
│   │   ├── vector_search.py    Vertex AI Vector Search client
│   │   ├── adk_client.py       Google ADK A2A client
│   │   └── llm.py              ChatVertexAI (Gemini) client
│   └── nodes/
│       ├── rag_retrieval.py    Node 1: embed + search
│       ├── evaluator.py        Node 2: LLM triage decision
│       ├── router.py           Conditional edge function
│       ├── risk_agent.py       Node 3: A2A call to specialist
│       └── synthesis.py        Node 4: compile final response
└── tests/
    ├── conftest.py             Shared fixtures
    ├── unit/
    │   ├── test_state.py
    │   ├── test_models.py
    │   ├── test_config.py
    │   ├── test_router.py
    │   └── test_nodes.py
    └── integration/
        └── test_graph.py
```
