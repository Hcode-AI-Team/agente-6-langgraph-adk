# LangGraph na prática — `add_node`, `add_edge` e `add_conditional_edges`

> **Aula didática (~1 hora)** usando o código real deste projeto: um
> orquestrador de atendimento do Banco BV que combina **RAG** (Vertex AI
> Vector Search) com **delegação A2A** (Google ADK) para um Agente de
> Risco Financeiro.
>
> Pré-requisitos: Python 3.11+ e leitura rápida do [README](../README.md)
> para entender o problema de negócio. Esta aula foca 100% em **como o
> grafo é montado e por quê**.

---

## Sumário

1. [Revisão: o que é um `StateGraph`](#1-revisão-o-que-é-um-stategraph)
2. [O estado compartilhado (`OrchestratorState`)](#2-o-estado-compartilhado-orchestratorstate)
3. [`add_node` — a unidade de trabalho do grafo](#3-add_node--a-unidade-de-trabalho-do-grafo)
4. [Dentro de cada nó — o que cada um faz](#4-dentro-de-cada-nó--o-que-cada-um-faz)
    - 4.1 [`node_rag_retrieval`](#41-node_rag_retrieval--busca-semântica-na-política-interna)
    - 4.2 [`node_evaluator`](#42-node_evaluator--triagem-com-saída-estruturada)
    - 4.3 [`node_risk_agent`](#43-node_risk_agent--delegação-a2a-para-o-especialista)
    - 4.4 [`node_synthesis`](#44-node_synthesis--redação-da-resposta-final)
5. [`add_edge` — conectando nós de forma determinística](#5-add_edge--conectando-nós-de-forma-determinística)
6. [`set_entry_point` e o nó terminal `END`](#6-set_entry_point-e-o-nó-terminal-end)
7. [`add_conditional_edges` — bifurcação baseada no estado](#7-add_conditional_edges--bifurcação-baseada-no-estado)
8. [Execução passo a passo — dois cenários reais](#8-execução-passo-a-passo--dois-cenários-reais)
9. [Erros comuns e como o projeto os evita](#9-erros-comuns-e-como-o-projeto-os-evita)
10. [Exercícios para fixar](#10-exercícios-para-fixar)

---

## 1. Revisão: o que é um `StateGraph`

O LangGraph é uma biblioteca para construir **máquinas de estado
orientadas a LLMs**. Diferente de uma `Chain` linear (`A → B → C`), um
`StateGraph` permite:

- **Bifurcações condicionais** (ex.: "se o RAG bastar, pular o
  especialista").
- **Ciclos controlados** (ex.: um agente que reflete e tenta de novo).
- **Estado compartilhado explícito** entre todos os nós.

A "receita" sempre é a mesma:

1. Definir um **schema de estado** (no nosso projeto,
   `OrchestratorState`, um `BaseModel` do Pydantic).
2. Criar um `StateGraph(schema)` vazio.
3. Registrar os **nós** com `add_node(nome, função)`.
4. Conectar os nós com **arestas**: `add_edge` (fixas) ou
   `add_conditional_edges` (dinâmicas).
5. Definir a **porta de entrada** (`set_entry_point`) e pelo menos uma
   **porta de saída** (aresta para `END`).
6. **Compilar** (`compile()`) para obter um app executável.

No nosso projeto, tudo isso acontece em um único arquivo enxuto
(`src/graph.py`). Vamos dissecá-lo.

---

## 2. O estado compartilhado (`OrchestratorState`)

Antes de falar dos nós, precisamos entender **o objeto que trafega
entre eles**. Todo nó recebe o estado atual e devolve um **dicionário
parcial** que o LangGraph mescla nesse estado antes de invocar o próximo
nó.

```34:60:src/state.py
class OrchestratorState(BaseModel):
    """Shared state that flows through every node in the LangGraph StateGraph.

    Each node receives the current state and returns a *partial* dict;
    LangGraph merges that dict into the state before calling the next node.
    Fields start with safe defaults so nodes that are skipped (e.g. the
    risk-agent node on simple queries) do not leave the state broken.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The account manager's original question — set once at entry.
    question: str

    # Documents retrieved by the RAG node.
    rag_context: list[RAGDocument] = Field(default_factory=list)

    # Set by the evaluator node; drives the conditional edge.
    requires_risk_assessment: bool = False
    evaluator_rationale: str = ""

    # Populated only when requires_risk_assessment is True and the A2A
    # call to the risk-agent node succeeds.
    risk_assessment_response: Optional[str] = None

    # Final answer compiled by the synthesis node.
    final_response: str = ""
```

Observações didáticas:

- **Todos os campos têm default** (exceto `question`). Isso é
  intencional: se o nó do especialista for pulado,
  `risk_assessment_response` continua `None` em vez de "quebrar" a
  síntese.
- **Cada nó só modifica os campos de sua responsabilidade**. O
  evaluator escreve em `requires_risk_assessment` e
  `evaluator_rationale`; o synthesis escreve em `final_response`; e
  assim por diante. Isso vira documentação viva do fluxo.
- O estado **não é um `dict` amorfo**. Usar Pydantic dá validação em
  tempo de execução e autocomplete no IDE.

Mentalize essa imagem: o estado é uma "prancheta" que passa de mão em
mão, e cada nó anota nela o resultado do seu trabalho.

---

## 3. `add_node` — a unidade de trabalho do grafo

Um **nó** no LangGraph é simplesmente uma função com a assinatura:

```python
def meu_no(state: OrchestratorState) -> dict:
    ...
    return {"campo_a_atualizar": valor}
```

Três regras de ouro:

1. **Entrada = estado completo** (tipado pelo schema).
2. **Saída = dicionário parcial** com os campos a atualizar. Não é
   preciso devolver o estado inteiro — o LangGraph faz o _merge_.
3. **Nome único** no grafo. Usamos o `enum Node` para garantir isso:

```8:19:src/constants.py
class Node(StrEnum):
    """Canonical names for every node in the StateGraph.

    Using an enum avoids hard-coded strings scattered across graph.py,
    router.py, and tests — a typo will raise AttributeError instead of
    silently routing to a nonexistent node.
    """

    RAG_RETRIEVAL = "rag_retrieval"
    EVALUATOR = "evaluator"
    RISK_AGENT = "risk_agent"
    SYNTHESIS = "synthesis"
```

> **Por que um enum e não strings soltas?** Strings soltas permitem
> erros silenciosos: um `"rag_retrieveal"` (sic) seria roteado para um
> nó inexistente e você só descobriria em runtime. O enum faz o Python
> gritar na importação.

No nosso `src/graph.py`, os quatro `add_node` aparecem assim:

```42:47:src/graph.py
    g = StateGraph(OrchestratorState)

    g.add_node(Node.RAG_RETRIEVAL, node_rag_retrieval)
    g.add_node(Node.EVALUATOR, node_evaluator)
    g.add_node(Node.RISK_AGENT, node_risk_agent)
    g.add_node(Node.SYNTHESIS, node_synthesis)
```

Cada chamada registra, no grafo `g`:

- o **nome** (chave interna do grafo);
- a **callable** que será executada quando o fluxo chegar ali.

Detalhe importante: `add_node` **não executa nada**. Ele apenas
registra a função na topologia. A execução só acontece depois do
`compile()` + `invoke()`.

---

## 4. Dentro de cada nó — o que cada um faz

Agora vamos abrir cada nó e entender:

- **qual sua responsabilidade**;
- **o que ele lê** do estado;
- **o que ele escreve** no estado;
- **por que foi desenhado assim**.

### 4.1 `node_rag_retrieval` — busca semântica na política interna

```12:20:src/nodes/rag_retrieval.py
def node_rag_retrieval(state: OrchestratorState) -> dict:
    """Embed the question and fetch the nearest policy chunks from Vector Search.

    Returns a partial state dict with `rag_context` populated.  An empty list
    is valid — the evaluator node handles the case where no documents are found.
    """
    logger.info("node_rag_retrieval | question=%r", state.question)
    documents = search_policies(state.question)
    return {"rag_context": documents}
```

**O que ele faz, passo a passo:**

1. **Lê** `state.question` (a dúvida do gestor de conta).
2. Delega para `search_policies(...)` em
   [`src/clients/vector_search.py`](../src/clients/vector_search.py),
   que:
    - converte a pergunta em vetor (`embed_query`, modelo
      `text-embedding-004`);
    - chama `find_neighbors` no
      `MatchingEngineIndexEndpoint` do Vertex AI;
    - desserializa a resposta em `list[RAGDocument]`.
3. **Escreve** no estado: `{"rag_context": documents}`.

**Decisões de design a observar:**

- **Retornar `[]` é válido**. O nó **não levanta exceção** se nada for
  encontrado. Isso transfere a decisão "e se o RAG não bastar?" para o
  evaluator, que é quem tem contexto para decidir.
- **Nada de LLM aqui**. Esse nó é puro "buscar documentos". Manter
  responsabilidade única facilita testar e trocar (por exemplo, por um
  banco vetorial diferente) sem tocar no grafo.

**O que muda no estado após este nó:**

| Campo           | Antes | Depois               |
| --------------- | ----- | -------------------- |
| `question`      | `"..."` | inalterado           |
| `rag_context`   | `[]`  | `list[RAGDocument]`  |
| demais campos   | defaults | inalterados          |

---

### 4.2 `node_evaluator` — triagem com saída estruturada

Este é o nó mais importante pedagogicamente: **ele é quem decide se
vamos bifurcar** o fluxo mais adiante.

```65:95:src/nodes/evaluator.py
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
```

**O que ele faz, passo a passo:**

1. **Lê** `state.question` e `state.rag_context`.
2. Monta um prompt com duas mensagens:
    - **system**: critérios do Banco para exigir análise de risco
      (crédito acima de limite padrão, workout, IFRS 9 etc.).
    - **user**: pergunta do gestor + trechos numerados do RAG.
3. Invoca o LLM com `.with_structured_output(RiskAssessment)`. Esse
   método força o modelo a devolver **JSON validado** pelo schema
   Pydantic `RiskAssessment`:

    ```13:33:src/models.py
    class RiskAssessment(BaseModel):
        """Structured decision returned by the evaluator node via `with_structured_output`.

        Using a Pydantic model here (rather than free-form JSON) guarantees that
        the LLM output is schema-validated before the conditional edge reads it.
        """

        # True forces routing to the ADK risk-agent node.
        requires_escalation: bool = Field(
            ...,
            description=(
                "True when the operation requires specialist risk analysis: "
                "high-value credit, debt restructuring, unusual exposure, "
                "covenants / IFRS 9 / Res. 4.966, or insufficient RAG coverage."
            ),
        )
        rationale: str = Field(
            ...,
            min_length=5,
            description="Concise explanation of the decision, citing the criteria that apply.",
        )
    ```

4. **Escreve** no estado `requires_risk_assessment` (booleano) e
   `evaluator_rationale` (justificativa textual).

**Por que isso importa para o grafo?** Porque o próximo passo não é uma
aresta fixa — é uma **aresta condicional** que lê
`state.requires_risk_assessment`. Se esse valor fosse uma string
_free-form_ ("talvez", "depende"...), a bifurcação seria frágil.
Usando `with_structured_output` + Pydantic, a decisão chega como um
`bool` garantido pelo schema.

**O que muda no estado após este nó:**

| Campo                        | Antes    | Depois            |
| ---------------------------- | -------- | ----------------- |
| `requires_risk_assessment`   | `False`  | `True` ou `False` |
| `evaluator_rationale`        | `""`     | texto justificando |

---

### 4.3 `node_risk_agent` — delegação A2A para o especialista

Este nó **só é executado** quando o evaluator decide `True`. Vamos
ignorar por ora _como_ o grafo chega aqui (é o assunto da
[seção 7](#7-add_conditional_edges--bifurcação-baseada-no-estado)) e
focar no que ele faz.

```25:52:src/nodes/risk_agent.py
def node_risk_agent(state: OrchestratorState) -> dict:
    """Invoke the remote risk-specialist agent and capture its assessment."""
    payload = A2APayload(
        question=state.question,
        # Convert RAGDocuments to PolicyChunks (wire-format model) so that the
        # A2A payload schema is independent of the internal state schema.
        policy_chunks=[PolicyChunk.from_document(doc) for doc in state.rag_context],
        session=A2ASession(
            # uuid4 hex gives a unique trace ID for each orchestrator invocation,
            # enabling end-to-end tracing across the orchestrator and risk-agent.
            trace_id=uuid.uuid4().hex,
            triage_rationale=state.evaluator_rationale,
        ),
    )

    logger.info("node_risk_agent | delegating to risk specialist (A2A)")

    try:
        assessment = query_risk_agent(payload)
    except A2AClientError as exc:
        # Log the error but allow the graph to continue to the synthesis node.
        logger.error("node_risk_agent | A2A call failed: %s", exc)
        assessment = (
            f"[Assessment unavailable — A2A call failed: {exc}]. "
            "Please inform the account manager that specialist review is pending."
        )

    return {"risk_assessment_response": assessment}
```

**O que ele faz, passo a passo:**

1. Monta um `A2APayload` com:
    - a pergunta original;
    - os trechos de política (convertidos de `RAGDocument` para
      `PolicyChunk`, o "formato de fio" da chamada A2A);
    - uma `A2ASession` com `trace_id` único e a justificativa do
      evaluator.
2. Chama `query_risk_agent(...)` do
   [`src/clients/adk_client.py`](../src/clients/adk_client.py), que
   usa o `RemoteA2aAgent` do Google ADK para conversar com o
   especialista remoto.
3. **Graceful degradation**: se a chamada falhar
   (`A2AClientError`), **captura** a exceção e guarda uma mensagem
   textual descritiva. O grafo **não explode** — o synthesis depois
   informará o gestor que o parecer está indisponível.
4. **Escreve** `risk_assessment_response` no estado.

**Por que essa decisão de design é relevante para o grafo?**

Porque mostra uma técnica importante: em vez de usar controle de fluxo
do grafo para tratar erro (por exemplo, um "edge condicional de
fallback"), preferimos **neutralizar o erro dentro do nó** e deixar o
fluxo seguir. O grafo fica mais simples; a política de degradação fica
localizada onde o erro acontece.

**O que muda no estado após este nó:**

| Campo                      | Antes  | Depois                                        |
| -------------------------- | ------ | --------------------------------------------- |
| `risk_assessment_response` | `None` | parecer do especialista **ou** mensagem de indisponibilidade |

---

### 4.4 `node_synthesis` — redação da resposta final

O último nó é o redator. Ele **sempre** é executado, receba ou não o
parecer do especialista.

```53:78:src/nodes/synthesis.py
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
```

**O que ele faz, passo a passo:**

1. **Lê** `question`, `rag_context` e `risk_assessment_response`.
2. Se o especialista não foi chamado, preenche com a string
   `"(not requested)"` — didática para o LLM saber que aquele campo
   não é um erro, apenas "não solicitado".
3. Invoca o LLM com instruções explícitas para:
    - citar trechos numerados `[n]`;
    - destacar o parecer do especialista em seção própria, **se
      existir**;
    - ser explícito quando o RAG não tiver cobertura.
4. **Escreve** `final_response`.

Observe que este nó é **idempotente em relação ao caminho**: ele roda
igual tendo ou não passado pelo risk_agent. Isso simplifica testes —
basta variar as entradas.

**O que muda no estado após este nó:**

| Campo            | Antes | Depois         |
| ---------------- | ----- | -------------- |
| `final_response` | `""`  | texto da resposta |

---

## 5. `add_edge` — conectando nós de forma determinística

Uma **aresta** (`edge`) diz ao LangGraph: "quando o nó A terminar,
execute o nó B".

Sintaxe:

```python
g.add_edge(origem, destino)
```

No nosso projeto temos três arestas fixas:

```50:50:src/graph.py
    g.add_edge(Node.RAG_RETRIEVAL, Node.EVALUATOR)
```

```63:64:src/graph.py
    g.add_edge(Node.RISK_AGENT, Node.SYNTHESIS)
    g.add_edge(Node.SYNTHESIS, END)
```

Traduzindo:

1. Depois do `rag_retrieval`, **sempre** vai para o `evaluator`.
2. Depois do `risk_agent`, **sempre** vai para o `synthesis`.
3. Depois do `synthesis`, **sempre** termina (`END`).

Repare que **nenhuma dessas arestas depende do estado**. São transições
determinísticas, "quem sai daqui, vai pra lá". Use `add_edge` sempre
que o próximo passo não depende de nenhuma decisão.

> **Dica didática:** uma heurística rápida para escolher entre
> `add_edge` e `add_conditional_edges`:
> - Se você consegue desenhar o próximo passo no papel **sem olhar o
>   estado**, use `add_edge`.
> - Se precisa de um "se/então" para decidir, use
>   `add_conditional_edges`.

---

## 6. `set_entry_point` e o nó terminal `END`

Todo grafo precisa de um **ponto de entrada**: por onde começa a
execução. No nosso projeto:

```49:49:src/graph.py
    g.set_entry_point(Node.RAG_RETRIEVAL)
```

Isso diz: "quando chamarem `app.invoke(state_inicial)`, comece pelo
`rag_retrieval`".

> Alternativa equivalente: `g.add_edge(START, Node.RAG_RETRIEVAL)`
> usando o sentinela `START` importado de `langgraph.graph`. São duas
> formas de dizer a mesma coisa; `set_entry_point` é mais legível.

E todo grafo precisa de pelo menos um **caminho até `END`** — o
sentinela que marca o fim da execução. No nosso código:

```64:64:src/graph.py
    g.add_edge(Node.SYNTHESIS, END)
```

Se você esquecer essa aresta, o `compile()` **reclama** que o grafo
tem nós sem rota de saída. É uma rede de proteção contra fluxos
infinitos.

---

## 7. `add_conditional_edges` — bifurcação baseada no estado

Chegamos ao coração do grafo: a **decisão dinâmica**.

No nosso caso, depois do `evaluator`, há dois caminhos possíveis:

- Se `requires_risk_assessment == True` → vai para o `risk_agent`.
- Caso contrário → vai direto para o `synthesis`.

A sintaxe é:

```python
g.add_conditional_edges(
    no_de_origem,
    funcao_de_roteamento,
    mapa_de_destinos,   # dict[retorno_da_funcao -> nome_do_no]
)
```

No nosso projeto:

```54:61:src/graph.py
    g.add_conditional_edges(
        Node.EVALUATOR,
        route_after_evaluation,
        {
            Node.RISK_AGENT: Node.RISK_AGENT,
            Node.SYNTHESIS: Node.SYNTHESIS,
        },
    )
```

Vamos quebrar isso em três peças:

### 7.1 O nó de origem

`Node.EVALUATOR` — é a partir daqui que vamos bifurcar.

### 7.2 A função de roteamento

Precisa ter assinatura `(state) -> algum_valor`. No projeto:

```9:16:src/nodes/router.py
def route_after_evaluation(state: OrchestratorState) -> Node:
    """Return the name of the next node based on the evaluator's decision.

    This is a pure function with no side effects — straightforward to unit-test
    without any mocks.  LangGraph calls it after `node_evaluator` completes and
    uses the returned value to look up the target in `add_conditional_edges`.
    """
    return Node.RISK_AGENT if state.requires_risk_assessment else Node.SYNTHESIS
```

Note três características **muito importantes**:

1. **É uma função pura**: só lê o estado e retorna um valor. Não faz
   IO, não chama LLM, não loga. Isso é proposital — funções puras são
   triviais de testar.
2. **Não modifica o estado**. O router **não é um nó**. Ele só decide
   "para onde ir", sem alterar a prancheta compartilhada.
3. **Retorna um `Node`** (enum), não uma string solta. Se você digitar
   errado, o Python reclama na hora da importação.

### 7.3 O mapa de destinos

```python
{
    Node.RISK_AGENT: Node.RISK_AGENT,
    Node.SYNTHESIS: Node.SYNTHESIS,
}
```

À primeira vista, isso parece redundante (`chave == valor`). Por que
não passar só a função?

O mapa tem duas funções pedagógicas/operacionais:

1. **Torna explícitos os destinos válidos.** Olhando só para o
   `add_conditional_edges`, você sabe exatamente quais nós podem ser
   acionados dali — sem precisar ler o código do router.
2. **Permite desacoplar o _retorno_ do router do _nome do nó_**. Você
   pode fazer a função retornar strings semânticas ("precisa_risco",
   "direto") e mapeá-las para os nomes reais. Exemplo:

    ```python
    def router(state):
        return "precisa_risco" if state.requires_risk_assessment else "direto"

    g.add_conditional_edges(
        Node.EVALUATOR,
        router,
        {
            "precisa_risco": Node.RISK_AGENT,
            "direto": Node.SYNTHESIS,
        },
    )
    ```

    No nosso projeto optamos por **retornar o próprio `Node`** para
    reduzir indireção — fica `chave == valor` e o código vira quase
    auto-documentado.

### 7.4 Visualizando a bifurcação

```
                     ┌──────────────┐
                     │  evaluator   │
                     └──────┬───────┘
                            │
              route_after_evaluation(state)
                            │
            ┌───────────────┴───────────────┐
            │                               │
   requires_risk_assessment          requires_risk_assessment
        == True                          == False
            │                               │
            ▼                               ▼
    ┌──────────────┐                ┌──────────────┐
    │  risk_agent  │                │  synthesis   │
    └──────┬───────┘                └──────┬───────┘
           │                               │
           └─────────────┬─────────────────┘
                         ▼
                   ┌──────────┐
                   │   END    │
                   └──────────┘
```

---

## 8. Execução passo a passo — dois cenários reais

Vamos simular duas perguntas diferentes passando pelo grafo. Isso
"amarra" os conceitos.

### Cenário A — Pergunta simples: RAG basta

**Pergunta:** _"Qual o prazo máximo de capital de giro no Middle
Market?"_

1. **Entrada.** O `main.py` chama `run(question=...)` → `app.invoke(...)`
   com `OrchestratorState(question="...")`.
2. **Entrada no grafo.** `set_entry_point` indica `rag_retrieval`.
3. **`node_rag_retrieval`** embeda a pergunta, consulta Vector Search,
   devolve 5 trechos relevantes. Estado: `rag_context=[...]`.
4. **Aresta fixa** `rag_retrieval → evaluator`.
5. **`node_evaluator`** vê que a pergunta é direta e os trechos
   cobrem o tema. O LLM devolve
   `RiskAssessment(requires_escalation=False, rationale="...")`.
   Estado: `requires_risk_assessment=False`.
6. **Aresta condicional**. `route_after_evaluation(state)` avalia
   `state.requires_risk_assessment == False` e retorna `Node.SYNTHESIS`.
7. **`node_synthesis`** redige a resposta citando `[1]`, `[2]` etc. e
   indica que não foi necessária análise do especialista.
   Estado: `final_response="..."`.
8. **Aresta fixa** `synthesis → END`. O `invoke` devolve o estado
   final.

**Nós executados:** 3 de 4 (o `risk_agent` foi pulado).

### Cenário B — Pergunta complexa: exige especialista

**Pergunta:** _"Posso aprovar R$ 10M de capital de giro para Corporate
com rating B e covenants atípicos?"_

1. **Entrada** no grafo como antes.
2. **`node_rag_retrieval`** traz trechos sobre política de crédito,
   limites, covenants. Vários são relevantes, mas insuficientes para
   o caso concreto.
3. **Aresta fixa** `rag_retrieval → evaluator`.
4. **`node_evaluator`**. O prompt do sistema lista justamente os
   critérios de escalação: "crédito acima de limites padrão",
   "exposição concentrada, garantias atípicas", "covenants". O LLM
   retorna `requires_escalation=True`.
5. **Aresta condicional**. Router retorna `Node.RISK_AGENT`.
6. **`node_risk_agent`** monta o `A2APayload` com a pergunta, os
   trechos e a justificativa do evaluator, e chama o especialista
   remoto via A2A. O parecer volta e é armazenado em
   `risk_assessment_response`.
7. **Aresta fixa** `risk_agent → synthesis`.
8. **`node_synthesis`** redige a resposta com duas seções: a análise
   baseada na política (com `[1]`, `[2]`...) e uma seção dedicada ao
   parecer do especialista.
9. **Aresta fixa** `synthesis → END`.

**Nós executados:** 4 de 4.

### Cenário B' — Quando o A2A falha

Pequena variação do B para mostrar _graceful degradation_.

No passo 6, `query_risk_agent` levanta `A2AClientError` (timeout ou
card indisponível). O `node_risk_agent` **captura** a exceção e
escreve no estado:

```python
"[Assessment unavailable — A2A call failed: <mensagem>]. Please inform the account manager that specialist review is pending."
```

O grafo **continua normalmente** para o `synthesis`, que usará essa
string como se fosse o parecer — e o prompt do sistema instrui o LLM a
ser transparente com o gestor sobre a indisponibilidade.

Lição: **erros de infraestrutura não precisam virar bifurcações do
grafo**. Tratados dentro do nó, o grafo segue simples e legível.

---

## 9. Erros comuns e como o projeto os evita

| Erro comum                                        | Como o projeto previne                                                                                                                                                   |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Digitar errado o nome de um nó                    | Usar o enum `Node` em `constants.py`. Typo vira `AttributeError` na importação.                                                                                          |
| LLM devolver JSON mal formado e quebrar o router  | Evaluator usa `.with_structured_output(RiskAssessment)`, validado por Pydantic.                                                                                          |
| Exceção em um nó de I/O derrubar o grafo inteiro  | `node_risk_agent` captura `A2AClientError` e escreve uma mensagem no estado; o fluxo segue.                                                                              |
| Campos "vazios" do estado gerarem crashes         | `OrchestratorState` tem defaults seguros para todos os campos opcionais.                                                                                                 |
| Recompilar o grafo toda vez que é usado           | `get_app()` é `@lru_cache(maxsize=1)`, então o `compile()` roda uma única vez por processo.                                                                              |
| Router que faz I/O e é difícil de testar          | `route_after_evaluation` é **função pura**, testada unitariamente sem mocks.                                                                                             |
| Acoplar formato de estado com formato de API      | Conversão explícita `RAGDocument → PolicyChunk` antes de enviar via A2A. Mudanças no estado interno não quebram o contrato com o especialista.                           |

---

## 10. Exercícios para fixar

> Para cada exercício, trabalhe em um _branch_ separado e rode os
> testes em `tests/` para validar.

### Exercício 1 — Adicionar um nó de _guardrail_ antes do RAG

Crie um nó `node_guardrail` que verifique se a pergunta contém dados
sensíveis (CPF, CNPJ). Se sim, devolve `final_response` informando que
o gestor deve anonimizar antes de reenviar, e o grafo deve ir direto
para `END`, pulando o RAG, o evaluator e o synthesis.

Dicas:

- Adicione um campo `blocked: bool = False` em `OrchestratorState`.
- Use `add_conditional_edges` logo após o guardrail, com um mapa
  `{True: END, False: Node.RAG_RETRIEVAL}`.

### Exercício 2 — Três caminhos no evaluator

Refatore para que o evaluator possa retornar **três** decisões:

- `"RAG_SUFFICIENT"` → `synthesis`;
- `"NEEDS_RISK"` → `risk_agent`;
- `"NEEDS_CLARIFICATION"` → um novo nó `node_clarify` que escreve em
  `final_response` uma pergunta de esclarecimento para o gestor, e
  encerra.

Isso força você a usar um mapa não-trivial em `add_conditional_edges`.

### Exercício 3 — Testar só a topologia

Escreva um teste que chame `build_graph().get_graph()` e verifique:

- todos os 4 nós estão registrados;
- há aresta `rag_retrieval → evaluator`;
- há aresta `risk_agent → synthesis`;
- há aresta `synthesis → END`;
- a aresta condicional a partir de `evaluator` mapeia apenas para
  `risk_agent` e `synthesis`.

O teste deve rodar sem nenhuma credencial GCP — é aí que brilha a
separação entre `build_graph()` e `compile()`.

---

### Fechamento

Se você chegou até aqui, já entende os três pilares da topologia de um
`StateGraph`:

- **`add_node(nome, fn)`** — registra unidades de trabalho cujo
  contrato é "recebo o estado, devolvo um dicionário parcial".
- **`add_edge(origem, destino)`** — liga nós quando o próximo passo é
  determinístico.
- **`add_conditional_edges(origem, router, mapa)`** — delega ao
  _router_ a decisão de "para onde ir", sempre com base no estado.

E, mais importante, viu que **um bom grafo é aquele onde a topologia
conta a história do negócio**: olhando `src/graph.py`, você lê a
lógica de triagem do Banco BV sem precisar abrir nenhum outro
arquivo. É esse o objetivo final — código que se explica sozinho.
