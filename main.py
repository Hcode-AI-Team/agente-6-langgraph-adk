"""CLI entry point for the Banco BV Credit-Policy Orchestrator.

Usage:
    python main.py --question "Posso aprovar R$10M de capital de giro para cliente X?"
    python main.py                           # interactive mode (gera thread_id automático)
    python main.py -q "..." --verbose        # mostra trace por nó
    python main.py -q "..." --thread my-id  # retoma sessão persistida pelo thread_id
"""

from __future__ import annotations

import argparse
import uuid
import warnings

from dotenv import load_dotenv

# Must run BEFORE any import that triggers google.adk, so that env vars like
# ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS are visible to its decorators.
load_dotenv(override=False)

warnings.filterwarnings(
    "ignore",
    message=r".*\[EXPERIMENTAL\] feature FeatureName\.PLUGGABLE_AUTH.*",
    category=UserWarning,
)

from pydantic import ValidationError  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402

from src.config import Settings, get_settings  # noqa: E402
from src.graph import get_app, run  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)
console = Console()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Banco BV Credit-Policy Orchestrator — LangGraph + Vertex RAG + ADK A2A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Pergunta do gestor de contas. Omita para modo interativo.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Imprime o delta de estado de cada nó durante a execução (streaming).",
    )
    parser.add_argument(
        "--thread",
        type=str,
        default=None,
        help=(
            "Thread ID para retomar uma sessão persistida. "
            "Se omitido, um novo ID é gerado automaticamente."
        ),
    )
    return parser.parse_args()


def _print_header(settings: Settings, thread_id: str) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("GCP project", str(settings.google_cloud_project))
    table.add_row("Region", str(settings.google_cloud_location))
    table.add_row("LLM model", str(settings.vertex_llm_model))
    table.add_row("Embedding", str(settings.vertex_embedding_model))
    table.add_row("Agent Card", str(settings.adk_risk_agent_card_url))
    table.add_row("Checkpointer DB", str(settings.checkpointer_db_path))
    table.add_row("Thread ID", f"[bold yellow]{thread_id}[/]")

    console.print(
        Panel(
            table,
            title="[bold green]Banco BV Credit-Policy Orchestrator[/]",
            subtitle="[dim]LangGraph + Vertex RAG + ADK A2A + SQLite Checkpointer[/]",
            border_style="green",
        )
    )


def _run_with_trace(question: str, thread_id: str) -> str:
    """Execute the graph in streaming mode, printing each node's state delta."""
    app = get_app()
    config = {"configurable": {"thread_id": thread_id}}
    console.print(Rule("[bold yellow]execution trace[/]", style="yellow"))
    final_response = ""

    for step in app.stream({"question": question}, config=config):
        for node_name, delta in step.items():
            keys = sorted(delta.keys()) if isinstance(delta, dict) else []
            console.print(f"  [cyan]{node_name}[/]: [dim]{keys}[/]")
            if isinstance(delta, dict) and "final_response" in delta:
                final_response = delta["final_response"]

    return final_response


def _execute(question: str, *, verbose: bool, thread_id: str) -> None:
    """Run the orchestrator and print the final response."""
    if verbose:
        response = _run_with_trace(question, thread_id)
    else:
        state = run(question, thread_id)
        response = state.final_response

    console.print()
    console.print(
        Panel(
            Markdown(response or "_(sem resposta)_"),
            title="[bold green]FINAL RESPONSE[/]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def main() -> int:
    args = _parse_args()

    try:
        settings = get_settings()
    except ValidationError as exc:
        err = Console(stderr=True)
        err.print(
            Panel(
                f"[red]{exc}[/]\n\n"
                "[dim]Copy .env.example to .env and fill in all required variables.[/]",
                title="[bold red]CONFIGURATION ERROR[/]",
                border_style="red",
            )
        )
        return 2

    # Use the provided thread_id or generate one for this session.
    # In interactive mode the same thread persists for the entire session,
    # enabling the checkpointer to accumulate conversation history.
    thread_id = args.thread or uuid.uuid4().hex[:12]
    _print_header(settings, thread_id)

    if args.question:
        _execute(args.question, verbose=args.verbose, thread_id=thread_id)
        return 0

    console.print(
        f"[bold]Interactive mode[/] — sessão [yellow]{thread_id}[/] "
        f"| use [dim]--thread {thread_id}[/] para retomar depois\n"
        "Digite [red]'exit'[/] para sair.\n"
    )
    while True:
        try:
            question = console.input("[bold cyan]Manager>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit", "sair"}:
            return 0
        _execute(question, verbose=args.verbose, thread_id=thread_id)


if __name__ == "__main__":
    raise SystemExit(main())
