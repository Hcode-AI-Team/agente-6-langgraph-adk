"""CLI entry point for the Banco BV Credit-Policy Orchestrator.

Usage:
    python main.py --question "Can I approve R$10M working capital for client X?"
    python main.py                      # interactive mode
    python main.py -q "..." --verbose   # show per-node execution trace
"""

from __future__ import annotations

import argparse
import sys
import warnings

from dotenv import load_dotenv

# Must run BEFORE any import that triggers google.adk, so that env vars like
# ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS are visible to its decorators.
load_dotenv(override=False)

# Silence the once-per-process "[EXPERIMENTAL] feature FeatureName.PLUGGABLE_AUTH
# is enabled" UserWarning emitted by google.adk.features._feature_decorator.
# The feature itself is left untouched (disabling it could break auth flows);
# only the noisy banner is filtered out.
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
from src.state import OrchestratorState  # noqa: E402

logger = get_logger(__name__)
console = Console()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Banco BV Credit-Policy Orchestrator — LangGraph + Vertex RAG + ADK A2A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Question from the account manager.  Omit to enter interactive mode.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-node state delta as the graph executes (streaming mode).",
    )
    return parser.parse_args()


def _print_header(settings: Settings) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("GCP project", str(settings.google_cloud_project))
    table.add_row("Region", str(settings.google_cloud_location))
    table.add_row("LLM model", str(settings.vertex_llm_model))
    table.add_row("Embedding", str(settings.vertex_embedding_model))
    table.add_row("Agent Card", str(settings.adk_risk_agent_card_url))

    console.print(
        Panel(
            table,
            title="[bold green]Banco BV Credit-Policy Orchestrator[/]",
            subtitle="[dim]LangGraph + Vertex RAG + ADK A2A[/]",
            border_style="green",
        )
    )


def _run_with_trace(question: str) -> str:
    """Execute the graph in streaming mode, printing each node's state delta."""
    app = get_app()
    console.print(Rule("[bold yellow]execution trace[/]", style="yellow"))
    accumulated: dict = {"question": question}

    for step in app.stream(OrchestratorState(question=question)):
        for node_name, delta in step.items():
            keys = sorted(delta.keys()) if isinstance(delta, dict) else []
            console.print(f"  [cyan]{node_name}[/]: [dim]{keys}[/]")
            if isinstance(delta, dict):
                accumulated.update(delta)

    return accumulated.get("final_response", "")


def _execute(question: str, *, verbose: bool) -> None:
    """Run the orchestrator and print the final response."""
    if verbose:
        response = _run_with_trace(question)
    else:
        state: OrchestratorState = run(question)
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

    _print_header(settings)

    if args.question:
        _execute(args.question, verbose=args.verbose)
        return 0

    console.print("[bold]Interactive mode[/] — type [red]'exit'[/] to quit.\n")
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
        _execute(question, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
