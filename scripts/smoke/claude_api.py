#!/usr/bin/env python3
"""Manual paid smoke check for the Claude strategy API."""

import os

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console


def main() -> int:
    """Run the explicit, paid API smoke test without import-time side effects."""

    load_dotenv()
    console = Console()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
    if not api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY not found in .env[/red]")
        return 1

    client = Anthropic(api_key=api_key, timeout=15.0, max_retries=0)
    test_prompt = """You are a poker range analyst. A player in UTG position raised 3bb preflop,
got 3-bet by the button, and then 4-bet all-in.

Give me their likely range in ONE LINE using standard notation (e.g., "AA, KK, QQ, AKs").
Just the range, nothing else."""

    console.print(f"[cyan]Testing Claude API ({model})...[/cyan]")
    try:
        response = client.messages.create(
            model=model,
            max_tokens=128,
            messages=[{"role": "user", "content": test_prompt}],
        )
        result = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not result:
            raise ValueError("Claude returned no text content")

        console.print("\n[green]✓ API Working![/green]")
        console.print("\n[yellow]Test: UTG 4-bet shove range[/yellow]")
        console.print(f"[white]{result}[/white]")
        return 0
    except Exception as error:
        console.print(f"[red]✗ Error: {error}[/red]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
