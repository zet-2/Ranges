#!/usr/bin/env python3
"""Quick test to verify OpenAI API is working for strategy."""

import os
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

load_dotenv()
console = Console()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    console.print("[red]Error: OPENAI_API_KEY not found in .env[/red]")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# Test strategy prompt
test_prompt = """You are a poker range analyst. A player in UTG position raised 3bb preflop, 
got 3-bet by the button, and then 4-bet all-in.

Give me their likely range in ONE LINE using standard notation (e.g., "AA, KK, QQ, AKs").
Just the range, nothing else."""

console.print("[cyan]Testing OpenAI API (GPT-4o-mini)...[/cyan]")

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": test_prompt}],
        temperature=0.1
    )
    console.print(f"\n[green]✓ API Working![/green]")
    console.print(f"\n[yellow]Test: UTG 4-bet shove range[/yellow]")
    console.print(f"[white]{response.choices[0].message.content.strip()}[/white]")
except Exception as e:
    console.print(f"[red]✗ Error: {e}[/red]")
