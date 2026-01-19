#!/usr/bin/env python3
"""Quick test to verify Gemini API is working with a sample poker scenario."""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from rich.console import Console

load_dotenv()
console = Console()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Test range estimation
test_prompt = """You are a poker range analyst. A player in UTG position raised 3bb preflop, 
got 3-bet by the button, and then 4-bet all-in.

Give me their likely range in ONE LINE using standard notation (e.g., "AA, KK, QQ, AKs").
Just the range, nothing else."""

console.print("[cyan]Testing Gemini API...[/cyan]")

try:
    response = model.generate_content(test_prompt)
    console.print(f"\n[green]✓ API Working![/green]")
    console.print(f"\n[yellow]Test: UTG 4-bet shove range[/yellow]")
    console.print(f"[white]{response.text.strip()}[/white]")
except Exception as e:
    console.print(f"[red]✗ Error: {e}[/red]")
