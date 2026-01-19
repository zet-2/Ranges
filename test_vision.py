#!/usr/bin/env python3
"""Test screenshot capture + Gemini Vision."""

import os
from dotenv import load_dotenv
import mss
from PIL import Image
import google.generativeai as genai
from rich.console import Console

load_dotenv()
console = Console()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

console.print("[cyan]Capturing screen...[/cyan]")

with mss.mss() as sct:
    screenshot = sct.grab(sct.monitors[1])
    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    img.save("screenshot_test.png")
    console.print("[green]✓ Screenshot saved: screenshot_test.png[/green]")

console.print("[cyan]Sending to Gemini Vision...[/cyan]")

try:
    response = model.generate_content([
        "Describe what you see in this screenshot in 2 sentences max. Is there a poker table visible?",
        img
    ])
    console.print(f"\n[yellow]Gemini says:[/yellow]\n{response.text.strip()}")
except Exception as e:
    console.print(f"[red]Error: {e}[/red]")
