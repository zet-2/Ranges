#!/usr/bin/env python3
"""Manual screenshot capture and Gemini Vision smoke check."""

import os
import io
from dotenv import load_dotenv
import mss
from PIL import Image
from google import genai
from google.genai import types
from rich.console import Console

def main() -> int:
    """Run the explicit, networked smoke test without import-time side effects."""

    load_dotenv()
    console = Console()
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(
            timeout=6500,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    try:
        console.print("[cyan]Capturing screen...[/cyan]")
        with mss.MSS() as sct:
            screenshot = sct.grab(sct.monitors[1])
            img = Image.frombytes(
                "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
            )
            img.save("screenshot_test.png")
            console.print("[green]✓ Screenshot saved: screenshot_test.png[/green]")

        console.print("[cyan]Sending to Gemini Vision...[/cyan]")
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=85)
        response = client.models.generate_content(
            model=model,
            contents=[
                "Describe what you see in this screenshot in 2 sentences max. Is there a poker table visible?",
                types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg"),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
        console.print(f"\n[yellow]Gemini says:[/yellow]\n{response.text.strip()}")
        return 0
    except Exception as error:
        console.print(f"[red]Error: {error}[/red]")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
