#!/usr/bin/env python3
"""
Calibration tool - Click the 4 corners of your PokerStars window.
"""
from pynput import mouse
from rich.console import Console

console = Console()
clicks = []
corner_names = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"]

def on_click(x, y, button, pressed):
    if pressed:
        clicks.append((x, y))
        console.print(f"[green]✓ {corner_names[len(clicks)-1]}: ({x}, {y})[/green]")
        if len(clicks) >= 4:
            return False

console.print("[bold cyan]PokerStars Window Calibrator[/bold cyan]\n")
console.print("Click the 4 corners of your PokerStars window:")
console.print("1. TOP-LEFT corner")
console.print("2. TOP-RIGHT corner") 
console.print("3. BOTTOM-LEFT corner")
console.print("4. BOTTOM-RIGHT corner\n")

with mouse.Listener(on_click=on_click) as listener:
    listener.join()

# Calculate region
left = min(clicks[0][0], clicks[2][0])
top = min(clicks[0][1], clicks[1][1])
right = max(clicks[1][0], clicks[3][0])
bottom = max(clicks[2][1], clicks[3][1])

width = right - left
height = bottom - top

console.print(f"\n[bold green]Your window region:[/bold green]")
console.print(f'WINDOW_REGION = {{"left": {left}, "top": {top}, "width": {width}, "height": {height}}}')
console.print("\n[yellow]Copy this line to poker_assistant.py (line 77)[/yellow]")
