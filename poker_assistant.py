#!/usr/bin/env python3
"""
Poker Range Assistant - Real-time PokerStars table analyzer
Uses Gemini Vision API to parse table state and estimate opponent ranges.
"""

import os
import sys
import time
import io
import base64
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import mss
import mss.tools
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
import google.generativeai as genai

load_dotenv()

console = Console()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    console.print("[red]Error: GEMINI_API_KEY not found in .env file[/red]")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


@dataclass
class TableState:
    """Parsed poker table state"""
    hero_cards: str = ""
    board: str = ""
    pot_size: str = ""
    hero_position: str = ""
    street: str = ""  # preflop, flop, turn, river
    villain_actions: list = None
    villain_positions: list = None
    hero_stack: str = ""
    villain_stacks: dict = None
    current_bet: str = ""
    hero_to_act: bool = False
    
    def __post_init__(self):
        if self.villain_actions is None:
            self.villain_actions = []
        if self.villain_positions is None:
            self.villain_positions = []
        if self.villain_stacks is None:
            self.villain_stacks = {}


PARSE_PROMPT = """Analyze this PokerStars poker table screenshot. Extract the following information in a structured format:

1. HERO_CARDS: The two cards the hero (bottom player) is holding. Format: "As Kh" (rank + suit letter: s=spades, h=hearts, d=diamonds, c=clubs). If not visible, say "unknown".

2. BOARD: Community cards on the table. Format: "Js Tc 4d 2h" or empty if preflop.

3. POT: Total pot size (number only, e.g., "150").

4. HERO_POSITION: Hero's position (BTN, SB, BB, UTG, MP, CO, etc.).

5. STREET: Current street (preflop, flop, turn, river).

6. VILLAIN_ACTIONS: List each villain's actions this hand in order. Format: "Position: action (amount)" 
   Example: "UTG: raise 3bb, CO: call, BTN: fold"

7. CURRENT_BET: Amount hero needs to call (0 if checking).

8. HERO_TO_ACT: Is it hero's turn to act? (yes/no)

9. ACTIVE_VILLAINS: Positions of players still in the hand.

10. STACK_SIZES: Approximate stack sizes of active players.

Respond in this EXACT format (keep the labels):
HERO_CARDS: [cards]
BOARD: [cards or empty]
POT: [amount]
HERO_POSITION: [position]
STREET: [street]
VILLAIN_ACTIONS: [actions]
CURRENT_BET: [amount]
HERO_TO_ACT: [yes/no]
ACTIVE_VILLAINS: [positions]
STACKS: [position: amount, ...]

If you cannot determine something clearly, use "unknown"."""


RANGE_PROMPT = """You are an expert poker range analyst. Based on the following game state, estimate the likely hand ranges for each active villain.

Game State:
- Street: {street}
- Pot: {pot}
- Board: {board}
- Villain Actions: {villain_actions}
- Active Villain Positions: {active_villains}

For each villain, provide:
1. Their estimated hand range using standard notation (e.g., "AA-TT, AKs-AJs, AKo-AQo, KQs")
2. Range as percentage of all hands (e.g., "~8%")
3. Brief reasoning

Consider:
- Position awareness (early position = tighter, late = wider)
- Action strength (raise = stronger than call, check-raise = very strong)
- Board texture interaction
- Typical player tendencies at low/mid stakes

Format:
VILLAIN [position]:
Range: [hands]
Percentage: [%]
Reasoning: [brief explanation]
---"""


def capture_screen(monitor_num: int = 1) -> Image.Image:
    """Capture screenshot of the specified monitor."""
    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_num >= len(monitors):
            monitor_num = 1  # Default to primary
        
        monitor = monitors[monitor_num]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return img


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def parse_table_response(response_text: str) -> TableState:
    """Parse Gemini's response into a TableState object."""
    state = TableState()
    
    lines = response_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("HERO_CARDS:"):
            state.hero_cards = line.split(":", 1)[1].strip()
        elif line.startswith("BOARD:"):
            state.board = line.split(":", 1)[1].strip()
        elif line.startswith("POT:"):
            state.pot_size = line.split(":", 1)[1].strip()
        elif line.startswith("HERO_POSITION:"):
            state.hero_position = line.split(":", 1)[1].strip()
        elif line.startswith("STREET:"):
            state.street = line.split(":", 1)[1].strip()
        elif line.startswith("VILLAIN_ACTIONS:"):
            state.villain_actions = line.split(":", 1)[1].strip()
        elif line.startswith("CURRENT_BET:"):
            state.current_bet = line.split(":", 1)[1].strip()
        elif line.startswith("HERO_TO_ACT:"):
            state.hero_to_act = "yes" in line.lower()
        elif line.startswith("ACTIVE_VILLAINS:"):
            villains = line.split(":", 1)[1].strip()
            state.villain_positions = [v.strip() for v in villains.split(",") if v.strip()]
        elif line.startswith("STACKS:"):
            stacks = line.split(":", 1)[1].strip()
            for pair in stacks.split(","):
                if ":" in pair:
                    pos, amt = pair.split(":", 1)
                    state.villain_stacks[pos.strip()] = amt.strip()
    
    return state


def analyze_table(img: Image.Image) -> Optional[TableState]:
    """Send screenshot to Gemini and parse table state."""
    try:
        response = model.generate_content([PARSE_PROMPT, img])
        return parse_table_response(response.text)
    except Exception as e:
        console.print(f"[red]Error analyzing table: {e}[/red]")
        return None


def estimate_ranges(state: TableState) -> str:
    """Get range estimates for active villains."""
    if not state.villain_positions or state.villain_positions == ["unknown"]:
        return "No active villains detected"
    
    prompt = RANGE_PROMPT.format(
        street=state.street,
        pot=state.pot_size,
        board=state.board if state.board else "none (preflop)",
        villain_actions=state.villain_actions,
        active_villains=", ".join(state.villain_positions)
    )
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error estimating ranges: {e}"


def calculate_simple_equity(hero_cards: str, villain_range: str, board: str) -> str:
    """
    Simplified equity estimation based on hand strength.
    For accurate equity, you'd use a proper equity calculator library.
    """
    # This is a simplified heuristic - for real use, integrate a proper equity calc
    premium_hands = ["AA", "KK", "QQ", "JJ", "AKs", "AKo"]
    strong_hands = ["TT", "99", "AQs", "AQo", "AJs", "KQs"]
    
    hero_clean = hero_cards.replace(" ", "").upper()
    
    # Very rough equity estimates
    if any(h in hero_clean for h in ["AA", "KK"]):
        return "~75-85% vs typical range"
    elif any(h in hero_clean for h in ["QQ", "JJ", "AK"]):
        return "~55-65% vs typical range"
    elif any(h in hero_clean for h in ["TT", "99", "AQ"]):
        return "~45-55% vs typical range"
    else:
        return "Equity depends on board texture"


def create_display(state: TableState, ranges: str, equity: str) -> Panel:
    """Create rich display panel for terminal output."""
    # Main info table
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Label", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("🃏 Hero Cards", f"[bold yellow]{state.hero_cards}[/bold yellow]")
    info_table.add_row("📍 Position", state.hero_position)
    info_table.add_row("🎯 Board", state.board if state.board else "[dim]preflop[/dim]")
    info_table.add_row("💰 Pot", f"[green]{state.pot_size}[/green]")
    info_table.add_row("📊 Street", state.street)
    info_table.add_row("🎲 To Call", state.current_bet)
    
    # Action status
    if state.hero_to_act:
        action_status = "[bold green]>>> YOUR ACTION <<<[/bold green]"
    else:
        action_status = "[dim]Waiting...[/dim]"
    
    # Build output
    output = f"{info_table}\n\n"
    output += f"[bold cyan]Villain Actions:[/bold cyan] {state.villain_actions}\n\n"
    output += f"[bold magenta]═══ RANGE ESTIMATES ═══[/bold magenta]\n{ranges}\n\n"
    output += f"[bold yellow]Equity:[/bold yellow] {equity}\n\n"
    output += action_status
    
    return Panel(output, title="[bold blue]♠ Poker Range Assistant ♠[/bold blue]", border_style="blue")


def main():
    """Main loop - capture and analyze poker table."""
    console.print(Panel.fit(
        "[bold green]Poker Range Assistant[/bold green]\n"
        "Analyzing PokerStars tables in real-time\n"
        "[dim]Press Ctrl+C to exit[/dim]",
        border_style="green"
    ))
    
    # Check for monitor selection
    with mss.mss() as sct:
        monitors = sct.monitors
        console.print(f"\n[cyan]Found {len(monitors) - 1} monitor(s)[/cyan]")
        for i, m in enumerate(monitors[1:], 1):
            console.print(f"  Monitor {i}: {m['width']}x{m['height']}")
    
    monitor_num = 1
    if len(sys.argv) > 1:
        try:
            monitor_num = int(sys.argv[1])
        except ValueError:
            pass
    
    console.print(f"\n[yellow]Watching monitor {monitor_num}. Pass monitor number as argument to change.[/yellow]")
    console.print("[dim]Starting in 3 seconds...[/dim]\n")
    time.sleep(3)
    
    last_state = None
    poll_interval = 1.5  # seconds
    
    try:
        while True:
            # Capture screen
            img = capture_screen(monitor_num)
            
            # Analyze with Gemini
            console.print("[dim]Analyzing...[/dim]", end="\r")
            state = analyze_table(img)
            
            if state and state.hero_cards and state.hero_cards != "unknown":
                # Get range estimates
                ranges = estimate_ranges(state)
                
                # Calculate equity
                equity = calculate_simple_equity(
                    state.hero_cards,
                    ranges,
                    state.board
                )
                
                # Display
                console.clear()
                console.print(create_display(state, ranges, equity))
                
                last_state = state
            else:
                if last_state:
                    console.print("[dim]No active hand detected, showing last state...[/dim]")
                else:
                    console.print("[yellow]Waiting for poker table... Make sure PokerStars is visible.[/yellow]", end="\r")
            
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        console.print("\n[green]Exiting Poker Range Assistant. Good luck at the tables! 🍀[/green]")


if __name__ == "__main__":
    main()
