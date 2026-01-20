#!/usr/bin/env python3
"""
Poker Range Assistant - Real-time PokerStars table analyzer
Press 'j' to analyze | 'p' for preflop chart | 'n' to add note | ESC to exit
"""

import os
import sys
import json
import time
import dataclasses
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import mss
from PIL import Image, ImageChops
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich import box
from pynput import keyboard
import google.generativeai as genai
from openai import OpenAI

load_dotenv()
console = Console()

# Configure Gemini (Vision)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    console.print("[red]Error: GEMINI_API_KEY not found in .env[/red]")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

# Configure OpenAI (Strategy)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    console.print("[yellow]Warning: OPENAI_API_KEY not found in .env (Strategy will fail)[/yellow]")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize Vision Model - Using fastest available
# gemini-2.0-flash is faster than 2.5-flash-image for vision tasks
vision_model = genai.GenerativeModel("gemini-2.0-flash")    
# strategy_model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.1})

# Data file path
DATA_FILE = os.path.join(os.path.dirname(__file__), "poker_data.json")


def load_data() -> dict:
    """Load poker data (ranges + notes)."""
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"ranges": {}, "notes": {}}


def save_data(data: dict):
    """Save poker data."""
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)



# ---------------------------------------------------------
# PROFESSIONAL DATA STRUCTURES
# ---------------------------------------------------------

@dataclass
class MetaInfo:
    blind_level: dict = dataclasses.field(default_factory=lambda: {"sb": 0.01, "bb": 0.02, "ante": 0})
    table_size: int = 6
    current_street: str = "PREFLOP" # PREFLOP, FLOP, TURN, RIVER

@dataclass
class BoardState:
    community_cards: list = dataclasses.field(default_factory=list)
    total_pot: float = 0.0
    current_pot_odds: float = 0.0

@dataclass
class Player:
    seat_index: int
    name: str = "" # Position Label (e.g. "BTN")
    username: str = "" # Actual Player Name (e.g. "Player123")
    stack_size: float = 0.0
    hole_cards: list = None # ["Ah", "Kd"] or None
    current_bet: float = 0.0 # Chips currently in front
    status: str = "FOLDED" # ACTIVE, FOLDED, ALL_IN, SITTING_OUT
    is_hero: bool = False
    is_dealer: bool = False # Helper field

@dataclass
class LastActionContext:
    aggressor_seat_index: int = -1
    last_aggressive_action: str = "NONE" # BET, RAISE, CHECK
    amount_to_call: float = 0.0

@dataclass
class GameSnapshot:
    hand_id: str = ""
    timestamp: str = ""
    meta_info: MetaInfo = dataclasses.field(default_factory=MetaInfo)
    board_state: BoardState = dataclasses.field(default_factory=BoardState)
    dealer_seat_index: int = 0
    action_on_seat_index: int = 0
    players: list[Player] = dataclasses.field(default_factory=list)
    last_action_context: LastActionContext = dataclasses.field(default_factory=LastActionContext)
    
    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=2)


@dataclass
class HandHistory:
    """Accumulates snapshots for a single hand."""
    hand_id: str = ""
    snapshots: list = dataclasses.field(default_factory=list)
    last_action_on: int = -1
    last_street: str = ""
    
    def is_new_hand(self, snapshot: GameSnapshot) -> bool:
        """Detect if this snapshot is from a new hand."""
        if not self.snapshots:
            return True
        
        last = self.snapshots[-1]
        
        # New hand if board goes from cards to empty
        if len(last.board_state.community_cards) > 0 and len(snapshot.board_state.community_cards) == 0:
            return True
        
        # New hand if hero's cards change
        last_hero = next((p for p in last.players if p.is_hero), None)
        new_hero = next((p for p in snapshot.players if p.is_hero), None)
        if last_hero and new_hero:
            if last_hero.hole_cards and new_hero.hole_cards:
                if last_hero.hole_cards != new_hero.hole_cards:
                    return True
        
        return False
    
    def is_new_turn(self, snapshot: GameSnapshot) -> bool:
        """Detect if action has moved to a new player or street changed."""
        # Street changed
        if snapshot.meta_info.current_street != self.last_street:
            return True
        
        # Action moved to new player
        if snapshot.action_on_seat_index != self.last_action_on:
            return True
        
        return False
    
    def add_snapshot(self, snapshot: GameSnapshot):
        """Add snapshot and update tracking."""
        self.snapshots.append(snapshot)
        self.last_action_on = snapshot.action_on_seat_index
        self.last_street = snapshot.meta_info.current_street
        self.hand_id = snapshot.hand_id
    
    def reset(self):
        """Start fresh for new hand."""
        self.snapshots = []
        self.last_action_on = -1
        self.last_street = ""
        self.hand_id = ""
    
    def to_json(self) -> str:
        """Format history for strategist."""
        history = {
            "hand_id": self.hand_id,
            "total_turns": len(self.snapshots),
            "turns": [dataclasses.asdict(s) for s in self.snapshots]
        }
        return json.dumps(history, indent=2)

    def to_min_json(self) -> str:
        """Minimized JSON for strategy prompt."""
        turns = []
        for s in self.snapshots:
            min_players = []
            for p in s.players:
                min_players.append({
                    "pos": p.name,
                    "stack": f"{p.stack_size:.1f}",
                    "bet": p.current_bet,
                    "status": p.status
                })
            turns.append({
                "street": s.meta_info.current_street,
                "board": s.board_state.community_cards,
                "pot": s.board_state.total_pot,
                "actor": s.action_on_seat_index,
                "players": min_players
            })
        return json.dumps({"hand_id": self.hand_id, "turns": turns}, indent=2)

    def summary(self) -> str:
        """Quick text summary of the hand so far."""
        if not self.snapshots:
            return "No history yet"
        
        lines = [f"Hand {self.hand_id} - {len(self.snapshots)} turns tracked"]
        for i, s in enumerate(self.snapshots):
            ctx = s.last_action_context
            lines.append(f"  Turn {i+1}: {s.meta_info.current_street} | Pot: {s.board_state.total_pot}bb | Action: Seat {s.action_on_seat_index+1}")
        return "\n".join(lines)


ANALYZE_PROMPT = """
Role: You are a Poker Vision Expert. Analyze the provided poker table images to generate a precise game state JSON.

INPUT CONTEXT:
- **Images 1, 2, 3, 4, 6:** Villain Seats (Extract Villain Name & Stack).
- **Image 5:** Hero Seat (Extract Hero Name & Stack).
- **Image 7:** Board.
- **Image 8:** Actions.
- The visual style is PokerStars "Carbon" (4-Color Deck).
- **COLOR KEY:** Gray/Black = Spades (s), Blue = Diamonds (d), Red = Hearts (h), Green = Clubs (c).

VISUAL ANALYSIS INSTRUCTIONS (STRICT HIERARCHY):

1. STEP ONE: DETERMINE PLAYER STATUS (Check in this exact order):
   - **ACTIVE:** - Visible **CARDS** above the nameplate.
     - Look for two **RED/PATTERNED CARD BACKS** (Opponent) OR two **FACE-UP CARDS** (Hero).
   - **FOLDED (In Game, No Cards):** - Player has a Name and a **VISIBLE STACK VALUE** (e.g., "18 BB") inside the pod.
     - BUT there are **NO CARDS** above the nameplate.
   - **SITTING OUT:** - Player has a Name, but there is **NO STACK VALUE** displayed under the name.
     - OR explicit text says "Sitting Out".
   - **EMPTY:** - No Name, no chips, black background.

2. STEP TWO: EXTRACT ATTRIBUTES (Independent of Status):
   - **USERNAME:** 
     - Extract the player's username from the individual seat crop (Images 1-6).
     - **LOCATION:** The name is ALWAYS located immediately ABOVE the stack value inside the player pod.
     - Extract the exact string. If no name, use null.
   - **STACK SIZE (CRITICAL - CONVERT TO BB):** 
     - **LOCATION:** The number INSIDE the black player pod, directly BELOW the username.
     - **IF CURRENCY DETECTED (e.g. "$2.00", "€5.00"):** Convert to BBs assuming Big Blind = 0.02 (e.g. 2.00 -> 100 BB).
     - **IF NUMBER ONLY:** Assume it is already in BBs.
   - **CURRENT BET (CONVERT TO BB):** 
     - **LOCATION:** Look for a SEPARATE "Pill" or Bubble located OUTSIDE the player pod, usually near the cards or center.
     - **WARNING:** Do NOT confuse the Stack Size (inside pod) with the Current Bet.
     - If the player is All-In, the Stack Size might be 0.0, and the chips are in the Bet Bubble.
     - Apply same conversion rules (Currency -> BB).
   - **DEALER BUTTON:** Look for a small white circular disk with a black "D". It can be next to any player type.

3. HERO IDENTIFICATION (Seat 5):
   - If Seat 5 is Hero, extract rank/suit of face-up cards.
   - Format: ["Ah", "Tc"] (Rank + Suit).

OUTPUT FORMAT (JSON ONLY):
{
  "seats": [
    {
      "seat_index": <0-5, corresponding to image order>,
      "name": <String or null>,
      "stack_size_bb": <Float (IN BBs)>,
      "current_bet_bb": <Float (IN BBs)>,
      "has_cards": <Boolean - True ONLY if cards visible>,
      "is_folded": <Boolean - True if Stack visible but NO cards>,
      "is_sitting_out": <Boolean - True if Name present but NO Stack>,
      "is_dealer": <Boolean - Check for 'D' button regardless of status>,
      "hole_cards": <["Rs", "Rs"] or null>
    }
  ],
  "board_cards": [<list of community cards>],
  "total_pot_bb": <Float in BB>,
  "hero_context": {
    "is_turn": <Boolean>,
    "action_options": ["Check", "Call", "Raise"]
  }
}
"""

# boolean, # True if buttons are visible in Image 8
# (Prompt for Strategy remains similar but consumes this JSON)


def get_position_label(seat_num: int, dealer_seat_index: int) -> str:
    """
    Calculate poker position label for a seat in 6-max.
    seat_num: 1-6 (human readable)
    dealer_seat_index: 0-5 (0-indexed)
    Returns: BTN, SB, BB, UTG, MP, CO
    """
    # Convert dealer to 1-indexed
    btn_seat = dealer_seat_index + 1
    
    # Calculate position offset from BTN
    # In 6-max clockwise: BTN -> SB -> BB -> UTG -> MP -> CO
    positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    
    # How many seats after BTN is this seat?
    offset = (seat_num - btn_seat) % 6
    return positions[offset]


def get_position_label_dynamic(seat_index: int, dealer_seat_index: int, active_seats: list) -> str:
    """
    Calculate position label considering only active (non-sitting-out) seats.
    seat_index: 0-5
    dealer_seat_index: 0-5
    active_seats: list of seat indices that are NOT sitting out
    """
    if not active_seats:
        return "?"
    
    # Sort active seats in clockwise order starting from dealer
    def distance_from_dealer(s):
        return (s - dealer_seat_index) % 6
    
    sorted_seats = sorted(active_seats, key=distance_from_dealer)
    
    # Position names based on number of active players
    num_players = len(sorted_seats)
    
    if num_players == 2:
        positions = ["BTN", "BB"]  # Heads up: BTN is SB
    elif num_players == 3:
        positions = ["BTN", "SB", "BB"]
    elif num_players == 4:
        positions = ["BTN", "SB", "BB", "CO"]
    elif num_players == 5:
        positions = ["BTN", "SB", "BB", "UTG", "CO"]
    else:  # 6 players
        positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    
    # Find this seat's position in the sorted order
    try:
        idx = sorted_seats.index(seat_index)
        return positions[idx] if idx < len(positions) else f"P{idx+1}"
    except ValueError:
        return "?"


# ... (ACTION_PROMPT remains same) ...


# ---------------------------------------------------------
# DEEP GAME RECONSTRUCTION (Removed old HandHistory - using new one above)
# ---------------------------------------------------------








# ---------------------------------------------------------
# REGION DEFINITIONS (Calibrated for User Screen)
# ---------------------------------------------------------
# Note: User must verify these coordinates via calibration tool if they drift.
# (Placeholders used where previous calibration was lost)
SEAT_ZONES = {
    "seat1": {"left": 39.83203125, "top": 153.62890625, "width": 283.7890625, "height": 135.609375},
    "seat2": {"left": 373.20703125, "top": 90.53515625, "width": 234.67578125, "height": 146.4453125},
    "seat3": {"left": 608.52734375, "top": 126.84765625, "width": 309.59375, "height": 165.41796875},
    "seat4": {"left": 626.46484375, "top": 355.68359375, "width": 314.81640625, "height": 119.01171875},
    "hero":  {"left": 335.921875, "top": 413.3359375, "width": 263.04296875, "height": 167.8828125}, # Seat 5
    "seat6": {"left": 15.7578125, "top": 346.51953125, "width": 340.4765625, "height": 125.2890625},
    "board": {"top": 249, "left": 308, "width": 346, "height": 155} 
}
BUTTONS_REGION = {"left": 472, "top": 579, "width": 483, "height": 141} # User provided

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug_images")

def clear_debug_images():
    """Clear all files in the debug_images directory."""
    if os.path.exists(DEBUG_DIR):
        for f in os.listdir(DEBUG_DIR):
            file_path = os.path.join(DEBUG_DIR, f)
            try:
                os.remove(file_path)
            except:
                pass
    else:
        os.makedirs(DEBUG_DIR, exist_ok=True)

def save_debug_images(captures: dict):
    """Save captured regions to debug_images/ for verification."""
    clear_debug_images()
    for name, img in captures.items():
        img.save(os.path.join(DEBUG_DIR, f"{name}.png"))
    console.print(f"[dim]📸 Saved {len(captures)} debug images to {DEBUG_DIR}[/dim]")


def compress_image(img: Image.Image, max_size: int = 400) -> Image.Image:
    """Compress image to reduce API latency."""
    # Resize to max dimension while keeping aspect ratio
    ratio = min(max_size / img.width, max_size / img.height)
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def capture_regions():
    """Captures all defined regions from the screen."""
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_num]
        mon_left = monitor["left"]
        mon_top = monitor["top"]
        
        captures = {}
        
        # 1. Capture Seats & Board
        for name, zone in SEAT_ZONES.items():
            region = {
                "top": mon_top + int(zone["top"]),
                "left": mon_left + int(zone["left"]),
                "width": int(zone["width"]),
                "height": int(zone["height"])
            }
            sct_img = sct.grab(region)
            captures[name] = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        # 2. Capture Buttons
        btn_reg = {
            "top": mon_top + int(BUTTONS_REGION["top"]),
            "left": mon_left + int(BUTTONS_REGION["left"]),
            "width": int(BUTTONS_REGION["width"]),
            "height": int(BUTTONS_REGION["height"])
        }
        sct_img = sct.grab(btn_reg)
        captures["buttons"] = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        
        # Save for debugging
        save_debug_images(captures)
        
        return captures



def parse_response(text: str) -> GameSnapshot:
    """Parses Vision Response into a GameSnapshot."""
    snapshot = GameSnapshot()
    snapshot.timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    snapshot.hand_id = str(int(time.time()))
    
    try:
        # Clean up JSON
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        data = json.loads(cleaned_text.strip())
        
        # 1. Parse Board & Meta
        board_cards = data.get("board_cards") or data.get("board") or []
        snapshot.board_state.community_cards = board_cards
        snapshot.board_state.total_pot = float(data.get("total_pot_bb") or data.get("pot") or 0)
        
        # Deduce Street
        bc_len = len(board_cards)
        if bc_len == 0: snapshot.meta_info.current_street = "PREFLOP"
        elif bc_len == 3: snapshot.meta_info.current_street = "FLOP"
        elif bc_len == 4: snapshot.meta_info.current_street = "TURN"
        elif bc_len == 5: snapshot.meta_info.current_street = "RIVER"
        
        # 2. Parse Players (Two-Pass for correct position labels)
        seats_data = data.get("seats") or data.get("players") or []
        active_indices = []
        temp_players = []
        
        # Pass 1: Create players and find dealer
        for p_data in seats_data:
            # Handle seat index variations
            if "seat_index" in p_data:
                idx = p_data["seat_index"]
            elif "seat" in p_data:
                idx = p_data["seat"] - 1 # Convert 1-based to 0-based
            else:
                idx = 0

            player = Player(seat_index=idx)
            
            # Store OCR Username (with validation)
            ocr_name = p_data.get("name")
            if ocr_name and is_valid_username(ocr_name):
                player.username = ocr_name
            else:
                player.username = f"Unknown_S{idx}"
            
            # Handle stack variations
            stack_val = p_data.get("stack_size_bb") or p_data.get("stack") or 0
            if isinstance(stack_val, str):
                # Clean string currency if needed (simple removal)
                stack_val = stack_val.replace('$','').replace('€','').strip()
            player.stack_size = float(stack_val)
            
            player.current_bet = float(p_data.get("current_bet_bb") or p_data.get("bet") or 0)
            player.is_hero = (idx == 4) # Assuming Seat 5 (Index 4) is Hero
            player.is_dealer = p_data.get("is_dealer") or p_data.get("dealer") or False
            
            # Hole Cards (Hero Only) - Check FIRST
            hc = p_data.get("hole_cards") or p_data.get("cards")
            if player.is_hero and hc and isinstance(hc, list):
                # Filter out "XX" placeholders if specific to hallucination
                player.hole_cards = [c for c in hc if c and c.upper() != "XX"]
            
            # Determine Status (Priority: hole_cards > sitting_out > folded > has_cards)
            # If hole cards are detected, player is ALWAYS active
            if player.hole_cards:
                player.status = "ACTIVE"
                active_indices.append(idx)
            elif p_data.get("is_sitting_out", False):
                player.status = "SITTING_OUT"
            elif p_data.get("is_folded", False):
                player.status = "FOLDED"
            elif p_data.get("has_cards", False) or (hc and "XX" in str(hc)): # Fallback for XX cards
                player.status = "ACTIVE"
                active_indices.append(idx)
            else:
                player.status = "FOLDED"  # No cards and not sitting out = folded
            
            # Track dealer
            if player.is_dealer:
                snapshot.dealer_seat_index = idx
                
            temp_players.append(player)
        
        # Pass 2: Assign position labels (skip sitting-out players)
        # First, get list of seat indices for non-sitting-out players
        active_seats = [p.seat_index for p in temp_players if p.status != "SITTING_OUT"]
        
        for player in temp_players:
            if player.status == "SITTING_OUT":
                player.name = "OUT"
            else:
                # Calculate position among only active seats
                player.name = get_position_label_dynamic(
                    player.seat_index, 
                    snapshot.dealer_seat_index, 
                    active_seats
                )
            snapshot.players.append(player)
            
        # 3. Post-Process Logic (The "Logic Layer")
        highest_bet = 0.0
        aggressor = -1
        
        for p in snapshot.players:
            if p.current_bet > highest_bet:
                highest_bet = p.current_bet
                aggressor = p.seat_index
        
        # Default deduction
        snapshot.last_action_context.amount_to_call = highest_bet
        snapshot.last_action_context.aggressor_seat_index = aggressor
        
        # 4. Hero Context (Override from Buttons)
        # FORCE HERO TURN as requested by user
        snapshot.action_on_seat_index = 4 # Hero is Seat 5 (idx 4)
        h_ctx = data.get("hero_context", {})
        # if h_ctx.get("is_turn"):
        #     snapshot.action_on_seat_index = 4 # Hero is Seat 5 (idx 4)
        
        # If call amount is explicitly seen on buttons, trust it over deduction
        call_amt = h_ctx.get("amount_to_call_bb")
        if call_amt is not None and call_amt > 0:
             snapshot.last_action_context.amount_to_call = float(call_amt)
        
        # Update Meta (Blind Levels hardcoded for now or parsed?)
        snapshot.meta_info.blind_level = {"sb": 0.01, "bb": 0.02, "ante": 0}

    except json.JSONDecodeError:
        console.print("[red]Failed to parse JSON response[/red]")
    except Exception as e:
        console.print(f"[red]Error parsing snapshot: {e}[/red]")
        
    return snapshot







def show_preflop_chart(position: str = None):
    """Display preflop range chart."""
    data = load_data()
    ranges = data.get("ranges", {}).get("6max", {})
    
    if position and position.upper() in ranges:
        pos = position.upper()
        r = ranges[pos]
        console.print(f"\n[bold cyan]═══ {pos} Preflop Ranges ═══[/bold cyan]")
        console.print(f"[green]Open:[/green] {r.get('open', 'N/A')}")
        if 'vs_3bet' in r:
            console.print(f"[yellow]vs 3bet:[/yellow] {r.get('vs_3bet', 'N/A')}")
        if 'defend_vs_open' in r:
            console.print(f"[yellow]Defend:[/yellow] {r.get('defend_vs_open', 'N/A')}")
        if '3bet' in r:
            console.print(f"[magenta]3bet:[/magenta] {r.get('3bet', 'N/A')}")
        console.print(f"[dim]{r.get('description', '')}[/dim]")
    else:
        # Show all positions
        console.print("\n[bold cyan]═══ 6-Max Preflop Ranges ═══[/bold cyan]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Position")
        table.add_column("Open Range")
        table.add_column("Description")
        
        for pos in ["UTG", "MP", "CO", "BTN", "SB", "BB"]:
            if pos in ranges:
                r = ranges[pos]
                open_range = r.get('open', r.get('defend_vs_open', 'N/A'))[:50] + "..."
                table.add_row(pos, open_range, r.get('description', ''))
        
        console.print(table)
    console.print("[dim]Press 'p' + position (e.g., 'pCO') for details[/dim]\n")


def show_player_notes():
    """Display all saved player notes."""
    data = load_data()
    notes = data.get("notes", {})
    
    if not notes:
        console.print("\n[yellow]No player notes saved yet.[/yellow]")
        console.print("[dim]Press 'n' to add a note[/dim]\n")
        return
    
    console.print("\n[bold cyan]═══ Player Notes ═══[/bold cyan]")
    for player, note_list in notes.items():
        console.print(f"[bold white]{player}:[/bold white]")
        for note in note_list[-5:]:  # Show last 5 notes
            console.print(f"  • {note}")
    console.print()


def add_player_note(player: str, note: str):
    """Add a note for a player."""
    data = load_data()
    if "notes" not in data:
        data["notes"] = {}
    if player not in data["notes"]:
        data["notes"][player] = []
    data["notes"][player].append(note)
    save_data(data)
    console.print(f"[green]✓ Note added for {player}[/green]")


def get_player_notes(players: list) -> str:
    """Get notes for active players."""
    data = load_data()
    notes = data.get("notes", {})
    
    found = []
    for player in players:
        if player in notes and notes[player]:
            # Get last note
            found.append(f"{player}: {notes[player][-1]}")
    
    return " | ".join(found) if found else ""


# Global state
monitor_num = 1
running = True
last_state = None
input_mode = None
note_player = ""



class GameRecorder:
    MAX_SESSION_FILES = 10  # Keep only the most recent sessions
    
    def __init__(self):
        self.filename = f"session_{int(time.time())}.jsonl" # JSON Lines formatted
        self.cleanup_old_sessions()
        self.log(f"Session Started: {self.filename}", is_meta=True)
    
    def cleanup_old_sessions(self):
        """Remove old session files, keeping only the most recent ones."""
        try:
            base_dir = os.path.dirname(__file__) or "."
            session_files = []
            
            for f in os.listdir(base_dir):
                if f.startswith("session_") and f.endswith(".jsonl"):
                    path = os.path.join(base_dir, f)
                    session_files.append((os.path.getmtime(path), path))
            
            # Sort by modification time (newest first)
            session_files.sort(reverse=True)
            
            # Delete old files beyond the limit
            for _, path in session_files[self.MAX_SESSION_FILES:]:
                try:
                    os.remove(path)
                except:
                    pass
                    
            deleted = len(session_files) - min(len(session_files), self.MAX_SESSION_FILES)
            if deleted > 0:
                console.print(f"[dim]🧹 Cleaned up {deleted} old session files[/dim]")
        except Exception:
            pass  # Don't fail if cleanup fails
    
    def log(self, content, is_meta=False):
        with open(self.filename, "a", encoding="utf-8") as f:
            if is_meta:
                f.write(json.dumps({"meta": content}) + "\n")
            else:
                f.write(json.dumps(content) + "\n")

    def update(self, snapshot: GameSnapshot):
        # Simply log the full snapshot
        # Convert to dict first
        data = dataclasses.asdict(snapshot)
        self.log(data)



def analyze_state(monitor: int) -> tuple[GameSnapshot, dict, float, float]:
    """
    1. Capture Regions
    2. Vision API -> GameSnapshot
    """
    t0 = time.time()
    
    # 1. Capture Direct Regions
    comps = capture_regions()
    t_capture = time.time() - t0
    
    # save_debug_images(comps) # Disabled
    
    # 2. Vision Pass 
    t1 = time.time()
    
    # Prepare and compress images for faster API response
    seat_imgs = [
        compress_image(comps["seat1"]), compress_image(comps["seat2"]), 
        compress_image(comps["seat3"]), compress_image(comps["seat4"]), 
        compress_image(comps["hero"]),  compress_image(comps["seat6"])
    ]
    
    vision_input = [ANALYZE_PROMPT] + seat_imgs + [
        compress_image(comps["board"]), 
        compress_image(comps["buttons"])
    ] 
    
    try:
        response = vision_model.generate_content(vision_input)
        # Save raw response for debugging
        with open("debug_vision_response.txt", "w") as f:
            f.write(response.text)
        console.print(f"[dim]Vision response saved to debug_vision_response.txt[/dim]")
        snapshot = parse_response(response.text)
        console.print(f"[dim]Parsed {len(snapshot.players)} players[/dim]")
    except Exception as e:
        console.print(f"[red]Vision Error: {e}[/red]")
        snapshot = GameSnapshot() 
        
    t_vision = time.time() - t1
    
    return snapshot, comps, t_capture, t_vision



def display_results(snapshot: GameSnapshot, analysis: str, t_cap, t_vis, t_strat, t_tot, metrics: dict = None, hand_rank: str = ""):
    """Synthetic display of GameSnapshot data."""
    console.clear()
    
    # 1. EXTRACT RECOMMENDATION
    recommendation = "Analyzing..."
    details = ""
    
    if analysis:
        lines = analysis.split('\n')
        action_line = next((line for line in lines if "**Action:**" in line), None)
        amount_line = next((line for line in lines if "**Amount:**" in line), None)
        
        if action_line:
            act = action_line.split(":", 1)[1].strip().replace("*", "")
            amt = ""
            if amount_line:
                raw_amt = amount_line.split(":", 1)[1].strip().replace("*", "")
                if "0" not in raw_amt and raw_amt.lower() != "n/a":
                    amt = " " + raw_amt
            recommendation = f"{act}{amt}"
            
            # Extract only the bullet points from reasoning
            reasoning_lines = [l.strip() for l in lines if l.strip().startswith("*")]
            details = "\n".join(reasoning_lines)

    # Header
    console.print(f"[dim]ID: {snapshot.hand_id} | {t_tot:.2f}s[/dim]")

    # ★ RECOMMENDATION PANEL ★
    if analysis:
        style = "bold white on red"
        rec_upper = recommendation.upper()
        if "FOLD" in rec_upper: style = "bold white on red"
        elif "CHECK" in rec_upper or "CALL" in rec_upper: style = "bold black on yellow"
        elif "RAISE" in rec_upper or "BET" in rec_upper: style = "bold white on green"
        console.print(Panel(Align.center(f"[bold]{recommendation}[/bold]"), style=style))

    # HUD Table
    hud = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE, expand=True, padding=(0,1))
    hud.add_column("HERO", justify="center")
    hud.add_column(f"BOARD ({snapshot.meta_info.current_street})", justify="center")
    hud.add_column("POT", justify="center")
    
    hero = next((p for p in snapshot.players if p.is_hero), None)
    hero_cards = " ".join(hero.hole_cards) if hero and hero.hole_cards else "—"
    hand_display = f"[bold white]{hero_cards}[/bold white]"
    if hand_rank: hand_display += f"\n[yellow]{hand_rank}[/yellow]"
    
    board_str = "[bold white]" + (" ".join(snapshot.board_state.community_cards) or "Pre") + "[/bold white]"
    hud.add_row(hand_display, board_str, f"[bold yellow]{snapshot.board_state.total_pot} BB[/bold yellow]")
    console.print(hud)

    # Metrics Grid
    if metrics:
        spr = metrics.get("spr", 0)
        odds = metrics.get("pot_odds_str", "N/A")
        eff = metrics.get("eff_stack", 0)
        spr_col = "green" if spr > 10 else "yellow" if spr > 3 else "red"
        
        m_grid = Table.grid(expand=True)
        m_grid.add_column(justify="center", ratio=1)
        m_grid.add_column(justify="center", ratio=1)
        m_grid.add_column(justify="center", ratio=1)
        m_grid.add_row(f"Eff: [white]{eff:.1f}[/white]", f"SPR: [{spr_col}]{spr:.1f}[/]", f"Odds: [cyan]{odds}[/]")
        console.print(m_grid)

    # Reasoning
    if details:
        console.print(Panel(details, border_style="dim", title="[dim]Logic[/]", title_align="left"))
    
    # Player Table (Very Compact)
    p_tab = Table(show_header=True, header_style="dim", box=None, expand=True, padding=(0,1))
    p_tab.add_column("P", width=2)
    p_tab.add_column("Name", style="italic", ratio=1)
    p_tab.add_column("Stack", justify="right")
    p_tab.add_column("Act", justify="right")
    
    for p in sorted(snapshot.players, key=lambda x: x.seat_index):
        if p.status == "SITTING_OUT" and not p.is_hero: continue
        pos = p.name + ("(D)" if p.is_dealer else "")
        act = "[bold yellow]HERO[/]" if p.is_hero else ("[dim]Fold[/]" if p.status == "FOLDED" else (f"[red]B:{p.current_bet}[/]" if p.current_bet > 0 else "[green]Check[/]"))
        p_tab.add_row(pos[:4], (p.username or "—")[:12], f"{p.stack_size:.0f}", act)
    
    console.print(p_tab)









# Global state
monitor_num = 1
running = True
last_state = None
last_captures = None
input_mode = None
note_player = ""
current_history = HandHistory()  # Accumulates snapshots for current hand
recorder = GameRecorder() # Enable logging


def parse_action_history(history_json: str) -> str:
    """Generates a human-readable summary of the action line from JSON history."""
    try:
        data = json.loads(history_json)
        turns = data.get("turns", [])
    except:
        return "History parsing failed."
        
    if not turns: return "No history."
    
    output_lines = []
    current_street = None
    street_actions = []
    
    for i, turn in enumerate(turns):
        meta = turn.get("meta_info", {})
        street = meta.get("current_street", "UNKNOWN")
        
        # New Street Handling
        if street != current_street:
            if current_street and street_actions:
                output_lines.append(f"{current_street}: {', '.join(street_actions)}.")
            current_street = street
            street_actions = []
            
        # Skip first snapshot as base
        if i == 0: continue
        
        prev = turns[i-1]
        curr = turn
        
        # Detect Actions
        # 1. Folds
        for p_idx, p_curr in enumerate(curr["players"]):
            # Safety check for list length
            if p_idx >= len(prev["players"]): break
            p_prev = prev["players"][p_idx]
            
            if p_curr["status"] == "FOLDED" and p_prev["status"] != "FOLDED":
                name = p_curr["name"] or f"S{p_curr['seat_index']+1}"
                street_actions.append(f"{name} Folds")

        # 2. Bets/Calls/Raises
        prev_max_bet = max((p["current_bet"] for p in prev["players"]), default=0.0)
        
        # Check who acted (Action moved from Prev Actor)
        # Or simply check state changes for all players
        for p_idx, p_curr in enumerate(curr["players"]):
            if p_idx >= len(prev["players"]): break
            p_prev = prev["players"][p_idx]
            
            if p_curr["status"] == "FOLDED": continue
            
            curr_bet = p_curr["current_bet"]
            prev_bet = p_prev["current_bet"]
            
            if curr_bet > prev_bet:
                name = p_curr["name"] or f"S{p_curr['seat_index']+1}"
                
                # Logic: Call vs Bet vs Raise
                if curr_bet <= prev_max_bet and prev_max_bet > 0:
                    street_actions.append(f"{name} Calls {curr_bet}")
                elif prev_max_bet == 0:
                    street_actions.append(f"{name} Bets {curr_bet}")
                else:
                    street_actions.append(f"{name} Raises to {curr_bet}")

        # 3. Checks
        # If action moved from A to B, and A didn't put more money, A checked.
        prev_actor_idx = prev["action_on_seat_index"]
        if prev_actor_idx != curr["action_on_seat_index"]:
             # Check if previous actor checked
             if prev_actor_idx < len(prev["players"]) and prev_actor_idx < len(curr["players"]):
                 p_prev_actor = prev["players"][prev_actor_idx]
                 p_curr_actor = curr["players"][prev_actor_idx]
                 
                 if p_curr_actor["status"] == "ACTIVE":
                     if p_curr_actor["current_bet"] == p_prev_actor["current_bet"]:
                         # No money added
                         if p_curr_actor["current_bet"] == prev_max_bet:
                             # Matches table max (or 0) -> Check
                             name = p_curr_actor["name"] or f"S{prev_actor_idx+1}"
                             street_actions.append(f"{name} Checks")

    # Flush last street
    if current_street and street_actions:
        output_lines.append(f"{current_street}: {', '.join(street_actions)}.")
        
    return "\n".join(output_lines)


def parse_pokerstars_files(directory: str) -> list:
    """Parses PokerStars text HH files into synthetic snapshots."""
    hands_data = []
    
    if not os.path.exists(directory):
        return []
        
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"): continue
        
        path = os.path.join(directory, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            raw_hands = content.split("PokerStars Hand #")
            for raw in raw_hands:
                if not raw.strip(): continue
                
                lines = raw.strip().split('\n')
                
                # Basic Parsing
                hand_id = lines[0].split(':')[0].strip()
                
                # Meta - Detect stake level
                sb_val = 0.01
                bb_val = 0.02
                
                # Skip Play Money files
                if "Play Money" in lines[0] or "100-200" in lines[0]:
                    continue
                    
                # Detect real money stakes
                if "€0.01/€0.02" in lines[0]: 
                    sb_val, bb_val = 0.01, 0.02
                elif "€0.02/€0.05" in lines[0]: 
                    sb_val, bb_val = 0.02, 0.05
                elif "€0.05/€0.10" in lines[0]:
                    sb_val, bb_val = 0.05, 0.10
                elif "€0.10/€0.25" in lines[0]:
                    sb_val, bb_val = 0.10, 0.25
                
                # Players
                players = []
                btn_seat = 1
                
                # Find Button
                for line in lines:
                    if "Seat #" in line and "is the button" in line:
                        try:
                            btn_seat = int(line.split("#")[1].split(" ")[0])
                        except: pass
                        break
                
                # Parse Seats
                active_seats = {}
                for line in lines:
                    if line.startswith("Seat ") and ": " in line and " in chips" in line:
                        try:
                            # Seat 1: Name (€2.00 in chips)
                            parts = line.split(": ")
                            seat_idx = int(parts[0].replace("Seat ", "")) - 1
                            rest = parts[1]
                            name = rest.split(" (")[0]
                            stack_str = rest.split(" (")[1].split(" in chips")[0]
                            # Clean currency
                            stack_str = stack_str.replace("€", "").replace("$", "")
                            stack = float(stack_str) / bb_val # Convert to BB
                            
                            p = {
                                "seat_index": seat_idx,
                                "name": name, # Position calc later or ignore
                                "username": name,
                                "stack_size": stack,
                                "current_bet": 0.0,
                                "status": "ACTIVE",
                                "is_hero": (name == "biba287"), # Hardcoded hero check
                                "is_dealer": (seat_idx == btn_seat - 1)
                            }
                            players.append(p)
                            active_seats[name] = p
                        except: pass
                
                if not players: continue
                
                # Scan for Preflop Max Bets
                max_bets = {p["username"]: 0.0 for p in players}
                aggressors = set()
                
                in_street = "PREFLOP"
                
                for line in lines:
                    if "*** HOLE CARDS ***" in line: in_street = "PREFLOP"
                    elif "*** FLOP ***" in line: break # Stop at flop for now (VPIP/PFR only needs Preflop)
                    elif "*** SUMMARY ***" in line: break
                    
                    if in_street == "PREFLOP":
                        # Parse Action: "Name: raises x to y" or "Name: bets x" or "Name: calls x"
                        if ": " in line:
                            parts = line.split(": ")
                            name = parts[0]
                            action = parts[1]
                            
                            if name in active_seats:
                                amt = 0.0
                                is_agg = False
                                
                                # raises €0.02 to €0.04
                                if "raises" in action:
                                    try:
                                        amt_str = action.split(" to ")[1].split(" ")[0].replace("€","").replace("$","")
                                        amt = float(amt_str) / bb_val
                                        is_agg = True
                                    except: pass
                                elif "bets" in action:
                                    try:
                                        amt_str = action.split(" ")[1].replace("€","").replace("$","")
                                        amt = float(amt_str) / bb_val
                                        is_agg = True
                                    except: pass
                                elif "calls" in action:
                                    try:
                                        amt_str = action.split(" ")[1].replace("€","").replace("$","")
                                        amt = float(amt_str) / bb_val
                                    except: pass
                                    
                                if amt > max_bets[name]:
                                    max_bets[name] = amt
                                if is_agg:
                                    aggressors.add(name)

                # Create Synthetic Snapshots
                # 1. Start (Zero bets)
                s1 = {
                    "hand_id": hand_id,
                    "timestamp": "",
                    "meta_info": {"current_street": "PREFLOP", "blind_level": {"bb": bb_val}},
                    "players": [p.copy() for p in players],
                    "dealer_seat_index": btn_seat - 1,
                    "last_action_context": {"aggressor_seat_index": -1}
                }
                
                # 2. End of Preflop (Max bets applied)
                players_end = []
                last_agg_seat = -1
                
                for p in players:
                    p_new = p.copy()
                    p_new["current_bet"] = max_bets.get(p["username"], 0.0)
                    players_end.append(p_new)
                    
                    if p["username"] in aggressors:
                        last_agg_seat = p["seat_index"]
                        
                s2 = {
                    "hand_id": hand_id,
                    "timestamp": "",
                    "meta_info": {"current_street": "PREFLOP", "blind_level": {"bb": bb_val}},
                    "players": players_end,
                    "dealer_seat_index": btn_seat - 1,
                    "last_action_context": {"aggressor_seat_index": last_agg_seat}
                }
                
                hands_data.append([s1, s2])
                
        except Exception:
            continue
            
    return hands_data


def load_all_sessions() -> list:
    """Loads all hands ONLY from PokerStars HH files (legacy folder)."""
    hands = {}
    base_dir = os.path.dirname(__file__)
            
    # PokerStars HH (The Source of Truth)
    try:
        ps_hands = parse_pokerstars_files(os.path.join(base_dir, "biba287"))
        # We don't print this every time to keep the UI clean, 
        # but the stats will be based on these hands.
        for h in ps_hands:
            if h: hands[h[0]["hand_id"]] = h
    except Exception:
        pass
        
    return list(hands.values())


def calculate_villain_stats(villain_name: str, past_hands_list: list, min_samples: int = 0) -> str:
    """Calculates VPIP, PFR, and AF for a player using Max Bet detection.
    
    Args:
        villain_name: The player's username to look up
        past_hands_list: List of hand histories
        min_samples: Minimum sample size required (returns None if not met)
    """
    if not past_hands_list or not villain_name or villain_name == "Unknown":
        return "Unknown (0 hands)"
    
    total_hands = 0
    vpip_hands = 0
    pfr_hands = 0
    # AF tracking (simplified for now as PFR covers preflop aggression)
    
    for snapshots in past_hands_list:
        if not snapshots: continue
        
        # 1. Get Hand Context
        first_ss = snapshots[0]
        players = first_ss.get("players", [])
        
        # Identify Villain in this hand
        villain = next((p for p in players if p.get("username", p.get("name")) == villain_name), None)
        if not villain: continue
        
        villain_seat = villain.get("seat_index")
        dealer_seat = first_ss.get("dealer_seat_index", 0)
        
        # Determine Blinds (6-max logic)
        sb_seat = (dealer_seat + 1) % 6
        bb_seat = (dealer_seat + 2) % 6
        
        is_sb = (villain_seat == sb_seat)
        is_bb = (villain_seat == bb_seat)
        
        # 2. Scan Preflop for Max Bet & Aggression
        max_bet = 0.0
        was_aggressor = False
        saw_preflop = False
        
        # Check all snapshots for this hand
        for s in snapshots:
            meta = s.get("meta_info", {})
            street = meta.get("current_street", "PREFLOP") # Default to PREFLOP if missing
            
            if street == "PREFLOP":
                saw_preflop = True
                p_snap = next((p for p in s.get("players", []) if p.get("seat_index") == villain_seat), None)
                
                if p_snap:
                    bet = float(p_snap.get("current_bet", 0.0))
                    if bet > max_bet: max_bet = bet
                    
                    # Check aggression context
                    ctx = s.get("last_action_context", {})
                    if ctx.get("aggressor_seat_index") == villain_seat:
                        was_aggressor = True
        
        # If we missed preflop but player is active post-flop, count as VPIP
        if not saw_preflop:
             # Look at last snapshot
             last = snapshots[-1]
             p_last = next((p for p in last.get("players", []) if p.get("seat_index") == villain_seat), None)
             if p_last and p_last.get("status") in ["ACTIVE", "ALL_IN"]:
                 vpip_hands += 1
                 total_hands += 1
             continue

        # 3. Calculate VPIP/PFR
        is_vpip = False
        is_pfr = False
        
        # Logic: 
        # SB is VPIP if they put in > 0.5 BB (Complete or Raise)
        # BB is VPIP if they put in > 1.0 BB (Raise or Call Raise). Checking option (1.0) is not VPIP.
        # Others are VPIP if > 0 (Call or Raise)
        
        if is_bb:
            if max_bet > 1.0: is_vpip = True
        elif is_sb:
            if max_bet > 0.5: is_vpip = True
        else:
            if max_bet > 0.0: is_vpip = True
            
        # PFR: Must be VPIP'd AND was the Aggressor (Raised) at some point
        # And max_bet must imply a raise (usually > 1.0)
        if is_vpip and was_aggressor and max_bet > 1.0:
            is_pfr = True
            
        total_hands += 1
        if is_vpip: vpip_hands += 1
        if is_pfr: pfr_hands += 1
            
    if total_hands < 1: 
        return None
    
    # Check min_samples requirement
    if min_samples > 0 and total_hands < min_samples:
        return None
        
    vpip = (vpip_hands / total_hands) * 100
    pfr = (pfr_hands / total_hands) * 100
    
    return f"VPIP: {vpip:.0f}% | PFR: {pfr:.0f}% (Sample: {total_hands} hands)"


def is_valid_username(name: str) -> bool:
    """Filters out invalid OCR detections (actions, positions, numbers)."""
    if not name or len(name) < 2: return False
    
    # Common OCR misreads
    BAD_KEYWORDS = {
        "CALL", "BET", "CHECK", "RAISE", "FOLD", "ALL-IN", "ALLIN",
        "BB", "SB", "UTG", "MP", "CO", "BTN", "DEALER",
        "UNKNOWN", "SEAT", "POT", "TOTAL", "CHECKING", "CALLING"
    }
    
    clean_name = name.upper().strip()
    
    # 1. Exact Keyword Match
    if clean_name in BAD_KEYWORDS: return False
    
    # 2. "Call (BB)" pattern (Action + Position)
    if "(" in clean_name or ")" in clean_name: return False
    
    # 3. Numeric (Stack read as name)
    try:
        float(name.replace("$","").replace("€","").replace(",","."))
        return False # It's a number
    except:
        pass
        
    # 4. Unknown_S prefix
    if clean_name.startswith("UNKNOWN_S"): return False
    
    return True


def save_villain_db(history_hands: list):
    """Aggregates and saves all villain stats to villain_stats.json for user inspection."""
    db = {}
    
    # 1. Identify all unique players
    all_players = set()
    for hand in history_hands:
        # Check first turn for player list (usually static per hand)
        if hand:
            for p in hand[0].get("players", []):
                # Prefer username, fallback to name if not hero
                if not p.get("is_hero"):
                    name = p.get("username") or p.get("name")
                    if name and is_valid_username(name):
                        all_players.add(name)

    # 2. Calculate stats for each
    for name in all_players:
        stats_str = calculate_villain_stats(name, history_hands)
        # Only store if stats were calculated (not None)
        if stats_str is not None:
            db[name] = stats_str
            
    # 3. Save
    try:
        with open("villain_stats.json", "w") as f:
            json.dump(db, f, indent=2, sort_keys=True)
    except Exception as e:
        console.print(f"[dim]Error saving villain DB: {e}[/dim]")


class HandEvaluator:
    """Pure Python Poker Hand Evaluator with 'Playing the Board' detection."""
    
    RANKS = '23456789TJQKA'
    
    @staticmethod
    def parse_card(card_str):
        if not card_str: return None, None
        card_str = card_str.strip().upper()
        
        # Handle '10' as 'T'
        if card_str.startswith("10"):
            r = "T"
            s = card_str[2:].lower() if len(card_str) > 2 else ""
        else:
            r = card_str[0]
            s = card_str[1:].lower() if len(card_str) > 1 else ""
            
        if r not in HandEvaluator.RANKS: return None, None
        return HandEvaluator.RANKS.index(r) + 2, s

    @staticmethod
    def evaluate(hero_cards, board_cards):
        """Evaluate hand strength, detecting when hero is 'playing the board'."""
        if not hero_cards: return "Unknown"
        
        all_cards = hero_cards + board_cards
        if not all_cards: return "No Cards"
        
        # Parse all cards
        parsed_all = [HandEvaluator.parse_card(c) for c in all_cards]
        parsed_all = [p for p in parsed_all if p[0] is not None]
        
        # Parse hero and board separately for "playing the board" detection
        hero_parsed = [HandEvaluator.parse_card(c) for c in hero_cards]
        hero_parsed = [p for p in hero_parsed if p[0] is not None]
        hero_ranks = set(p[0] for p in hero_parsed)
        hero_suits = set(p[1] for p in hero_parsed)
        
        board_parsed = [HandEvaluator.parse_card(c) for c in board_cards]
        board_parsed = [p for p in board_parsed if p[0] is not None]
        board_ranks = [p[0] for p in board_parsed]
        board_suits = [p[1] for p in board_parsed]
        
        if len(parsed_all) < 2: return "High Card"

        ranks = sorted([p[0] for p in parsed_all], reverse=True)
        suits = [p[1] for p in parsed_all]
        
        # ============ FLUSH DETECTION ============
        flush_suit = None
        for s in 'shdc':
            if suits.count(s) >= 5:
                flush_suit = s
                break
        
        # Check if hero contributes to the flush
        hero_contributes_flush = False
        if flush_suit:
            hero_flush_cards = [p for p in hero_parsed if p[1] == flush_suit]
            board_flush_count = board_suits.count(flush_suit)
            # Hero contributes if board has < 5 flush cards OR hero has higher flush card
            if board_flush_count < 5 and len(hero_flush_cards) > 0:
                hero_contributes_flush = True
            elif board_flush_count >= 5:
                # Board has 5+ flush cards - check if hero improves it
                board_flush_ranks = sorted([p[0] for p in board_parsed if p[1] == flush_suit], reverse=True)[:5]
                hero_flush_ranks = [p[0] for p in hero_flush_cards]
                for hr in hero_flush_ranks:
                    if hr > min(board_flush_ranks):
                        hero_contributes_flush = True
                        break
        
        # ============ STRAIGHT DETECTION ============
        unique_ranks = sorted(list(set(ranks)), reverse=True)
        straight_high = None
        
        # Ace low straight check (5, 4, 3, 2, A)
        if {14, 2, 3, 4, 5}.issubset(set(unique_ranks)):
            straight_high = 5
            
        for i in range(len(unique_ranks) - 4):
            window = unique_ranks[i:i+5]
            if window[0] - window[4] == 4:
                straight_high = window[0]
                break
        
        # Check if hero contributes to the straight
        hero_contributes_straight = False
        if straight_high:
            board_unique = sorted(list(set(board_ranks)), reverse=True)
            # Check if board alone makes the straight
            board_has_straight = False
            if {14, 2, 3, 4, 5}.issubset(set(board_unique)):
                board_has_straight = True
            for i in range(len(board_unique) - 4):
                if board_unique[i] - board_unique[i+4] == 4:
                    board_has_straight = True
                    break
            
            if not board_has_straight:
                hero_contributes_straight = True
            else:
                # Board has a straight - check if hero improves it
                for hr in hero_ranks:
                    if hr > straight_high and hr == straight_high + 1:
                        hero_contributes_straight = True
                        break
                        
        # ============ STRAIGHT FLUSH ============
        if flush_suit and straight_high:
            flush_cards = [p[0] for p in parsed_all if p[1] == flush_suit]
            flush_ranks_sorted = sorted(list(set(flush_cards)), reverse=True)
            sf_high = None
            if {14, 2, 3, 4, 5}.issubset(set(flush_ranks_sorted)): sf_high = 5
            for i in range(len(flush_ranks_sorted) - 4):
                if flush_ranks_sorted[i] - flush_ranks_sorted[i+4] == 4:
                    sf_high = flush_ranks_sorted[i]
                    break
            
            if sf_high:
                # Check if board alone has straight flush
                board_flush = [p[0] for p in board_parsed if p[1] == flush_suit]
                board_flush_sorted = sorted(list(set(board_flush)), reverse=True)
                board_sf = None
                if len(board_flush_sorted) >= 5:
                    if {14, 2, 3, 4, 5}.issubset(set(board_flush_sorted)): board_sf = 5
                    for i in range(len(board_flush_sorted) - 4):
                        if board_flush_sorted[i] - board_flush_sorted[i+4] == 4:
                            board_sf = board_flush_sorted[i]
                            break
                
                if board_sf and board_sf >= sf_high:
                    return f"Nothing (Board Straight Flush {HandEvaluator.rank_name(board_sf)} High)"
                return f"Straight Flush ({HandEvaluator.rank_name(sf_high)} High)"

        # ============ QUADS ============
        rank_counts = {r: ranks.count(r) for r in set(ranks)}
        quads = [r for r, c in rank_counts.items() if c == 4]
        if quads:
            quad_rank = max(quads)
            if board_ranks.count(quad_rank) == 4:
                return f"Nothing (Board Quads {HandEvaluator.rank_name(quad_rank)}s)"
            return f"Four of a Kind ({HandEvaluator.rank_name(quad_rank)}s)"
        
        # ============ FULL HOUSE ============
        trips = sorted([r for r, c in rank_counts.items() if c >= 3], reverse=True)
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        
        if trips:
            high_trip = trips[0]
            if len(trips) > 1 or pairs:
                # Check if board alone has full house
                board_counts = {r: board_ranks.count(r) for r in set(board_ranks)}
                board_trips = [r for r, c in board_counts.items() if c >= 3]
                board_pairs = [r for r, c in board_counts.items() if c == 2]
                
                if board_trips and (len(board_trips) > 1 or board_pairs):
                    if high_trip in board_trips:
                        return f"Nothing (Board Full House {HandEvaluator.rank_name(high_trip)}s full)"
                
                return f"Full House ({HandEvaluator.rank_name(high_trip)}s full)"
        
        # ============ FLUSH ============
        if flush_suit:
            if not hero_contributes_flush:
                return f"Nothing (Board Flush)"
            # Get hero's highest flush card
            hero_flush = [p[0] for p in hero_parsed if p[1] == flush_suit]
            if hero_flush:
                return f"Flush ({HandEvaluator.rank_name(max(hero_flush))} High)"
            return "Flush"
        
        # ============ STRAIGHT ============
        if straight_high:
            if not hero_contributes_straight:
                return f"Nothing (Board Straight {HandEvaluator.rank_name(straight_high)} High)"
            return f"Straight ({HandEvaluator.rank_name(straight_high)} High)"
        
        # ============ TRIPS ============
        if trips:
            trip_rank = trips[0]
            if board_ranks.count(trip_rank) == 3:
                return f"Nothing (Board Trips {HandEvaluator.rank_name(trip_rank)}s)"
            return f"Three of a Kind ({HandEvaluator.rank_name(trip_rank)}s)"
        
        # ============ TWO PAIR ============
        if len(pairs) >= 2:
            high_pair = pairs[0]
            low_pair = pairs[1]
            # Check if both pairs are on board
            if board_ranks.count(high_pair) >= 2 and board_ranks.count(low_pair) >= 2:
                return f"Nothing (Board Two Pair)"
            return f"Two Pair ({HandEvaluator.rank_name(high_pair)}s & {HandEvaluator.rank_name(low_pair)}s)"
        
        # ============ ONE PAIR ============
        if pairs:
            pair_rank = pairs[0]
            if board_ranks.count(pair_rank) >= 2:
                return f"Nothing (Board Pair {HandEvaluator.rank_name(pair_rank)}s)"
            return f"Pair of {HandEvaluator.rank_name(pair_rank)}s"
        
        # ============ HIGH CARD ============
        high_rank = ranks[0]
        hero_high = max(hero_ranks) if hero_ranks else 0
        
        if high_rank > hero_high:
            return f"Nothing (Board {HandEvaluator.rank_name(high_rank)} High)"
        
        return f"High Card ({HandEvaluator.rank_name(hero_high)})"

    @staticmethod
    def rank_name(r):
        return {11:'J', 12:'Q', 13:'K', 14:'A'}.get(r, str(r))


# Global Settings
PROMPT_MODE = "FAST" # Options: "COACH", "FAST" - FAST for quick decisions

def generate_strategy_prompt_fast(history_json: str, action_history: str, villain_stats: str, 
                             snapshot: GameSnapshot, metrics: dict, current_action: str, hand_rank: str) -> str:
    """Compact prompt for fast, decisive answers at micro-stakes."""
    
    hero = next((p for p in snapshot.players if p.is_hero), None)
    hero_cards = " ".join(hero.hole_cards) if hero and hero.hole_cards else "Unknown"
    hero_pos = hero.name if hero else "Unknown"
    board = " ".join(snapshot.board_state.community_cards) or "Preflop"
    street = snapshot.meta_info.current_street
    final_pot = metrics.get("final_pot", 0.0)
    spr = metrics.get("spr", 0.0)
    call_amount = snapshot.last_action_context.amount_to_call
    
    # Aggressor
    agg_name = "Unknown"
    if snapshot.last_action_context.aggressor_seat_index != -1:
        p = next((p for p in snapshot.players if p.seat_index == snapshot.last_action_context.aggressor_seat_index), None)
        if p: agg_name = p.username

    # Position tag
    pos_tag = "IP" if hero_pos in ["BTN", "CO"] else "OOP"
    
    # Bet size category
    bet_cat = ""
    if call_amount > 0 and final_pot > 0:
        ratio = call_amount / final_pot
        if ratio > 0.8: bet_cat = "[OVERBET=STRONG]"
        elif ratio > 0.5: bet_cat = "[BIG BET]"

    return f"""
MICRO-STAKES WINNING STRATEGY (NL2-NL10). One decision only.

STATE: {street} | {hero_cards} ({hero_pos}/{pos_tag}) | Board: {board}
HAND: {hand_rank} | Pot: {final_pot:.0f}BB | SPR: {spr:.1f}
FACING: {current_action} {bet_cat}
VILLAIN: {villain_stats}

═══ WINNING AT MICRO-STAKES ═══

PREFLOP:
- Open 3x with pairs, broadways, suited connectors in position
- 3-bet premium hands (QQ+, AK) for value, NOT as bluff
- Call raises with implied odds hands (small pairs, suited connectors) if SPR > 10
- FOLD trash. Don't defend blinds wide vs raises.

POSTFLOP OFFENSE (How to WIN pots):
- C-bet 50-66% pot on dry boards (K72r, A83r) with any 2 cards
- Value bet 66-80% pot with top pair+ vs calling stations
- Bet THREE streets for value with 2pair+ (they call with worse)
- Size UP with strong hands - fish call any size
- River: Go for thin value (top pair good kicker vs fish)

POSTFLOP DEFENSE (How to NOT LOSE stacks):
- Big bet from passive player = FOLD all but monsters
- "Nothing" hand vs any bet = FOLD (no bluff catching vs fish)
- SPR < 1 = Only stack off with top 10% hands
- Don't call to "keep them honest" - they're not bluffing

POSITION RULES:
- IP (BTN/CO): Bet for thin value, call wider, bluff occasionally
- OOP (SB/BB): Check strong hands to trap, bet only for value

OUTPUT:
**Action:** [Fold/Check/Call/Bet/Raise]
**Size:** [BB or 0]
**Why:** [One sentence max - the key reason]
"""


def generate_strategy_prompt(history_json: str, action_history: str, villain_stats: str, 
                             snapshot: GameSnapshot, metrics: dict, current_action: str, hand_rank: str) -> str:
    """Constructs the final prompt for the Strategy LLM with enhanced decision logic."""
    
    if PROMPT_MODE == "FAST":
        return generate_strategy_prompt_fast(history_json, action_history, villain_stats, snapshot, metrics, current_action, hand_rank)

    hero = next((p for p in snapshot.players if p.is_hero), None)
    hero_cards = " ".join(hero.hole_cards) if hero and hero.hole_cards else "Unknown"
    hero_pos = hero.name if hero else "Unknown"
    board = " ".join(snapshot.board_state.community_cards) or "Preflop"
    street = snapshot.meta_info.current_street
    
    final_pot = metrics.get("final_pot", snapshot.board_state.total_pot)
    eff_stack = metrics.get("eff_stack", 0.0)
    spr = metrics.get("spr", 0.0)
    odds_str = metrics.get("pot_odds_str", "N/A")
    call_amount = snapshot.last_action_context.amount_to_call

    # Find Aggressor Name and Stats
    aggressor_name = "Unknown"
    agg_idx = snapshot.last_action_context.aggressor_seat_index
    if agg_idx != -1:
        agg_p = next((p for p in snapshot.players if p.seat_index == agg_idx), None)
        if agg_p:
            aggressor_name = agg_p.username or agg_p.name
    
    # Calculate bet sizing category
    bet_sizing_category = "N/A"
    if call_amount > 0 and final_pot > 0:
        bet_ratio = call_amount / (final_pot - call_amount) if final_pot > call_amount else call_amount / final_pot
        if bet_ratio < 0.4:
            bet_sizing_category = "SMALL (< 40% pot) - Often weak or blocking bet"
        elif bet_ratio < 0.75:
            bet_sizing_category = "MEDIUM (40-75% pot) - Standard value/bluff"
        elif bet_ratio < 1.2:
            bet_sizing_category = "LARGE (75-120% pot) - Polarized range"
        else:
            bet_sizing_category = "OVERBET (> pot) - Very strong or bluff, rarely medium strength"

    # Position context
    position_context = ""
    if hero_pos in ["BTN", "CO"]:
        position_context = "IN POSITION (IP) - Can control pot, bluff more, call wider"
    elif hero_pos in ["SB", "BB"]:
        position_context = "OUT OF POSITION (OOP) - Play tighter, check-raise strong hands, avoid bloating pot with marginal hands"
    elif hero_pos in ["UTG", "MP"]:
        position_context = "EARLY POSITION - Play tight, strongest ranges only"

    return f"""
ROLE: Expert Micro-Stakes Poker Coach (NL2-NL10 Specialist).
OBJECTIVE: Maximize EV while PROTECTING STACK. Provide optimal decision for {street}.

GAME CONTEXT:
- This is LOW STAKES (NL2-NL10 online poker)
- Villains are often passive calling stations who don't fold
- Big bets usually mean BIG HANDS (don't hero call)
- Bluffing rarely works vs fish - prefer value betting thin

═══════════════════════════════════════════════════════════════
[CURRENT STATE]
Street: {street}
Board: {board}
Hero: {hero_cards} ({hero_pos})
Hand Strength: {hand_rank}
Pot: {final_pot:.2f} BB
Facing: {current_action}
Aggressor: {aggressor_name}

[POSITION]
{position_context}

[STACK METRICS]
Effective Stack: {eff_stack:.1f} BB
SPR: {spr:.2f}
Pot Odds: {odds_str}
Bet Sizing: {bet_sizing_category}

[VILLAIN READS]
{villain_stats}
INTERPRETATION:
- VPIP > 50% = Fish/Calling Station (don't bluff, value bet relentlessly)
- VPIP < 20% = Nit (respect their bets, fold marginal hands)
- PFR close to VPIP = Aggressive (can 3-bet bluff, but respect postflop aggression)
- PFR = 0% or very low = Passive (their bets/raises = STRONG hands)

[HAND HISTORY]
{action_history}
═══════════════════════════════════════════════════════════════

DECISION FRAMEWORK:

1. **STACK PRESERVATION (CRITICAL)**
   - SPR < 1: You are committed. Only continue with TOP of range.
   - SPR < 3: One bet commits you. Big decisions = need strong hands.
   - SPR > 10: Deep stacked. Can play speculative hands, implied odds matter.
   - If facing > 50% of your stack: Need VERY strong hand to continue.

2. **RELATIVE HAND STRENGTH**
   - "Flush" on 4-flush board = Check if your flush card is low (beaten by higher flush)
   - "Two Pair" on paired board = Possible full house for villain
   - "Top Pair" facing big bet = Often just a bluff-catcher
   - Ask: "What hands does villain bet this big that I BEAT?"

3. **POSITION ADJUSTMENTS**
   - OOP (SB/BB): Check-call or check-raise more, lead less
   - IP (BTN/CO): Can bet for thin value, bluff catch profitably
   - OOP vs IP aggression: Require stronger hands to continue

4. **LOW STAKES EXPLOITS**
   - Villain bets BIG = They have it. Don't be a hero.
   - Villain checks twice = They're weak. Bet for value or bluff.
   - Passive villain suddenly raises = FOLD (unless you have the nuts).
   - Calling stations: Never bluff, value bet any pair+

5. **STREET-SPECIFIC LOGIC**
   - PREFLOP: Position > Cards. 3-bet IP, flat OOP.
   - FLOP: Bet strong draws and value. Check/fold air.
   - TURN: Polarize. Bet your value and bluffs, check SDV.
   - RIVER: NO MORE CARDS. Value bet or bluff, don't "block bet."

OUTPUT FORMAT (Strict):

### DECISION
**Action:** [Check/Call/Fold/Bet/Raise]
**Amount:** [Specific BB amount, or 0 if Check/Fold]

### REASONING
*   **Hand Class:** [Nuts / Strong Value / Marginal / Bluff-Catcher / Air]
*   **Villain Read:** [What their action tells us in 1 sentence]
*   **Risk Assessment:** [Stack preservation consideration in 1 sentence]
*   **Final Logic:** [Why this action wins you money or saves your stack]
"""


def run_analysis_flow(mode: str = "debug"):
    """
    Main Logic Flow:
    1. debug -> Capture + Vision + Accumulate History + Display Table
    2. strategy -> Use accumulated history for strategy advice
    """
    global last_state, current_history
    
    start_total = time.time()
    
    if mode == "debug":
        console.print("\n[dim]📸 SNAPSHOT (Vision Check)...[/dim]")
    else:
        console.print("\n[dim]🧠 STRATEGY (Thinking)...[/dim]")
    
    try:
        # 1. Capture & Vision
        snapshot, comps, t_capture, t_vision = analyze_state(monitor_num)
        
        # --- ID CONSISTENCY (Fix for Session Stats) ---
        # If we are already tracking a hand, ensure the new snapshot shares the SAME ID.
        if len(current_history.snapshots) > 0 and current_history.hand_id:
            snapshot.hand_id = current_history.hand_id
        
        # --- BOARD PERSISTENCE (Anti-Hallucination) ---
        if len(current_history.snapshots) > 0:
            prev_snapshot = current_history.snapshots[-1]
            prev_board = prev_snapshot.board_state.community_cards
            curr_board = snapshot.board_state.community_cards
            
            # If we have cards from previous turns, they shouldn't change
            if len(curr_board) >= len(prev_board) and len(prev_board) > 0:
                # Keep the old cards as they were, only take the NEW cards from the current vision
                new_cards = curr_board[len(prev_board):]
                snapshot.board_state.community_cards = prev_board + new_cards

        last_state = snapshot
        
        # 2. Hand History - Always add snapshot (user presses 'n' for new hand)
        current_history.add_snapshot(snapshot)
        recorder.update(snapshot) # Log to file
        console.print(f"[dim]📝 Turn {len(current_history.snapshots)} added to history[/dim]")
        
        analysis = ""
        t_strategy = 0.0
        metrics = None
        hand_rank = ""  # Initialize here to avoid reference error
        
        # 3. Strategy (Optional) - Uses FULL HISTORY
        if mode == "strategy":
            hero = next((p for p in snapshot.players if p.is_hero), None)
            
            if not hero or not hero.hole_cards:
                analysis = "No Hero Hand Detected."
            else:
                t2 = time.time()
                
                # --- INTELLIGENT POT TRACKING ---
                # Fixes vision glitches where pot size momentarily drops (e.g. 8.0 -> 3.5)
                current_street = snapshot.meta_info.current_street
                max_seen_pot = 0.0
                
                # Scan history for the highest pot value recorded on THIS street
                for s in current_history.snapshots:
                    if s.meta_info.current_street == current_street:
                         if s.board_state.total_pot > max_seen_pot:
                             max_seen_pot = s.board_state.total_pot
                
                # If current snapshot has a valid pot, compare it too
                if snapshot.board_state.total_pot > max_seen_pot:
                    max_seen_pot = snapshot.board_state.total_pot
                
                # Calculate Effective Pot (Center + Active Bets)
                current_bets = sum(p.current_bet for p in snapshot.players)
                final_pot = max_seen_pot + current_bets
                
                # --- PYTHON MATH LAYER (Derived Metrics) ---
                # 1. Effective Stack (Hero vs Deepest Active Villain)
                active_villains = [p for p in snapshot.players if not p.is_hero and p.status in ["ACTIVE", "ALL_IN"]]
                max_villain_stack = max((p.stack_size for p in active_villains), default=0.0)
                eff_stack = min(hero.stack_size, max_villain_stack) if active_villains else hero.stack_size
                
                # 2. SPR (Stack-to-Pot Ratio)
                spr = (eff_stack / final_pot) if final_pot > 0 else 0.0
                
                # 3. Pot Odds
                call_amount = snapshot.last_action_context.amount_to_call
                if call_amount > 0:
                    # Risk / (Reward + Risk)
                    pot_odds_pct = (call_amount / (final_pot + call_amount)) * 100
                    pot_odds_ratio = f"{(final_pot / call_amount):.1f}:1"
                    pot_odds_str = f"{pot_odds_pct:.1f}% ({pot_odds_ratio})"
                else:
                    pot_odds_pct = 0.0
                    pot_odds_ratio = "N/A"
                    pot_odds_str = "N/A (Facing 0 bet)"
                
                # Store metrics for display and prompt
                metrics = {
                    "final_pot": final_pot,
                    "eff_stack": eff_stack,
                    "spr": spr,
                    "pot_odds_pct": pot_odds_pct,
                    "pot_odds_ratio": pot_odds_ratio,
                    "pot_odds_str": pot_odds_str
                }

                # --- VILLAIN PROFILING (DISABLED FOR SPEED) ---
                # history_hands = load_all_sessions()
                # save_villain_db(history_hands)
                # villain_profiles = []
                # for p in snapshot.players:
                #     if p.is_hero or p.username in ["biba287"]:
                #         continue
                #     if p.status in ["ACTIVE", "ALL_IN"]:
                #         lookup_name = p.username if p.username else p.name
                #         if not is_valid_username(lookup_name):
                #             continue
                #         stats = calculate_villain_stats(lookup_name, history_hands, min_samples=10)
                #         if stats:
                #             villain_profiles.append(f"- {lookup_name} ({p.name}): {stats}")
                # villain_context = "\n".join(villain_profiles) if villain_profiles else "- Unknown/Random"
                
                villain_context = ""  # Disabled for speed

                # Define current action for the prompt
                current_action = f"Facing bet/raise of {call_amount} BB" if call_amount > 0 else "Checked to Hero"
                
                # Calculate Hand Rank
                hand_rank = HandEvaluator.evaluate(hero.hole_cards, snapshot.board_state.community_cards)

                # Build strategy prompt
                history_full = current_history.to_json()
                history_min = current_history.to_min_json()
                action_summary = parse_action_history(history_full)
                
                strategy_prompt = generate_strategy_prompt(
                    history_min, action_summary, villain_context, 
                    snapshot, metrics, current_action, hand_rank
                )
                
                # Debug prompt
                with open("debug_strategy_prompt.txt", "w") as f:
                    f.write(strategy_prompt)
                
                try:
                    if openai_client:
                        resp = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "ROLE: Expert Poker Assistant & GTO Solver. OBJECTIVE: Analyze the hand and provide the EV-maximizing decision."},
                                {"role": "user", "content": strategy_prompt}
                            ],
                            temperature=0.1
                        )
                        analysis = resp.choices[0].message.content.strip()
                    else:
                        analysis = "Strategy Error: OPENAI_API_KEY missing."
                except Exception as e:
                    analysis = f"Strategy Error: {e}"
                t_strategy = time.time() - t2

        t_total = time.time() - start_total
        
        # 4. Display
        display_results(snapshot, analysis, t_capture, t_vision, t_strategy, t_total, metrics, hand_rank)
        
        # Show history summary
        # console.print(f"\n[dim]{current_history.summary()}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")



def on_press(key):
    global running, input_mode, note_player
    try:
        if hasattr(key, 'char') and key.char:
            char = key.char.lower()
            
            if char == 'j':
                # J = Strategy Mode
                run_analysis_flow(mode="strategy")
                
            elif char == 'l':
                # L = Debug Snapshot (Vision Check)
                run_analysis_flow(mode="debug")
                    
            elif char == 'p':
                # Show preflop chart for last detected position
                pos = None
                if last_state:
                     hero = next((p for p in last_state.players if p.is_hero), None)
                     pos = hero.name if hero else None
                show_preflop_chart(pos)
            elif char == 'n':
                # Reset hand history (NEW HAND)
                current_history.reset()
                console.print("\n[bold green]🆕 HAND HISTORY RESET - Ready for new hand![/bold green]")
            elif char == 'm':
                # Toggle Prompt Mode
                global PROMPT_MODE
                if PROMPT_MODE == "COACH":
                    PROMPT_MODE = "FAST"
                else:
                    PROMPT_MODE = "COACH"
                console.print(f"\n[bold magenta]Mode Switched: {PROMPT_MODE}[/bold magenta]")
            elif char == 'q':
                running = False
                return False
                
        elif key == keyboard.Key.esc:
            running = False
            return False
    except AttributeError:
        pass


def main():
    global monitor_num
    
    # Parse monitor arg
    if len(sys.argv) > 1:
        try:
            monitor_num = int(sys.argv[1])
        except ValueError:
            pass
    
    # Show monitors
    with mss.mss() as sct:
        console.print(f"[cyan]Monitors: {len(sct.monitors) - 1}[/cyan]")
        for i, m in enumerate(sct.monitors[1:], 1):
            console.print(f"  {i}: {m['width']}x{m['height']}")
    
    console.print(f"\n[bold green]♠ Poker Range Assistant ♠[/bold green]")
    console.print(f"[white]Monitor: {monitor_num}[/white]")
    console.print(f"[yellow]'l' SNAPSHOT | 'j' STRATEGY | 'n' NEW HAND | 'p' CHART | ESC quit[/yellow]\n")
    
    # Start keyboard listener
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    
    console.print("\n[green]Good luck at the tables! 🍀[/green]")


if __name__ == "__main__":
    main()

