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
from pynput import keyboard
import google.generativeai as genai

load_dotenv()
console = Console()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    console.print("[red]Error: GEMINI_API_KEY not found in .env[/red]")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Dual Models
# Switching to 2.0 Flash for Vision as well for speed (3.0 Preview is high latency)
vision_model = genai.GenerativeModel("gemini-2.0-flash-exp")    
strategy_model = genai.GenerativeModel("gemini-2.0-flash")

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
    name: str = ""
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
- You are analyzing individual seat crops (Images 1-6), the Board (Image 7), and Actions (Image 8).
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
   - **STACK SIZE (CRITICAL - CONVERT TO BB):** 
     - Extract the number inside the player pod.
     - **IF CURRENCY DETECTED (e.g. "$2.00", "€5.00"):** Convert to BBs assuming Big Blind = 0.02 (e.g. 2.00 -> 100 BB).
     - **IF NUMBER ONLY:** Assume it is already in BBs.
   - **CURRENT BET (CONVERT TO BB):** 
     - Look for the bet bubble/pill. Apply same conversion rules (Currency -> BB).
     - *Crucial:* Check for this bubble for ALL players.
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
    def __init__(self):
        self.filename = f"session_{int(time.time())}.jsonl" # JSON Lines formatted
        self.log(f"Session Started: {self.filename}", is_meta=True)
    
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
        # (This is done in log)



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
    
    # Prepare images for Gemini
    # Images 1-6: Seats 1-6.
    # Image 7: Board.
    # Image 8: Pot Info (Reuse board image).
    
    seat_imgs = [
        comps["seat1"], comps["seat2"], comps["seat3"], 
        comps["seat4"], comps["hero"],  comps["seat6"]
    ]
    
    vision_input = [ANALYZE_PROMPT] + seat_imgs + [comps["board"], comps["buttons"]] 
    
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


def analyze_and_display(monitor: int):
    """Main function for single-shot analysis (Hotkey J)."""
    global last_state, live_mode, live_builder
    
    start_total = time.time()
    console.print("\n[dim]Analyzing (Full 6-Max Grid)...[/dim]")
    
    # MODE A: LIVE DATA (Strategy on existing history)
    if live_mode and live_builder:
        # Use simple Live Logic for now until fully refactored
        console.print("[dim]Using Live Data...[/dim]")
        state = live_state
        if not state: 
            return
    else:
        # MODE B: SNAPSHOT
        try:
            state, comps, t_capture, t_vision = analyze_state(monitor)
            last_state = state
            
            hero = next((p for p in state.players if p.is_hero), None)
            if not hero or not hero.hole_cards:
                 console.print("[yellow]No hand detected.[/yellow]")
            else:
                # Strategy logic needs update to accept GameSnapshot?
                # For now pass the whole snapshot to a prompt builder?
                # or just display.
                # Let's keep Strategy simple for a moment.
                pass
                
            t_total = time.time() - start_total
            display_results(state, "", t_capture, t_vision, 0, t_total)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def display_results(snapshot: GameSnapshot, analysis: str, t_cap, t_vis, t_strat, t_tot, metrics: dict = None):
    """Comprehensive display of GameSnapshot data."""
    console.clear()
    
    # 1. EXTRACT RECOMMENDATION (At a glance!)
    recommendation = "Analyzing..."
    details = analysis
    
    if analysis:
        lines = analysis.split('\n')
        if lines and "RECOMMENDATION" in lines[0].upper():
            recommendation = lines[0]
            details = "\n".join(lines[1:])
        else:
            # Fallback if model didn't follow strict format
            recommendation = lines[0] if lines else "No advice."
            details = "\n".join(lines[1:])

    # Header
    console.print(f"\n[dim]ID: {snapshot.hand_id} | {t_tot:.2f}s[/dim]")

    # ★★★ BIG SUGGESTION BOX ★★★
    if analysis:
        style = "bold white on red"
        if "FOLD" in recommendation.upper(): style = "bold white on red"
        elif "CHECK" in recommendation.upper(): style = "bold black on yellow"
        elif "CALL" in recommendation.upper(): style = "bold black on yellow"
        elif "RAISE" in recommendation.upper(): style = "bold white on green"
        elif "BET" in recommendation.upper(): style = "bold white on green"
        
        console.print()
        console.print(f" {recommendation} ", style=style, justify="center")
        console.print()

    # ----- COMPACT HUD -----
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)
    
    # Hero Hand
    hero = next((p for p in snapshot.players if p.is_hero), None)
    hero_cards = " ".join(hero.hole_cards) if hero and hero.hole_cards else "—"
    
    # Board
    board_str = " ".join(snapshot.board_state.community_cards) or "Preflop"
    
    # Pot
    pot = f"{snapshot.board_state.total_pot} BB"
    
    grid.add_row(
        f"[bold cyan]HERO[/bold cyan]\n[bold white]{hero_cards}[/bold white]",
        f"[bold cyan]BOARD ({snapshot.meta_info.current_street})[/bold cyan]\n[bold white]{board_str}[/bold white]",
        f"[bold cyan]POT[/bold cyan]\n[bold yellow]{pot}[/bold yellow]"
    )
    
    # Metrics Row
    if metrics:
        spr = metrics.get("spr", 0)
        odds_pct = metrics.get("pot_odds_pct", 0)
        odds_ratio = metrics.get("pot_odds_ratio", "N/A")
        eff = metrics.get("eff_stack", 0)
        
        spr_color = "green" if spr > 10 else "yellow" if spr > 3 else "red"
        
        grid.add_row(
            f"[dim]Eff Stack:[/dim] [white]{eff:.1f} BB[/white]",
            f"[dim]SPR:[/dim] [{spr_color}]{spr:.2f}[/{spr_color}]",
            f"[dim]Odds:[/dim] [cyan]{odds_pct:.1f}% ({odds_ratio})[/cyan]"
        )

    console.print(grid)
    console.print()

    # ----- STRATEGY DETAILS -----
    if details:
        console.print(f"[dim]Analysis:[/dim]")
        console.print(details.strip())
        console.print()
    
    # ----- COMPACT PLAYER TABLE (Background info) -----
    # Only show if requested or in debug, or make it very small
    # console.print(f"[dim]Active Players:[/dim]")
    table = Table(show_header=True, header_style="dim", box=None, padding=(0,1))
    table.add_column("Seat", style="dim", width=4)
    table.add_column("Pos", style="magenta", width=6)
    table.add_column("Stack", justify="right", width=8)
    table.add_column("Act", width=12) # Action/Status
    
    for p in sorted(snapshot.players, key=lambda x: x.seat_index):
        if p.status == "SITTING_OUT": continue
        
        seat_num = str(p.seat_index + 1)
        pos = p.name or "?"
        stack = f"{p.stack_size:.0f}"
        
        # Status/Action
        if p.is_hero: act = "[bold yellow]HERO[/bold yellow]"
        elif p.status == "FOLDED": act = "[dim]Fold[/dim]"
        elif p.current_bet > 0: act = f"[red]Bet {p.current_bet}[/red]"
        else: act = "[green]Check/Wait[/green]"
        
        if p.is_dealer: pos += " (D)"
        
        table.add_row(seat_num, pos, stack, act)
    
    console.print(table)
    console.print(f"[dim]════════════════════════════════[/dim]")









# Global state
monitor_num = 1
running = True
last_state = None
last_captures = None
input_mode = None
note_player = ""
current_history = HandHistory()  # Accumulates snapshots for current hand
recorder = GameRecorder() # Enable logging


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
                
                # Store metrics for display
                metrics = {
                    "eff_stack": eff_stack,
                    "spr": spr,
                    "pot_odds_pct": pot_odds_pct,
                    "pot_odds_ratio": pot_odds_ratio
                }

                # Define current action for the prompt
                current_action = f"Facing bet/raise of {call_amount} BB" if call_amount > 0 else "Checked to Hero"

                # Build strategy prompt with FULL HISTORY
                history_json = current_history.to_json()
                strategy_prompt = f"""
                You are a professional poker strategist. 
                
                FULL HAND HISTORY:
                {history_json}
                
                CURRENT SITUATION:
                - Board: {" ".join(snapshot.board_state.community_cards)}
                - Hero: {" ".join(hero.hole_cards)} ({hero.name})
                - Position: {hero.name}
                - Pot: {final_pot:.2f} BB
                - Call: {call_amount} BB
                - Action: {current_action}
                - Villain Profile: Unknown/Random
                
                MATH & METRICS (Calculated):
                - Effective Stack: {eff_stack:.1f} BB
                - SPR: {spr:.2f}
                - Pot Odds: {pot_odds_str}
                
                OUTPUT FORMAT (STRICT):
                Line 1: RECOMMENDATION: [ACTION] [SIZING] (e.g. "RECOMMENDATION: FOLD")
                Line 2+: Max 3 bullet points of KEY reasoning.
                
                DO NOT summarize the hand history.
                DO NOT estimate specific equity percentages.
                BE INSTANT.
                """
                
                # Debug prompt
                with open("debug_strategy_prompt.txt", "w") as f:
                    f.write(strategy_prompt)
                
                try:
                    resp = strategy_model.generate_content(strategy_prompt)
                    analysis = resp.text.strip()
                except Exception as e:
                    analysis = f"Strategy Error: {e}"
                t_strategy = time.time() - t2

        t_total = time.time() - start_total
        
        # 4. Display
        display_results(snapshot, analysis, t_capture, t_vision, t_strategy, t_total, metrics)
        
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

