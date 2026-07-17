#!/usr/bin/env python3
"""Poker Range Assistant for a user-controlled poker simulator.

Press ``j`` to analyze, ``p`` for a preflop chart, ``n`` for a new hand, and
Escape to exit.
"""

import os
import sys
import json
import time
import io
import base64
import itertools
import math
import re
import threading
import dataclasses
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
import cv2
import numpy as np
from dotenv import load_dotenv
import mss
from PIL import Image, ImageChops, ImageDraw, ImageFont
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich import box
from pynput import keyboard
from google import genai
from google.genai import types
from anthropic import Anthropic
from live_gto import (
    LiveDecisionState,
    LiveGTOConfig,
    LiveGTOConfigurationError,
    LiveGTOStatus,
    LiveGTORouter,
)
from preflop_observation import (
    PreflopObservationError,
    current_decision as current_preflop_observation,
    terminal_from_history as terminal_preflop_observation,
)

load_dotenv()
console = Console()

# Configure Gemini (Vision)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_MEDIA_RESOLUTION = os.getenv("GEMINI_MEDIA_RESOLUTION", "high").lower()
VISION_LAYOUT = os.getenv("VISION_LAYOUT", "mosaic").lower()
SAVE_DEBUG_IMAGES = os.getenv("SAVE_DEBUG_IMAGES", "0").lower() in {
    "1", "true", "yes", "on"
}
OFFLINE_STRATEGY_ONLY = os.getenv("POKER_ASSISTANT_OFFLINE", "0").lower() in {
    "1", "true", "yes", "on"
}
FAST_REQUEST_TIMEOUT_SECONDS = max(
    12.0,
    float(
        os.getenv(
            "FAST_REQUEST_TIMEOUT_SECONDS",
            os.getenv("FAST_PARALLEL_TIMEOUT_SECONDS", "12.0"),
        )
    ),
)
FAST_PROVIDER_TIMEOUT_SECONDS = max(
    0.25,
    FAST_REQUEST_TIMEOUT_SECONDS - 0.5,
)
GEMINI_TIMEOUT_MS = min(
    max(10000, int(os.getenv("GEMINI_TIMEOUT_MS", "10000"))),
    int(FAST_PROVIDER_TIMEOUT_SECONDS * 1000),
)
if not GEMINI_API_KEY and not OFFLINE_STRATEGY_ONLY:
    console.print("[red]Error: GEMINI_API_KEY not found in .env[/red]")
    sys.exit(1)

gemini_client = (
    genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            timeout=GEMINI_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    if GEMINI_API_KEY
    else None
)

# Configure Claude (Strategy)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HERO_USERNAME = os.getenv("HERO_USERNAME", "biba287")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
CLAUDE_FAST_MODEL = os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-4-5")
CLAUDE_FAST_TIMEOUT_SECONDS = min(
    max(0.25, float(os.getenv("CLAUDE_FAST_TIMEOUT_SECONDS", "6.5"))),
    FAST_PROVIDER_TIMEOUT_SECONDS,
)
CLAUDE_COACH_TIMEOUT_SECONDS = float(
    os.getenv("CLAUDE_COACH_TIMEOUT_SECONDS", "15.0")
)
if not ANTHROPIC_API_KEY:
    console.print("[yellow]Warning: ANTHROPIC_API_KEY not found in .env (Strategy will fail)[/yellow]")

anthropic_client = (
    Anthropic(
        api_key=ANTHROPIC_API_KEY,
        timeout=CLAUDE_COACH_TIMEOUT_SECONDS,
        max_retries=0,
    )
    if ANTHROPIC_API_KEY
    else None
)

# Strategy backend. GTO is opt-in and requires a truthful acknowledgement that
# the target is a user-controlled simulator. HYBRID tries the exact supported
# HU/OOP street-root node first and falls back to Claude everywhere else.
STRATEGY_BACKEND = os.getenv("STRATEGY_BACKEND", "CLAUDE").strip().upper()
if STRATEGY_BACKEND not in {"CLAUDE", "HYBRID", "GTO"}:
    console.print(
        f"[yellow]Warning: unknown STRATEGY_BACKEND={STRATEGY_BACKEND!r}; "
        "using CLAUDE[/yellow]"
    )
    STRATEGY_BACKEND = "CLAUDE"

live_gto_router = None
live_gto_config_error = ""
try:
    _live_gto_config = LiveGTOConfig.from_env(os.path.dirname(__file__))
    if STRATEGY_BACKEND in {"HYBRID", "GTO"}:
        live_gto_router = LiveGTORouter(_live_gto_config)
except LiveGTOConfigurationError as error:
    live_gto_config_error = str(error)
    console.print(f"[yellow]Warning: live GTO disabled: {error}[/yellow]")

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
    status: str = "FOLDED" # ACTIVE, FOLDED, ALL_IN, SITTING_OUT, EMPTY
    is_hero: bool = False
    is_dealer: bool = False # Helper field
    visible_action: str = "" # CHECK, CALL, BET, RAISE, FOLD when visibly overlaid
    cards_confirmed_locally: bool = False

@dataclass
class LastActionContext:
    aggressor_seat_index: int = -1
    last_aggressive_action: str = "NONE" # BET, RAISE, CHECK
    amount_to_call: float = 0.0
    hero_action_options: list = dataclasses.field(default_factory=list)

@dataclass
class GameSnapshot:
    hand_id: str = ""
    timestamp: str = ""
    meta_info: MetaInfo = dataclasses.field(default_factory=MetaInfo)
    board_state: BoardState = dataclasses.field(default_factory=BoardState)
    dealer_seat_index: int = -1
    action_on_seat_index: int = -1
    players: list[Player] = dataclasses.field(default_factory=list)
    last_action_context: LastActionContext = dataclasses.field(default_factory=LastActionContext)
    vision_error: str = ""
    
    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=2)


def confirmed_preflop_to_flop_transition(
    previous: GameSnapshot,
    current: GameSnapshot,
) -> bool:
    """Recognize one hand's PREFLOP->FLOP transition despite card OCR noise."""

    if (
        previous.meta_info.current_street != "PREFLOP"
        or current.meta_info.current_street != "FLOP"
        or previous.board_state.community_cards
        or len(current.board_state.community_cards) != 3
        or previous.dealer_seat_index < 0
        or previous.dealer_seat_index != current.dealer_seat_index
    ):
        return False
    previous_hero = next((player for player in previous.players if player.is_hero), None)
    current_hero = next((player for player in current.players if player.is_hero), None)
    if not previous_hero or not current_hero or previous_hero.seat_index != current_hero.seat_index:
        return False

    def occupied_layout(snapshot: GameSnapshot) -> dict[int, str]:
        return {
            player.seat_index: player.name
            for player in snapshot.players
            if player.status not in {"EMPTY", "SITTING_OUT"}
        }

    return occupied_layout(previous) == occupied_layout(current)


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
        
        last_board = last.board_state.community_cards
        current_board = snapshot.board_state.community_cards
        board_continues = bool(
            last_board
            and len(current_board) >= len(last_board)
            and current_board[:len(last_board)] == last_board
        )
        preflop_to_flop_continues = confirmed_preflop_to_flop_transition(
            last, snapshot
        )

        # New hole cards normally identify a new hand. Once the same postflop
        # board is visibly continuing, however, a one-suit OCR disagreement is
        # noise rather than a new deal.
        last_hero = next((p for p in last.players if p.is_hero), None)
        new_hero = next((p for p in snapshot.players if p.is_hero), None)
        if last_hero and new_hero:
            if (
                last_hero.hole_cards
                and new_hero.hole_cards
                and len(last_hero.hole_cards) == 2
                and len(new_hero.hole_cards) == 2
            ):
                if (
                    set(last_hero.hole_cards) != set(new_hero.hole_cards)
                    and not board_continues
                    and not preflop_to_flop_continues
                ):
                    return True

        # A board rollback alone can be one bad OCR frame. Treat it as a new
        # hand only when the dealer also moved.
        board_rolled_back = bool(
            last.board_state.community_cards
            and not snapshot.board_state.community_cards
        )
        dealer_moved = bool(
            last.dealer_seat_index >= 0
            and snapshot.dealer_seat_index >= 0
            and last.dealer_seat_index != snapshot.dealer_seat_index
        )
        if board_rolled_back and dealer_moved:
            return True
        if (
            dealer_moved
            and not last.board_state.community_cards
            and not snapshot.board_state.community_cards
            and last.meta_info.current_street == "PREFLOP"
            and snapshot.meta_info.current_street == "PREFLOP"
        ):
            return True
        if (
            dealer_moved
            and new_hero
            and new_hero.hole_cards
            and (not last_hero or not last_hero.hole_cards)
        ):
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
                    "seat": p.seat_index,
                    "pos": p.name,
                    "username": p.username,
                    "is_hero": p.is_hero,
                    "stack": f"{p.stack_size:.1f}",
                    "bet": p.current_bet,
                    "status": p.status,
                    "visible_action": p.visible_action,
                })
            turns.append({
                "street": s.meta_info.current_street,
                "board": s.board_state.community_cards,
                "pot": s.board_state.total_pot,
                "actor": s.action_on_seat_index,
                "dealer": s.dealer_seat_index,
                "amount_to_call": s.last_action_context.amount_to_call,
                "legal_actions": s.last_action_context.hero_action_options,
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
- **SUITS:** Read the actual suit glyph shape. The client may use a two-color deck:
  red can be hearts or diamonds and black can be spades or clubs. Never infer a
  suit from color alone.

VISUAL ANALYSIS INSTRUCTIONS (STRICT HIERARCHY):

1. STEP ONE: DETERMINE PLAYER STATUS (Check in this exact order):
   - **ACTIVE:** - Visible **CARDS** above the nameplate.
     - Look for two **RED/PATTERNED CARD BACKS** (Opponent) OR two **FACE-UP CARDS** (Hero).
     - In opponent crops, even two partially visible wide red rectangles at the top are card backs: set has_cards=true and is_folded=false.
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
   - **TOTAL POT:** Read the central `Pot:` label exactly. Do not calculate it and do not add current bets to it. Example: `Pot: 3.5 BB` means total_pot_bb=3.5.
   - **DEALER BUTTON:** Look for a small white circular disk with a black "D". It can be next to any player type.
   - **VISIBLE ACTION:** Text such as Check, Fold, Call, Bet, or Raise may temporarily replace a username. Put it in visible_action; do not use it as the username.

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
      "is_empty": <Boolean - True only when the seat has no player>,
      "is_all_in": <Boolean>,
      "is_dealer": <Boolean - Check for 'D' button regardless of status>,
      "visible_action": <"CHECK"|"CALL"|"BET"|"RAISE"|"FOLD" or null>,
      "hole_cards": <["Rs", "Rs"] or null>
    }
  ],
  "board_cards": [<list of community cards>],
  "total_pot_bb": <Float in BB>,
  "hero_context": {
    "is_turn": <Boolean>,
    "control_mode": <"ACTION_BUTTONS" only for the large bottom buttons; "PREACTION_CHECKBOXES" for small Check/Check-Fold/Call-Any boxes; otherwise "NONE">,
    "action_options": ["Check", "Call", "Raise"],
    "amount_to_call_bb": <Float, 0 when Check is available>
  }
}

IMPORTANT BUTTON RULE:
- Set is_turn=true only when the large primary action buttons are visible.
- Small checkbox controls labelled Check, Check/Fold, Call Any, or Fold to any bet are pre-action controls, not a current Hero decision. For those use control_mode="PREACTION_CHECKBOXES", is_turn=false, and action_options=[].
"""

# Low-latency wire format. The schema deliberately avoids redundant booleans
# and repeated null hole-card fields: less generated JSON means the state is
# available sooner, while parse_response() still materializes the same domain
# objects used by the rest of the application.
FAST_ANALYZE_PROMPT = """Analyze the three labelled PokerStars Carbon images.
Image 1 is the table mosaic. Image 2 repeats HERO, BOARD, and ACTIONS. Image 3 is a lossless enlarged HERO+BOARD card crop and is authoritative for ranks and suit glyphs. Labels map to seats S0-S5; HERO is S4.
Read each actual suit glyph shape; red may be hearts or diamonds and black may be spades or clubs. Emit cards as rank+suit.
All six-seat arrays are ordered S0,S1,S2,S3,S4,S5. n=usernames, s=stacks inside pods, w=separate current bets, x=status codes, v=visible action overlays. d is the dealer seat index or -1.
x codes: A=cards visible/active, F=name+stack but no cards/folded, I=all-in, S=sitting out, E=empty.
p is the displayed total Pot value exactly; never calculate it or add bets.
h contains only Hero's two cards. o contains only the large current action buttons; ignore small Check/Check-Fold/Call-Any checkboxes. c is the incremental amount Hero must call, or 0 when Check is available.
Use empty strings for an absent username/action and return only the requested structured data."""

FAST_VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "n": {
            "type": "array",
            "minItems": 6,
            "maxItems": 6,
            "items": {"type": "string"},
        },
        "s": {
            "type": "array",
            "minItems": 6,
            "maxItems": 6,
            "items": {"type": "number", "minimum": 0},
        },
        "w": {
            "type": "array",
            "minItems": 6,
            "maxItems": 6,
            "items": {"type": "number", "minimum": 0},
        },
        "x": {
            "type": "array",
            "minItems": 6,
            "maxItems": 6,
            "items": {"type": "string", "enum": ["A", "F", "I", "S", "E"]},
        },
        "d": {"type": "integer", "minimum": -1, "maximum": 5},
        "v": {
            "type": "array",
            "minItems": 6,
            "maxItems": 6,
            "items": {
                "type": "string",
                "enum": ["", "CHECK", "CALL", "BET", "RAISE", "FOLD"],
            },
        },
        "h": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "Rank+suit card code such as Ah or Tc",
            },
            "maxItems": 2,
        },
        "b": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "Rank+suit card code such as Ah or Tc",
            },
            "maxItems": 5,
        },
        "p": {"type": "number", "minimum": 0},
        "o": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["CHECK", "CALL", "FOLD", "BET", "RAISE"],
            },
            "maxItems": 3,
        },
        "c": {"type": "number", "minimum": 0},
    },
    "required": ["n", "s", "w", "x", "d", "v", "h", "b", "p", "o", "c"],
    "additionalProperties": False,
}

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
    if not active_seats or dealer_seat_index < 0:
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

# Each opponent capture maps directly to the zero-based seat index used by the
# vision response. Hero is intentionally excluded: their face-up cards are read
# separately by Gemini.
OPPONENT_CAPTURE_SEATS = {
    "seat1": 0,
    "seat2": 1,
    "seat3": 2,
    "seat4": 3,
    "seat6": 5,
}

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


def build_vision_mosaic(captures: dict) -> Image.Image:
    """Combine all semantic crops into one native-resolution labelled image.

    Gemini allocates visual processing per image part. A single mosaic retains
    the source pixels used for OCR while avoiding the overhead of eight
    independent image parts.
    """
    canvas = Image.new("RGB", (1080, 600), (18, 20, 23))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # Pillow versions before the sized default font.
        font = ImageFont.load_default()

    cells = [
        ("S0", "seat1", (8, 8, 360, 203)),
        ("S1", "seat2", (364, 8, 716, 203)),
        ("S2", "seat3", (720, 8, 1072, 203)),
        ("S3", "seat4", (8, 207, 360, 402)),
        ("HERO S4", "hero", (364, 207, 716, 402)),
        ("S5", "seat6", (720, 207, 1072, 402)),
        ("BOARD", "board", (8, 406, 420, 594)),
        ("ACTIONS", "buttons", (424, 406, 1072, 594)),
    ]

    for label, capture_name, (left, top, right, bottom) in cells:
        draw.rectangle((left, top, right, bottom), outline=(82, 91, 101), width=2)
        draw.text((left + 7, top + 4), label, fill=(100, 220, 235), font=font)
        image = captures[capture_name].convert("RGB")
        available_width = right - left - 12
        available_height = bottom - top - 28
        scale = min(1.0, available_width / image.width, available_height / image.height)
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        paste_x = left + 6 + (available_width - image.width) // 2
        paste_y = top + 24 + (available_height - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))

    return canvas


def build_vision_core_detail(captures: dict) -> Image.Image:
    """Give cards and action labels their own visual-token allocation."""
    canvas = Image.new("RGB", (600, 650), (18, 20, 23))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    cells = [
        ("HERO S4 — READ SUIT GLYPHS", "hero", (8, 8, 592, 248)),
        ("BOARD — READ CARDS AND POT", "board", (8, 252, 592, 452)),
        ("ACTIONS — LARGE BUTTONS ONLY", "buttons", (8, 456, 592, 642)),
    ]
    for label, capture_name, (left, top, right, bottom) in cells:
        draw.rectangle((left, top, right, bottom), outline=(82, 91, 101), width=2)
        draw.text((left + 7, top + 4), label, fill=(100, 220, 235), font=font)
        image = captures[capture_name].convert("RGB")
        available_width = right - left - 12
        available_height = bottom - top - 28
        scale = min(1.5, available_width / image.width, available_height / image.height)
        if abs(scale - 1.0) > 0.01:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        paste_x = left + 6 + (available_width - image.width) // 2
        paste_y = top + 24 + (available_height - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))
    return canvas


def build_vision_card_detail(captures: dict) -> Image.Image:
    """Losslessly enlarge only Hero and board cards for suit classification."""
    canvas = Image.new("RGB", (1000, 600), (18, 20, 23))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()

    hero = captures["hero"].convert("RGB")
    hero = hero.crop(
        (
            int(hero.width * 0.12),
            0,
            int(hero.width * 0.88),
            int(hero.height * 0.72),
        )
    )
    board = captures["board"].convert("RGB")

    cells = [
        ("HERO CARDS — GLYPH SHAPE IS AUTHORITATIVE", hero, (8, 8, 992, 292)),
        ("BOARD CARDS — MATCH SUIT GLYPH SHAPES", board, (8, 300, 992, 592)),
    ]
    for label, image, (left, top, right, bottom) in cells:
        draw.rectangle((left, top, right, bottom), outline=(82, 91, 101), width=2)
        draw.text((left + 8, top + 5), label, fill=(100, 220, 235), font=font)
        available_width = right - left - 16
        available_height = bottom - top - 38
        scale = min(3.0, available_width / image.width, available_height / image.height)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.NEAREST,
        )
        paste_x = left + 8 + (available_width - image.width) // 2
        paste_y = top + 32 + (available_height - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))
    return canvas


def gemini_media_resolution() -> types.MediaResolution:
    """Map the environment setting to a supported Gemini media resolution."""
    return {
        "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
        "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    }.get(
        GEMINI_MEDIA_RESOLUTION,
        types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    )


def image_to_jpeg_bytes(img: Image.Image) -> bytes:
    """Encode a PIL image once for either vision provider."""
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def image_to_gemini_part(img: Image.Image) -> types.Part:
    """Encode a PIL image as an inline JPEG part for the Gemini API."""
    return types.Part.from_bytes(
        data=image_to_jpeg_bytes(img), mime_type="image/jpeg"
    )


def image_to_gemini_png_part(img: Image.Image) -> types.Part:
    """Encode tiny card glyphs losslessly so JPEG cannot alter their shape."""
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="PNG", optimize=True)
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")


def detect_opponent_card_backs(img: Image.Image) -> bool:
    """Detect PokerStars Carbon's paired red opponent card backs locally.

    Opponent crops show each face-down card as a broad red/pink rectangle near
    the upper part of the image. Requiring two similarly sized, horizontally
    adjacent components avoids confusing a red username, chip, or card pip for
    a live hand. This is positive evidence only: a negative result never marks
    a player folded.
    """
    rgb = np.asarray(img.convert("RGB"))
    if rgb.size == 0:
        return False

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    red_mask = (
        ((hue <= 12) | (hue >= 165))
        & (saturation >= 60)
        & (value >= 60)
    ).astype(np.uint8)

    _, _, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    candidates = []
    min_area = max(120, int(0.005 * img.width * img.height))
    for x, y, width, height, area in stats[1:]:
        if (
            area < min_area
            or width < 0.07 * img.width
            or height < 0.06 * img.height
            or width > 0.45 * img.width
            or height > 0.35 * img.height
        ):
            continue
        aspect_ratio = width / height
        fill_ratio = area / (width * height)
        if not 1.2 <= aspect_ratio <= 3.5 or fill_ratio < 0.40:
            continue
        candidates.append(
            (int(x), int(y), int(width), int(height), int(area))
        )

    for first, second in itertools.combinations(candidates, 2):
        left, right = sorted((first, second), key=lambda component: component[0])
        lx, ly, lw, lh, larea = left
        rx, ry, rw, rh, rarea = right

        area_similarity = max(larea, rarea) / min(larea, rarea)
        center_y_delta = abs((ly + lh / 2) - (ry + rh / 2))
        horizontal_gap = rx - (lx + lw)

        if (
            area_similarity <= 2.0
            and center_y_delta <= 0.35 * max(lh, rh)
            and -0.40 * min(lw, rw) <= horizontal_gap <= 0.50 * max(lw, rw)
        ):
            return True

    return False


def detect_locally_dealt_seats(captures: dict) -> set[int]:
    """Return opponent seats with deterministic face-down-card evidence."""
    return {
        seat_index
        for capture_name, seat_index in OPPONENT_CAPTURE_SEATS.items()
        if captures.get(capture_name) is not None
        and detect_opponent_card_backs(captures[capture_name])
    }


def detect_hero_action_buttons(img: Image.Image) -> bool:
    """Return True only for PokerStars' large, actionable bottom buttons.

    The client also shows small pre-action checkboxes such as Check/Fold and
    Call Any. Those must never be mistaken for a live Hero decision.
    """
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    bright = (gray >= 110).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    large_buttons = 0
    for x, y, width, height, area in stats[1:]:
        if (
            y >= 0.18 * img.height
            and width >= 0.24 * img.width
            and height >= 0.24 * img.height
            and area >= 0.025 * img.width * img.height
            and area / (width * height) >= 0.70
        ):
            large_buttons += 1
    return large_buttons >= 2


def grab_zone_set(monitor: int, zones: dict) -> dict:
    """Grab one atomic bounding frame and crop named regions in memory."""
    with mss.MSS() as sct:
        selected_monitor = sct.monitors[monitor]
        mon_left = selected_monitor["left"]
        mon_top = selected_monitor["top"]
        normalized = {
            name: {
                "left": int(zone["left"]),
                "top": int(zone["top"]),
                "width": int(zone["width"]),
                "height": int(zone["height"]),
            }
            for name, zone in zones.items()
        }
        min_left = min(zone["left"] for zone in normalized.values())
        min_top = min(zone["top"] for zone in normalized.values())
        max_right = max(
            zone["left"] + zone["width"] for zone in normalized.values()
        )
        max_bottom = max(
            zone["top"] + zone["height"] for zone in normalized.values()
        )
        bounding_region = {
            "left": mon_left + min_left,
            "top": mon_top + min_top,
            "width": max_right - min_left,
            "height": max_bottom - min_top,
        }
        grabbed = sct.grab(bounding_region)
        frame = Image.frombytes("RGB", grabbed.size, grabbed.bgra, "raw", "BGRX")

        captures = {}
        for name, zone in normalized.items():
            left = zone["left"] - min_left
            top = zone["top"] - min_top
            captures[name] = frame.crop(
                (
                    left,
                    top,
                    left + zone["width"],
                    top + zone["height"],
                )
            )
        return captures


def capture_validation_regions(monitor: int) -> dict:
    """Atomically recapture state-critical regions without debug writes."""
    return grab_zone_set(
        monitor,
        {
            "board": SEAT_ZONES["board"],
            "hero": SEAT_ZONES["hero"],
            "buttons": BUTTONS_REGION,
        },
    )


def image_change_fraction(
    before: Image.Image,
    after: Image.Image,
    *,
    ignore_top: float = 0.0,
    ignore_bottom: float = 0.0,
) -> float:
    """Measure meaningful fixed-region pixel change after mild denoising."""
    if before.size != after.size:
        return 1.0

    before_gray = cv2.cvtColor(np.asarray(before.convert("RGB")), cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(np.asarray(after.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height = before_gray.shape[0]
    top = int(height * ignore_top)
    bottom = height - int(height * ignore_bottom)
    if bottom <= top:
        return 1.0

    before_gray = cv2.GaussianBlur(before_gray[top:bottom], (5, 5), 0)
    after_gray = cv2.GaussianBlur(after_gray[top:bottom], (5, 5), 0)
    delta = cv2.absdiff(before_gray, after_gray)
    return float(np.mean(delta >= 24))


def image_edge_change_fraction(
    before: Image.Image,
    after: Image.Image,
    *,
    ignore_top: float = 0.0,
) -> float:
    """Compare UI text/borders while ignoring hover fill-color changes."""
    if before.size != after.size:
        return 1.0
    before_gray = cv2.cvtColor(np.asarray(before.convert("RGB")), cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(np.asarray(after.convert("RGB")), cv2.COLOR_RGB2GRAY)
    top = int(before_gray.shape[0] * ignore_top)
    before_edges = cv2.Canny(before_gray[top:], 60, 140)
    after_edges = cv2.Canny(after_gray[top:], 60, 140)
    changed_edges = cv2.bitwise_xor(before_edges, after_edges)
    return float(np.mean(changed_edges > 0))


def hero_card_edge_change_fraction(
    before: Image.Image,
    after: Image.Image,
) -> float:
    """Compare the stable card area while excluding stack/time-bank animation."""
    if before.size != after.size:
        return 1.0
    box = (
        int(before.width * 0.12),
        0,
        int(before.width * 0.88),
        int(before.height * 0.62),
    )
    return image_edge_change_fraction(before.crop(box), after.crop(box))


def stale_validation_metrics(before: dict, after: dict) -> dict:
    """Return diagnostics for tuning freshness checks without another API call."""
    return {
        "board_pixel_change": image_change_fraction(
            before["board"], after["board"]
        ),
        "hero_card_edge_change": hero_card_edge_change_fraction(
            before["hero"], after["hero"]
        ),
        "buttons_before_detected": detect_hero_action_buttons(before["buttons"]),
        "buttons_after_detected": detect_hero_action_buttons(after["buttons"]),
        "button_edge_change": image_edge_change_fraction(
            before["buttons"], after["buttons"], ignore_top=0.08
        ),
        "button_pixel_change": image_change_fraction(
            before["buttons"], after["buttons"], ignore_top=0.08
        ),
    }


def save_stale_debug_capture(
    before: dict,
    after: dict,
    reasons: list[str],
    stage: str,
) -> None:
    """Save only rejected before/after frames so false positives are inspectable."""
    stale_dir = os.path.join(DEBUG_DIR, "stale")
    os.makedirs(stale_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
    run_dir = os.path.join(stale_dir, f"{run_id}_{stage}")
    os.makedirs(run_dir, exist_ok=True)
    for name in ("board", "hero", "buttons"):
        before[name].save(os.path.join(run_dir, f"before_{name}.png"))
        after[name].save(os.path.join(run_dir, f"after_{name}.png"))
    payload = {
        "run_id": run_id,
        "stage": stage,
        "reasons": reasons,
        **stale_validation_metrics(before, after),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as file:
        json.dump(payload, file, indent=2)
    with open(os.path.join(stale_dir, "latest.txt"), "w") as file:
        file.write(run_dir + "\n")


def table_state_change_reasons(before: dict, after: dict) -> list[str]:
    """Identify changes that make a pending strategy response obsolete."""
    required = {"board", "hero", "buttons"}
    if not required.issubset(before) or not required.issubset(after):
        return ["validation capture incomplete"]

    reasons = []
    if image_change_fraction(before["board"], after["board"]) >= 0.025:
        reasons.append("board or pot changed")
    if hero_card_edge_change_fraction(before["hero"], after["hero"]) >= 0.005:
        reasons.append("Hero cards changed")
    before_buttons = detect_hero_action_buttons(before["buttons"])
    after_buttons = detect_hero_action_buttons(after["buttons"])
    button_edge_change = image_edge_change_fraction(
        before["buttons"], after["buttons"], ignore_top=0.08
    )
    if before_buttons and not after_buttons:
        # PokerStars dims the same buttons when window focus or hover changes.
        # Treat them as gone only when their structure also disappeared.
        if button_edge_change >= 0.006:
            reasons.append("Hero action buttons disappeared")
    elif before_buttons and after_buttons and button_edge_change >= 0.00045:
        # The button rectangles can remain fixed while their legal actions or
        # call/raise amounts change. Reject even a small structural redraw in
        # that case; focus dimming is handled by the branch above.
        reasons.append("Hero action controls changed")
    return reasons


def apply_local_card_back_evidence(
    snapshot: GameSnapshot,
    captures: dict,
) -> list[int]:
    """Correct false folded/empty statuses using deterministic card evidence."""
    corrected_seats = []
    players_by_seat = {player.seat_index: player for player in snapshot.players}

    for seat_index in detect_locally_dealt_seats(captures):
        player = players_by_seat.get(seat_index)
        if player is None:
            continue
        player.cards_confirmed_locally = True
        if player.status not in {"ACTIVE", "ALL_IN"}:
            player.status = "ACTIVE"
            corrected_seats.append(seat_index)

    return corrected_seats


def canonical_poker_action(value) -> str:
    """Normalize one immediate poker action and reject pre-action controls."""
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    if not text or "/" in text or " any" in text or "to any" in text:
        return ""
    for action in ("fold", "check", "call", "bet", "raise"):
        if text == action or text.startswith(f"{action} ") or text.startswith(f"{action} to"):
            return action.upper()
    if text in {"all-in", "all in", "allin"}:
        return "ALL-IN"
    return ""

def capture_regions():
    """Atomically capture all calibrated regions from one screen frame."""
    captures = grab_zone_set(
        monitor_num,
        {**SEAT_ZONES, "buttons": BUTTONS_REGION},
    )
    if SAVE_DEBUG_IMAGES:
        save_debug_images(captures)
    return captures


def expand_fast_vision_payload(data: dict) -> dict:
    """Expand the compact Gemini wire schema into the legacy parser shape."""
    if not isinstance(data, dict) or "s" not in data:
        return data

    status_flags = {
        "A": {"has_cards": True},
        "F": {"is_folded": True},
        "I": {"has_cards": True, "is_all_in": True},
        "S": {"is_sitting_out": True},
        "E": {"is_empty": True},
    }
    hero_cards = data.get("h") or []
    raw_seats = data.get("s") or []
    if raw_seats and not isinstance(raw_seats[0], dict):
        names = data.get("n") or []
        bets = data.get("w") or []
        statuses = data.get("x") or []
        actions = data.get("v") or []
        dealer_index = data.get("d", -1)
        raw_seats = [
            {
                "i": index,
                "n": names[index] if index < len(names) else "",
                "s": raw_seats[index],
                "b": bets[index] if index < len(bets) else 0,
                "x": statuses[index] if index < len(statuses) else "E",
                "d": index == dealer_index,
                "a": actions[index] if index < len(actions) else "",
            }
            for index in range(min(6, len(raw_seats)))
        ]

    seats = []
    for seat_data in raw_seats:
        if not isinstance(seat_data, dict):
            continue
        try:
            seat_index = int(seat_data.get("i"))
        except (TypeError, ValueError):
            continue
        status = str(seat_data.get("x") or "E").upper()
        expanded = {
            "seat_index": seat_index,
            "name": seat_data.get("n") or None,
            "stack_size_bb": seat_data.get("s") or 0,
            "current_bet_bb": seat_data.get("b") or 0,
            "is_dealer": bool(seat_data.get("d")),
            "visible_action": seat_data.get("a") or None,
            "has_cards": False,
            "is_folded": False,
            "is_all_in": False,
            "is_sitting_out": False,
            "is_empty": False,
        }
        expanded.update(status_flags.get(status, status_flags["E"]))
        if seat_index == 4:
            expanded["hole_cards"] = hero_cards
        seats.append(expanded)

    return {
        "seats": seats,
        "board_cards": data.get("b") or [],
        "total_pot_bb": data.get("p") or 0,
        "hero_context": {
            "is_turn": True,
            "action_options": data.get("o") or [],
            "amount_to_call_bb": data.get("c") or 0,
        },
    }



def parse_response(
    text: str,
    locally_dealt_seats: Optional[set[int]] = None,
    hero_turn_confirmed: Optional[bool] = None,
) -> GameSnapshot:
    """Parses Vision Response into a GameSnapshot."""
    snapshot = GameSnapshot()
    local_card_evidence_available = locally_dealt_seats is not None
    locally_dealt_seats = locally_dealt_seats or set()
    snapshot.timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    snapshot.hand_id = str(int(time.time()))
    
    try:
        # Clean up JSON
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        
        data = expand_fast_vision_payload(json.loads(cleaned_text.strip()))
        
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
            visible_action = canonical_poker_action(p_data.get("visible_action"))
            name_as_action = canonical_poker_action(ocr_name)
            if name_as_action:
                visible_action = visible_action or name_as_action
                ocr_name = None
            if ocr_name and is_valid_username(ocr_name):
                player.username = ocr_name
            else:
                player.username = f"Unknown_S{idx}"
            player.visible_action = visible_action
            
            # Handle stack variations
            stack_val = p_data.get("stack_size_bb") or p_data.get("stack") or 0
            if isinstance(stack_val, str):
                # Clean string currency if needed (simple removal)
                stack_val = stack_val.replace('$','').replace('€','').strip()
            player.stack_size = float(stack_val)
            
            player.current_bet = float(p_data.get("current_bet_bb") or p_data.get("bet") or 0)
            player.is_hero = (idx == 4) # Assuming Seat 5 (Index 4) is Hero
            if player.is_hero:
                player.username = HERO_USERNAME
            player.is_dealer = p_data.get("is_dealer") or p_data.get("dealer") or False
            player.cards_confirmed_locally = idx in locally_dealt_seats
            
            # Hole Cards (Hero Only) - Check FIRST
            hc = p_data.get("hole_cards") or p_data.get("cards")
            if player.is_hero and hc and isinstance(hc, list):
                # Filter out "XX" placeholders if specific to hallucination
                player.hole_cards = [c for c in hc if c and c.upper() != "XX"]
            
            # Local card-back pixels are stronger status evidence than Gemini's
            # semantic flags. Apply them before EMPTY/FOLDED so position labels
            # are also calculated from the corrected state.
            if p_data.get("is_all_in", False) or (
                idx in locally_dealt_seats
                and player.stack_size <= 0.05
                and player.current_bet > 0
            ):
                player.status = "ALL_IN"
                active_indices.append(idx)
            elif player.hole_cards:
                player.status = "ACTIVE"
                active_indices.append(idx)
            elif idx in locally_dealt_seats:
                player.status = "ACTIVE"
                active_indices.append(idx)
            elif (
                not ocr_name
                and player.stack_size == 0
                and (
                    p_data.get("is_empty", False)
                    or not p_data.get("has_cards", False)
                )
            ):
                player.status = "EMPTY"
            elif p_data.get("is_sitting_out", False):
                player.status = "SITTING_OUT"
            elif local_card_evidence_available and not player.is_hero:
                # For calibrated opponent crops, visible card-back pixels are
                # stronger evidence than a VLM status flag. At Hero's turn an
                # occupied opponent without cards has folded.
                player.status = "FOLDED"
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
        active_seats = [
            p.seat_index for p in temp_players
            if p.status not in {"SITTING_OUT", "EMPTY"}
        ]
        
        for player in temp_players:
            if not player.is_hero and player.username == HERO_USERNAME:
                player.username = f"Unknown_S{player.seat_index}"
            if player.status in {"SITTING_OUT", "EMPTY"}:
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
        highest_bet = max((p.current_bet for p in snapshot.players), default=0.0)

        # Default deduction: Hero only owes the difference between the table's
        # highest visible bet and chips Hero already has in front.
        hero = next((p for p in snapshot.players if p.is_hero), None)
        hero_bet = hero.current_bet if hero else 0.0
        deduced_call_amount = max(0.0, highest_bet - hero_bet)
        snapshot.last_action_context.amount_to_call = deduced_call_amount
        # One static frame cannot distinguish a blind, bet, raise, or caller.
        # Aggression is populated only from observed history elsewhere.
        snapshot.last_action_context.aggressor_seat_index = -1

        # PokerStars' pot cannot be smaller than chips visibly committed on
        # the table. Repair dropped digits such as `3` instead of `30` using
        # the conservative visible-bet lower bound.
        visible_bets = sum(player.current_bet for player in snapshot.players)
        if snapshot.board_state.total_pot + 0.15 < visible_bets:
            snapshot.board_state.total_pot = visible_bets
        
        # 4. Hero Context (Override from Buttons)
        h_ctx = data.get("hero_context", {})
        is_hero_turn = (
            hero_turn_confirmed
            if hero_turn_confirmed is not None
            else h_ctx.get("is_turn") is True
        )
        snapshot.action_on_seat_index = 4 if is_hero_turn else -1
        canonical_options = []
        if is_hero_turn:
            for option in h_ctx.get("action_options") or []:
                canonical = canonical_poker_action(option)
                display_action = canonical.title() if canonical else ""
                if display_action and display_action not in canonical_options:
                    canonical_options.append(display_action)
        snapshot.last_action_context.hero_action_options = canonical_options
        # if h_ctx.get("is_turn"):
        #     snapshot.action_on_seat_index = 4 # Hero is Seat 5 (idx 4)
        
        # A visible Check is definitive. Otherwise prefer a positive amount
        # read from the buttons, but never let a missing/zero OCR value erase a
        # positive contribution difference visible on the table.
        call_amt = h_ctx.get("amount_to_call_bb") if is_hero_turn else None
        if "Check" in snapshot.last_action_context.hero_action_options:
            snapshot.last_action_context.amount_to_call = 0.0
        elif call_amt is not None and float(call_amt) > 0:
            snapshot.last_action_context.amount_to_call = float(call_amt)
        else:
            snapshot.last_action_context.amount_to_call = deduced_call_amount

        # Fundamental NLHE actions do not depend on OCR wording: when facing a
        # wager Hero can always fold or call; with no wager Hero can check.
        if is_hero_turn:
            options = snapshot.last_action_context.hero_action_options
            if snapshot.last_action_context.amount_to_call > 0:
                for required_action in ("Fold", "Call"):
                    if required_action not in options:
                        options.insert(0 if required_action == "Fold" else 1, required_action)
            elif "Check" not in options:
                options.insert(0, "Check")
        
        # Update Meta (Blind Levels hardcoded for now or parsed?)
        snapshot.meta_info.blind_level = {"sb": 0.01, "bb": 0.02, "ante": 0}

    except json.JSONDecodeError as error:
        snapshot.vision_error = f"Gemini returned invalid JSON: {error}"
        console.print(f"[red]{snapshot.vision_error}[/red]")
    except Exception as e:
        snapshot.vision_error = f"Could not parse Gemini state: {e}"
        console.print(f"[red]{snapshot.vision_error}[/red]")
        
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



def build_vision_request(captures: dict) -> tuple[list, types.GenerateContentConfig]:
    """Build either the compact one-image request or the legacy request."""
    if VISION_LAYOUT == "legacy":
        seat_images = [
            compress_image(captures["seat1"]),
            compress_image(captures["seat2"]),
            compress_image(captures["seat3"]),
            compress_image(captures["seat4"]),
            compress_image(captures["hero"]),
            compress_image(captures["seat6"]),
        ]
        images = seat_images + [
            compress_image(captures["board"]),
            compress_image(captures["buttons"]),
        ]
        return (
            [ANALYZE_PROMPT] + [image_to_gemini_part(image) for image in images],
            types.GenerateContentConfig(
                max_output_tokens=1200,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL
                ),
            ),
        )

    mosaic = build_vision_mosaic(captures)
    core_detail = build_vision_core_detail(captures)
    card_detail = build_vision_card_detail(captures)
    return (
        [
            FAST_ANALYZE_PROMPT,
            image_to_gemini_part(mosaic),
            image_to_gemini_part(core_detail),
            image_to_gemini_png_part(card_detail),
        ],
        types.GenerateContentConfig(
            max_output_tokens=400,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL
            ),
            media_resolution=gemini_media_resolution(),
            response_mime_type="application/json",
            response_json_schema=FAST_VISION_SCHEMA,
        ),
    )


def analyze_captures(
    comps: dict,
    *,
    hero_turn_confirmed: Optional[bool] = None,
) -> tuple[GameSnapshot, float]:
    """Run Gemini on an already captured, internally consistent frame."""
    t1 = time.time()
    locally_dealt_seats = detect_locally_dealt_seats(comps)
    if hero_turn_confirmed is None:
        hero_turn_confirmed = detect_hero_action_buttons(comps["buttons"])
    
    try:
        vision_input, vision_config = build_vision_request(comps)
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=vision_input,
            config=vision_config,
        )
        if not response.text:
            raise ValueError("Gemini returned no text content")
        # Save raw response for debugging
        with open("debug_vision_response.txt", "w") as f:
            f.write(response.text)
        console.print(f"[dim]Vision response saved to debug_vision_response.txt[/dim]")
        usage = getattr(response, "usage_metadata", None)
        if usage:
            console.print(
                "[dim]Vision tokens: "
                f"input={getattr(usage, 'prompt_token_count', '?')} "
                f"output={getattr(usage, 'candidates_token_count', '?')} "
                f"thoughts={getattr(usage, 'thoughts_token_count', 0) or 0}[/dim]"
            )
        snapshot = parse_response(
            response.text,
            locally_dealt_seats,
            hero_turn_confirmed=hero_turn_confirmed,
        )
        console.print(f"[dim]Parsed {len(snapshot.players)} players[/dim]")
    except Exception as e:
        console.print(f"[red]Vision Error: {e}[/red]")
        snapshot = GameSnapshot(
            vision_error=f"Gemini request failed ({type(e).__name__}): {e}"
        )
        
    t_vision = time.time() - t1
    return snapshot, t_vision


def analyze_state(
    monitor: int,
    *,
    require_hero_turn: bool = False,
) -> tuple[GameSnapshot, dict, float, float]:
    """
    1. Capture Regions
    2. Vision API -> GameSnapshot
    """
    t0 = time.time()
    comps = capture_regions()
    t_capture = time.time() - t0
    hero_turn_confirmed = detect_hero_action_buttons(comps["buttons"])

    # Strategy requests are useful only during a real Hero decision. This
    # avoids spending several seconds on pre-action checkbox states.
    if require_hero_turn and not hero_turn_confirmed:
        return GameSnapshot(), comps, t_capture, 0.0

    snapshot, t_vision = analyze_captures(
        comps,
        hero_turn_confirmed=hero_turn_confirmed,
    )
    return snapshot, comps, t_capture, t_vision



def display_results(
    snapshot: GameSnapshot,
    analysis: str,
    t_cap,
    t_vis,
    t_strat,
    t_tot,
    metrics: dict = None,
    hand_rank: str = "",
    strategy_source: str = "",
):
    """Synthetic display of GameSnapshot data."""
    console.clear()
    
    # 1. EXTRACT RECOMMENDATION
    recommendation = "Analyzing..."
    details = ""
    
    if analysis:
        if analysis == "No current Hero decision detected.":
            recommendation = "WAIT"
            details = analysis
        elif analysis == "No Hero Hand Detected." or analysis.startswith(
            ("Strategy Error:", "Vision Error:")
        ):
            recommendation = "RETRY"
            details = analysis
        lines = analysis.split('\n')
        action_line = next((line for line in lines if "Action:" in line), None)
        amount_line = next(
            (line for line in lines if "Amount:" in line or "Size:" in line),
            None,
        )
        
        if action_line and analysis != "No current Hero decision detected.":
            act = action_line.split(":", 1)[1].strip().replace("*", "")
            amt = ""
            if amount_line:
                raw_amt = amount_line.split(":", 1)[1].strip().replace("*", "")
                normalized_amt = raw_amt.lower().replace(" ", "")
                if normalized_amt not in {"0", "0bb", "n/a", "na"}:
                    amt = " " + raw_amt
            recommendation = f"{act}{amt}"
            
            # Extract only the bullet points from reasoning
            reasoning_lines = [l.strip() for l in lines if l.strip().startswith("*")]
            details = "\n".join(reasoning_lines)
        elif (
            not analysis.startswith(("Strategy Error:", "Vision Error:"))
            and analysis != "No current Hero decision detected."
            and analysis != "No Hero Hand Detected."
        ):
            recommendation = "Advice received"
            details = analysis

    # Header
    timing_mode = f" | {strategy_source}" if strategy_source and t_strat > 0 else ""
    console.print(
        f"[dim]ID: {snapshot.hand_id} | Total {t_tot:.2f}s{timing_mode} | "
        f"Capture {t_cap:.2f}s | Vision {t_vis:.2f}s | Strategy {t_strat:.2f}s[/dim]"
    )

    # ★ RECOMMENDATION PANEL ★
    if analysis:
        style = "bold white on red"
        rec_upper = recommendation.upper()
        if "FOLD" in rec_upper: style = "bold white on red"
        elif "CHECK" in rec_upper or "CALL" in rec_upper: style = "bold black on yellow"
        elif "RAISE" in rec_upper or "BET" in rec_upper: style = "bold white on green"
        elif "WAIT" in rec_upper: style = "bold black on cyan"
        elif "RETRY" in rec_upper: style = "bold white on red"
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
        if p.status in {"SITTING_OUT", "EMPTY"} and not p.is_hero:
            continue
        pos = p.name + ("(D)" if p.is_dealer else "")
        act = (
            "[bold yellow]HERO[/]" if p.is_hero
            else "[dim]Fold[/]" if p.status == "FOLDED"
            else "[magenta]All-in[/]" if p.status == "ALL_IN"
            else f"[red]B:{p.current_bet}[/]" if p.current_bet > 0
            else "[green]Active[/]"
        )
        p_tab.add_row(pos[:4], (p.username or "—")[:12], f"{p.stack_size:.0f}", act)
    
    console.print(p_tab)


def display_stale_result(
    t_cap,
    t_vis,
    t_strat,
    t_tot,
    reasons: list[str],
    strategy_source: str = "",
):
    """Show a neutral warning without rendering an obsolete poker action."""
    console.clear()
    timing_mode = f" | {strategy_source}" if strategy_source and t_strat > 0 else ""
    console.print(
        f"[dim]TABLE CHANGED | Total {t_tot:.2f}s{timing_mode} | Capture {t_cap:.2f}s | "
        f"Vision {t_vis:.2f}s | Strategy {t_strat:.2f}s[/dim]"
    )
    reason_text = ", ".join(reasons)
    console.print(
        Panel(
            Align.center(
                "[bold]ADVICE DISCARDED[/bold]\n"
                "The table changed while analysis was running. Press J again "
                f"on the current decision.\n[dim]{reason_text}[/dim]"
            ),
            style="bold black on yellow",
        )
    )









# Global state
monitor_num = 1
running = True
last_state = None
last_captures = None
input_mode = None
note_player = ""
current_history = HandHistory()  # Accumulates snapshots for current hand
recorder = None  # Initialized in main to keep imports side-effect free
analysis_lock = threading.Lock()


def preserve_hero_cards_on_continuing_board(
    snapshot: GameSnapshot,
    history: HandHistory,
) -> bool:
    """Keep validated hole cards across a confirmed same-hand board advance."""
    if not history.snapshots:
        return False
    previous = history.snapshots[-1]
    previous_board = previous.board_state.community_cards
    current_board = snapshot.board_state.community_cards
    board_continues = bool(
        previous_board
        and len(current_board) >= len(previous_board)
        and current_board[:len(previous_board)] == previous_board
    )
    if not board_continues and not confirmed_preflop_to_flop_transition(
        previous, snapshot
    ):
        return False

    prior_hero = next((player for player in previous.players if player.is_hero), None)
    current_hero = next((player for player in snapshot.players if player.is_hero), None)
    if not prior_hero or not current_hero or len(prior_hero.hole_cards or []) != 2:
        return False
    if current_hero.hole_cards == prior_hero.hole_cards:
        return False
    current_hero.hole_cards = list(prior_hero.hole_cards)
    return True


def reconcile_snapshot_with_history(snapshot: GameSnapshot, history: HandHistory):
    """Preserve irreversible hand facts across noisy vision snapshots."""
    if not history.snapshots or history.is_new_hand(snapshot):
        return

    previous = history.snapshots[-1]
    previous_by_seat = {player.seat_index: player for player in previous.players}
    if snapshot.dealer_seat_index < 0:
        snapshot.dealer_seat_index = next(
            (
                state.dealer_seat_index
                for state in reversed(history.snapshots)
                if state.dealer_seat_index >= 0
            ),
            -1,
        )

    for player in snapshot.players:
        prior = previous_by_seat.get(player.seat_index)
        if not prior:
            continue

        # Reject an impossible OCR collapse to zero. A real all-in of the
        # previous stack must be reflected by comparable pot growth.
        pot_growth = max(
            0.0,
            snapshot.board_state.total_pot - previous.board_state.total_pot,
        )
        if (
            not player.is_hero
            and player.cards_confirmed_locally
            and player.stack_size <= 0.05
            and prior.stack_size > 1.0
            and prior.status == "ACTIVE"
            and pot_growth + 1.0 < prior.stack_size * 0.5
        ):
            player.stack_size = prior.stack_size
            player.status = "ACTIVE"

        seat_history = [
            past_player
            for state in history.snapshots
            for past_player in state.players
            if past_player.seat_index == player.seat_index
        ]
        stable_position = next(
            (
                past_player.name
                for past_player in seat_history
                if past_player.name not in {"", "?", "OUT"}
            ),
            "",
        )
        stable_username = next(
            (
                past_player.username
                for past_player in reversed(seat_history)
                if past_player.username
                and not past_player.username.startswith("Unknown_")
                and (past_player.is_hero or past_player.username != HERO_USERNAME)
            ),
            "",
        )
        past_statuses = {past_player.status for past_player in seat_history}

        # A seat that was empty throughout this hand cannot join after the
        # flop. Vision may start reading its username once the layout settles,
        # but that player was never dealt in and must not acquire a position or
        # turn a verified heads-up pot into a multiway one.
        if (
            not player.is_hero
            and snapshot.board_state.community_cards
            and past_statuses
            and past_statuses.issubset({"EMPTY", "SITTING_OUT"})
            and not player.cards_confirmed_locally
        ):
            player.name = "OUT"
            player.status = "SITTING_OUT"
            player.current_bet = 0.0
            continue

        # Positions are fixed when the hand is dealt. A player leaving their
        # seat mid-hand cannot turn UTG into the BB, for example.
        if stable_position:
            player.name = stable_position
        # A username cannot move to another physical seat during one hand.
        # Keep the first reliable seat mapping even when later OCR returns a
        # plausible but conflicting name from a neighboring crop.
        if stable_username:
            player.username = stable_username
        if player.is_hero:
            player.username = HERO_USERNAME
        elif player.username == HERO_USERNAME:
            player.username = f"Unknown_S{player.seat_index}"
        if any(past_player.is_dealer for past_player in seat_history):
            player.is_dealer = True

        # Folded/all-in are absorbing states until a confirmed new hand.
        if "ALL_IN" in past_statuses:
            player.status = "ALL_IN"
        elif player.cards_confirmed_locally:
            if player.status != "ALL_IN":
                player.status = "ACTIVE"
        elif "FOLDED" in past_statuses:
            player.status = "FOLDED"
        elif "ACTIVE" in past_statuses and player.status in {"EMPTY", "SITTING_OUT"}:
            player.status = "FOLDED"
            if player.stack_size == 0:
                player.stack_size = prior.stack_size


def parse_action_history(history_json: str) -> str:
    """Summarize only betting changes actually observed between snapshots."""
    try:
        data = json.loads(history_json)
        turns = data.get("turns", [])
    except (TypeError, json.JSONDecodeError):
        return "History parsing failed."

    if len(turns) < 2:
        return "No reliable action sequence observed."

    actions_by_street = {}

    def player_label(player):
        position = player.get("name") or f"S{player.get('seat_index', -1) + 1}"
        username = player.get("username")
        return f"{position} ({username})" if username and not username.startswith("Unknown_") else position

    for previous, current in zip(turns, turns[1:]):
        previous_street = previous.get("meta_info", {}).get("current_street", "UNKNOWN")
        current_street = current.get("meta_info", {}).get("current_street", "UNKNOWN")
        if previous_street != current_street:
            continue

        street_actions = actions_by_street.setdefault(current_street, [])
        previous_by_seat = {
            player.get("seat_index"): player for player in previous.get("players", [])
        }
        current_by_seat = {
            player.get("seat_index"): player for player in current.get("players", [])
        }
        previous_max = max(
            (float(player.get("current_bet", 0.0)) for player in previous_by_seat.values()),
            default=0.0,
        )

        new_visible_actions = {}
        visible_numeric_seats = set()
        for seat, player in current_by_seat.items():
            previous_player = previous_by_seat.get(seat)
            if not previous_player:
                continue
            if player.get("status") == "FOLDED" and previous_player.get("status") != "FOLDED":
                street_actions.append(f"{player_label(player)} folds")

            visible_action = player.get("visible_action") or ""
            previous_visible = previous_player.get("visible_action") or ""
            if visible_action and visible_action != previous_visible:
                new_visible_actions[seat] = (player, visible_action)
            if (
                visible_action in {"CALL", "BET", "RAISE"}
                and visible_action != previous_visible
            ):
                visible_numeric_seats.add(seat)

        increased = []
        for seat, player in current_by_seat.items():
            previous_player = previous_by_seat.get(seat)
            if not previous_player or player.get("status") == "FOLDED":
                continue
            current_bet = float(player.get("current_bet", 0.0))
            previous_bet = float(previous_player.get("current_bet", 0.0))
            if current_bet > previous_bet and seat not in visible_numeric_seats:
                increased.append((seat, player, current_bet))

        above_previous_max = [item for item in increased if item[2] > previous_max]
        if above_previous_max:
            new_max = max(amount for _, _, amount in above_previous_max)
            leaders = [
                item for item in above_previous_max
                if abs(item[2] - new_max) < 1e-6
            ]
            if len(leaders) == 1:
                aggressor = leaders[0][1]
                verb = "bets" if previous_max == 0 else "raises to"
                street_actions.append(f"{player_label(aggressor)} {verb} {new_max:g} BB")
            else:
                labels = ", ".join(player_label(player) for _, player, _ in leaders)
                street_actions.append(f"{labels} reach {new_max:g} BB (order unknown)")

        for _, player, amount in increased:
            if amount <= previous_max and previous_max > 0:
                street_actions.append(f"{player_label(player)} calls to {amount:g} BB")

        for player, visible_action in new_visible_actions.values():
            if visible_action == "FOLD" and player.get("status") == "FOLDED":
                continue
            amount = float(player.get("current_bet", 0.0))
            if visible_action == "CHECK":
                action_text = "checks"
            elif visible_action == "CALL":
                action_text = f"calls to {amount:g} BB" if amount > 0 else "calls"
            elif visible_action == "BET":
                action_text = f"bets {amount:g} BB" if amount > 0 else "bets"
            elif visible_action == "RAISE":
                action_text = f"raises to {amount:g} BB" if amount > 0 else "raises"
            else:
                action_text = visible_action.lower()
            street_actions.append(f"{player_label(player)} {action_text}")

    output = [
        f"{street}: {', '.join(actions)}."
        for street, actions in actions_by_street.items() if actions
    ]
    return "\n".join(output) if output else "No reliable betting actions observed."


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
        "UNKNOWN", "SEAT", "EMPTY", "EMPTY SEAT", "POT", "TOTAL",
        "CHECKING", "CALLING"
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
    """Deterministic Hold'em made-hand, draw, and board-texture analysis."""

    RANKS = "23456789TJQKA"
    STRAIGHT_WINDOWS = [
        {14, 2, 3, 4, 5},
        *[set(range(low, low + 5)) for low in range(2, 11)],
    ]

    @staticmethod
    def parse_card(card_str):
        if not card_str:
            return None, None
        card_str = card_str.strip().upper()
        if card_str.startswith("10"):
            rank = "T"
            suit = card_str[2:].lower() if len(card_str) > 2 else ""
        else:
            rank = card_str[0]
            suit = card_str[1:].lower() if len(card_str) > 1 else ""
        if rank not in HandEvaluator.RANKS or suit not in {"s", "h", "d", "c"}:
            return None, None
        return HandEvaluator.RANKS.index(rank) + 2, suit

    @staticmethod
    def _unique_cards(card_strings):
        cards = []
        seen = set()
        for card_string in card_strings or []:
            card = HandEvaluator.parse_card(card_string)
            if card[0] is not None and card not in seen:
                cards.append(card)
                seen.add(card)
        return cards

    @staticmethod
    def _straight_high(ranks):
        rank_set = set(ranks)
        highs = []
        for window in HandEvaluator.STRAIGHT_WINDOWS:
            if window.issubset(rank_set):
                highs.append(5 if 14 in window and 2 in window else max(window))
        return max(highs, default=0)

    @staticmethod
    def _rank_five(cards):
        ranks = [rank for rank, _ in cards]
        suits = [suit for _, suit in cards]
        counts = {rank: ranks.count(rank) for rank in set(ranks)}
        groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
        flush = len(set(suits)) == 1
        straight_high = HandEvaluator._straight_high(ranks)

        if flush and straight_high:
            return (8, straight_high)
        if groups[0][0] == 4:
            quad = groups[0][1]
            kicker = max(rank for rank in ranks if rank != quad)
            return (7, quad, kicker)
        if groups[0][0] == 3 and groups[1][0] == 2:
            return (6, groups[0][1], groups[1][1])
        if flush:
            return (5, *sorted(ranks, reverse=True))
        if straight_high:
            return (4, straight_high)
        if groups[0][0] == 3:
            trip = groups[0][1]
            kickers = sorted((rank for rank in ranks if rank != trip), reverse=True)
            return (3, trip, *kickers)
        pair_ranks = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
        if len(pair_ranks) == 2:
            kicker = max(rank for rank in ranks if rank not in pair_ranks)
            return (2, pair_ranks[0], pair_ranks[1], kicker)
        if len(pair_ranks) == 1:
            pair = pair_ranks[0]
            kickers = sorted((rank for rank in ranks if rank != pair), reverse=True)
            return (1, pair, *kickers)
        return (0, *sorted(ranks, reverse=True))

    @staticmethod
    def _best_rank(cards):
        if len(cards) >= 5:
            return max(HandEvaluator._rank_five(list(combo)) for combo in itertools.combinations(cards, 5))

        ranks = [rank for rank, _ in cards]
        counts = {rank: ranks.count(rank) for rank in set(ranks)}
        groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
        if groups and groups[0][0] == 4:
            return (7, groups[0][1], 0)
        if groups and groups[0][0] == 3:
            return (3, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
        pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
        if len(pairs) >= 2:
            return (2, pairs[0], pairs[1], 0)
        if pairs:
            return (1, pairs[0], *sorted((r for r in ranks if r != pairs[0]), reverse=True))
        return (0, *sorted(ranks, reverse=True)) if ranks else (0, 0)

    @staticmethod
    def _format_rank(rank_tuple):
        category = rank_tuple[0]
        name = HandEvaluator.rank_name
        if category == 8:
            return f"Straight Flush ({name(rank_tuple[1])}-high)"
        if category == 7:
            return f"Four of a Kind ({name(rank_tuple[1])}s)"
        if category == 6:
            return f"Full House ({name(rank_tuple[1])}s full of {name(rank_tuple[2])}s)"
        if category == 5:
            return f"Flush ({name(rank_tuple[1])}-high)"
        if category == 4:
            return f"Straight ({name(rank_tuple[1])}-high)"
        if category == 3:
            return f"Three of a Kind ({name(rank_tuple[1])}s)"
        if category == 2:
            return f"Two Pair ({name(rank_tuple[1])}s and {name(rank_tuple[2])}s)"
        if category == 1:
            kicker = f", {name(rank_tuple[2])} kicker" if len(rank_tuple) > 2 else ""
            return f"Pair of {name(rank_tuple[1])}s{kicker}"
        return f"High Card ({name(rank_tuple[1])}-high)" if len(rank_tuple) > 1 else "High Card"

    @staticmethod
    def detect_draws(hero_cards, board_cards, made_category=None):
        """Return Hero-owned flop/turn draws; never infer board-only draws."""
        if len(board_cards or []) not in {3, 4}:
            return []

        hero = HandEvaluator._unique_cards(hero_cards)
        board = HandEvaluator._unique_cards(board_cards)
        all_cards = HandEvaluator._unique_cards((hero_cards or []) + (board_cards or []))
        if not hero or len(board) != len(board_cards or []):
            return []

        made_category = made_category if made_category is not None else HandEvaluator._best_rank(all_cards)[0]
        all_ranks = {rank for rank, _ in all_cards}
        board_ranks = {rank for rank, _ in board}
        hero_unique_ranks = {rank for rank, _ in hero} - board_ranks

        straight_label = None
        if made_category < 4:
            candidates = []
            for window in HandEvaluator.STRAIGHT_WINDOWS:
                missing = window - all_ranks
                if len(missing) == 1 and hero_unique_ranks.intersection(window):
                    candidates.append((frozenset(window & all_ranks), next(iter(missing))))

            missing_by_present = {}
            for present, missing_rank in candidates:
                missing_by_present.setdefault(present, set()).add(missing_rank)
            out_ranks = {missing for _, missing in candidates}
            open_ended = any(len(missing) >= 2 for missing in missing_by_present.values())

            if out_ranks:
                ordered_outs = sorted(out_ranks, key=lambda rank: 1 if rank == 14 else rank)
                outs_text = " or ".join(HandEvaluator.rank_name(rank) for rank in ordered_outs)
                nominal_outs = sum(4 - sum(1 for card in all_cards if card[0] == rank) for rank in out_ranks)
                if open_ended:
                    straight_label = f"Open-ended straight draw ({outs_text}; {nominal_outs} nominal outs)"
                elif len(out_ranks) >= 2:
                    straight_label = f"Double-gutshot straight draw ({outs_text}; {nominal_outs} nominal outs)"
                else:
                    straight_label = f"Gutshot straight draw ({outs_text}; {nominal_outs} nominal outs)"

        flush_label = None
        if made_category < 5:
            for suit in "shdc":
                suited_cards = [card for card in all_cards if card[1] == suit]
                hero_has_suit = any(card[1] == suit for card in hero)
                if len(suited_cards) == 4 and hero_has_suit:
                    flush_label = f"Flush draw ({13 - len(suited_cards)} nominal outs)"
                    break

        if straight_label and flush_label:
            return [f"Combo draw: {straight_label} + {flush_label}"]
        return [draw for draw in (straight_label, flush_label) if draw]

    @staticmethod
    def evaluate_details(hero_cards, board_cards):
        hero = HandEvaluator._unique_cards(hero_cards)
        board = HandEvaluator._unique_cards(board_cards)
        all_cards = HandEvaluator._unique_cards((hero_cards or []) + (board_cards or []))
        if not hero:
            return {"made_hand": "Unknown", "made_category": -1, "draws": [], "summary": "Unknown"}

        best_rank = HandEvaluator._best_rank(all_cards)
        made_hand = HandEvaluator._format_rank(best_rank)
        plays_board = False
        if len(board) == 5:
            plays_board = HandEvaluator._best_rank(board) == best_rank
            if plays_board:
                made_hand = f"Playing board: {made_hand}"

        draws = HandEvaluator.detect_draws(hero_cards, board_cards, best_rank[0])
        summary = made_hand if not draws else f"{made_hand} | {'; '.join(draws)}"
        return {
            "made_hand": made_hand,
            "made_category": best_rank[0],
            "plays_board": plays_board,
            "draws": draws,
            "summary": summary,
        }

    @staticmethod
    def evaluate(hero_cards, board_cards):
        return HandEvaluator.evaluate_details(hero_cards, board_cards)["summary"]

    @staticmethod
    def board_texture(board_cards):
        board = HandEvaluator._unique_cards(board_cards)
        if not board:
            return "Preflop"

        ranks = [rank for rank, _ in board]
        suits = [suit for _, suit in board]
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        max_suit_count = max((suits.count(suit) for suit in set(suits)), default=0)

        if max_suit_count >= 4:
            suit_texture = "four-flush"
        elif max_suit_count == 3:
            suit_texture = "monotone/three-flush"
        elif max_suit_count == 2:
            suit_texture = "two-tone"
        else:
            suit_texture = "rainbow"

        max_rank_count = max(rank_counts.values(), default=1)
        pair_count = sum(1 for count in rank_counts.values() if count == 2)
        if max_rank_count >= 3:
            pairing = "trips on board"
        elif pair_count >= 2:
            pairing = "double-paired"
        elif pair_count == 1:
            pairing = "paired"
        else:
            pairing = "unpaired"

        board_rank_set = set(ranks)
        max_window_overlap = max(
            (len(window.intersection(board_rank_set)) for window in HandEvaluator.STRAIGHT_WINDOWS),
            default=0,
        )
        close_pair = any(abs(a - b) <= 2 for a, b in itertools.combinations(board_rank_set, 2))
        if max_window_overlap >= 3:
            connectivity = "connected"
        elif close_pair:
            connectivity = "semi-connected"
        else:
            connectivity = "disconnected"

        flush_draw_present = max_suit_count >= 2 and len(board) < 5
        if flush_draw_present and connectivity == "connected":
            volatility = "wet"
        elif flush_draw_present or connectivity != "disconnected":
            volatility = "dynamic"
        else:
            volatility = "dry"

        return (
            f"{HandEvaluator.rank_name(max(ranks))}-high, {pairing}, "
            f"{suit_texture}, {connectivity}, {volatility}"
        )

    @staticmethod
    def rank_name(rank):
        return {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}.get(rank, str(rank))


@dataclass
class StrategyContext:
    opponents_in_hand: int
    opponents_eligible_to_act: int
    opponents_before_hero: Optional[int]
    opponents_after_hero: Optional[int]
    pot_type: str
    relative_position: str
    action_flow: str
    board_texture: str
    made_hand: str
    draws: str
    preflop_aggressor: str
    hero_was_pfa: Optional[bool]
    current_street_aggressor: str
    legal_actions: str

    @property
    def cbet_eligibility(self):
        if self.hero_was_pfa is True:
            return "YES"
        if self.hero_was_pfa is False:
            return "NO"
        return "UNKNOWN"

    @property
    def opponents_before_hero_label(self):
        return (
            str(self.opponents_before_hero)
            if self.opponents_before_hero is not None else "UNKNOWN"
        )

    @property
    def opponents_after_hero_label(self):
        return (
            str(self.opponents_after_hero)
            if self.opponents_after_hero is not None else "UNKNOWN"
        )


def infer_preflop_aggressor(history: HandHistory, current_snapshot: GameSnapshot):
    """Conservatively infer the last preflop raiser from sequential snapshots."""
    preflop = [
        snapshot for snapshot in history.snapshots
        if snapshot.meta_info.current_street == "PREFLOP"
    ]
    aggressor_seat = None

    for previous, current in zip(preflop, preflop[1:]):
        previous_by_seat = {player.seat_index: player for player in previous.players}
        previous_max = max((player.current_bet for player in previous.players), default=0.0)
        raiser_candidates = []
        for player in current.players:
            previous_player = previous_by_seat.get(player.seat_index)
            previous_bet = previous_player.current_bet if previous_player else 0.0
            if player.current_bet > previous_bet and player.current_bet > previous_max:
                raiser_candidates.append(player)

        if raiser_candidates:
            highest = max(player.current_bet for player in raiser_candidates)
            highest_candidates = [
                player for player in raiser_candidates
                if abs(player.current_bet - highest) < 1e-6
            ]
            aggressor_seat = highest_candidates[0].seat_index if len(highest_candidates) == 1 else None

    if aggressor_seat is not None:
        player = next(
            (player for player in current_snapshot.players if player.seat_index == aggressor_seat),
            None,
        )
        label = (player.name or player.username) if player else f"Seat {aggressor_seat + 1}"
        hero = next((player for player in current_snapshot.players if player.is_hero), None)
        return label, bool(hero and hero.seat_index == aggressor_seat)

    # If a preflop state was observed and the hand reached the flop without any
    # contribution above one blind, this is a limped/unraised pot.
    if preflop and current_snapshot.meta_info.current_street != "PREFLOP":
        max_observed_bet = max(
            (player.current_bet for snapshot in preflop for player in snapshot.players),
            default=0.0,
        )
        if max_observed_bet <= 1.0:
            return "NONE (unraised pot)", False

    return "UNKNOWN (not observed)", None


def relative_postflop_position(snapshot: GameSnapshot, hero: Player, acting_villains: list[Player]):
    """Return IP/OOP/SANDWICHED from actual live-seat postflop order."""
    if snapshot.meta_info.current_street == "PREFLOP":
        return f"PREFLOP POSITION: {hero.name or 'UNKNOWN'}"
    if snapshot.dealer_seat_index < 0 or not acting_villains:
        return "UNKNOWN"

    players = [hero, *acting_villains]

    def action_order(player):
        distance = (player.seat_index - snapshot.dealer_seat_index) % 6
        return 6 if distance == 0 else distance

    ordered = sorted(players, key=action_order)
    hero_index = ordered.index(hero)
    if hero_index == 0:
        return "OOP"
    if hero_index == len(ordered) - 1:
        return "IP"
    return "SANDWICHED"


def opponents_around_hero(
    snapshot: GameSnapshot,
    hero: Player,
    acting_villains: list[Player],
) -> tuple[Optional[int], Optional[int]]:
    """Count eligible opponents positioned before/after Hero this street."""
    if not acting_villains:
        return 0, 0
    if snapshot.dealer_seat_index < 0:
        return None, None

    players = [hero, *acting_villains]
    if snapshot.meta_info.current_street == "PREFLOP":
        preflop_order = {"UTG": 0, "MP": 1, "CO": 2, "BTN": 3, "SB": 4, "BB": 5}

        def order_key(player):
            if player.name in preflop_order:
                return preflop_order[player.name]
            distance = (player.seat_index - snapshot.dealer_seat_index) % 6
            return (distance - 3) % 6
    else:
        def order_key(player):
            distance = (player.seat_index - snapshot.dealer_seat_index) % 6
            return 6 if distance == 0 else distance

    ordered = sorted(players, key=order_key)
    hero_index = ordered.index(hero)
    return hero_index, len(ordered) - hero_index - 1


def build_strategy_context(
    snapshot: GameSnapshot,
    history: HandHistory,
    hand_details: dict,
) -> StrategyContext:
    hero = next((player for player in snapshot.players if player.is_hero), None)
    villains = [
        player for player in snapshot.players
        if not player.is_hero and player.status in {"ACTIVE", "ALL_IN"}
    ]
    acting_villains = [player for player in villains if player.status == "ACTIVE"]
    if len(villains) == 0:
        pot_type = "NO ACTIVE OPPONENT"
    elif len(villains) == 1:
        pot_type = "HEADS-UP"
    else:
        pot_type = "MULTIWAY"

    preflop_aggressor, hero_was_pfa = infer_preflop_aggressor(history, snapshot)
    current_aggressor = "NONE/UNKNOWN"
    aggressor_seat = snapshot.last_action_context.aggressor_seat_index
    if aggressor_seat >= 0:
        aggressor = next(
            (player for player in snapshot.players if player.seat_index == aggressor_seat),
            None,
        )
        if aggressor:
            current_aggressor = aggressor.name or aggressor.username or f"Seat {aggressor_seat + 1}"
    if current_aggressor == "NONE/UNKNOWN":
        visible_aggressors = [
            player for player in snapshot.players
            if player.visible_action in {"BET", "RAISE"}
        ]
        if len(visible_aggressors) == 1:
            aggressor = visible_aggressors[0]
            current_aggressor = (
                aggressor.name or aggressor.username or f"Seat {aggressor.seat_index + 1}"
            )

    legal_actions = snapshot.last_action_context.hero_action_options
    before_hero, after_hero = (
        opponents_around_hero(snapshot, hero, acting_villains)
        if hero else (0, 0)
    )
    if before_hero is None or after_hero is None:
        action_flow = "Action order is unknown because the dealer is not confirmed"
    elif snapshot.action_on_seat_index != 4:
        action_flow = "Hero turn is not confirmed"
    elif snapshot.meta_info.current_street == "PREFLOP":
        action_flow = (
            f"Hero decision confirmed; {before_hero} eligible opponents are positioned "
            f"before Hero and {after_hero} after Hero preflop. Exact prior actions are unknown."
        )
    elif (
        snapshot.last_action_context.amount_to_call == 0
        and "Check" in legal_actions
    ):
        action_flow = (
            f"No wager to call; action reached Hero through {before_hero} opponents, "
            f"with {after_hero} eligible opponents positioned after Hero."
        )
    else:
        action_flow = (
            f"Hero decision confirmed; {before_hero} eligible opponents are positioned "
            f"before Hero and {after_hero} after Hero."
        )
    return StrategyContext(
        opponents_in_hand=len(villains),
        opponents_eligible_to_act=len(acting_villains),
        opponents_before_hero=before_hero,
        opponents_after_hero=after_hero,
        pot_type=pot_type,
        relative_position=(
            relative_postflop_position(snapshot, hero, villains)
            if hero else "UNKNOWN"
        ),
        action_flow=action_flow,
        board_texture=HandEvaluator.board_texture(snapshot.board_state.community_cards),
        made_hand=hand_details["made_hand"],
        draws="; ".join(hand_details["draws"]) if hand_details["draws"] else "NONE",
        preflop_aggressor=preflop_aggressor,
        hero_was_pfa=hero_was_pfa,
        current_street_aggressor=current_aggressor,
        legal_actions=", ".join(legal_actions) if legal_actions else "UNKNOWN",
    )


def strategy_pot_for_street(snapshot: GameSnapshot, history: HandHistory) -> float:
    """Use the current total, carrying an older high only when corroborated."""
    street = snapshot.meta_info.current_street
    past_values = [
        state.board_state.total_pot
        for state in history.snapshots
        if state.meta_info.current_street == street
        and state is not snapshot
        and state.board_state.total_pot > 0
    ]
    current = snapshot.board_state.total_pot

    value_buckets = {}
    for value in past_values:
        bucket = round(value, 1)
        value_buckets.setdefault(bucket, []).append(value)
    corroborated = [
        max(values) for values in value_buckets.values() if len(values) >= 2
    ]

    if current > 0:
        # A single old OCR spike must not poison the entire street. Preserve a
        # higher prior total only after it was independently seen at least twice.
        stable_highs = [value for value in corroborated if value > current]
        return max(stable_highs, default=current)
    if corroborated:
        return max(corroborated)
    return past_values[-1] if past_values else 0.0


@dataclass
class PreparedStrategyState:
    """Deterministic facts shared by Claude and the local GTO router."""

    hero: Player
    active_villains: list[Player]
    final_pot: float
    effective_stack: float
    hand_details: dict
    hand_rank: str
    strategy_context: StrategyContext
    metrics: dict
    current_action: str


def prepare_strategy_state(
    snapshot: GameSnapshot,
    history: HandHistory,
) -> PreparedStrategyState:
    hero = next((player for player in snapshot.players if player.is_hero), None)
    if not hero or len(hero.hole_cards or []) != 2:
        raise ValueError("Structured strategy evaluation requires two Hero cards")

    final_pot = strategy_pot_for_street(snapshot, history)
    active_villains = [
        player for player in snapshot.players
        if not player.is_hero and player.status in {"ACTIVE", "ALL_IN"}
    ]
    max_villain_stack = max(
        (player.stack_size for player in active_villains),
        default=0.0,
    )
    effective_stack = (
        min(hero.stack_size, max_villain_stack)
        if active_villains else hero.stack_size
    )
    hand_details = HandEvaluator.evaluate_details(
        hero.hole_cards,
        snapshot.board_state.community_cards,
    )
    hand_rank = hand_details["summary"]
    strategy_context = build_strategy_context(snapshot, history, hand_details)
    spr = (effective_stack / final_pot) if final_pot > 0 else 0.0

    call_amount = snapshot.last_action_context.amount_to_call
    if call_amount > 0:
        pot_odds_pct = (call_amount / (final_pot + call_amount)) * 100
        pot_odds_ratio = f"{(final_pot / call_amount):.1f}:1"
        pot_odds_str = f"{pot_odds_pct:.1f}% ({pot_odds_ratio})"
        current_action = f"Facing {call_amount:.2f} BB to call"
    else:
        pot_odds_pct = 0.0
        pot_odds_ratio = "N/A"
        pot_odds_str = "N/A (Facing 0 bet)"
        if any(
            canonical_poker_action(option) == "CHECK"
            for option in snapshot.last_action_context.hero_action_options
        ):
            current_action = "No wager to call; Check is available"
        else:
            current_action = "No wager detected; prior checks were not observed"

    metrics = {
        "final_pot": final_pot,
        "eff_stack": effective_stack,
        "spr": spr,
        "pot_odds_pct": pot_odds_pct,
        "pot_odds_ratio": pot_odds_ratio,
        "pot_odds_str": pot_odds_str,
        "made_hand": hand_details["made_hand"],
        "draws": hand_details["draws"],
        "board_texture": strategy_context.board_texture,
        "opponents_in_hand": strategy_context.opponents_in_hand,
        "relative_position": strategy_context.relative_position,
    }
    return PreparedStrategyState(
        hero=hero,
        active_villains=active_villains,
        final_pot=final_pot,
        effective_stack=effective_stack,
        hand_details=hand_details,
        hand_rank=hand_rank,
        strategy_context=strategy_context,
        metrics=metrics,
        current_action=current_action,
    )


def sanitize_strategy_response(analysis: str, strategy_context: StrategyContext) -> str:
    """Remove unsupported claims and reject actions unavailable in the client."""
    sanitized = analysis
    sanitized = re.sub(
        r"\bpremium\s+(?:ace-?high|high card)\b",
        "ace-high",
        sanitized,
        flags=re.IGNORECASE,
    )
    if strategy_context.hero_was_pfa is not True:
        sanitized = re.sub(
            r"\bcontinuation[ -]?bet\b", "bet", sanitized, flags=re.IGNORECASE
        )
        sanitized = re.sub(r"\bc-?bet\b", "bet", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(
            r"\b(?:we|hero)\s+(?:have|has)\s+(?:the\s+)?initiative\b",
            "initiative is not confirmed",
            sanitized,
            flags=re.IGNORECASE,
        )

    # Eligibility is not turn order. Replace invented "behind/to act" counts
    # with the deterministic number positioned after Hero.
    if strategy_context.opponents_after_hero is not None:
        after_count = strategy_context.opponents_after_hero
        after_phrase = (
            f"{after_count} eligible opponent"
            f"{'s' if after_count != 1 else ''} positioned after Hero"
        )
        count_word = r"(?:\d+|zero|one|two|three|four|five)"
        sanitized = re.sub(
            rf"\b{count_word}\s+(?:opponents?|others|players?|hands?)\s+"
            rf"(?:still\s+)?(?:able\s+)?to act\b",
            after_phrase,
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            rf"\b{count_word}\s+(?:live\s+)?(?:opponents?|players?|hands?)\s+behind\b",
            after_phrase,
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            rf"\b{count_word}\s+(?:eligible\s+)?(?:opponents?|players?|hands?)\s+after Hero\b",
            after_phrase,
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            r"\bopponents?\s+(?:still\s+)?(?:able\s+)?to act\b",
            after_phrase,
            sanitized,
            flags=re.IGNORECASE,
        )

    action_match = re.search(
        r"^\s*\*{0,2}Action:\*{0,2}\s*(.*?)\s*$",
        sanitized,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not action_match:
        return "Strategy Error: Claude response is missing a valid Action line."

    raw_action = action_match.group(1).strip().strip("*").strip().lower()
    strict_actions = {
        "fold": "FOLD",
        "check": "CHECK",
        "call": "CALL",
        "bet": "BET",
        "raise": "RAISE",
        "all-in": "ALL-IN",
        "all in": "ALL-IN",
    }
    suggested = strict_actions.get(raw_action, "")
    legal = {
        canonical_poker_action(action)
        for action in strategy_context.legal_actions.split(",")
    }
    legal.discard("")
    if suggested not in legal:
        legal_display = ", ".join(sorted(action.title() for action in legal))
        return (
            f"Strategy Error: Claude suggested {raw_action or 'an unknown action'}, "
            f"but legal actions are {legal_display or 'unknown'}."
        )
    return sanitized


def build_parallel_strategy_prompt(history_hint: str = "") -> str:
    """Prompt Haiku to make a decision directly from the same captured frame."""
    return """Read the labelled PokerStars mosaic and choose one practical micro-stakes NLHE action.
Seats S0 through S5 are clockwise table order; Hero is S4. Use the visible D marker to determine position.
Read suit glyph shapes, not color alone: red may be hearts or diamonds; black may be spades or clubs.
Use only the large current ACTIONS buttons. Ignore Check/Fold, Call Any, and other small pre-action checkboxes.
Never invent prior actions. Multiway bluff less; prefer clear value and avoid marginal pot inflation.
No prior-hand or unconfirmed action history is supplied in FAST mode.

For Raise, Size means the final raise-to total in BB.
Return exactly these twelve fields, one per line:
**Seen Hero:** [two rank+suit cards]
**Seen Board:** [rank+suit cards, or Preflop]
**Seen Pot:** [total pot in BB]
**Seen Call:** [incremental call amount in BB, or 0]
**Seen Stack:** [Hero stack in BB]
**Seen Live Seats:** [active/all-in opponents as S0:A or S3:I, excluding Hero; or None]
**Seen Effective Stack:** [Hero versus deepest live opponent, in BB]
**Seen Dealer:** [S0-S5]
**Options:** [all large current action options]
**Action:** [Fold/Check/Call/Bet/Raise]
**Size:** [specific BB amount, or 0 for Check/Fold]
**Why:** [one short sentence]"""


def request_parallel_strategy(
    captures: dict,
    history_hint: str,
) -> tuple[str, float, str]:
    """Ask low-latency Claude to inspect the frame concurrently with Gemini."""
    prompt = build_parallel_strategy_prompt(history_hint)
    if not anthropic_client:
        return "Strategy Error: ANTHROPIC_API_KEY missing.", 0.0, prompt

    mosaic_bytes = image_to_jpeg_bytes(build_vision_mosaic(captures))
    started = time.time()
    try:
        response = anthropic_client.with_options(
            timeout=CLAUDE_FAST_TIMEOUT_SECONDS,
            max_retries=0,
        ).messages.create(
            model=CLAUDE_FAST_MODEL,
            max_tokens=192,
            system=(
                "You are a precise real-time poker decision assistant. Never invent "
                "cards, buttons, action history, or legal actions."
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(mosaic_bytes).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        analysis = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not analysis:
            raise ValueError("Claude returned no text content")
    except Exception as error:
        analysis = f"Strategy Error: {error}"
    return analysis, time.time() - started, prompt


def extract_named_strategy_field(
    text: str,
    label: str,
    following_labels: tuple[str, ...],
) -> str:
    """Extract a markdown field even if the model joins fields on one line."""
    start = re.search(
        rf"\*{{0,2}}{re.escape(label)}:\*{{0,2}}\s*",
        text,
        flags=re.IGNORECASE,
    )
    if not start:
        return ""
    tail = text[start.end():]
    end_positions = []
    for next_label in following_labels:
        match = re.search(
            rf"\*{{0,2}}{re.escape(next_label)}:\*{{0,2}}",
            tail,
            flags=re.IGNORECASE,
        )
        if match:
            end_positions.append(match.start())
    end = min(end_positions) if end_positions else len(tail)
    return tail[:end].strip(" \t\r\n/;|")


def cards_from_strategy_field(value: str) -> list[str]:
    """Normalize ASCII or glyph suit notation from Claude's visual report."""
    value = value.translate(str.maketrans({"♠": "s", "♥": "h", "♦": "d", "♣": "c"}))
    cards = []
    for rank, suit in re.findall(
        r"(?<![A-Za-z0-9])(10|[2-9TJQKA])\s*([shdc])\b",
        value,
        flags=re.IGNORECASE,
    ):
        normalized_rank = "T" if rank == "10" else rank.upper()
        cards.append(f"{normalized_rank}{suit.lower()}")
    return cards


PARALLEL_STRATEGY_LABELS = (
    "Seen Hero",
    "Seen Board",
    "Seen Pot",
    "Seen Call",
    "Seen Stack",
    "Seen Live Seats",
    "Seen Effective Stack",
    "Seen Dealer",
    "Options",
    "Action",
    "Size",
    "Why",
)


def parallel_strategy_fields(raw_analysis: str) -> dict[str, str]:
    """Parse every required field from the speculative visual strategy."""
    return {
        label: extract_named_strategy_field(
            raw_analysis,
            label,
            PARALLEL_STRATEGY_LABELS[index + 1:],
        )
        for index, label in enumerate(PARALLEL_STRATEGY_LABELS)
    }


def strict_bb_amount(value: str) -> Optional[float]:
    """Parse a complete non-negative BB value; reject signs and other units."""
    cleaned = value.strip().strip("[]").strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:BB)?", cleaned, re.IGNORECASE)
    return float(match.group(1)) if match else None


def strict_live_seats(value: str) -> Optional[dict[int, str]]:
    """Parse a compact opponent map such as `S0:A, S3:I`."""
    cleaned = value.strip().strip("[]").strip()
    if cleaned.lower() in {"none", "no opponents", "empty"}:
        return {}
    if not cleaned:
        return None

    live_seats = {}
    for item in cleaned.split(","):
        match = re.fullmatch(
            r"S?([0-5])\s*:\s*(A|I|ACTIVE|ALL[- ]?IN)",
            item.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        seat_index = int(match.group(1))
        if seat_index in live_seats or seat_index == 4:
            return None
        status = match.group(2).upper().replace("-", "").replace(" ", "")
        live_seats[seat_index] = "I" if status == "ALLIN" or status == "I" else "A"
    return live_seats


CARD_CODE_RE = re.compile(r"^[2-9TJQKA][shdc]$")


def validate_snapshot_candidate(
    snapshot: GameSnapshot,
    *,
    require_hero_hand: bool = False,
) -> list[str]:
    """Reject malformed Vision state before it can affect hand history."""
    if snapshot.vision_error:
        return [snapshot.vision_error]
    errors = []
    players = snapshot.players
    seat_indices = [player.seat_index for player in players]
    if len(players) != 6 or set(seat_indices) != set(range(6)):
        errors.append("Vision did not return six distinct seats S0-S5")

    board = snapshot.board_state.community_cards
    if len(board) not in {0, 3, 4, 5}:
        errors.append("board must contain 0, 3, 4, or 5 cards")

    hero = next((player for player in players if player.is_hero), None)
    hero_cards = hero.hole_cards if hero and hero.hole_cards else []
    if require_hero_hand and len(hero_cards) != 2:
        errors.append("Hero must have exactly two cards")

    all_cards = [*hero_cards, *board]
    malformed_cards = [
        str(card) for card in all_cards
        if not isinstance(card, str) or not CARD_CODE_RE.fullmatch(card)
    ]
    if malformed_cards:
        errors.append("cards must use rank+suit notation such as Ah or Tc")
    if len(all_cards) != len(set(all_cards)):
        errors.append("the same card appears more than once")

    amounts = [
        ("pot", snapshot.board_state.total_pot),
        ("call amount", snapshot.last_action_context.amount_to_call),
    ]
    for player in players:
        amounts.extend(
            (
                (f"S{player.seat_index} stack", player.stack_size),
                (f"S{player.seat_index} bet", player.current_bet),
            )
        )
    for label, value in amounts:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            errors.append(f"{label} must be a finite non-negative number")

    if snapshot.dealer_seat_index not in {-1, 0, 1, 2, 3, 4, 5}:
        errors.append("dealer seat must be S0-S5 or unknown")
    if snapshot.action_on_seat_index not in {-1, 4}:
        errors.append("action may only be on Hero or unknown")

    options = {
        canonical_poker_action(option)
        for option in snapshot.last_action_context.hero_action_options
    }
    options.discard("")
    call_amount = snapshot.last_action_context.amount_to_call
    if call_amount > 0 and "CHECK" in options:
        errors.append("Check cannot be legal while facing a positive call amount")
    if (
        board
        and call_amount > 0
        and snapshot.board_state.total_pot <= call_amount + 0.15
    ):
        # PokerStars' displayed pot already includes the outstanding wager.
        # If it is no larger than the amount Hero must call, Vision mixed two
        # different frames (or misread one of the values), so neither pot odds
        # nor a solver root can be trusted.
        errors.append("postflop displayed total pot must exceed the call amount")
    return errors


def validate_parallel_observation(
    raw_analysis: str,
    snapshot: GameSnapshot,
) -> str:
    """Validate Haiku's decision inputs against the immutable Gemini state."""
    if raw_analysis.startswith("Strategy Error:"):
        return raw_analysis
    fields = parallel_strategy_fields(raw_analysis)
    if any(not fields[label] for label in PARALLEL_STRATEGY_LABELS):
        return "Strategy Error: Claude's parallel response is missing a required field."

    hero = next((player for player in snapshot.players if player.is_hero), None)
    expected_hero = hero.hole_cards if hero and hero.hole_cards else []
    seen_hero = cards_from_strategy_field(fields["Seen Hero"])
    seen_board = cards_from_strategy_field(fields["Seen Board"])
    expected_board = snapshot.board_state.community_cards
    if len(seen_hero) != 2 or set(seen_hero) != set(expected_hero):
        return "Strategy Error: Claude and Gemini disagreed on Hero's cards."
    if seen_board != expected_board:
        return "Strategy Error: Claude and Gemini disagreed on the board."

    seen_pot = strict_bb_amount(fields["Seen Pot"])
    seen_call = strict_bb_amount(fields["Seen Call"])
    seen_stack = strict_bb_amount(fields["Seen Stack"])
    if seen_pot is None or abs(seen_pot - snapshot.board_state.total_pot) > max(
        0.15, snapshot.board_state.total_pot * 0.02
    ):
        return "Strategy Error: Claude and Gemini disagreed on the pot."
    if seen_call is None or abs(
        seen_call - snapshot.last_action_context.amount_to_call
    ) > 0.10:
        return "Strategy Error: Claude and Gemini disagreed on the call amount."
    if not hero or seen_stack is None or abs(seen_stack - hero.stack_size) > 0.20:
        return "Strategy Error: Claude and Gemini disagreed on Hero's stack."

    seen_live_seats = strict_live_seats(fields["Seen Live Seats"])
    if seen_live_seats is None:
        return "Strategy Error: Claude returned an invalid live-seat map."
    expected_live_seats = {
        player.seat_index: "I" if player.status == "ALL_IN" else "A"
        for player in snapshot.players
        if not player.is_hero and player.status in {"ACTIVE", "ALL_IN"}
    }
    if seen_live_seats != expected_live_seats:
        return "Strategy Error: Claude and Gemini disagreed on live seats."

    active_villains = [
        player for player in snapshot.players
        if not player.is_hero and player.status in {"ACTIVE", "ALL_IN"}
    ]
    expected_effective_stack = (
        min(hero.stack_size, max(player.stack_size for player in active_villains))
        if hero and active_villains
        else hero.stack_size if hero
        else 0.0
    )
    seen_effective_stack = strict_bb_amount(fields["Seen Effective Stack"])
    if (
        seen_effective_stack is None
        or abs(seen_effective_stack - expected_effective_stack) > 0.20
    ):
        return "Strategy Error: Claude and Gemini disagreed on effective stack."

    dealer_match = re.fullmatch(
        r"S?([0-5])",
        fields["Seen Dealer"].strip().strip("[]").strip(),
        flags=re.IGNORECASE,
    )
    if not dealer_match or int(dealer_match.group(1)) != snapshot.dealer_seat_index:
        return "Strategy Error: Claude and Gemini disagreed on the dealer."

    option_words = re.findall(
        r"\b(?:fold|check|call|bet|raise)\b",
        fields["Options"],
        flags=re.IGNORECASE,
    )
    seen_options = {canonical_poker_action(option) for option in option_words}
    expected_options = {
        canonical_poker_action(option)
        for option in snapshot.last_action_context.hero_action_options
    }
    seen_options.discard("")
    expected_options.discard("")
    if seen_options != expected_options:
        return "Strategy Error: Claude and Gemini disagreed on the legal actions."
    return ""


def validate_parallel_action(
    raw_analysis: str,
    snapshot: GameSnapshot,
) -> str:
    """Validate the speculative action and amount without mutating state."""
    fields = parallel_strategy_fields(raw_analysis)
    expected_options = {
        canonical_poker_action(option)
        for option in snapshot.last_action_context.hero_action_options
    }
    expected_options.discard("")

    action_text = fields["Action"].strip().strip("[]").strip()
    if not re.fullmatch(r"(?:Fold|Check|Call|Bet|Raise)", action_text, re.I):
        return "Strategy Error: Claude's parallel response has no valid action."
    action = action_text.upper()
    if action not in expected_options:
        return "Strategy Error: Claude's action was not independently confirmed as legal."

    numeric_size = strict_bb_amount(fields["Size"])
    if numeric_size is None:
        return "Strategy Error: Claude returned an invalid BB size."

    return validate_action_size(action, numeric_size, snapshot)


def validate_action_size(
    action: str,
    numeric_size: float,
    snapshot: GameSnapshot,
) -> str:
    """Validate one normalized poker action amount against the visible state."""
    if action not in {"FOLD", "CHECK", "CALL", "BET", "RAISE"}:
        return "Strategy Error: Claude returned an unsupported action."

    hero = next((player for player in snapshot.players if player.is_hero), None)
    hero_total = (hero.stack_size + hero.current_bet) if hero else 0.0
    call_amount = snapshot.last_action_context.amount_to_call
    if action in {"CHECK", "FOLD"}:
        if numeric_size != 0:
            return "Strategy Error: Check or Fold must have size 0."
    elif action == "CALL":
        if call_amount <= 0 or abs(numeric_size - call_amount) > 0.10:
            return "Strategy Error: Claude's call size disagreed with the call amount."
    elif numeric_size <= 0 or numeric_size > hero_total + 0.01:
        return "Strategy Error: Claude's bet or raise size is outside Hero's stack."
    elif action == "BET":
        minimum_bet = min(1.0, hero_total)
        if numeric_size + 0.01 < minimum_bet:
            return "Strategy Error: Claude's bet is below the legal minimum."
    else:
        highest_bet = max(
            (player.current_bet for player in snapshot.players),
            default=0.0,
        )
        conservative_min_raise_to = highest_bet + max(1.0, call_amount)
        is_short_all_in = abs(numeric_size - hero_total) <= 0.01
        if (
            numeric_size <= highest_bet + 0.01
            or (
                numeric_size + 0.01 < conservative_min_raise_to
                and not is_short_all_in
            )
        ):
            return "Strategy Error: Claude's raise is below the legal minimum."
    return ""


def validate_strategy_amount(
    analysis: str,
    snapshot: GameSnapshot,
) -> str:
    """Apply the same deterministic size checks to sequential COACH output."""
    if analysis.startswith("Strategy Error:"):
        return analysis
    action_match = re.search(
        r"^\s*\*{0,2}Action:\*{0,2}\s*(Fold|Check|Call|Bet|Raise)\s*$",
        analysis,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    amount_match = re.search(
        r"^\s*\*{0,2}(?:Amount|Size):\*{0,2}\s*(.*?)\s*$",
        analysis,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not action_match or not amount_match:
        return "Strategy Error: Claude response is missing a valid action amount."
    action = action_match.group(1).upper()
    legal_actions = {
        canonical_poker_action(option)
        for option in snapshot.last_action_context.hero_action_options
    }
    legal_actions.discard("")
    numeric_size = strict_bb_amount(amount_match.group(1).strip().strip("*"))
    if numeric_size is None:
        return "Strategy Error: Claude returned an invalid BB size."
    hero = next((player for player in snapshot.players if player.is_hero), None)
    hero_total = (hero.stack_size + hero.current_bet) if hero else 0.0
    call_amount = snapshot.last_action_context.amount_to_call
    all_in_equivalent = (
        "ALL-IN" in legal_actions
        and action == ("RAISE" if call_amount > 0 else "BET")
        and abs(numeric_size - hero_total) <= 0.01
    )
    if action not in legal_actions and not all_in_equivalent:
        return (
            "Strategy Error: selected action was not independently confirmed "
            "as legal."
        )
    size_error = validate_action_size(
        action,
        numeric_size,
        snapshot,
    )
    return size_error or analysis


def nominal_draw_outs(hand_details: dict) -> int:
    """Return the strongest locally calculated nominal draw count."""
    counts = []
    for draw in hand_details.get("draws") or []:
        match = re.search(r"(\d+)\s+nominal outs", draw, flags=re.IGNORECASE)
        if match:
            counts.append(int(match.group(1)))
    return max(counts, default=0)


def hero_is_playing_board_pair(hero_cards: list[str], board_cards: list[str]) -> bool:
    """True for a single pair supplied by the board rather than Hero's cards."""
    hero = HandEvaluator._unique_cards(hero_cards)
    board = HandEvaluator._unique_cards(board_cards)
    all_cards = HandEvaluator._unique_cards((hero_cards or []) + (board_cards or []))
    if not hero or HandEvaluator._best_rank(all_cards)[0] != 1:
        return False
    paired_board_ranks = {
        rank for rank, _ in board
        if sum(1 for board_rank, _ in board if board_rank == rank) >= 2
    }
    return bool(paired_board_ranks) and not any(
        rank in paired_board_ranks for rank, _ in hero
    )


def apply_deterministic_strategy_guard(
    analysis: str,
    snapshot: GameSnapshot,
    strategy_context: StrategyContext,
    hand_details: dict,
    metrics: dict,
) -> str:
    """Override a narrow, provably unsupported draw-based multiway call."""
    if analysis.startswith("Strategy Error:"):
        return analysis
    action_match = re.search(
        r"^\s*\*{0,2}Action:\*{0,2}\s*(Fold|Check|Call|Bet|Raise)\s*$",
        analysis,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not action_match or action_match.group(1).upper() != "CALL":
        return analysis

    outs = nominal_draw_outs(hand_details)
    pot_odds = float(metrics.get("pot_odds_pct", 0.0) or 0.0)
    hero = next((player for player in snapshot.players if player.is_hero), None)
    turn_board_pair_guard = (
        snapshot.meta_info.current_street == "TURN"
        and hero
        and hero_is_playing_board_pair(
            hero.hole_cards,
            snapshot.board_state.community_cards,
        )
        and strategy_context.opponents_in_hand >= 2
        and 0 < outs <= 4
        and pot_odds >= 25.0
    )
    flop_weak_gutshot_guard = bool(
        snapshot.meta_info.current_street == "FLOP"
        and hero
        and hand_details.get("made_category") == 0
        and strategy_context.opponents_in_hand >= 2
        and 0 < outs <= 4
    )
    if turn_board_pair_guard or flop_weak_gutshot_guard:
        unseen_cards = 52 - len(hero.hole_cards) - len(
            snapshot.board_state.community_cards
        )
        if snapshot.meta_info.current_street == "FLOP" and unseen_cards > 1:
            miss_turn = (unseen_cards - outs) / unseen_cards
            miss_river = (unseen_cards - 1 - outs) / (unseen_cards - 1)
            hit_pct = (1.0 - miss_turn * miss_river) * 100
            horizon = "by the river"
        else:
            hit_pct = (outs / unseen_cards * 100) if unseen_cards > 0 else 0.0
            horizon = "on the river"
        if flop_weak_gutshot_guard and pot_odds < hit_pct + 2.0:
            return analysis
        return (
            "**Action:** Fold\n"
            "**Size:** 0\n"
            f"**Why:** The board pair is shared, and the {outs}-out draw hits only "
            f"{hit_pct:.1f}% {horizon} versus {pot_odds:.1f}% pot odds in a "
            "multiway pot."
        ) if turn_board_pair_guard else (
            "**Action:** Fold\n"
            "**Size:** 0\n"
            f"**Why:** A-high with a {outs}-out gutshot improves only {hit_pct:.1f}% "
            f"by the river versus {pot_odds:.1f}% pot odds, with dirty overcard outs "
            "in a multiway pot."
        )
    return analysis


def validate_parallel_candidate(
    raw_analysis: str,
    snapshot: GameSnapshot,
) -> str:
    """Cross-check all FAST output needed before committing the observation."""
    observation_error = validate_parallel_observation(raw_analysis, snapshot)
    if observation_error:
        return observation_error
    return validate_parallel_action(raw_analysis, snapshot)


def validate_parallel_strategy(
    raw_analysis: str,
    snapshot: GameSnapshot,
    strategy_context: StrategyContext,
) -> str:
    """Cross-check Claude's visual reading against Gemini before using advice."""
    candidate_error = validate_parallel_candidate(raw_analysis, snapshot)
    if candidate_error:
        return candidate_error
    fields = parallel_strategy_fields(raw_analysis)
    hero = next((player for player in snapshot.players if player.is_hero), None)
    action = fields["Action"].strip().strip("[]").strip().upper()

    if action in {"CHECK", "FOLD"}:
        size = "0"
    elif action == "CALL":
        size = f"{snapshot.last_action_context.amount_to_call:g} BB"
    else:
        numeric_size = strict_bb_amount(fields["Size"])
        size = f"{numeric_size:g} BB"
    normalized = (
        f"**Action:** {action.title()}\n"
        f"**Size:** {size}\n"
        f"**Why:** {fields['Why']}"
    )
    return sanitize_strategy_response(normalized, strategy_context)


# Global Settings
PROMPT_MODE = os.getenv("PROMPT_MODE", "FAST").strip().upper()
if PROMPT_MODE not in {"FAST", "COACH"}:
    console.print(
        f"[yellow]Warning: unknown PROMPT_MODE={PROMPT_MODE!r}; using FAST[/yellow]"
    )
    PROMPT_MODE = "FAST"

def generate_strategy_prompt_fast(history_json: str, action_history: str, villain_stats: str, 
                             snapshot: GameSnapshot, metrics: dict, current_action: str,
                             hand_rank: str, strategy_context: StrategyContext) -> str:
    """Compact prompt for fast, decisive answers at micro-stakes."""
    
    hero = next((p for p in snapshot.players if p.is_hero), None)
    hero_cards = " ".join(hero.hole_cards) if hero and hero.hole_cards else "Unknown"
    hero_pos = hero.name if hero else "Unknown"
    board = " ".join(snapshot.board_state.community_cards) or "Preflop"
    street = snapshot.meta_info.current_street
    final_pot = metrics.get("final_pot", 0.0)
    spr = metrics.get("spr", 0.0)
    call_amount = snapshot.last_action_context.amount_to_call
    
    # Bet size category
    bet_cat = ""
    if call_amount > 0 and final_pot > 0:
        ratio = call_amount / final_pot
        if ratio > 0.8: bet_cat = "[OVERBET=STRONG]"
        elif ratio > 0.5: bet_cat = "[BIG BET]"

    return f"""Choose one EV-maximizing NL2-NL10 poker action.

STATE: {street} | {hero_cards} ({hero_pos}) | Board: {board}
TABLE: {strategy_context.pot_type}
OPPONENTS IN HAND: {strategy_context.opponents_in_hand}
ELIGIBLE OPPONENTS: {strategy_context.opponents_eligible_to_act} (not folded/all-in; NOT a remaining-action count)
POSITIONED BEFORE/AFTER HERO: {strategy_context.opponents_before_hero_label}/{strategy_context.opponents_after_hero_label}
CURRENT ACTION PASS: {strategy_context.action_flow}
RELATIVE POSITION: {strategy_context.relative_position}
MADE HAND: {strategy_context.made_hand}
DRAWS: {strategy_context.draws}
BOARD TEXTURE: {strategy_context.board_texture}
POT: {final_pot:.2f}BB | SPR: {spr:.1f}
FACING: {current_action} {bet_cat}
LEGAL ACTIONS: {strategy_context.legal_actions}
PREFLOP AGGRESSOR: {strategy_context.preflop_aggressor}
CURRENT STREET AGGRESSOR: {strategy_context.current_street_aggressor}
C-BET TERMINOLOGY ALLOWED: {strategy_context.cbet_eligibility}
OBSERVED ACTIONS: {action_history or "No reliable action sequence observed."}
VILLAIN READS: {villain_stats or "No reliable profile."}

RULES:
- Use only LEGAL ACTIONS. Treat supplied hand, draws, texture, counts and position as authoritative.
- Never call ELIGIBLE OPPONENTS "still to act"; only AFTER HERO describes positional order.
- Never invent actions, initiative or reads. Use "c-bet" only when explicitly allowed.
- Treat OPPONENTS IN HAND and POSITIONED BEFORE/AFTER HERO as exact; never substitute seated or folded players.
- Do not invent outs: mention an out count only when DRAWS supplies it. `DRAWS: NONE` does not mean "0 outs".
- Preflop, do not call pocket pairs 22-TT "premium" and do not claim future postflop IP/OOP from the preflop label alone.
- Multiway: bluff less and require stronger value/continues. A draw is not air; compare price and outs.
- At micro stakes prefer clear value, avoid marginal pot inflation, and respect large bets without evidence of bluffs.

OUTPUT:
**Action:** [Fold/Check/Call/Bet/Raise]
**Size:** [BB or 0]
**Why:** [One sentence max - the key reason]
"""


def generate_strategy_prompt(history_json: str, action_history: str, villain_stats: str,
                             snapshot: GameSnapshot, metrics: dict, current_action: str,
                             hand_rank: str, strategy_context: StrategyContext,
                             prompt_mode: Optional[str] = None) -> str:
    """Constructs the final prompt for the Strategy LLM with enhanced decision logic."""

    selected_mode = (prompt_mode or PROMPT_MODE).upper()
    if selected_mode not in {"FAST", "COACH"}:
        raise ValueError(f"Unsupported strategy prompt mode: {selected_mode}")
    if selected_mode == "FAST":
        return generate_strategy_prompt_fast(
            history_json, action_history, villain_stats, snapshot, metrics,
            current_action, hand_rank, strategy_context
        )

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
Made Hand: {strategy_context.made_hand}
Draws: {strategy_context.draws}
Board Texture: {strategy_context.board_texture}
Pot: {final_pot:.2f} BB
Facing: {current_action}
Aggressor: {aggressor_name}
Legal Actions: {strategy_context.legal_actions}

[TABLE AND POSITION]
Pot Type: {strategy_context.pot_type}
Opponents in Hand: {strategy_context.opponents_in_hand}
Eligible Opponents: {strategy_context.opponents_eligible_to_act} (NOT a remaining-action count)
Opponents Positioned Before Hero: {strategy_context.opponents_before_hero_label}
Opponents Positioned After Hero: {strategy_context.opponents_after_hero_label}
Current Action Pass: {strategy_context.action_flow}
Relative Position: {strategy_context.relative_position}
Preflop Aggressor: {strategy_context.preflop_aggressor}
Current Street Aggressor: {strategy_context.current_street_aggressor}
C-bet Terminology Allowed: {strategy_context.cbet_eligibility}

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
   - A draw is not air. Use the supplied draw type and price rather than the made-hand label alone.

3. **POSITION ADJUSTMENTS**
   - Use the supplied Relative Position; a position label alone does not prove IP/OOP
   - OOP or sandwiched: check more and bluff less, especially multiway
   - IP heads-up: apply more pressure when initiative and texture support it
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

6. **RELIABILITY AND MULTIWAY LOGIC**
   - Never invent an action, aggressor, check, or player tendency that is not supplied
   - Use "c-bet" only when C-bet Terminology Allowed is YES
   - Multiway: bluff much less and require stronger value/continue thresholds
   - Treat Made Hand, Draws, Board Texture, and opponent count as authoritative

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


STRATEGY_SYSTEM_PROMPT = (
    "You are an expert poker decision assistant. Use the supplied deterministic "
    "facts as authoritative. Never invent action history, turn order, or initiative. "
    "Use only listed legal actions and return exactly the requested format."
)


@dataclass
class StrategyEvaluation:
    """Complete, inspectable result of one structured strategy decision."""

    mode: str
    model: str
    prompt: str
    raw_analysis: str
    sanitized_analysis: str
    validated_analysis: str
    final_analysis: str
    metrics: dict
    hand_rank: str
    hand_details: dict
    strategy_context: StrategyContext
    latency_seconds: float
    error: str = ""
    source: str = ""


def evaluate_strategy_snapshot(
    snapshot: GameSnapshot,
    history: HandHistory,
    *,
    mode: Optional[str] = None,
    client=None,
    action_history_override: Optional[str] = None,
    villain_context: str = "",
    apply_guards: bool = True,
) -> StrategyEvaluation:
    """Evaluate one already-reconstructed state without capture or Vision.

    This is the shared strategy entry point for both the live flow and offline
    benchmarks. The caller owns ``snapshot`` and ``history``; neither is
    mutated here.
    """

    selected_mode = (mode or PROMPT_MODE).upper()
    if selected_mode not in {"FAST", "COACH"}:
        raise ValueError(f"Unsupported strategy mode: {selected_mode}")

    started = time.perf_counter()
    prepared = prepare_strategy_state(snapshot, history)
    hero = prepared.hero
    final_pot = prepared.final_pot
    eff_stack = prepared.effective_stack
    hand_details = prepared.hand_details
    hand_rank = prepared.hand_rank
    strategy_context = prepared.strategy_context
    metrics = prepared.metrics
    current_action = prepared.current_action

    history_full = history.to_json()
    history_min = history.to_min_json()
    action_summary = (
        action_history_override
        if action_history_override is not None
        else parse_action_history(history_full)
    )
    strategy_prompt = generate_strategy_prompt(
        history_min,
        action_summary,
        villain_context,
        snapshot,
        metrics,
        current_action,
        hand_rank,
        strategy_context,
        prompt_mode=selected_mode,
    )

    model = CLAUDE_FAST_MODEL if selected_mode == "FAST" else CLAUDE_MODEL
    max_tokens = 160 if selected_mode == "FAST" else 600
    selected_client = client if client is not None else anthropic_client
    raw_analysis = ""
    sanitized_analysis = ""
    validated_analysis = ""
    final_analysis = ""
    error_text = ""

    try:
        if selected_client is None:
            raise RuntimeError("ANTHROPIC_API_KEY missing")
        request_client = selected_client
        if selected_mode == "FAST" and hasattr(selected_client, "with_options"):
            request_client = selected_client.with_options(
                timeout=CLAUDE_FAST_TIMEOUT_SECONDS,
                max_retries=0,
            )
        response = request_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=STRATEGY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": strategy_prompt}],
        )
        raw_analysis = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not raw_analysis:
            raise ValueError("Claude returned no text content")
        sanitized_analysis = sanitize_strategy_response(
            raw_analysis,
            strategy_context,
        )
        validated_analysis = validate_strategy_amount(
            sanitized_analysis,
            snapshot,
        )
        final_analysis = (
            apply_deterministic_strategy_guard(
                validated_analysis,
                snapshot,
                strategy_context,
                hand_details,
                metrics,
            )
            if apply_guards else validated_analysis
        )
    except Exception as error:
        error_text = str(error)
        final_analysis = f"Strategy Error: {error_text}"

    return StrategyEvaluation(
        mode=selected_mode,
        model=model,
        prompt=strategy_prompt,
        raw_analysis=raw_analysis,
        sanitized_analysis=sanitized_analysis,
        validated_analysis=validated_analysis,
        final_analysis=final_analysis,
        metrics=metrics,
        hand_rank=hand_rank,
        hand_details=hand_details,
        strategy_context=strategy_context,
        latency_seconds=time.perf_counter() - started,
        error=error_text,
        source=(
            f"FAST {CLAUDE_FAST_MODEL}"
            if selected_mode == "FAST"
            else f"COACH {CLAUDE_MODEL}"
        ),
    )


def live_gto_street_root_confirmed(
    snapshot: GameSnapshot,
    history: HandHistory,
) -> bool:
    """Accept repeated identical captures, reject progressed same-street nodes."""

    def signature(state: GameSnapshot):
        players = tuple(
            sorted(
                (
                    player.seat_index,
                    player.status,
                    round(player.stack_size, 3),
                    round(player.current_bet, 3),
                )
                for player in state.players
            )
        )
        return (
            tuple(state.board_state.community_cards),
            round(state.board_state.total_pot, 3),
            round(state.last_action_context.amount_to_call, 3),
            players,
        )

    current_signature = signature(snapshot)
    for previous in history.snapshots:
        if previous is snapshot:
            continue
        if previous.meta_info.current_street != snapshot.meta_info.current_street:
            continue
        if tuple(previous.board_state.community_cards) != tuple(
            snapshot.board_state.community_cards
        ):
            continue
        if signature(previous) != current_signature:
            return False
    return True


def _live_bb_units(value) -> int:
    """Normalize noisy binary floats to the UI's centi-BB precision."""

    return int(
        (Decimal(str(value)) * Decimal(100)).quantize(
            Decimal(1), rounding=ROUND_HALF_UP
        )
    )


def _live_same_street_root(
    snapshot: GameSnapshot,
    history: HandHistory,
    hero: Player,
    villain: Player,
) -> Optional[GameSnapshot]:
    """Find a prior, fully conserved Hero-OOP street-root observation."""

    current_root_pot = _live_bb_units(snapshot.board_state.total_pot) - sum(
        _live_bb_units(player.current_bet) for player in snapshot.players
    )
    if current_root_pot <= 0:
        return None

    for previous in reversed(history.snapshots):
        if previous is snapshot:
            continue
        if previous.meta_info.current_street != snapshot.meta_info.current_street:
            continue
        if tuple(previous.board_state.community_cards) != tuple(
            snapshot.board_state.community_cards
        ):
            continue
        if (
            previous.dealer_seat_index < 0
            or previous.dealer_seat_index != snapshot.dealer_seat_index
        ):
            continue
        prior_hero = next(
            (player for player in previous.players if player.seat_index == hero.seat_index),
            None,
        )
        prior_villain = next(
            (
                player
                for player in previous.players
                if player.seat_index == villain.seat_index
            ),
            None,
        )
        if not prior_hero or not prior_villain or not prior_hero.is_hero:
            continue
        prior_live_villains = [
            player
            for player in previous.players
            if not player.is_hero and player.status in {"ACTIVE", "ALL_IN"}
        ]
        if [player.seat_index for player in prior_live_villains] != [
            prior_villain.seat_index
        ]:
            continue
        if previous.action_on_seat_index != prior_hero.seat_index:
            continue
        if relative_postflop_position(previous, prior_hero, [prior_villain]) != "OOP":
            continue
        prior_legal = {
            canonical_poker_action(action)
            for action in previous.last_action_context.hero_action_options
        }
        if not {"CHECK", "BET"}.issubset(prior_legal) and not {
            "CHECK", "ALL-IN"
        }.issubset(prior_legal):
            continue
        if _live_bb_units(previous.last_action_context.amount_to_call) != 0:
            continue
        if any(_live_bb_units(player.current_bet) != 0 for player in previous.players):
            continue
        if canonical_poker_action(prior_villain.visible_action):
            continue
        if _live_bb_units(previous.board_state.total_pot) != current_root_pot:
            continue
        if _live_bb_units(prior_hero.stack_size) != (
            _live_bb_units(hero.stack_size) + _live_bb_units(hero.current_bet)
        ):
            continue
        if _live_bb_units(prior_villain.stack_size) != (
            _live_bb_units(villain.stack_size) + _live_bb_units(villain.current_bet)
        ):
            continue
        return previous
    return None


def build_live_gto_state(
    snapshot: GameSnapshot,
    history: HandHistory,
    prepared: PreparedStrategyState,
) -> LiveDecisionState:
    preflop_observation = None
    preflop_mapping_error = ""
    try:
        if snapshot.meta_info.current_street == "PREFLOP":
            preflop_observation = current_preflop_observation(snapshot)
        else:
            preflop_observation = terminal_preflop_observation(
                history,
                snapshot,
                require_heads_up=True,
            )
    except PreflopObservationError as error:
        preflop_mapping_error = str(error)

    nonfolded_villains = [
        player for player in snapshot.players
        if not player.is_hero and player.status in {"ACTIVE", "ALL_IN"}
    ]
    active_villains = [
        player for player in nonfolded_villains if player.status == "ACTIVE"
    ]
    villain = active_villains[0] if len(active_villains) == 1 else (
        nonfolded_villains[0] if len(nonfolded_villains) == 1 else None
    )

    if snapshot.meta_info.current_street == "PREFLOP":
        deepest_villain = max(
            nonfolded_villains,
            key=lambda player: player.stack_size,
            default=None,
        )
        return LiveDecisionState(
            hand_id=snapshot.hand_id,
            street="PREFLOP",
            board=(),
            hero_combo=tuple(prepared.hero.hole_cards),
            hero_position=(prepared.hero.name or "").upper(),
            villain_position=(deepest_villain.name or "").upper()
            if deepest_villain else "",
            hero_is_oop=False,
            active_villains=len(nonfolded_villains),
            pot_bb=Decimal(str(prepared.final_pot)),
            hero_stack_bb=Decimal(str(prepared.hero.stack_size)),
            villain_stack_bb=Decimal(str(deepest_villain.stack_size))
            if deepest_villain else Decimal(0),
            hero_current_bet_bb=Decimal(str(prepared.hero.current_bet)),
            villain_current_bet_bb=max(
                (Decimal(str(player.current_bet)) for player in nonfolded_villains),
                default=Decimal(0),
            ),
            amount_to_call_bb=Decimal(
                str(snapshot.last_action_context.amount_to_call)
            ),
            legal_actions=tuple(
                canonical_poker_action(action)
                for action in snapshot.last_action_context.hero_action_options
            ),
            street_root_confirmed=False,
            preflop_observation=preflop_observation,
            preflop_mapping_error=preflop_mapping_error,
        )

    current_total_units = _live_bb_units(snapshot.board_state.total_pot)
    contribution_units = {
        player.seat_index: _live_bb_units(player.current_bet)
        for player in snapshot.players
    }
    hero_bet_units = contribution_units.get(prepared.hero.seat_index, 0)
    villain_bet_units = contribution_units.get(villain.seat_index, 0) if villain else 0
    other_bet_units = sum(
        amount
        for seat, amount in contribution_units.items()
        if seat not in {
            prepared.hero.seat_index,
            villain.seat_index if villain else -1,
        }
    )
    call_units = _live_bb_units(snapshot.last_action_context.amount_to_call)
    root_pot_units = current_total_units - sum(contribution_units.values())
    root_hero_stack_units = (
        _live_bb_units(prepared.hero.stack_size) + hero_bet_units
    )
    root_villain_stack_units = (
        _live_bb_units(villain.stack_size) + villain_bet_units if villain else 0
    )
    action_history: tuple[str, ...] = ()
    observed_bet_units = 0
    mapping_error = ""
    root_confirmed = live_gto_street_root_confirmed(snapshot, history)

    if villain is None:
        mapping_error = "the live state does not identify exactly one HU villain"
    elif other_bet_units:
        mapping_error = "a folded/third player still has a same-street contribution"
    elif call_units == 0 and hero_bet_units == 0 and villain_bet_units == 0:
        root_pot_units = current_total_units
        if prepared.strategy_context.relative_position == "OOP":
            if not root_confirmed:
                mapping_error = "same-street history does not confirm an untouched OOP root"
            elif canonical_poker_action(villain.visible_action):
                mapping_error = "villain overlay contradicts an untouched OOP root"
        elif prepared.strategy_context.relative_position == "IP":
            if not root_confirmed:
                mapping_error = "a prior same-street Hero decision makes IP-after-check ambiguous"
            else:
                action_history = ("CHECK",)
                villain_action = canonical_poker_action(villain.visible_action)
                if villain_action not in {"", "CHECK"}:
                    mapping_error = "villain overlay contradicts the inferred OOP check"
        else:
            mapping_error = "postflop HU position is not confirmed"
    elif (
        call_units > 0
        and hero_bet_units == 0
        and villain_bet_units > 0
        and call_units == villain_bet_units
    ):
        observed_bet_units = villain_bet_units
        villain_action = canonical_poker_action(villain.visible_action)
        if villain_action not in {"", "BET"}:
            mapping_error = "villain overlay contradicts the inferred first bet"
        elif prepared.strategy_context.relative_position == "IP":
            if not root_confirmed:
                mapping_error = "a prior same-street Hero decision makes the first OOP bet ambiguous"
            else:
                action_history = ("BET",)
        elif prepared.strategy_context.relative_position == "OOP":
            prior_root = _live_same_street_root(
                snapshot,
                history,
                prepared.hero,
                villain,
            )
            if prior_root is None:
                mapping_error = "OOP facing an IP bet requires a conserved prior Hero-check root"
            else:
                prior_hero = next(player for player in prior_root.players if player.is_hero)
                prior_villain = next(
                    player
                    for player in prior_root.players
                    if player.seat_index == villain.seat_index
                )
                root_pot_units = _live_bb_units(prior_root.board_state.total_pot)
                root_hero_stack_units = _live_bb_units(prior_hero.stack_size)
                root_villain_stack_units = _live_bb_units(prior_villain.stack_size)
                action_history = ("CHECK", "BET")
        else:
            mapping_error = "postflop HU position is not confirmed"
    else:
        mapping_error = (
            "current contributions require an unsupported prior call, bet, or raise"
        )

    if not mapping_error and root_pot_units <= 0:
        mapping_error = "inclusive pot does not leave a positive street-root pot"

    return LiveDecisionState(
        hand_id=snapshot.hand_id,
        street=snapshot.meta_info.current_street,
        board=tuple(snapshot.board_state.community_cards),
        hero_combo=tuple(prepared.hero.hole_cards),
        hero_position=(prepared.hero.name or "").upper(),
        villain_position=(villain.name or "").upper() if villain else "",
        hero_is_oop=prepared.strategy_context.relative_position == "OOP",
        active_villains=len(nonfolded_villains),
        pot_bb=Decimal(root_pot_units) / Decimal(100),
        hero_stack_bb=Decimal(root_hero_stack_units) / Decimal(100),
        villain_stack_bb=Decimal(root_villain_stack_units) / Decimal(100),
        hero_current_bet_bb=Decimal(hero_bet_units) / Decimal(100),
        villain_current_bet_bb=Decimal(villain_bet_units) / Decimal(100),
        amount_to_call_bb=Decimal(call_units) / Decimal(100),
        legal_actions=tuple(
            canonical_poker_action(action)
            for action in snapshot.last_action_context.hero_action_options
        ),
        street_root_confirmed=root_confirmed,
        action_history=action_history,
        observed_bet_to_bb=Decimal(observed_bet_units) / Decimal(100),
        mapping_error=mapping_error,
        preflop_observation=preflop_observation,
        preflop_mapping_error=preflop_mapping_error,
    )


def evaluate_strategy_backend(
    snapshot: GameSnapshot,
    history: HandHistory,
    *,
    mode: Optional[str] = None,
    backend: Optional[str] = None,
    router: Optional[LiveGTORouter] = None,
    client=None,
) -> StrategyEvaluation:
    """Use GTO on supported simulator nodes and make every fallback explicit."""

    selected_backend = (backend or STRATEGY_BACKEND).upper()
    if selected_backend == "CLAUDE":
        return evaluate_strategy_snapshot(snapshot, history, mode=mode, client=client)
    if selected_backend not in {"HYBRID", "GTO"}:
        raise ValueError(f"Unsupported strategy backend: {selected_backend}")

    started = time.perf_counter()
    prepared = prepare_strategy_state(snapshot, history)
    selected_router = router if router is not None else live_gto_router
    if selected_router is None:
        reason = live_gto_config_error or "live GTO router is not configured"
        outcome = None
    else:
        live_state = build_live_gto_state(snapshot, history, prepared)
        outcome = selected_router.evaluate(live_state)
        reason = outcome.reason

    if outcome is not None and outcome.solved:
        validated_gto_analysis = validate_strategy_amount(
            outcome.analysis,
            snapshot,
        )
        if validated_gto_analysis.startswith("Strategy Error:"):
            reason = (
                "solver action failed live legality validation: "
                + validated_gto_analysis.removeprefix("Strategy Error:").strip()
            )
        else:
            solver_mode = (
                "APPROXIMATE_SOLVER"
                if outcome.source.lower().startswith("approximate")
                else "GTO"
            )
            prompt = json.dumps(
                {
                    "backend": solver_mode,
                    "usage": "user_controlled_simulator_live",
                    "state": dataclasses.asdict(live_state),
                    "spec_key": outcome.spec.cache_key if outcome.spec else "",
                },
                indent=2,
                default=str,
            )
            return StrategyEvaluation(
                mode=solver_mode,
                model=getattr(outcome, "model", "") or "b-inary/postflop-solver",
                prompt=prompt,
                raw_analysis=outcome.analysis,
                sanitized_analysis=outcome.analysis,
                validated_analysis=validated_gto_analysis,
                final_analysis=validated_gto_analysis,
                metrics=prepared.metrics,
                hand_rank=prepared.hand_rank,
                hand_details=prepared.hand_details,
                strategy_context=prepared.strategy_context,
                latency_seconds=time.perf_counter() - started,
                source=outcome.source,
            )

    status = (
        "failed"
        if outcome is not None and outcome.solved
        else outcome.status.value.lower()
        if outcome is not None
        else "unavailable"
    )
    fallback_reason = f"GTO {status}: {reason}"
    if selected_backend == "HYBRID":
        fallback = evaluate_strategy_snapshot(
            snapshot,
            history,
            mode=mode,
            client=client,
        )
        fallback.latency_seconds = time.perf_counter() - started
        fallback.source = f"{fallback.source} fallback — {fallback_reason}"
        fallback.final_analysis = (
            f"{fallback.final_analysis}\n"
            f"* **Backend:** Claude fallback — {fallback_reason}"
        )
        return fallback

    return StrategyEvaluation(
        mode="GTO",
        model="b-inary/postflop-solver",
        prompt="",
        raw_analysis="",
        sanitized_analysis="",
        validated_analysis="",
        final_analysis=f"Strategy Error: {fallback_reason}",
        metrics=prepared.metrics,
        hand_rank=prepared.hand_rank,
        hand_details=prepared.hand_details,
        strategy_context=prepared.strategy_context,
        latency_seconds=time.perf_counter() - started,
        error=fallback_reason,
        source=fallback_reason,
    )


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
        t_vision = 0.0
        validated_captures = None
        snapshot, comps, t_capture, t_vision = analyze_state(
            monitor_num,
            require_hero_turn=(mode == "strategy"),
        )

        # Local preflight: pre-action checkboxes and already-finished turns do
        # not need a Gemini or Claude request and must not enter hand history.
        if mode == "strategy" and not detect_hero_action_buttons(comps["buttons"]):
            t_total = time.time() - start_total
            display_results(
                GameSnapshot(),
                "No current Hero decision detected.",
                t_capture,
                t_vision,
                0.0,
                t_total,
            )
            return

        # Validate the candidate before it can reset or mutate hand history.
        # A street transition during vision discards the entire observation.
        if mode == "strategy":
            if validated_captures is None:
                validated_captures = capture_validation_regions(monitor_num)
            stale_reasons = table_state_change_reasons(comps, validated_captures)
            if stale_reasons:
                save_stale_debug_capture(
                    comps,
                    validated_captures,
                    stale_reasons,
                    "after_vision",
                )
                t_total = time.time() - start_total
                display_stale_result(
                    t_capture,
                    t_vision,
                    0.0,
                    t_total,
                    stale_reasons,
                )
                return

        if mode == "strategy":
            preserve_hero_cards_on_continuing_board(snapshot, current_history)

        candidate_errors = validate_snapshot_candidate(
            snapshot,
            require_hero_hand=(mode == "strategy"),
        )
        if candidate_errors:
            t_total = time.time() - start_total
            display_results(
                snapshot,
                "Vision Error: " + "; ".join(candidate_errors),
                t_capture,
                t_vision,
                0.0,
                t_total,
            )
            return

        if mode == "strategy":
            candidate_hero = next(
                (player for player in snapshot.players if player.is_hero),
                None,
            )
            if not candidate_hero or len(candidate_hero.hole_cards or []) != 2:
                t_total = time.time() - start_total
                display_results(
                    snapshot,
                    "No Hero Hand Detected.",
                    t_capture,
                    t_vision,
                    0.0,
                    t_total,
                )
                return

            if (
                snapshot.action_on_seat_index != 4
                or not snapshot.last_action_context.hero_action_options
            ):
                t_total = time.time() - start_total
                display_results(
                    snapshot,
                    "No current Hero decision detected.",
                    t_capture,
                    t_vision,
                    0.0,
                    t_total,
                )
                return

        if current_history.snapshots and current_history.is_new_hand(snapshot):
            current_history.reset()
            console.print("[dim]New hand detected; history reset[/dim]")
        else:
            reconcile_snapshot_with_history(snapshot, current_history)
        
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
        if recorder:
            recorder.update(snapshot)  # Log to file
        console.print(f"[dim]📝 Turn {len(current_history.snapshots)} added to history[/dim]")
        
        analysis = ""
        t_strategy = 0.0
        strategy_source = ""
        metrics = None
        hand_rank = ""  # Initialize here to avoid reference error
        
        # 3. Strategy (Optional) - Uses FULL HISTORY
        if mode == "strategy":
            hero = next((p for p in snapshot.players if p.is_hero), None)
            
            if not hero or not hero.hole_cards:
                analysis = "No Hero Hand Detected."
            elif (
                snapshot.action_on_seat_index != 4
                or not snapshot.last_action_context.hero_action_options
            ):
                analysis = "No current Hero decision detected."
            else:
                strategy_result = evaluate_strategy_backend(
                    snapshot,
                    current_history,
                    mode=PROMPT_MODE,
                )
                analysis = strategy_result.final_analysis
                metrics = strategy_result.metrics
                hand_rank = strategy_result.hand_rank
                t_strategy = strategy_result.latency_seconds
                strategy_source = strategy_result.source
                strategy_prompt = strategy_result.prompt

                with open("debug_strategy_prompt.txt", "w") as f:
                    f.write(strategy_prompt)
                with open("debug_strategy_response.txt", "w") as f:
                    f.write(analysis)

                final_captures = capture_validation_regions(monitor_num)
                stale_reasons = table_state_change_reasons(
                    validated_captures, final_captures
                )
                if stale_reasons:
                    save_stale_debug_capture(
                        validated_captures,
                        final_captures,
                        stale_reasons,
                        "before_display",
                    )
                    t_total = time.time() - start_total
                    display_stale_result(
                        t_capture,
                        t_vision,
                        t_strategy,
                        t_total,
                        stale_reasons,
                        strategy_source,
                    )
                    return

        t_total = time.time() - start_total
        
        # 4. Display
        display_results(
            snapshot,
            analysis,
            t_capture,
            t_vision,
            t_strategy,
            t_total,
            metrics,
            hand_rank,
            strategy_source,
        )
        
        # Show history summary
        # console.print(f"\n[dim]{current_history.summary()}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")



def start_analysis_flow(mode: str) -> None:
    """Run one analysis off the keyboard listener and ignore queued repeats."""
    if not analysis_lock.acquire(blocking=False):
        console.print("[dim]Analysis already running; key ignored.[/dim]")
        return

    def worker():
        try:
            run_analysis_flow(mode=mode)
        finally:
            analysis_lock.release()

    threading.Thread(target=worker, daemon=True).start()


def on_press(key):
    global running, input_mode, note_player
    try:
        if hasattr(key, 'char') and key.char:
            char = key.char.lower()
            
            if char == 'j':
                # J = Strategy Mode
                start_analysis_flow(mode="strategy")
                
            elif char == 'l':
                # L = Debug Snapshot (Vision Check)
                start_analysis_flow(mode="debug")
                    
            elif char == 'p':
                if analysis_lock.locked():
                    console.print("[dim]Analysis running; chart key ignored.[/dim]")
                    return
                # Show preflop chart for last detected position
                pos = None
                if last_state:
                     hero = next((p for p in last_state.players if p.is_hero), None)
                     pos = hero.name if hero else None
                show_preflop_chart(pos)
            elif char == 'n':
                if analysis_lock.locked():
                    console.print("[dim]Analysis running; reset key ignored.[/dim]")
                    return
                # Reset hand history (NEW HAND)
                current_history.reset()
                console.print("\n[bold green]🆕 HAND HISTORY RESET - Ready for new hand![/bold green]")
            elif char == 'm':
                if analysis_lock.locked():
                    console.print("[dim]Analysis running; mode key ignored.[/dim]")
                    return
                # Toggle Prompt Mode
                global PROMPT_MODE
                if PROMPT_MODE == "COACH":
                    PROMPT_MODE = "FAST"
                else:
                    PROMPT_MODE = "COACH"
                model_label = (
                    f"{CLAUDE_FAST_MODEL}, sequential"
                    if PROMPT_MODE == "FAST"
                    else CLAUDE_MODEL
                )
                console.print(
                    f"\n[bold magenta]Mode Switched: {PROMPT_MODE} "
                    f"({model_label})[/bold magenta]"
                )
            elif char == 'q':
                running = False
                return False
                
        elif key == keyboard.Key.esc:
            running = False
            return False
    except AttributeError:
        pass


def main():
    global monitor_num, recorder

    if recorder is None:
        recorder = GameRecorder()
    
    # Parse monitor arg
    if len(sys.argv) > 1:
        try:
            monitor_num = int(sys.argv[1])
        except ValueError:
            pass
    
    # Show monitors
    with mss.MSS() as sct:
        console.print(f"[cyan]Monitors: {len(sct.monitors) - 1}[/cyan]")
        for i, m in enumerate(sct.monitors[1:], 1):
            console.print(f"  {i}: {m['width']}x{m['height']}")
    
    console.print(f"\n[bold green]♠ Poker Range Assistant ♠[/bold green]")
    console.print(f"[white]Monitor: {monitor_num}[/white]")
    console.print(
        f"[white]Mode: {PROMPT_MODE} | Backend: {STRATEGY_BACKEND}[/white]"
    )
    console.print(
        "[yellow]'l' SNAPSHOT | 'j' STRATEGY | 'n' NEW HAND | "
        "'p' CHART | 'm' CLAUDE MODE | ESC quit[/yellow]\n"
    )
    
    # Start keyboard listener
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    
    console.print("\n[green]Good luck at the tables! 🍀[/green]")


if __name__ == "__main__":
    main()
