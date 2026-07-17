#!/usr/bin/env python3
"""Offline, resumable PokerBench evaluation for the configured Claude models.

This module never captures the screen and never calls Gemini. It downloads only
the two held-out structured CSV splits from a pinned PokerBench revision, builds
neutral solver-alignment prompts, and measures agreement with the published
solver labels. PokerBench contains one selected action per case, not per-action
EVs or full mixed frequencies, so the report intentionally avoids claiming GTO
regret or exploitability.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Iterable, Optional

from anthropic import Anthropic
from dotenv import load_dotenv


POKERBENCH_REVISION = "7ac61f961c81a50fc0f667820b2fb0e432dfec0d"
ADAPTER_VERSION = 2
PROMPT_VERSION = 3
CACHE_SCHEMA_VERSION = 3
RESULT_SCHEMA_VERSION = 1
SELECTION_SCHEME = "proportional_stratified_v1"
RESPONSE_FORMAT = "structured_json_direct_v1"

DATASET_URL_ROOT = (
    "https://huggingface.co/datasets/RZ412/PokerBench/resolve/"
    f"{POKERBENCH_REVISION}"
)


@dataclass(frozen=True)
class DatasetFile:
    filename: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"{DATASET_URL_ROOT}/{self.filename}?download=true"


DATASET_FILES = {
    "preflop": DatasetFile(
        "preflop_1k_test_set_game_scenario_information.csv",
        81_300,
        "38710b92bbefbaa881c8b16e0d617152dc3606866ed9a88b3687b1194f2203e5",
    ),
    "postflop": DatasetFile(
        "postflop_10k_test_set_game_scenario_information.csv",
        1_667_973,
        "cfa73be8948c9927e6f3a7d73c26dd23160d04bb9cc9691ff6408c04cb4178a2",
    ),
}

SYSTEM_PROMPT = """You are being evaluated offline on solver-labelled No-Limit Hold'em decisions.
Treat the supplied state and legal decisions as authoritative. Use no population reads or exploitative assumptions.
Choose exactly one listed legal decision. Set amount to the listed numeric size
for BET or RAISE and to 0 for FOLD, CHECK, CALL, or ALL_IN.
"""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["FOLD", "CHECK", "CALL", "BET", "RAISE", "ALL_IN"],
        },
        "amount": {"type": "number"},
    },
    "required": ["action", "amount"],
    "additionalProperties": False,
}

DECISION_OUTPUT_CONFIG = {
    "format": {
        "type": "json_schema",
        "schema": DECISION_SCHEMA,
    }
}


class PokerBenchDataError(ValueError):
    """Raised when a source row cannot be safely interpreted."""


@dataclass(frozen=True)
class Move:
    action: str
    amount: Optional[Decimal] = None

    def display(self) -> str:
        if self.action == "ALL_IN":
            return "All-in"
        if self.amount is None:
            return self.action.title()
        amount = format(self.amount.normalize(), "f")
        if self.action == "RAISE":
            return f"Raise to {amount} BB"
        return f"Bet {amount} BB"

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "amount": str(self.amount) if self.amount is not None else None,
        }


@dataclass(frozen=True)
class PokerBenchCase:
    case_id: str
    split: str
    source_index: int
    street: str
    prompt: str
    target: Move
    legal_moves: tuple[Move, ...]
    metadata: dict


@dataclass(frozen=True)
class Completion:
    text: str
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    response_model: str = ""
    stop_reason: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_file(path: Path, descriptor: DatasetFile) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != descriptor.size:
        raise PokerBenchDataError(
            f"{path.name}: expected {descriptor.size} bytes, found {size}"
        )
    digest = _sha256(path)
    if digest != descriptor.sha256:
        raise PokerBenchDataError(
            f"{path.name}: SHA-256 mismatch ({digest})"
        )


def download_dataset(
    data_dir: Path,
    *,
    force: bool = False,
    split: str = "both",
) -> dict[str, Path]:
    """Download and verify only the two held-out structured CSV files."""

    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    wanted = (
        DATASET_FILES.items()
        if split == "both"
        else [(split, DATASET_FILES[split])]
    )
    for current_split, descriptor in wanted:
        destination = data_dir / descriptor.filename
        if destination.exists() and not force:
            verify_dataset_file(destination, descriptor)
            downloaded[current_split] = destination
            continue

        request = urllib.request.Request(
            descriptor.url,
            headers={"User-Agent": "Ranges-PokerBench/1.0"},
        )
        temporary_path = None
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=data_dir,
                    prefix=f".{descriptor.filename}.",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        temporary.write(chunk)
            verify_dataset_file(temporary_path, descriptor)
            temporary_path.replace(destination)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
        downloaded[current_split] = destination
    return downloaded


def dataset_paths(
    data_dir: Path,
    *,
    split: str = "both",
    require_all: bool = True,
) -> dict[str, Path]:
    paths = {
        split: data_dir / descriptor.filename
        for split, descriptor in DATASET_FILES.items()
    }
    if require_all:
        wanted = DATASET_FILES if split == "both" else {split: DATASET_FILES[split]}
        for current_split, descriptor in wanted.items():
            verify_dataset_file(paths[current_split], descriptor)
    return paths


def _clean_spaces(value: object) -> str:
    return re.sub(r"\s+", " ", str("" if value is None else value)).strip()


def _decimal(value: object, *, field: str) -> Decimal:
    text = _clean_spaces(value).replace(" ", "")
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise PokerBenchDataError(f"{field}: invalid number {value!r}") from error
    if not number.is_finite() or number < 0:
        raise PokerBenchDataError(f"{field}: expected a finite non-negative number")
    return number


def _parse_cards(value: object, *, count: int, field: str) -> tuple[str, ...]:
    text = re.sub(r"\s+", "", str(value or ""))
    cards = tuple(re.findall(r"[2-9TJQKA][cdhs]", text, flags=re.IGNORECASE))
    canonical = tuple(card[0].upper() + card[1].lower() for card in cards)
    if len(canonical) != count or "".join(cards).lower() != text.lower():
        raise PokerBenchDataError(
            f"{field}: expected {count} canonical cards, found {value!r}"
        )
    if len(set(canonical)) != len(canonical):
        raise PokerBenchDataError(f"{field}: duplicate card")
    return canonical


def _parse_position(value: object, *, role_only: bool = False) -> str:
    position = re.sub(r"\s+", "", str(value or "")).upper()
    allowed = {"IP", "OOP"} if role_only else {"UTG", "HJ", "CO", "BTN", "SB", "BB"}
    if position not in allowed:
        raise PokerBenchDataError(f"invalid position {value!r}")
    return position


def parse_move(value: object, *, split: str) -> Move:
    """Normalize one PokerBench label or legal move."""

    text = _clean_spaces(value).strip("[]").strip()
    compact = text.lower().replace("-", "_").replace(" ", "")
    plain = {
        "fold": "FOLD",
        "check": "CHECK",
        "call": "CALL",
        "allin": "ALL_IN",
        "all_in": "ALL_IN",
    }
    if compact in plain:
        return Move(plain[compact])

    if split == "preflop":
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*bb", text, flags=re.IGNORECASE)
        if match:
            amount = _decimal(match.group(1), field="move size")
            if amount <= 0:
                raise PokerBenchDataError("raise size must be positive")
            return Move("RAISE", amount)
    else:
        match = re.fullmatch(
            r"(bet|raise)\s+(\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            amount = _decimal(match.group(2), field="move size")
            if amount <= 0:
                raise PokerBenchDataError(f"{match.group(1)} size must be positive")
            return Move(
                match.group(1).upper(),
                amount,
            )

    raise PokerBenchDataError(f"unsupported {split} move {value!r}")


def parse_available_moves(value: object, *, split: str) -> tuple[Move, ...]:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise PokerBenchDataError("available_moves is not a literal list") from error
    if not isinstance(parsed, list) or not parsed or not all(
        isinstance(item, str) for item in parsed
    ):
        raise PokerBenchDataError("available_moves must be a non-empty string list")
    moves = tuple(parse_move(item, split=split) for item in parsed)
    if len(set(moves)) != len(moves):
        raise PokerBenchDataError("available_moves contains duplicates")
    return moves


def _canonical_action_line(value: object) -> str:
    text = _clean_spaces(value)
    return text if text else "No prior voluntary action"


def _truncate_postflop_action(
    value: object,
    street: str,
    *,
    expected_dealt: tuple[str, ...] = (),
) -> str:
    """Remove any source tokens that occur after the evaluated street."""

    wanted_deals = {"FLOP": 0, "TURN": 1, "RIVER": 2}[street]
    tokens = [token.strip() for token in str(value or "").split("/") if token.strip()]
    kept = []
    dealt_cards = []
    deal_count = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.lower() == "dealcards":
            if deal_count >= wanted_deals:
                break
            if index + 1 >= len(tokens):
                raise PokerBenchDataError("postflop_action ends after dealcards")
            dealt = _parse_cards(
                tokens[index + 1],
                count=1,
                field="postflop dealt card",
            )[0]
            kept.extend(["dealcards", dealt])
            dealt_cards.append(dealt)
            deal_count += 1
            index += 2
            continue
        kept.append(_clean_spaces(token))
        index += 1
    if deal_count != wanted_deals:
        raise PokerBenchDataError(
            f"postflop_action contains {deal_count} dealt cards before {street}, "
            f"expected {wanted_deals}"
        )
    if expected_dealt and tuple(dealt_cards) != expected_dealt:
        raise PokerBenchDataError(
            "postflop_action dealt cards disagree with the board columns"
        )
    return "/".join(kept) if kept else "No prior postflop action"


def _build_preflop_prompt(
    *,
    previous_action: str,
    hero_position: str,
    holding: tuple[str, ...],
    pot: Decimal,
    players: int,
    legal_moves: tuple[Move, ...],
) -> str:
    legal = ", ".join(move.display() for move in legal_moves)
    return f"""Game: six-handed No-Limit Hold'em, 100 BB starting stacks, 0.5/1 BB blinds.
Street: PREFLOP
Previous action: {previous_action}
Hero position: {hero_position}
Hero cards: {' '.join(holding)}
Players represented in this branch: {players}
Current pot: {pot} BB
Legal decisions: {legal}
Choose the solver-aligned decision from the legal list."""


def _build_postflop_prompt(
    *,
    preflop_action: str,
    postflop_action: str,
    street: str,
    board: tuple[str, ...],
    hero_role: str,
    holding: tuple[str, ...],
    pot: Decimal,
    aggressor_role: str,
    legal_moves: tuple[Move, ...],
) -> str:
    legal = ", ".join(move.display() for move in legal_moves)
    return f"""Game: six-handed No-Limit Hold'em, 100 BB starting stacks.
Street: {street}
Preflop action: {preflop_action}
Postflop action through this decision: {postflop_action}
Board now: {' '.join(board)}
Hero role: {hero_role}
Hero cards: {' '.join(holding)}
Source aggressor role: {aggressor_role}
Current pot: {pot} BB
Legal decisions: {legal}
Choose the solver-aligned decision from the legal list."""


def _source_index(row: dict, fallback: int) -> int:
    raw = _clean_spaces(row.get(""))
    if not raw:
        return fallback
    if not raw.isdigit():
        raise PokerBenchDataError(f"invalid source index {raw!r}")
    return int(raw)


def load_preflop_cases(path: Path) -> tuple[list[PokerBenchCase], list[dict]]:
    cases = []
    quarantined = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "prev_line", "hero_pos", "hero_holding", "correct_decision",
            "num_players", "num_bets", "available_moves", "pot_size",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise PokerBenchDataError("preflop CSV has an unexpected schema")
        for row_number, row in enumerate(reader, start=2):
            try:
                source_index = _source_index(row, row_number - 2)
                position = _parse_position(row["hero_pos"])
                holding = _parse_cards(
                    row["hero_holding"], count=2, field="hero_holding"
                )
                players = int(_clean_spaces(row["num_players"]))
                if not 1 <= players <= 6:
                    raise PokerBenchDataError("num_players must be between 1 and 6")
                legal_moves = parse_available_moves(
                    row["available_moves"], split="preflop"
                )
                target = parse_move(row["correct_decision"], split="preflop")
                if target not in legal_moves:
                    raise PokerBenchDataError("target is not one of the legal moves")
                pot = _decimal(row["pot_size"], field="pot_size")
                previous = _canonical_action_line(row["prev_line"])
                prompt = _build_preflop_prompt(
                    previous_action=previous,
                    hero_position=position,
                    holding=holding,
                    pot=pot,
                    players=players,
                    legal_moves=legal_moves,
                )
                cases.append(PokerBenchCase(
                    case_id=f"pokerbench:{POKERBENCH_REVISION}:preflop:{source_index}",
                    split="preflop",
                    source_index=source_index,
                    street="PREFLOP",
                    prompt=prompt,
                    target=target,
                    legal_moves=legal_moves,
                    metadata={
                        "hero_position": position,
                        "holding": list(holding),
                        "pot_bb": str(pot),
                        "previous_action": previous,
                        "num_players": players,
                    },
                ))
            except (PokerBenchDataError, ValueError) as error:
                quarantined.append({
                    "split": "preflop",
                    "row": row_number,
                    "source_index": row.get("", ""),
                    "error": str(error),
                })
    return cases, quarantined


def load_postflop_cases(path: Path) -> tuple[list[PokerBenchCase], list[dict]]:
    cases = []
    quarantined = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "preflop_action", "board_flop", "board_turn", "board_river",
            "aggressor_position", "postflop_action", "evaluation_at",
            "available_moves", "pot_size", "hero_position", "holding",
            "correct_decision",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise PokerBenchDataError("postflop CSV has an unexpected schema")
        for row_number, row in enumerate(reader, start=2):
            try:
                source_index = _source_index(row, row_number - 2)
                street = _clean_spaces(row["evaluation_at"]).upper()
                if street not in {"FLOP", "TURN", "RIVER"}:
                    raise PokerBenchDataError(f"invalid evaluation_at {street!r}")
                flop = _parse_cards(row["board_flop"], count=3, field="board_flop")
                board = list(flop)
                if street in {"TURN", "RIVER"}:
                    board.extend(_parse_cards(
                        row["board_turn"], count=1, field="board_turn"
                    ))
                if street == "RIVER":
                    board.extend(_parse_cards(
                        row["board_river"], count=1, field="board_river"
                    ))
                if len(set(board)) != len(board):
                    raise PokerBenchDataError("board contains a duplicate card")
                holding = _parse_cards(row["holding"], count=2, field="holding")
                if set(board) & set(holding):
                    raise PokerBenchDataError("Hero and board contain a duplicate card")
                hero_role = _parse_position(row["hero_position"], role_only=True)
                aggressor_role = _parse_position(
                    row["aggressor_position"], role_only=True
                )
                legal_moves = parse_available_moves(
                    row["available_moves"], split="postflop"
                )
                target = parse_move(row["correct_decision"], split="postflop")
                if target not in legal_moves:
                    raise PokerBenchDataError("target is not one of the legal moves")
                pot = _decimal(row["pot_size"], field="pot_size")
                preflop_action = _canonical_action_line(row["preflop_action"])
                postflop_action = _truncate_postflop_action(
                    row["postflop_action"],
                    street,
                    expected_dealt=tuple(board[3:]),
                )
                prompt = _build_postflop_prompt(
                    preflop_action=preflop_action,
                    postflop_action=postflop_action,
                    street=street,
                    board=tuple(board),
                    hero_role=hero_role,
                    holding=holding,
                    pot=pot,
                    aggressor_role=aggressor_role,
                    legal_moves=legal_moves,
                )
                cases.append(PokerBenchCase(
                    case_id=f"pokerbench:{POKERBENCH_REVISION}:postflop:{source_index}",
                    split="postflop",
                    source_index=source_index,
                    street=street,
                    prompt=prompt,
                    target=target,
                    legal_moves=legal_moves,
                    metadata={
                        "street": street,
                        "board": board,
                        "holding": list(holding),
                        "hero_role": hero_role,
                        "aggressor_role": aggressor_role,
                        "pot_bb": str(pot),
                        "preflop_action": preflop_action,
                        "postflop_action": postflop_action,
                    },
                ))
            except (PokerBenchDataError, ValueError) as error:
                quarantined.append({
                    "split": "postflop",
                    "row": row_number,
                    "source_index": row.get("", ""),
                    "error": str(error),
                })
    return cases, quarantined


def load_cases(
    paths: dict[str, Path],
    *,
    split: str = "both",
) -> tuple[list[PokerBenchCase], list[dict]]:
    cases = []
    quarantined = []
    if split in {"preflop", "both"}:
        loaded, rejected = load_preflop_cases(paths["preflop"])
        cases.extend(loaded)
        quarantined.extend(rejected)
    if split in {"postflop", "both"}:
        loaded, rejected = load_postflop_cases(paths["postflop"])
        cases.extend(loaded)
        quarantined.extend(rejected)
    seen = set()
    duplicates = [case.case_id for case in cases if case.case_id in seen or seen.add(case.case_id)]
    if duplicates:
        raise PokerBenchDataError(f"duplicate case id: {duplicates[0]}")
    return cases, quarantined


def parse_model_move(text: str) -> Move:
    """Parse the strict benchmark format and a small set of safe variants."""

    raw = str(text or "").strip()
    if not raw:
        raise PokerBenchDataError("empty model response")

    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise PokerBenchDataError("invalid structured JSON response") from error
        if not isinstance(payload, dict) or set(payload) != {"action", "amount"}:
            raise PokerBenchDataError(
                "structured response must contain only action and amount"
            )
        action = payload["action"]
        amount_value = payload["amount"]
        if action not in {"FOLD", "CHECK", "CALL", "BET", "RAISE", "ALL_IN"}:
            raise PokerBenchDataError("structured response has an invalid action")
        if isinstance(amount_value, bool) or not isinstance(
            amount_value, (int, float)
        ):
            raise PokerBenchDataError("structured response amount must be numeric")
        amount = _decimal(amount_value, field="model amount")
        if action in {"BET", "RAISE"}:
            if amount <= 0:
                raise PokerBenchDataError(f"{action} requires a positive size")
            return Move(action, amount)
        if amount != 0:
            raise PokerBenchDataError(f"{action} must have size 0")
        return Move(action)

    action_matches = re.findall(
        r"^\s*\**\s*Action\s*:\s*\**\s*"
        r"(FOLD|CHECK|CALL|BET|RAISE|ALL[-_ ]?IN)"
        r"(?:\s+(?:to\s+)?(\d+(?:\.\d+)?))?\s*\**\s*$",
        raw,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if len(action_matches) == 1:
        action_text, inline_size = action_matches[0]
        action = action_text.upper().replace("-", "_").replace(" ", "_")
        size_matches = re.findall(
            r"^\s*\**\s*(?:Size|Amount)\s*:\s*\**\s*"
            r"(\d+(?:\.\d+)?)\s*(?:BB)?\s*\**\s*$",
            raw,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if len(size_matches) > 1:
            raise PokerBenchDataError("multiple size lines")
        if inline_size and size_matches:
            inline_amount = _decimal(inline_size, field="inline model size")
            line_amount = _decimal(size_matches[0], field="model size")
            if inline_amount != line_amount:
                raise PokerBenchDataError("conflicting inline and Size amounts")
        size_text = size_matches[0] if size_matches else inline_size
        if action in {"BET", "RAISE"}:
            if not size_text:
                raise PokerBenchDataError(f"{action} requires a positive size")
            amount = _decimal(size_text, field="model size")
            if amount <= 0:
                raise PokerBenchDataError(f"{action} requires a positive size")
            return Move(action, amount)
        if size_text and _decimal(size_text, field="model size") != 0:
            raise PokerBenchDataError(f"{action} must have size 0")
        return Move(action)

    one_line = re.sub(r"[.*`#]", "", raw).strip()
    if "\n" not in one_line:
        compact = one_line.lower().replace("-", "_").replace(" ", "")
        if compact in {"fold", "check", "call", "allin", "all_in"}:
            return parse_move(one_line, split="postflop")
        match = re.fullmatch(
            r"(bet|raise)\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*(?:bb)?",
            one_line,
            flags=re.IGNORECASE,
        )
        if match:
            amount = _decimal(match.group(2), field="model size")
            if amount <= 0:
                raise PokerBenchDataError(
                    f"{match.group(1).upper()} requires a positive size"
                )
            return Move(
                match.group(1).upper(),
                amount,
            )
    raise PokerBenchDataError("response is missing one unambiguous Action line")


def move_is_legal(move: Move, legal_moves: Iterable[Move]) -> bool:
    candidates = [candidate for candidate in legal_moves if candidate.action == move.action]
    if not candidates:
        return False
    if any(candidate.amount is not None for candidate in candidates):
        return move.amount is not None and move in candidates
    return move.amount is None


def cache_key(
    case: PokerBenchCase,
    *,
    model: str,
    max_tokens: int,
) -> str:
    payload = {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "dataset_revision": POKERBENCH_REVISION,
        "case_id": case.case_id,
        "prompt": case.prompt,
        "system": SYSTEM_PROMPT,
        "response_format": RESPONSE_FORMAT,
        "decision_schema": DECISION_SCHEMA,
        "model": model,
        "max_tokens": max_tokens,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class JsonlCache:
    """Small append-only cache suitable for resumable 11k-case runs."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict] = {}
        self.corrupt_lines = 0
        self.stale_lines = 0
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("schema_version") != CACHE_SCHEMA_VERSION:
                            self.stale_lines += 1
                            continue
                        if (
                            not isinstance(entry.get("key"), str)
                            or not isinstance(entry.get("completion"), dict)
                        ):
                            raise ValueError("invalid cache entry")
                        completion = entry["completion"]
                        if (
                            not isinstance(completion.get("text"), str)
                            or not isinstance(
                                completion.get("latency_seconds", 0), (int, float)
                            )
                            or not isinstance(
                                completion.get("input_tokens", 0), int
                            )
                            or not isinstance(
                                completion.get("output_tokens", 0), int
                            )
                            or not isinstance(
                                completion.get("response_model", ""), str
                            )
                            or not isinstance(
                                completion.get("stop_reason", ""), str
                            )
                        ):
                            raise ValueError("invalid cached completion")
                        self.entries[entry["key"]] = completion
                    except (json.JSONDecodeError, TypeError, ValueError):
                        self.corrupt_lines += 1

    def get(self, key: str) -> Optional[Completion]:
        value = self.entries.get(key)
        if value is None:
            return None
        return Completion(
            text=value["text"],
            latency_seconds=float(value.get("latency_seconds", 0.0)),
            input_tokens=int(value.get("input_tokens", 0)),
            output_tokens=int(value.get("output_tokens", 0)),
            response_model=str(value.get("response_model", "")),
            stop_reason=str(value.get("stop_reason", "")),
        )

    def put(self, key: str, completion: Completion) -> None:
        value = {
            "text": completion.text,
            "latency_seconds": completion.latency_seconds,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "response_model": completion.response_model,
            "stop_reason": completion.stop_reason,
        }
        entry = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": key,
            "completion": value,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        self.entries[key] = value


class AnthropicCompleter:
    def __init__(self, api_key: str, *, timeout: float, max_retries: int):
        self.client = Anthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
    ) -> Completion:
        started = time.perf_counter()
        request = {
            "model": model,
            "max_tokens": max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": DECISION_OUTPUT_CONFIG,
        }
        if re.fullmatch(r"claude-sonnet-5(?:-[A-Za-z0-9]+)?", model):
            request["thinking"] = {"type": "disabled"}
        response = self.client.messages.create(
            **request,
        )
        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            latency_seconds=time.perf_counter() - started,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            response_model=str(getattr(response, "model", "") or ""),
            stop_reason=str(getattr(response, "stop_reason", "") or ""),
        )


def select_cases(
    cases: list[PokerBenchCase],
    *,
    limit: Optional[int],
    seed: int,
) -> list[PokerBenchCase]:
    """Select a deterministic, proportionally stratified representative subset."""

    if limit is None or limit >= len(cases):
        return list(cases)
    if limit <= 0:
        raise ValueError("limit must be positive")
    groups: dict[tuple, list[PokerBenchCase]] = defaultdict(list)
    for case in cases:
        groups[(case.split, case.street, case.target.action)].append(case)
    randomizer = random.Random(seed)
    for group in groups.values():
        randomizer.shuffle(group)
    total = len(cases)
    quotas = {}
    remainders = []
    for key in sorted(groups):
        exact = limit * len(groups[key]) / total
        quota = math.floor(exact)
        quotas[key] = quota
        remainders.append((exact - quota, key))
    unassigned = limit - sum(quotas.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if unassigned == 0:
            break
        quotas[key] += 1
        unassigned -= 1

    selected = [
        case
        for key in sorted(groups)
        for case in groups[key][:quotas[key]]
    ]
    randomizer.shuffle(selected)
    return selected


def _result_base(case: PokerBenchCase, model: str) -> dict:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "case_id": case.case_id,
        "split": case.split,
        "street": case.street,
        "model": model,
        "target": case.target.to_dict(),
        "legal_moves": [move.to_dict() for move in case.legal_moves],
        "metadata": case.metadata,
    }


def _failed_result(
    case: PokerBenchCase,
    model: str,
    *,
    status: str,
    error: str,
) -> dict:
    result = _result_base(case, model)
    result.update({
        "status": status,
        "cached": False,
        "raw_response": "",
        "prediction": None,
        "parse_error": error,
        "legal": False,
        "action_correct": False,
        "exact_correct": False,
        "latency_seconds": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "response_model": "",
        "stop_reason": "",
    })
    return result


def run_cases(
    cases: list[PokerBenchCase],
    *,
    model: str,
    completer,
    cache: JsonlCache,
    max_tokens: int = 80,
    offline: bool = False,
    max_consecutive_provider_errors: int = 5,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    if max_consecutive_provider_errors <= 0:
        raise ValueError("max_consecutive_provider_errors must be positive")
    results = []
    total = len(cases)
    consecutive_provider_errors = 0
    for index, case in enumerate(cases, start=1):
        result = _result_base(case, model)
        key = cache_key(
            case,
            model=model,
            max_tokens=max_tokens,
        )
        completion = cache.get(key)
        cached = completion is not None
        if completion is None and offline:
            results.append(_failed_result(
                case,
                model,
                status="CACHE_MISS",
                error="offline cache miss",
            ))
            if progress:
                progress(index, total)
            continue
        if completion is None:
            try:
                completion = completer.complete(
                    case.prompt,
                    model=model,
                    max_tokens=max_tokens,
                )
                cache.put(key, completion)
                consecutive_provider_errors = 0
            except Exception as error:
                consecutive_provider_errors += 1
                results.append(_failed_result(
                    case,
                    model,
                    status="PROVIDER_ERROR",
                    error=str(error),
                ))
                if progress:
                    progress(index, total)
                if (
                    consecutive_provider_errors
                    >= max_consecutive_provider_errors
                ):
                    for remaining_case in cases[index:]:
                        results.append(_failed_result(
                            remaining_case,
                            model,
                            status="ABORTED_PROVIDER_ERRORS",
                            error=(
                                "circuit breaker opened after "
                                f"{consecutive_provider_errors} consecutive "
                                "provider errors"
                            ),
                        ))
                    if progress and index < total:
                        progress(total, total)
                    return results
                continue

        try:
            prediction = parse_model_move(completion.text)
            parse_error = ""
            status = "OK"
            legal = move_is_legal(prediction, case.legal_moves)
            action_correct = prediction.action == case.target.action
            exact_correct = prediction == case.target
        except PokerBenchDataError as error:
            prediction = None
            parse_error = str(error)
            status = "PARSE_ERROR"
            legal = False
            action_correct = False
            exact_correct = False

        result.update({
            "status": status,
            "cached": cached,
            "raw_response": completion.text,
            "prediction": prediction.to_dict() if prediction else None,
            "parse_error": parse_error,
            "legal": legal,
            "action_correct": action_correct,
            "exact_correct": exact_correct,
            "latency_seconds": completion.latency_seconds,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "response_model": completion.response_model,
            "stop_reason": completion.stop_reason,
        })
        results.append(result)
        if progress:
            progress(index, total)
    return results


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _aggregate(results: list[dict]) -> dict:
    total = len(results)
    counts = Counter(result["status"] for result in results)
    successful = [result for result in results if result["status"] == "OK"]
    responses = [
        result for result in results
        if result["status"] in {"OK", "PARSE_ERROR"}
    ]
    uncached_latencies = [
        result["latency_seconds"] for result in responses if not result["cached"]
    ]
    sized_targets = [
        result for result in results if result["target"]["amount"] is not None
    ]
    return {
        "cases": total,
        "status_counts": dict(sorted(counts.items())),
        "response_rate": len(responses) / total if total else 0.0,
        "coverage_rate": len(successful) / total if total else 0.0,
        "legal_move_rate": (
            sum(bool(result["legal"]) for result in results) / total if total else 0.0
        ),
        "action_agreement": (
            sum(bool(result["action_correct"]) for result in results) / total
            if total else 0.0
        ),
        "exact_decision_agreement": (
            sum(bool(result["exact_correct"]) for result in results) / total
            if total else 0.0
        ),
        "sized_exact_agreement": (
            sum(bool(result["exact_correct"]) for result in sized_targets)
            / len(sized_targets) if sized_targets else 0.0
        ),
        "sized_cases": len(sized_targets),
        "cache_hits": sum(bool(result["cached"]) for result in results),
        "input_tokens": sum(int(result["input_tokens"]) for result in results),
        "output_tokens": sum(int(result["output_tokens"]) for result in results),
        "uncached_latency_p50_seconds": (
            statistics.median(uncached_latencies) if uncached_latencies else 0.0
        ),
        "uncached_latency_p95_seconds": _percentile(uncached_latencies, 0.95),
    }


def summarize(
    results: list[dict],
    *,
    model: str,
    quarantined: list[dict],
    run_config: Optional[dict] = None,
) -> dict:
    breakdown = {}
    strata = defaultdict(list)
    for result in results:
        strata[f"split:{result['split']}"].append(result)
        strata[f"street:{result['street']}"].append(result)
        strata[f"target:{result['target']['action']}"].append(result)
    for label in sorted(strata):
        breakdown[label] = _aggregate(strata[label])

    confusion = Counter()
    for result in results:
        target = result["target"]["action"]
        if result["status"] != "OK":
            predicted = result["status"]
        else:
            predicted = result["prediction"]["action"]
        confusion[f"{target}->{predicted}"] += 1

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "RZ412/PokerBench",
            "revision": POKERBENCH_REVISION,
            "license": "Apache-2.0",
            "adapter_version": ADAPTER_VERSION,
            "prompt_version": PROMPT_VERSION,
            "files": {
                split: {
                    "filename": descriptor.filename,
                    "size": descriptor.size,
                    "sha256": descriptor.sha256,
                }
                for split, descriptor in DATASET_FILES.items()
            },
            "limitations": [
                "One selected solver label per case; no full mixed policy.",
                "No per-action EV, regret, or exploitability measurement.",
                "Postflop scenarios are represented as IP versus OOP.",
                "Public benchmark cases may have appeared in model training data.",
                "This tests a neutral standalone prompt, not the live app pipeline.",
            ],
        },
        "model": model,
        "run_config": run_config or {},
        "overall": _aggregate(results),
        "breakdown": breakdown,
        "confusion": dict(sorted(confusion.items())),
        "quarantined_rows": len(quarantined),
        "quarantine_examples": quarantined[:20],
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _print_summary(report: dict) -> None:
    overall = report["overall"]
    print(f"\nModel: {report['model']}")
    print(f"Cases: {overall['cases']}")
    print(f"Action agreement: {overall['action_agreement']:.1%}")
    print(f"Exact decision agreement: {overall['exact_decision_agreement']:.1%}")
    print(f"Legal move rate: {overall['legal_move_rate']:.1%}")
    print(f"Coverage: {overall['coverage_rate']:.1%}")
    incomplete = overall["cases"] - overall["status_counts"].get("OK", 0)
    print(f"Run complete: {'yes' if incomplete == 0 else f'no ({incomplete} missing)'}")
    print(
        "Uncached latency p50/p95: "
        f"{overall['uncached_latency_p50_seconds']:.2f}s / "
        f"{overall['uncached_latency_p95_seconds']:.2f}s"
    )
    print(f"Quarantined source rows: {report['quarantined_rows']}")


def _default_progress(index: int, total: int) -> None:
    if index == 1 or index == total or index % 10 == 0:
        print(f"\rEvaluated {index}/{total}", end="", flush=True)
        if index == total:
            print()


def _resolve_models(choice: str, explicit_model: Optional[str]) -> list[str]:
    fast = os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-4-5")
    coach = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
    if explicit_model:
        if choice == "both":
            raise ValueError("--model-id cannot be combined with --model both")
        return [explicit_model]
    if choice == "fast":
        return [fast]
    if choice == "coach":
        return [coach]
    return list(dict.fromkeys([fast, coach]))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Claude evaluation on PokerBench solver labels",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmark_data/pokerbench"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download pinned test CSVs")
    download.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate", help="parse and validate source rows")
    validate.add_argument(
        "--split", choices=["preflop", "postflop", "both"], default="both"
    )
    validate.add_argument("--no-download", action="store_true")

    run = subparsers.add_parser("run", help="run or resume a model benchmark")
    run.add_argument(
        "--split", choices=["preflop", "postflop", "both"], default="both"
    )
    run.add_argument("--model", choices=["fast", "coach", "both"], default="fast")
    run.add_argument("--model-id")
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=_positive_int, default=100)
    selection.add_argument(
        "--all", action="store_true", help="evaluate every loaded case"
    )
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--max-tokens", type=_positive_int, default=80)
    run.add_argument("--timeout", type=_positive_float, default=20.0)
    run.add_argument("--retries", type=_nonnegative_int, default=2)
    run.add_argument(
        "--max-consecutive-errors", type=_positive_int, default=5
    )
    run.add_argument(
        "--offline",
        action="store_true",
        help="disable all network calls and use existing dataset/cache only",
    )
    run.add_argument("--no-download", action="store_true")
    run.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/pokerbench")
    )
    run.add_argument(
        "--cache", type=Path, default=Path("benchmark_results/pokerbench/cache.jsonl")
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        paths = download_dataset(args.data_dir, force=args.force, split="both")
        for split, path in paths.items():
            print(f"{split}: {path}")
        return 0

    if not args.no_download and not getattr(args, "offline", False):
        download_dataset(args.data_dir, split=args.split)
    paths = dataset_paths(args.data_dir, split=args.split)
    cases, quarantined = load_cases(paths, split=args.split)
    print(f"Loaded {len(cases)} cases; quarantined {len(quarantined)} rows")

    if args.command == "validate":
        by_split = Counter(case.split for case in cases)
        by_street = Counter(case.street for case in cases)
        print("By split:", dict(sorted(by_split.items())))
        print("By street:", dict(sorted(by_street.items())))
        if quarantined:
            print("First quarantine reasons:")
            for item in quarantined[:10]:
                print(f"  {item}")
        return 1 if quarantined else 0

    selected = select_cases(
        cases,
        limit=None if args.all else args.limit,
        seed=args.seed,
    )
    try:
        models = _resolve_models(args.model, args.model_id)
    except ValueError as error:
        parser.error(str(error))
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not args.offline and not api_key:
        print("ANTHROPIC_API_KEY is required unless --offline is used", file=sys.stderr)
        return 2
    completer = (
        AnthropicCompleter(
            api_key,
            timeout=args.timeout,
            max_retries=args.retries,
        )
        if api_key else None
    )
    cache = JsonlCache(args.cache)
    if cache.corrupt_lines:
        print(f"Warning: skipped {cache.corrupt_lines} corrupt cache lines")
    if cache.stale_lines:
        print(
            f"Info: ignored {cache.stale_lines} cache entries from an older "
            "benchmark format"
        )

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_config = {
        "split": args.split,
        "source_cases": len(cases),
        "selected_cases": len(selected),
        "selection_scheme": (
            "full" if len(selected) == len(cases) else SELECTION_SCHEME
        ),
        "requested_limit": None if args.all else args.limit,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "response_format": RESPONSE_FORMAT,
        "timeout_seconds": args.timeout,
        "provider_retries": args.retries,
        "max_consecutive_provider_errors": args.max_consecutive_errors,
        "offline_cache_only": args.offline,
    }
    incomplete_run = False
    for model in models:
        results = run_cases(
            selected,
            model=model,
            completer=completer,
            cache=cache,
            max_tokens=args.max_tokens,
            offline=args.offline,
            max_consecutive_provider_errors=args.max_consecutive_errors,
            progress=_default_progress,
        )
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        prefix = args.output_dir / f"{run_stamp}_{safe_model}"
        result_path = prefix.with_suffix(".jsonl")
        report_path = prefix.with_name(prefix.name + "_report.json")
        _write_jsonl(result_path, results)
        report = summarize(
            results,
            model=model,
            quarantined=quarantined,
            run_config={**run_config, "requested_model": model},
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _print_summary(report)
        print(f"Results: {result_path}")
        print(f"Report: {report_path}")
        statuses = report["overall"]["status_counts"]
        if statuses.get("OK", 0) != report["overall"]["cases"]:
            incomplete_run = True
    return 3 if incomplete_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
