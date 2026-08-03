"""Live GTO routing for an explicitly user-controlled poker simulator.

A validated six-max preflop blueprint supplies mixed preflop policies and the
two surviving reach ranges for eligible postflop hands.  Only conservatively
reconstructed heads-up postflop nodes are solved locally.  Every descendant
query retains the exact action path from the OOP street root; sparse screen
snapshots are never promoted into an invented raise/call history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from typing import Callable

from gto_hand_history import PublicHandHistory
from preflop_blueprint import (
    BlueprintError,
    BlueprintNode,
    PokerStudyBlueprintStore,
)
from preflop_history import (
    PreflopHistoryError,
    PreflopHistoryResolver,
    PreflopResolution,
    ReachResolution,
    canonical_position,
    expand_position_reach,
    hand_class_for_cards,
    parse_action_label,
)
from preflop_observation import ObservedPreflopState

from gto_oracle import (
    Action,
    ActionKind,
    AllocationMode,
    BetSizingConfig,
    ContinuationResult,
    ContinuationSpec,
    EngineClient,
    EngineClientError,
    EngineResponseError,
    OracleCache,
    OracleValidationError,
    PlayerBetSizes,
    PlayerRange,
    Position,
    SolveParameters,
    SolveResult,
    SolveSpec,
    Street,
    StreetBetSizes,
    TreeConfig,
    WeightedCombo,
)


PINNED_SOLVER_COMMIT = "9d1509fe5077d019825f833eed04b16d342dfda1"
_REPOSITORY_ENGINE = (
    Path(__file__).resolve().parent
    / "gto_oracle_engine"
    / "target"
    / "release"
    / "gto-oracle-engine"
)
_VERIFIED_TEMP_ENGINE = Path(
    "/private/tmp/oracle-engine-target/release/gto-oracle-engine"
)
DEFAULT_ENGINE = (
    _REPOSITORY_ENGINE
    if _REPOSITORY_ENGINE.is_file()
    else _VERIFIED_TEMP_ENGINE
)
RANKS = "AKQJT98765432"
SUITS = "cdhs"
_CARD_RE = re.compile(r"^[2-9TJQKA][cdhs]$")
_CLASS_RE = re.compile(r"^([2-9TJQKA])([2-9TJQKA])([so]?)$")
POSITION_ONLY_HANDOFF_SOURCE = "gto_hu_position_continuity"


class LiveGTOStatus(str, Enum):
    SOLVED = "SOLVED"
    DISABLED = "DISABLED"
    UNSUPPORTED = "UNSUPPORTED"
    CACHE_MISS = "CACHE_MISS"
    FAILED = "FAILED"


class LiveGTOConfigurationError(ValueError):
    """The opt-in or local range profile is invalid."""


class LiveGTORangeError(ValueError):
    """A position chart cannot be expanded into a solver range."""


class LiveGTOCacheError(RuntimeError):
    """The local solver cache could not be opened or decoded safely."""


@dataclass(frozen=True, slots=True)
class LiveGTOConfig:
    enabled: bool = False
    owned_simulator_acknowledged: bool = False
    engine_path: Path = DEFAULT_ENGINE
    cache_path: Path = Path("gto_live_cache.sqlite3")
    range_data_path: Path = Path("poker_data.json")
    range_source: str = "blueprint"
    blueprint_cache_path: Path = Path("preflop_blueprint_cache")
    blueprint_allow_network: bool = False
    blueprint_match_mode: str = "exact"
    blueprint_min_tolerance_bb: Decimal = Decimal("0.05")
    blueprint_size_tolerance_pct: Decimal = Decimal(0)
    blueprint_max_stack_error_pct: Decimal = Decimal(25)
    blueprint_network_timeout_seconds: Decimal = Decimal(10)
    chip_scale: int = 100
    bet_size_pct: Decimal = Decimal("50")
    target_exploitability_pct: Decimal = Decimal("0.1")
    max_iterations: int = 1_000_000
    turn_timeout_seconds: Decimal = Decimal("5")
    river_timeout_seconds: Decimal = Decimal("2")
    flop_timeout_seconds: Decimal = Decimal("180")
    flop_cache_only: bool = False
    rake_rate_pct: Decimal = Decimal(5)
    rake_cap_bb: Decimal = Decimal("0.5")
    mix_secret: bytes = field(
        default_factory=lambda: secrets.token_bytes(32),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.enabled and not self.owned_simulator_acknowledged:
            raise LiveGTOConfigurationError(
                "GTO live requires GTO_OWNED_SIMULATOR_ACK=1"
            )
        if not isinstance(self.chip_scale, int) or not 1 <= self.chip_scale <= 1_000_000:
            raise LiveGTOConfigurationError("GTO chip_scale must be 1..1,000,000")
        if not Decimal(1) <= self.bet_size_pct <= Decimal(500):
            raise LiveGTOConfigurationError("GTO bet size must be between 1% and 500%")
        if not Decimal(0) < self.target_exploitability_pct <= Decimal(100):
            raise LiveGTOConfigurationError(
                "GTO target exploitability must be in (0, 100]"
            )
        if not 1 <= self.max_iterations <= 1_000_000:
            raise LiveGTOConfigurationError("GTO max iterations must be 1..1,000,000")
        for name, value in (
            ("flop timeout", self.flop_timeout_seconds),
            ("turn timeout", self.turn_timeout_seconds),
            ("river timeout", self.river_timeout_seconds),
        ):
            if not value.is_finite() or value <= 0:
                raise LiveGTOConfigurationError(f"{name} must be finite and positive")
        if (
            not self.rake_rate_pct.is_finite()
            or not Decimal(0) <= self.rake_rate_pct <= Decimal(100)
        ):
            raise LiveGTOConfigurationError("GTO rake rate must be between 0 and 100")
        if not self.rake_cap_bb.is_finite() or self.rake_cap_bb < 0:
            raise LiveGTOConfigurationError(
                "GTO rake cap must be finite and non-negative"
            )
        if not isinstance(self.mix_secret, bytes) or len(self.mix_secret) < 32:
            raise LiveGTOConfigurationError(
                "GTO mix secret must contain at least 32 private bytes"
            )
        if self.range_source not in {"charts", "blueprint"}:
            raise LiveGTOConfigurationError(
                "GTO range source must be 'charts' or 'blueprint'"
            )
        if self.blueprint_match_mode not in {"exact", "abstract"}:
            raise LiveGTOConfigurationError(
                "preflop blueprint match mode must be 'exact' or 'abstract'"
            )
        if (
            not self.blueprint_min_tolerance_bb.is_finite()
            or self.blueprint_min_tolerance_bb < 0
        ):
            raise LiveGTOConfigurationError(
                "preflop blueprint minimum tolerance must be finite and non-negative"
            )
        if (
            not self.blueprint_size_tolerance_pct.is_finite()
            or not Decimal(0) <= self.blueprint_size_tolerance_pct <= Decimal(100)
        ):
            raise LiveGTOConfigurationError(
                "preflop blueprint size tolerance must be between 0% and 100%"
            )
        if (
            not self.blueprint_max_stack_error_pct.is_finite()
            or not Decimal(0) <= self.blueprint_max_stack_error_pct <= Decimal(100)
        ):
            raise LiveGTOConfigurationError(
                "preflop blueprint maximum stack error must be between 0% and 100%"
            )
        if (
            not self.blueprint_network_timeout_seconds.is_finite()
            or self.blueprint_network_timeout_seconds <= 0
        ):
            raise LiveGTOConfigurationError(
                "preflop blueprint network timeout must be finite and positive"
            )

    @classmethod
    def from_env(cls, base_dir: str | Path = ".") -> "LiveGTOConfig":
        base = Path(base_dir)

        def flag(name: str, default: str = "0") -> bool:
            return os.getenv(name, default).strip().lower() in {
                "1", "true", "yes", "on"
            }

        def decimal(name: str, default: str) -> Decimal:
            try:
                value = Decimal(os.getenv(name, default).strip())
            except (InvalidOperation, AttributeError) as error:
                raise LiveGTOConfigurationError(f"{name} must be a decimal") from error
            if not value.is_finite():
                raise LiveGTOConfigurationError(f"{name} must be finite")
            return value

        def path(name: str, default: str) -> Path:
            raw = Path(os.getenv(name, default).strip()).expanduser()
            return raw if raw.is_absolute() else base / raw

        try:
            max_iterations = int(os.getenv("GTO_MAX_ITERATIONS", "1000000"))
        except ValueError as error:
            raise LiveGTOConfigurationError(
                "GTO_MAX_ITERATIONS must be an integer"
            ) from error
        raw_mix_secret = os.getenv("GTO_MIX_SECRET", "")
        if raw_mix_secret and len(raw_mix_secret.encode("utf-8")) < 32:
            raise LiveGTOConfigurationError(
                "GTO_MIX_SECRET must contain at least 32 UTF-8 bytes"
            )
        mix_secret = (
            hashlib.sha256(raw_mix_secret.encode("utf-8")).digest()
            if raw_mix_secret
            else secrets.token_bytes(32)
        )

        return cls(
            enabled=flag("GTO_LIVE_ENABLED"),
            owned_simulator_acknowledged=flag("GTO_OWNED_SIMULATOR_ACK"),
            engine_path=path("GTO_ENGINE_PATH", str(DEFAULT_ENGINE)),
            cache_path=path("GTO_CACHE_PATH", "gto_live_cache.sqlite3"),
            range_data_path=path("GTO_RANGE_DATA_PATH", "poker_data.json"),
            range_source=os.getenv("GTO_RANGE_SOURCE", "blueprint").strip().lower(),
            blueprint_cache_path=path(
                "PREFLOP_BLUEPRINT_CACHE_PATH", "preflop_blueprint_cache"
            ),
            blueprint_allow_network=flag("PREFLOP_BLUEPRINT_ALLOW_NETWORK"),
            blueprint_match_mode=os.getenv(
                "PREFLOP_BLUEPRINT_MATCH_MODE", "exact"
            ).strip().lower(),
            blueprint_min_tolerance_bb=decimal(
                "PREFLOP_BLUEPRINT_MIN_TOLERANCE_BB", "0.05"
            ),
            blueprint_size_tolerance_pct=decimal(
                "PREFLOP_BLUEPRINT_SIZE_TOLERANCE_PCT", "0"
            ),
            blueprint_max_stack_error_pct=decimal(
                "PREFLOP_BLUEPRINT_MAX_STACK_ERROR_PCT", "25"
            ),
            blueprint_network_timeout_seconds=decimal(
                "PREFLOP_BLUEPRINT_NETWORK_TIMEOUT_SECONDS", "10"
            ),
            bet_size_pct=decimal("GTO_BET_SIZE_PCT", "50"),
            target_exploitability_pct=decimal(
                "GTO_TARGET_EXPLOITABILITY_PCT", "0.1"
            ),
            max_iterations=max_iterations,
            flop_timeout_seconds=decimal("GTO_FLOP_TIMEOUT_SECONDS", "180"),
            turn_timeout_seconds=decimal("GTO_TURN_TIMEOUT_SECONDS", "5"),
            river_timeout_seconds=decimal("GTO_RIVER_TIMEOUT_SECONDS", "2"),
            flop_cache_only=flag("GTO_FLOP_CACHE_ONLY", "0"),
            rake_rate_pct=decimal("GTO_RAKE_RATE_PCT", "5"),
            rake_cap_bb=decimal("GTO_RAKE_CAP_BB", "0.5"),
            mix_secret=mix_secret,
        )

    def timeout_for(self, street: Street) -> Decimal:
        return {
            Street.FLOP: self.flop_timeout_seconds,
            Street.TURN: self.turn_timeout_seconds,
            Street.RIVER: self.river_timeout_seconds,
        }[street]


@dataclass(frozen=True, slots=True)
class LiveDecisionState:
    hand_id: str
    street: str
    board: tuple[str, ...]
    hero_combo: tuple[str, str]
    hero_position: str
    villain_position: str
    hero_is_oop: bool
    active_villains: int
    pot_bb: Decimal
    hero_stack_bb: Decimal
    villain_stack_bb: Decimal
    hero_current_bet_bb: Decimal
    villain_current_bet_bb: Decimal
    amount_to_call_bb: Decimal
    legal_actions: tuple[str, ...]
    street_root_confirmed: bool
    # Compact, scale-neutral path from the OOP street root.  BET is sized by
    # ``observed_bet_to_bb``.  The only accepted paths are (), (CHECK,),
    # (BET,), and (CHECK, BET); raises and calls are deliberately unsupported.
    action_history: tuple[str, ...] = ()
    observed_bet_to_bb: Decimal = Decimal(0)
    mapping_error: str = ""
    preflop_observation: ObservedPreflopState | None = None
    preflop_mapping_error: str = ""
    # Optional lossless preflop-to-current-node transcript.  The current
    # conservative HU router does not invent one from sparse screenshots, but
    # full/stateful server backends require it and can replay it independently.
    public_hand: PublicHandHistory | None = None


@dataclass(frozen=True, slots=True)
class RangeBundle:
    oop: PlayerRange
    ip: PlayerRange
    profile_id: str
    hero_combo_injected: bool
    provenance: str = ""
    approximations: tuple[str, ...] = ()
    approximate: bool = False


@dataclass(frozen=True, slots=True)
class LiveGTOOutcome:
    status: LiveGTOStatus
    reason: str
    latency_seconds: float
    analysis: str = ""
    source: str = ""
    model: str = ""
    cache_hit: bool = False
    spec: SolveSpec | ContinuationSpec | None = None
    result: SolveResult | ContinuationResult | None = None
    approximate: bool = False
    # Remote responses intentionally do not return the full (potentially very
    # large) SolveSpec.  Preserve its canonical key so local audit records can
    # still tie a recommendation to the exact server-side solve.
    spec_key: str = ""

    @property
    def solved(self) -> bool:
        return self.status is LiveGTOStatus.SOLVED

    @property
    def effective_spec_key(self) -> str:
        return self.spec.cache_key if self.spec is not None else self.spec_key


@dataclass(frozen=True, slots=True)
class BlueprintPolicyAction:
    """One normalized action for Hero's exact 169-grid hand class."""

    label: str
    kind: str
    frequency: Decimal
    source_target_bb: Decimal | None = None
    display_target_bb: Decimal | None = None
    ev: Decimal | None = None


@dataclass(frozen=True, slots=True)
class BlueprintDecisionBundle:
    """A verified preflop node and its conditional mixed strategy."""

    stack: int
    history: str
    hand_class: str
    actions: tuple[BlueprintPolicyAction, ...]
    profile_id: str
    provenance: str
    approximations: tuple[str, ...]
    resolution: PreflopResolution
    approximate: bool = False


def _normalized_legal_actions(actions: tuple[str, ...]) -> frozenset[str]:
    return frozenset(
        action.upper().replace("-", "_")
        for action in actions
        if isinstance(action, str) and action.strip()
    )


def _verified_hu_handoff_reason(state: LiveDecisionState) -> str:
    """Return why the preflop handoff cannot seed the current HU projection."""

    if state.preflop_mapping_error:
        return (
            "verified heads-up preflop handoff is unavailable: "
            f"{state.preflop_mapping_error}"
        )
    observation = state.preflop_observation
    if not isinstance(observation, ObservedPreflopState):
        return "verified heads-up preflop handoff is missing"
    position_only = (
        observation.provenance.source == POSITION_ONLY_HANDOFF_SOURCE
    )
    if not observation.terminal and not position_only:
        return "verified preflop handoff is not terminal"
    if len(observation.live_positions) < 2:
        return (
            "verified preflop handoff has fewer than two surviving positions: found "
            f"{len(observation.live_positions)} surviving positions"
        )
    try:
        expected_positions = frozenset(
            (
                canonical_position(state.hero_position),
                canonical_position(state.villain_position),
            )
        )
    except PreflopHistoryError as error:
        return f"postflop positions cannot verify the heads-up handoff: {error}"
    if len(expected_positions) != 2:
        return "Hero and villain positions do not identify two distinct players"
    if not expected_positions.issubset(observation.live_positions):
        return (
            "current Hero/villain positions were not both preflop survivors"
        )
    observed_hand_id = observation.provenance.hand_id.strip()
    if state.hand_id and observed_hand_id and observed_hand_id != state.hand_id:
        return "verified preflop handoff belongs to a different hand"
    return ""


def _require_verified_hu_handoff(
    state: LiveDecisionState,
) -> ObservedPreflopState:
    reason = _verified_hu_handoff_reason(state)
    if reason:
        raise LiveGTORangeError(reason)
    assert state.preflop_observation is not None
    return state.preflop_observation


def _ordered_combo(cards: tuple[str, str]) -> tuple[str, str]:
    return WeightedCombo(cards).cards


def _class_combos(hand_class: str) -> set[tuple[str, str]]:
    match = _CLASS_RE.fullmatch(hand_class)
    if not match:
        raise LiveGTORangeError(f"invalid hand class {hand_class!r}")
    first, second, shape = match.groups()
    if first == second:
        if shape:
            raise LiveGTORangeError("pairs cannot use suited/offsuit suffixes")
        return {
            _ordered_combo((first + SUITS[left], second + SUITS[right]))
            for left in range(len(SUITS))
            for right in range(left + 1, len(SUITS))
        }
    if RANKS.index(first) > RANKS.index(second):
        raise LiveGTORangeError(f"hand class ranks must be descending: {hand_class}")
    if shape == "s":
        return {_ordered_combo((first + suit, second + suit)) for suit in SUITS}
    if shape == "o":
        return {
            _ordered_combo((first + first_suit, second + second_suit))
            for first_suit in SUITS
            for second_suit in SUITS
            if first_suit != second_suit
        }
    return _class_combos(first + second + "s") | _class_combos(first + second + "o")


def _rank_path(start: str, end: str) -> list[str]:
    start_index = RANKS.index(start)
    end_index = RANKS.index(end)
    step = 1 if end_index >= start_index else -1
    return list(RANKS[start_index : end_index + step : step])


def _expand_class_range(start: str, end: str) -> list[str]:
    start_match = _CLASS_RE.fullmatch(start)
    end_match = _CLASS_RE.fullmatch(end)
    if not start_match or not end_match:
        raise LiveGTORangeError(f"invalid class range {start}-{end}")
    a1, a2, a_shape = start_match.groups()
    b1, b2, b_shape = end_match.groups()
    if a_shape != b_shape:
        raise LiveGTORangeError(f"range suffixes differ in {start}-{end}")
    if a1 == a2 and b1 == b2:
        return [rank + rank for rank in _rank_path(a1, b1)]
    if a1 == b1:
        return [a1 + rank + a_shape for rank in _rank_path(a2, b2)]
    first_path = _rank_path(a1, b1)
    second_path = _rank_path(a2, b2)
    if len(first_path) != len(second_path):
        raise LiveGTORangeError(f"unsupported diagonal range {start}-{end}")
    return [first + second + a_shape for first, second in zip(first_path, second_path)]


def expand_range_text(
    text: str,
    *,
    dead_cards: tuple[str, ...] = (),
) -> tuple[WeightedCombo, ...]:
    """Expand the compact chart notation used by ``poker_data.json``."""

    if not isinstance(text, str) or not text.strip():
        raise LiveGTORangeError("range text cannot be empty")
    dead = set(dead_cards)
    if any(not _CARD_RE.fullmatch(card) for card in dead):
        raise LiveGTORangeError("dead cards use invalid notation")
    combos: dict[tuple[str, str], Decimal] = {}
    for raw_token in text.split(","):
        token = raw_token.strip().replace(" ", "")
        if not token:
            continue
        weight = Decimal(1)
        if ":" in token:
            token, weight_text = token.rsplit(":", 1)
            try:
                weight = Decimal(weight_text)
            except InvalidOperation as error:
                raise LiveGTORangeError(f"invalid range weight {weight_text!r}") from error
            if not weight.is_finite() or not Decimal(0) < weight <= Decimal(1):
                raise LiveGTORangeError("range weights must be in (0, 1]")
        if len(token) == 4 and _CARD_RE.fullmatch(token[:2]) and _CARD_RE.fullmatch(token[2:]):
            token_combos = {_ordered_combo((token[:2], token[2:]))}
        else:
            classes = (
                _expand_class_range(*token.split("-", 1))
                if "-" in token
                else [token]
            )
            token_combos = set().union(*(_class_combos(item) for item in classes))
        for combo in token_combos:
            if dead.intersection(combo):
                continue
            combos[combo] = max(combos.get(combo, Decimal(0)), weight)
    if not combos:
        raise LiveGTORangeError("range has no combos after blocker filtering")
    return tuple(WeightedCombo(cards, weight) for cards, weight in combos.items())


class PositionChartRangeProvider:
    """Build a named approximate postflop profile from local position charts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _charts(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            charts = data["ranges"]["6max"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise LiveGTORangeError(
                f"cannot load 6-max range charts from {self.path}: {error}"
            ) from error
        if not isinstance(charts, dict):
            raise LiveGTORangeError("6-max range charts must be an object")
        return charts

    @staticmethod
    def _chart_text(charts: dict, position: str) -> tuple[str, str]:
        position = position.upper()
        chart = charts.get(position)
        if not isinstance(chart, dict):
            raise LiveGTORangeError(f"no position chart for {position or 'UNKNOWN'}")
        preferred = "defend_vs_open" if position == "BB" else "open"
        for source_kind in dict.fromkeys(
            (preferred, "open", "defend_vs_open")
        ):
            value = chart.get(source_kind)
            if isinstance(value, str) and value.strip():
                return value, source_kind
        raise LiveGTORangeError(f"position {position} has no usable range")

    def ranges_for(self, state: LiveDecisionState) -> RangeBundle:
        observation = _require_verified_hu_handoff(state)
        charts = self._charts()
        oop_position = state.hero_position if state.hero_is_oop else state.villain_position
        ip_position = state.villain_position if state.hero_is_oop else state.hero_position
        oop_text, oop_kind = self._chart_text(charts, oop_position)
        ip_text, ip_kind = self._chart_text(charts, ip_position)
        oop_combos = list(expand_range_text(oop_text, dead_cards=state.board))
        ip_combos = list(expand_range_text(ip_text, dead_cards=state.board))
        hero_combo = _ordered_combo(state.hero_combo)
        hero_range = oop_combos if state.hero_is_oop else ip_combos
        injected = not any(combo.cards == hero_combo for combo in hero_range)
        if injected:
            hero_range.append(WeightedCombo(hero_combo, Decimal(1)))
        profile = (
            f"position-charts-v1:{oop_position}.{oop_kind}"
            f"_vs_{ip_position}.{ip_kind}"
        )
        approximations = [
            "static position charts do not reconstruct the observed preflop action path"
        ]
        if observation.provenance.source == POSITION_ONLY_HANDOFF_SOURCE:
            approximations.append(
                "the terminal preflop wager was unavailable; Hero and villain "
                "positions are anchored to a same-hand preflop capture"
            )
        if injected:
            approximations.append(
                "Hero's observed combo was absent from the chart and was injected at full weight"
            )
        return RangeBundle(
            oop=PlayerRange(Position.OOP, tuple(oop_combos)),
            ip=PlayerRange(Position.IP, tuple(ip_combos)),
            profile_id=profile,
            hero_combo_injected=injected,
            provenance=(
                "local static position charts using actual six-max entries "
                f"{oop_position}.{oop_kind} and {ip_position}.{ip_kind}"
            ),
            approximations=tuple(approximations),
            approximate=True,
        )


class BlueprintRangeProvider:
    """Use a validated six-max preflop blueprint instead of static charts.

    PokerStudy's NL v2 artifact is a fixed 5% rake / 0.5 BB cap, uniform-stack
    abstraction.  ``exact`` mode enforces that profile and exact stack bucket.
    ``abstract`` mode permits a unique nearest-bucket/action-path match but
    records every mismatch in the returned provenance.
    """

    SOURCE_RAKE_RATE_PCT = Decimal(5)
    SOURCE_RAKE_CAP_BB = Decimal("0.5")

    def __init__(
        self,
        config: LiveGTOConfig,
        *,
        store: PokerStudyBlueprintStore | None = None,
    ) -> None:
        self.config = config
        self.store = store or PokerStudyBlueprintStore(
            config.blueprint_cache_path,
            allow_network=config.blueprint_allow_network,
            timeout_seconds=float(config.blueprint_network_timeout_seconds),
        )
        self.resolver = PreflopHistoryResolver(
            self.store,
            contribution_tolerance=config.blueprint_min_tolerance_bb,
        )

    @staticmethod
    def _require_observation(state: LiveDecisionState) -> ObservedPreflopState:
        if state.street.upper() != "PREFLOP":
            return _require_verified_hu_handoff(state)
        if state.preflop_observation is None:
            detail = state.preflop_mapping_error or "no preflop observation was captured"
            raise LiveGTORangeError(detail)
        return state.preflop_observation

    def _profile_notes(self) -> list[str]:
        notes: list[str] = []
        if self.config.blueprint_match_mode == "abstract":
            notes.append(
                "abstract blueprint matching projects observed stacks and action sizes "
                "onto a source bucket"
            )
        rake_mismatch = self.config.rake_rate_pct != self.SOURCE_RAKE_RATE_PCT
        cap_mismatch = self.config.rake_cap_bb != self.SOURCE_RAKE_CAP_BB
        if self.config.blueprint_match_mode == "exact" and (
            rake_mismatch or cap_mismatch
        ):
            raise LiveGTORangeError(
                "the blueprint profile is fixed at 5% rake with a 0.5 BB cap"
            )
        if rake_mismatch or cap_mismatch:
            notes.append(
                "source rake is 5% capped at 0.5 BB; configured postflop rake differs"
            )
        return notes

    def _select_stack(
        self,
        observation: ObservedPreflopState,
        hero_position: str,
    ) -> tuple[int, list[str]]:
        manifest = self.store.manifest()
        if set(manifest.positions) != {"UTG", "HJ", "CO", "BTN", "SB", "BB"}:
            raise LiveGTORangeError("blueprint manifest is not the expected six-max game")
        stacks = observation.initial_stack_map
        hero = canonical_position(hero_position)
        hero_stack = stacks[hero]
        relevant_positions = observation.live_positions
        effective_stacks = tuple(
            min(hero_stack, stacks[position])
            for position in relevant_positions
            if position != hero
        ) or (hero_stack,)
        if any(value <= 0 for value in effective_stacks):
            raise LiveGTORangeError("effective preflop stacks must be positive")

        def bucket_error(stack: int) -> Decimal:
            candidate = Decimal(stack)
            return max(
                abs(candidate - value) * Decimal(100) / value
                for value in effective_stacks
            )

        selected = min(manifest.stacks, key=lambda stack: (bucket_error(stack), stack))
        # Exact mode is a uniform six-stack profile, so use Hero's starting
        # stack deterministically and let the all-six check below reject any
        # mismatch.  Abstract bucket selection already minimizes the maximum
        # error across every relevant effective stack.
        reference = hero_stack
        notes = self._profile_notes()

        if self.config.blueprint_match_mode == "exact":
            if abs(Decimal(selected) - reference) > self.config.blueprint_min_tolerance_bb:
                raise LiveGTORangeError(
                    f"no exact blueprint stack for the observed {reference} BB effective stack"
                )
            mismatched = {
                position: value
                for position, value in stacks.items()
                if abs(value - Decimal(selected)) > self.config.blueprint_min_tolerance_bb
            }
            if mismatched:
                raise LiveGTORangeError(
                    "the exact uniform-stack blueprint does not match all six starting stacks"
                )
        else:
            values = tuple(stacks.values())
            maximum_error = bucket_error(selected)
            if maximum_error > self.config.blueprint_max_stack_error_pct:
                raise LiveGTORangeError(
                    f"nearest {selected} BB blueprint bucket is "
                    f"{format(maximum_error.normalize(), 'f')}% away from at least "
                    "one relevant effective stack"
                )
            if any(value != Decimal(selected) for value in values):
                notes.append(
                    f"uniform {selected} BB bucket used for observed starting stacks "
                    f"{format(min(values).normalize(), 'f')}-"
                    f"{format(max(values).normalize(), 'f')} BB"
                )
        return selected, notes

    def _match_tolerances(
        self, observation: ObservedPreflopState
    ) -> dict[str, Decimal]:
        if self.config.blueprint_match_mode == "exact":
            return {
                position: self.config.blueprint_min_tolerance_bb
                for position in observation.contribution_map
            }
        result: dict[str, Decimal] = {}
        for position, value in observation.contribution_map.items():
            relative = (
                Decimal(0)
                if value <= Decimal(1)
                else value * self.config.blueprint_size_tolerance_pct / Decimal(100)
            )
            result[position] = max(
                self.config.blueprint_min_tolerance_bb, relative
            )
        return result

    def _resolve(
        self,
        state: LiveDecisionState,
    ) -> tuple[ObservedPreflopState, PreflopResolution, list[str]]:
        observation = self._require_observation(state)
        stack, notes = self._select_stack(observation, state.hero_position)
        tolerances = self._match_tolerances(observation)
        if observation.terminal:
            terminal_resolver = (
                self.resolver.resolve_hu_handoff
                if len(observation.live_positions) == 2
                else self.resolver.resolve_terminal_handoff
            )
            resolution = terminal_resolver(
                stack=stack,
                observed_contributions=observation.contribution_map,
                observed_folded=observation.folded,
                survivors=observation.live_positions,
                observed_all_in=observation.all_in,
                tolerance=tolerances,
            )
        else:
            if observation.actor != canonical_position(state.hero_position):
                raise LiveGTORangeError("preflop observation actor is not Hero")
            resolution = self.resolver.resolve_decision(
                stack=stack,
                expected_actor=observation.actor,
                observed_contributions=observation.contribution_map,
                observed_folded=observation.folded,
                observed_all_in=observation.all_in,
                tolerance=tolerances,
            )

            hero = canonical_position(state.hero_position)
            observed_contributions = observation.contribution_map
            actual_highest = max(observed_contributions.values(), default=Decimal(0))
            expected_call = actual_highest - observed_contributions[hero]
            direct_tolerance = self.config.blueprint_min_tolerance_bb
            if abs(state.amount_to_call_bb - expected_call) > direct_tolerance:
                raise LiveGTORangeError(
                    "visible call amount disagrees with Hero's table contribution"
                )
            if (
                abs(state.hero_current_bet_bb - observed_contributions[hero])
                > direct_tolerance
            ):
                raise LiveGTORangeError(
                    "Hero current bet disagrees with the preflop observation"
                )
            if (
                abs(
                    state.hero_stack_bb
                    + state.hero_current_bet_bb
                    - observation.initial_stack_map[hero]
                )
                > direct_tolerance
            ):
                raise LiveGTORangeError(
                    "Hero remaining stack and contribution do not conserve chips"
                )
            if hero in observation.all_in:
                raise LiveGTORangeError("Hero is already all-in and cannot act")
            legal = {
                action.upper().replace("-", "_")
                for action in state.legal_actions
                if action
            }
            if expected_call > direct_tolerance:
                if "CALL" not in legal or "CHECK" in legal:
                    raise LiveGTORangeError(
                        "Hero action buttons disagree with a facing-bet node"
                    )
            elif "CHECK" not in legal or "CALL" in legal:
                raise LiveGTORangeError(
                    "Hero action buttons disagree with a no-wager node"
                )

        deltas = {
            position: abs(
                observation.contribution_map[position]
                - resolution.state.contribution_map[position]
            )
            for position in resolution.state.contribution_map
        }
        largest_delta = max(deltas.values(), default=Decimal(0))
        if largest_delta > Decimal("0.05"):
            notes.append(
                "preflop contribution sizes mapped to the unique blueprint path "
                f"(maximum delta {format(largest_delta.normalize(), 'f')} BB)"
            )
        return observation, resolution, notes

    def _provenance(
        self,
        stack: int,
        histories: tuple[str, ...],
    ) -> str:
        manifest = self.store.manifest()
        digests = [self.store.node(stack, history).response_sha256 for history in histories]
        artifact_hash = hashlib.sha256("".join(digests).encode("ascii")).hexdigest()
        return (
            f"PokerStudy NL v2 generated {manifest.generated_at}; MonkerSolver 6-max "
            f"5%/0.5BB; stack={stack}; artifact={artifact_hash[:16]}"
        )

    @staticmethod
    def _policy_target(
        resolution: PreflopResolution,
        action,
    ) -> Decimal | None:
        parsed = parse_action_label(action.label)
        if parsed.kind == "raise":
            assert parsed.size_pct is not None
            actor = resolution.state.actor
            if actor is None:
                raise LiveGTORangeError("preflop decision has no actor")
            current = resolution.state.contribution_map[actor]
            call_amount = resolution.state.highest_bet - current
            pot_after_call = resolution.state.pot + call_amount
            return resolution.state.highest_bet + (
                pot_after_call * parsed.size_pct / Decimal(100)
            )
        if parsed.kind == "all_in":
            return Decimal(resolution.stack)
        return None

    @staticmethod
    def _observed_last_full_raise(
        observation: ObservedPreflopState,
        resolution: PreflopResolution,
    ) -> Decimal:
        """Replay observable raise targets, rejecting paths whose sizes were lost.

        A terminal contribution records an earlier raise target only while that
        actor has not subsequently called, raised, or moved all-in. Abstract
        matching may otherwise borrow the source tree's smaller raise increment
        and authorize a below-minimum live re-raise.
        """

        observed = observation.contribution_map
        parsed_steps = tuple(
            (step, parse_action_label(step.action))
            for step in resolution.steps
        )
        last_money_action: dict[str, int] = {}
        for index, (step, parsed) in enumerate(parsed_steps):
            if parsed.kind in {"call", "raise", "all_in"}:
                last_money_action[step.actor] = index

        highest = Decimal(1)
        last_full_raise = Decimal(1)
        tolerance = Decimal("0.01")
        for index, (step, parsed) in enumerate(parsed_steps):
            if parsed.kind not in {"raise", "all_in"}:
                continue
            if last_money_action.get(step.actor) != index:
                raise LiveGTORangeError(
                    "abstract raise legality cannot reconstruct an earlier observed "
                    f"raise target for {step.actor}"
                )
            target = observed.get(step.actor)
            if target is None or not target.is_finite() or target < 0:
                raise LiveGTORangeError(
                    f"abstract raise legality is missing {step.actor}'s observed target"
                )
            if target <= highest + tolerance:
                if parsed.kind == "all_in" and target <= highest + tolerance:
                    continue
                raise LiveGTORangeError(
                    f"abstract raise legality cannot reconcile {step.actor}'s "
                    "observed raise target"
                )
            increment = target - highest
            if parsed.kind == "raise" and increment + tolerance < last_full_raise:
                raise LiveGTORangeError(
                    f"observed {step.actor} raise increment is below the live minimum"
                )
            if increment + tolerance >= last_full_raise:
                last_full_raise = increment
            highest = target

        actual_highest = max(observed.values(), default=Decimal(0))
        if abs(highest - actual_highest) > tolerance:
            raise LiveGTORangeError(
                "abstract raise legality cannot reconstruct the actual last full raise"
            )
        return last_full_raise

    @staticmethod
    def _legal_preflop_kind(kind: str, legal_actions: tuple[str, ...]) -> bool:
        legal = {
            action.upper().replace("-", "_")
            for action in legal_actions
            if action
        }
        if kind == "raise":
            return bool({"RAISE", "BET"} & legal)
        if kind == "allin":
            return bool({"ALL_IN", "RAISE", "BET"} & legal)
        return kind.upper() in legal

    def preflop_policy_for(self, state: LiveDecisionState) -> BlueprintDecisionBundle:
        if state.street.upper() != "PREFLOP":
            raise LiveGTORangeError("preflop policy requested for a postflop state")
        observation, resolution, notes = self._resolve(state)
        node = self.store.node(resolution.stack, resolution.history)
        hand_class = hand_class_for_cards(*state.hero_combo)

        prior_reach = self.resolver.walk_reaches(resolution).for_position(
            state.hero_position
        )[hand_class]
        weighted_actions = []
        total = Decimal(0)
        for action in node.actions:
            weight = dict(action.weights).get(hand_class, Decimal(0))
            if weight <= 0:
                continue
            total += weight
            weighted_actions.append((action, weight))
        if total <= 0 or prior_reach <= 0:
            raise LiveGTORangeError(
                f"Hero {canonical_position(state.hero_position)} {hand_class} has zero blueprint reach"
            )
        if abs(total - prior_reach) > Decimal("0.01"):
            raise LiveGTORangeError(
                f"blueprint action reach {total} disagrees with prior reach {prior_reach}"
            )

        hero_total = state.hero_stack_bb + state.hero_current_bet_bb
        actual_highest = max(observation.contribution_map.values(), default=Decimal(0))
        observed_last_full_raise: Decimal | None = None
        policies: list[BlueprintPolicyAction] = []
        for action, weight in weighted_actions:
            policy_kind = (
                "check"
                if action.kind == "call" and state.amount_to_call_bb == 0
                else action.kind
            )
            if not self._legal_preflop_kind(policy_kind, state.legal_actions):
                raise LiveGTORangeError(
                    f"blueprint action {action.label} is absent from the visible Hero buttons"
                )
            source_target = self._policy_target(resolution, action)
            display_target = source_target
            if action.kind == "allin":
                display_target = hero_total
            if display_target is not None:
                is_short_all_in = (
                    action.kind == "allin"
                    and abs(display_target - hero_total) <= Decimal("0.01")
                )
                if (
                    action.kind != "allin"
                    and display_target >= hero_total - Decimal("0.01")
                ):
                    raise LiveGTORangeError(
                        "a normal blueprint raise would collapse into an actual all-in"
                    )
                if (
                    action.kind != "allin"
                    and self.config.blueprint_match_mode == "abstract"
                ):
                    if observed_last_full_raise is None:
                        observed_last_full_raise = self._observed_last_full_raise(
                            observation, resolution
                        )
                        notes.append(
                            "raise legality checked against the reconstructed observed "
                            f"last full raise of "
                            f"{format(observed_last_full_raise.normalize(), 'f')} BB"
                        )
                    minimum_raise_to = actual_highest + observed_last_full_raise
                else:
                    minimum_raise_to = (
                        actual_highest + resolution.state.last_full_raise
                    )
                if display_target > hero_total + Decimal("0.01"):
                    raise LiveGTORangeError(
                        f"blueprint raise-to {display_target} BB exceeds Hero's actual stack"
                    )
                if (
                    display_target <= actual_highest
                    or (
                        display_target + Decimal("0.01") < minimum_raise_to
                        and not is_short_all_in
                    )
                ):
                    raise LiveGTORangeError(
                        f"blueprint raise-to {display_target} BB is not legal in the observed game"
                    )
            ev = None if action.evs is None else dict(action.evs).get(hand_class)
            policies.append(
                BlueprintPolicyAction(
                    label=action.label,
                    kind=policy_kind,
                    frequency=weight / total,
                    source_target_bb=source_target,
                    display_target_bb=display_target,
                    ev=ev,
                )
            )

        histories = tuple((*self.resolver.walk_reaches(resolution).artifact_nodes, node.history))
        provenance = self._provenance(resolution.stack, histories)
        profile = f"pokerstudy-nl-v2:{resolution.stack}bb:{resolution.history}"
        return BlueprintDecisionBundle(
            stack=resolution.stack,
            history=resolution.history,
            hand_class=hand_class,
            actions=tuple(policies),
            profile_id=profile,
            provenance=provenance,
            approximations=tuple(notes),
            resolution=resolution,
            approximate=self.config.blueprint_match_mode == "abstract",
        )

    def ranges_for(self, state: LiveDecisionState) -> RangeBundle:
        if state.street.upper() == "PREFLOP":
            raise LiveGTORangeError("postflop ranges requested for a preflop state")
        observation, resolution, notes = self._resolve(state)
        if not observation.terminal:
            raise LiveGTORangeError("postflop range construction requires a terminal preflop path")
        oop_position = canonical_position(
            state.hero_position if state.hero_is_oop else state.villain_position
        )
        ip_position = canonical_position(
            state.villain_position if state.hero_is_oop else state.hero_position
        )
        current_positions = {oop_position, ip_position}
        if not current_positions.issubset(observation.live_positions):
            raise LiveGTORangeError(
                "current HU seats were not both preflop survivors"
            )
        projected_positions = observation.live_positions - current_positions
        if projected_positions:
            notes.append(
                "HU projection begins after postflop fold(s) by "
                + ", ".join(
                    position
                    for position in (
                        "UTG",
                        "HJ",
                        "CO",
                        "BTN",
                        "SB",
                        "BB",
                    )
                    if position in projected_positions
                )
                + "; their postflop folding strategy is not range-conditioned"
            )

        reach = self.resolver.walk_reaches(
            resolution,
            hero_position=state.hero_position,
            hero_cards=state.hero_combo,
        )
        dead_cards = tuple(state.board)
        oop_weights = expand_position_reach(reach, oop_position, dead_cards=dead_cards)
        ip_weights = expand_position_reach(reach, ip_position, dead_cards=dead_cards)
        oop_combos = tuple(
            WeightedCombo(cards, weight) for cards, weight in oop_weights.items()
        )
        ip_combos = tuple(
            WeightedCombo(cards, weight) for cards, weight in ip_weights.items()
        )
        if not oop_combos or not ip_combos:
            raise LiveGTORangeError("a surviving blueprint range is empty after blockers")

        notes.append("folded-card bunching is omitted by the HU postflop solve")
        if state.street.upper() in {"TURN", "RIVER"}:
            notes.append(
                "ranges are preflop-conditioned but not conditioned on prior postflop actions"
            )
        provenance = self._provenance(resolution.stack, reach.artifact_nodes)
        return RangeBundle(
            oop=PlayerRange(Position.OOP, oop_combos),
            ip=PlayerRange(Position.IP, ip_combos),
            profile_id=(
                f"pokerstudy-nl-v2:{resolution.stack}bb:{resolution.history}:"
                f"{oop_position}-vs-{ip_position}"
            ),
            hero_combo_injected=False,
            provenance=provenance,
            approximations=tuple(notes),
            # The six-max-origin HU solve omits the four folded ranges'
            # card-removal distribution. Even an exact stack/action-path match
            # is therefore not an exact continuation of the source game.
            approximate=True,
        )


def _units(value: Decimal, scale: int, field: str) -> int:
    if not value.is_finite() or value <= 0:
        raise LiveGTOConfigurationError(f"{field} must be finite and positive")
    result = int((value * scale).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    if result <= 0:
        raise LiveGTOConfigurationError(f"{field} rounds to zero solver units")
    return result


def _nonnegative_units(value: Decimal, scale: int, field: str) -> int:
    if not value.is_finite() or value < 0:
        raise LiveGTOConfigurationError(f"{field} must be finite and non-negative")
    return int((value * scale).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _street_sizing_config(
    street: Street,
    *,
    default_bet: str,
    observed_bet_units: int,
) -> BetSizingConfig:
    """Keep the base menu while adding an exact traversed off-tree size.

    Replacing the menu with the observed size would condition Villain's range
    on a different game in which no alternative bet existed. The additive size
    is therefore added alongside the configured base size and traversed exactly.
    """

    standard = PlayerBetSizes(default_bet, "2.5x")
    current = (
        PlayerBetSizes(f"{default_bet}, {observed_bet_units}c", "2.5x")
        if observed_bet_units
        else standard
    )
    flop = StreetBetSizes(current, current) if street is Street.FLOP else StreetBetSizes(standard, standard)
    turn = StreetBetSizes(current, current) if street is Street.TURN else StreetBetSizes(standard, standard)
    river = StreetBetSizes(current, current) if street is Street.RIVER else StreetBetSizes(standard, standard)
    return BetSizingConfig(
        flop=flop,
        turn=turn,
        river=river,
        flop_donk_sizes=None,
        turn_donk_sizes="",
        river_donk_sizes="",
    )


def build_live_spec(
    state: LiveDecisionState,
    ranges: RangeBundle,
    config: LiveGTOConfig,
) -> SolveSpec:
    try:
        street = Street(state.street.upper())
    except ValueError as error:
        raise LiveGTOConfigurationError(f"unsupported street {state.street!r}") from error
    pot = _units(state.pot_bb, config.chip_scale, "pot")
    effective_stack = _units(
        min(state.hero_stack_bb, state.villain_stack_bb),
        config.chip_scale,
        "effective stack",
    )
    observed_bet = _nonnegative_units(
        state.observed_bet_to_bb,
        config.chip_scale,
        "observed bet",
    )
    requested_bet = int(
        (Decimal(pot) * config.bet_size_pct / Decimal(100)).quantize(
            Decimal(1), rounding=ROUND_HALF_UP
        )
    )
    requested_bet = max(1, requested_bet)
    legal = _normalized_legal_actions(state.legal_actions)
    if state.amount_to_call_bb > 0:
        if not {"FOLD", "CALL"}.issubset(legal):
            raise LiveGTOConfigurationError(
                "the visible controls do not authorize modeled fold/call branches"
            )
    elif "CHECK" not in legal:
        raise LiveGTOConfigurationError(
            "the visible controls do not authorize a modeled check branch"
        )
    if "BET" not in legal and "ALL_IN" in legal:
        aggressive = Action(ActionKind.ALL_IN, effective_stack)
    elif requested_bet >= effective_stack:
        aggressive = Action(ActionKind.ALL_IN, effective_stack)
    else:
        aggressive = Action(ActionKind.BET, requested_bet)
    size_text = f"{format(config.bet_size_pct.normalize(), 'f')}%"
    history = []
    for item in state.action_history:
        if item == "CHECK":
            history.append(Action(ActionKind.CHECK))
        elif item == "BET":
            if observed_bet <= 0:
                raise LiveGTOConfigurationError(
                    "a BET history requires a positive observed bet"
                )
            history.append(
                Action(
                    ActionKind.ALL_IN if observed_bet == effective_stack else ActionKind.BET,
                    observed_bet,
                )
            )
        else:
            raise LiveGTOConfigurationError(
                f"unsupported live action-history item {item!r}"
            )

    if state.amount_to_call_bb > 0:
        facing_bet = _units(
            state.amount_to_call_bb,
            config.chip_scale,
            "amount to call",
        )
        modeled_actions = [Action(ActionKind.FOLD), Action(ActionKind.CALL)]
        if effective_stack > facing_bet:
            if "RAISE" in legal:
                raise_to = min(
                    effective_stack,
                    int(
                        (Decimal(facing_bet) * Decimal("2.5")).quantize(
                            Decimal(1), rounding=ROUND_HALF_UP
                        )
                    ),
                )
                modeled_actions.append(
                    Action(
                        ActionKind.ALL_IN
                        if raise_to == effective_stack
                        else ActionKind.RAISE,
                        raise_to,
                    )
                )
            elif "ALL_IN" in legal:
                modeled_actions.append(
                    Action(ActionKind.ALL_IN, effective_stack)
                )
            else:
                raise LiveGTOConfigurationError(
                    "the visible controls do not authorize a modeled raise branch"
                )
        legal_action_kinds = tuple(action.kind for action in modeled_actions)
    else:
        if aggressive.kind is ActionKind.BET and "BET" not in legal:
            raise LiveGTOConfigurationError(
                "the visible controls do not authorize a modeled bet branch"
            )
        if (
            aggressive.kind is ActionKind.ALL_IN
            and not ({"BET", "ALL_IN"} & legal)
        ):
            raise LiveGTOConfigurationError(
                "the visible controls do not authorize a modeled all-in branch"
            )
        facing_bet = 0
        modeled_actions = [Action(ActionKind.CHECK), aggressive]
        legal_action_kinds = (ActionKind.CHECK, aggressive.kind)

    parameters = SolveParameters(
        chip_scale=config.chip_scale,
        chip_unit="centi-BB" if config.chip_scale == 100 else "scaled-BB",
        bet_sizes=_street_sizing_config(
            street,
            default_bet=size_text,
            observed_bet_units=observed_bet,
        ),
        add_allin_threshold=Decimal(0),
        force_allin_threshold=Decimal(0),
        merging_threshold=Decimal(0),
        target_exploitability_pct=config.target_exploitability_pct,
        max_iterations=config.max_iterations,
        allocation_mode=AllocationMode.UNCOMPRESSED_F32,
        solver_name="b-inary/postflop-solver",
        solver_commit=PINNED_SOLVER_COMMIT,
    )
    return SolveSpec(
        street=street,
        board=state.board,
        acting_player=Position.OOP if state.hero_is_oop else Position.IP,
        oop_range=ranges.oop,
        ip_range=ranges.ip,
        tree=TreeConfig(
            pot=pot,
            effective_stack=effective_stack,
            facing_bet=facing_bet,
            legal_action_kinds=legal_action_kinds,
            modeled_actions=tuple(modeled_actions),
            action_history=tuple(history),
            rake_rate_pct=config.rake_rate_pct,
            rake_cap=int(
                (config.rake_cap_bb * config.chip_scale).quantize(
                    Decimal(1), rounding=ROUND_HALF_UP
                )
            ),
        ),
        parameters=parameters,
    )


def _eligibility_reason(state: LiveDecisionState) -> str:
    if state.mapping_error:
        return state.mapping_error
    if state.street.upper() == "PREFLOP":
        return "preflop is not supported by the local postflop solver"
    if state.street.upper() not in {"FLOP", "TURN", "RIVER"}:
        return "street is unknown"
    expected_board = {"FLOP": 3, "TURN": 4, "RIVER": 5}[state.street.upper()]
    if len(state.board) != expected_board:
        return "board is incomplete"
    if state.active_villains != 1:
        return "the solver requires exactly one active villain"
    handoff_reason = _verified_hu_handoff_reason(state)
    if handoff_reason:
        return handoff_reason
    if state.pot_bb <= 0 or min(state.hero_stack_bb, state.villain_stack_bb) <= 0:
        return "street-root pot and effective stack must be positive"
    monetary = (
        state.pot_bb,
        state.hero_stack_bb,
        state.villain_stack_bb,
        state.hero_current_bet_bb,
        state.villain_current_bet_bb,
        state.amount_to_call_bb,
        state.observed_bet_to_bb,
    )
    if any(not value.is_finite() or value < 0 for value in monetary):
        return "live monetary values must be finite and non-negative"
    try:
        cards = (*state.board, *state.hero_combo)
        if len(cards) != len(set(cards)) or any(not _CARD_RE.fullmatch(card) for card in cards):
            return "cards are invalid or repeated"
    except TypeError:
        return "cards are invalid"

    legal = _normalized_legal_actions(state.legal_actions)
    history = tuple(action.upper() for action in state.action_history)
    allowed_histories = {(), ("CHECK",), ("BET",), ("CHECK", "BET")}
    if history not in allowed_histories:
        return "only root, check, and first-bet histories are supported"

    no_wager = state.amount_to_call_bb == 0
    no_contributions = (
        state.hero_current_bet_bb == 0 and state.villain_current_bet_bb == 0
    )
    if history == ():
        if not state.hero_is_oop:
            return "an empty action history must be an OOP street root"
        if not state.street_root_confirmed:
            return "same-street history does not confirm an untouched OOP root"
        if not no_wager or not no_contributions:
            return "street contributions prove this is not an untouched root node"
    elif history == ("CHECK",):
        if state.hero_is_oop:
            return "after one root check the current player must be IP"
        if not no_wager or not no_contributions:
            return "IP-after-check requires zero street contributions"
    else:
        if history == ("BET",) and state.hero_is_oop:
            return "a first OOP bet must face IP"
        if history == ("CHECK", "BET") and not state.hero_is_oop:
            return "check-bet must return action to OOP"
        observed = state.observed_bet_to_bb
        if observed <= 0:
            return "a bet history requires a positive observed bet"
        if state.hero_current_bet_bb != 0:
            return "prior Hero bets/raises are not supported"
        if state.villain_current_bet_bb != observed:
            return "villain contribution disagrees with the observed bet-to amount"
        if state.amount_to_call_bb != observed:
            return "incremental call amount disagrees with the observed bet"
        if observed > min(state.hero_stack_bb, state.villain_stack_bb):
            return "observed bet exceeds the reconstructed effective stack"

    if no_wager:
        if "CHECK" not in legal or not ({"BET", "ALL_IN"} & legal):
            return "the current no-wager decision is not check/bet"
    else:
        if not {"FOLD", "CALL"}.issubset(legal):
            return "a facing-bet decision must visibly allow fold and call"
        effective_stack = min(state.hero_stack_bb, state.villain_stack_bb)
        if effective_stack > state.observed_bet_to_bb and not (
            {"RAISE", "ALL_IN"} & legal
        ):
            return "a non-all-in facing-bet node has no confirmed raise action"
    return ""


def _private_roll(secret: bytes, seed: str) -> Decimal:
    digest = hmac.digest(secret, seed.encode("utf-8"), "sha256")
    return Decimal(int.from_bytes(digest[:8], "big")) / Decimal(2**64)


def _select_private_mix(
    items,
    frequency: Callable[[object], Decimal],
    seed: str,
    secret: bytes,
):
    weighted = []
    for item in items:
        weight = frequency(item)
        if weight > 0:
            weighted.append((item, weight))
    total = sum((weight for _, weight in weighted), Decimal(0))
    if total <= 0:
        raise LiveGTORangeError("mixed strategy has no positive probability mass")
    roll = _private_roll(secret, seed)
    threshold = roll * total
    cumulative = Decimal(0)
    for item, weight in weighted:
        cumulative += weight
        if threshold < cumulative:
            return item, roll
    return weighted[-1][0], roll


def _stable_action(policy, seed: str, secret: bytes):
    return _select_private_mix(
        policy.action_values,
        lambda value: value.frequency,
        seed,
        secret,
    )


def _stable_blueprint_action(
    bundle: BlueprintDecisionBundle,
    seed: str,
    secret: bytes,
) -> tuple[BlueprintPolicyAction, Decimal]:
    return _select_private_mix(
        bundle.actions,
        lambda action: action.frequency,
        seed,
        secret,
    )


def _blueprint_action_label(
    action: BlueprintPolicyAction,
    state: LiveDecisionState,
) -> tuple[str, str]:
    if action.kind == "fold":
        return "Fold", "0"
    if action.kind == "check":
        return "Check", "0"
    if action.kind == "call":
        return "Call", f"{format(state.amount_to_call_bb.normalize(), 'f')} BB"
    if action.kind in {"raise", "allin"}:
        if action.display_target_bb is None:
            raise LiveGTORangeError("a blueprint raise has no raise-to amount")
        return "Raise", f"{format(action.display_target_bb.normalize(), 'f')} BB"
    raise LiveGTORangeError(f"unsupported blueprint policy kind {action.kind!r}")


def _format_blueprint_analysis(
    bundle: BlueprintDecisionBundle,
    selected: BlueprintPolicyAction,
    roll: Decimal,
    state: LiveDecisionState,
) -> str:
    action, size = _blueprint_action_label(selected, state)
    mix_name = (
        "Approximate blueprint mix"
        if bundle.approximate
        else "Blueprint mix"
    )
    mixes: list[str] = []
    for candidate in bundle.actions:
        label, amount = _blueprint_action_label(candidate, state)
        suffix = f" {amount}" if amount != "0" else ""
        mixes.append(f"{label}{suffix} {float(candidate.frequency) * 100:.1f}%")
    caveats = (
        "; ".join(bundle.approximations)
        if bundle.approximations
        else "exact stack/action-path match to the fixed source abstraction"
    )
    return (
        f"**Action:** {action}\n"
        f"**Size:** {size}\n"
        f"**Why:** Solver-derived preflop mix for {bundle.hand_class} at the uniquely "
        f"reconstructed node `{bundle.history}`. The displayed action is a stable "
        "roll from the full mixed strategy.\n"
        f"* **{mix_name}:** {' | '.join(mixes)}\n"
        f"* **Profile:** {bundle.profile_id}\n"
        f"* **Provenance:** {bundle.provenance}\n"
        f"* **Approximation boundary:** {caveats}. This is a fixed-tree blueprint, "
        "not a proof of exact full-game 6-max Nash equilibrium.\n"
        f"* **Stable roll:** {float(roll) * 100:.2f}%"
    )


def _action_label(
    action: Action,
    scale: int,
    *,
    amount_to_call_bb: Decimal,
) -> tuple[str, str]:
    if action.kind is ActionKind.CHECK:
        return "Check", "0"
    if action.kind is ActionKind.FOLD:
        return "Fold", "0"
    if action.kind is ActionKind.CALL:
        return "Call", f"{format(amount_to_call_bb.normalize(), 'f')} BB"
    if action.kind in {ActionKind.BET, ActionKind.RAISE, ActionKind.ALL_IN}:
        amount = Decimal(action.amount or 0) / Decimal(scale)
        is_raise = action.kind is ActionKind.RAISE or (
            action.kind is ActionKind.ALL_IN and amount_to_call_bb > 0
        )
        return (
            "Raise" if is_raise else "Bet",
            f"{format(amount.normalize(), 'f')} BB",
        )
    return action.kind.value.title(), "0"


def _format_analysis(
    policy,
    selected,
    roll: Decimal,
    config: LiveGTOConfig,
    ranges: RangeBundle,
    result: SolveResult | ContinuationResult,
    source: str,
    state: LiveDecisionState,
) -> str:
    action, size = _action_label(
        selected.action,
        config.chip_scale,
        amount_to_call_bb=state.amount_to_call_bb,
    )
    mixes = []
    for value in policy.action_values:
        label, amount = _action_label(
            value.action,
            config.chip_scale,
            amount_to_call_bb=state.amount_to_call_bb,
        )
        suffix = f" {amount}" if amount != "0" else ""
        mixes.append(f"{label}{suffix} {float(value.frequency) * 100:.1f}%")
    extra = dict(result.metadata.extra)
    exploitability_pct = extra.get("exploitability_pct_of_pot", "unknown")
    injected = "; observed Hero combo added" if ranges.hero_combo_injected else ""
    provenance = ranges.provenance or "local position charts"
    approximation_text = (
        "; ".join(ranges.approximations)
        if ranges.approximations
        else "no additional recorded approximation"
    )
    sizing_assumption = (
        f"configured {format(config.bet_size_pct.normalize(), 'f')}% pot first bet "
        f"plus exact observed {format(state.observed_bet_to_bb.normalize(), 'f')} BB size"
        if state.observed_bet_to_bb > 0
        else f"{format(config.bet_size_pct.normalize(), 'f')}% pot first bet"
    )
    mix_name = "Approximate solver mix" if ranges.approximate else "GTO mix"
    solver_description = (
        "Approximate local solver mix"
        if ranges.approximate
        else "Local solver mix"
    )
    conditional_ranges = getattr(result, "conditional_ranges", ())
    if conditional_ranges:
        range_counts = ", ".join(
            f"{item.position.value} {len(item.combos)} combos"
            for item in conditional_ranges
        )
        continuity_line = (
            "* **Range continuity:** action-conditioned from the verified "
            f"flop root through every recorded action/card ({range_counts})\n"
        )
    else:
        continuity_line = ""
    return (
        f"**Action:** {action}\n"
        f"**Size:** {size}\n"
        f"**Why:** {solver_description} for the named {ranges.profile_id} range profile. "
        "The displayed action is a stable roll from the full mixed strategy.\n"
        f"* **{mix_name}:** {' | '.join(mixes)}\n"
        f"* **Source:** {source}; target {config.target_exploitability_pct}% pot; "
        f"achieved {exploitability_pct}% pot; {result.metadata.iterations} iterations\n"
        f"* **Range assumption:** {ranges.profile_id}{injected}\n"
        f"* **Range provenance:** {provenance}\n"
        f"* **Approximation boundary:** {approximation_text}\n"
        f"{continuity_line}"
        f"* **Tree assumption:** {sizing_assumption}; one 2.5x response raise when legal\n"
        f"* **Stable roll:** {float(roll) * 100:.2f}%"
    )


def _postflop_source(*, cache_hit: bool, approximate: bool) -> str:
    prefix = "APPROXIMATE_SOLVER" if approximate else "GTO"
    return f"{prefix} {'cache' if cache_hit else 'fresh'}"


def _cache_get(
    cache: OracleCache,
    spec: SolveSpec,
    *,
    digest: str,
) -> SolveResult | None:
    try:
        return cache.get(
            spec,
            expected_binary_sha256=digest,
            expected_execution_context="owned_simulator",
        )
    except (
        sqlite3.Error,
        json.JSONDecodeError,
        InvalidOperation,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        raise LiveGTOCacheError(
            f"live solver cache entry is corrupt or unreadable: {error}"
        ) from error


def _cache_put(cache: OracleCache, spec: SolveSpec, result: SolveResult) -> None:
    try:
        cache.put(spec, result)
    except (
        sqlite3.Error,
        json.JSONDecodeError,
        InvalidOperation,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        raise LiveGTOCacheError(
            f"live solver cache write failed: {error}"
        ) from error


def _continuation_cache_get(
    cache: OracleCache,
    spec: ContinuationSpec,
    *,
    digest: str,
) -> ContinuationResult | None:
    try:
        return cache.get_continuation(
            spec,
            expected_binary_sha256=digest,
            expected_execution_context="owned_simulator",
        )
    except (
        sqlite3.Error,
        json.JSONDecodeError,
        InvalidOperation,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        raise LiveGTOCacheError(
            "live continuation cache entry is corrupt or unreadable: "
            f"{error}"
        ) from error


def _continuation_cache_put(
    cache: OracleCache,
    spec: ContinuationSpec,
    result: ContinuationResult,
) -> None:
    try:
        cache.put_continuation(spec, result)
    except (
        sqlite3.Error,
        json.JSONDecodeError,
        InvalidOperation,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        raise LiveGTOCacheError(
            f"live continuation cache write failed: {error}"
        ) from error


class LiveGTORouter:
    """Cache-first solver router for the supported owned-simulator node."""

    def __init__(
        self,
        config: LiveGTOConfig,
        *,
        range_provider: object | None = None,
        engine_factory: Callable[..., EngineClient] = EngineClient,
    ) -> None:
        self.config = config
        if range_provider is not None:
            self.range_provider = range_provider
        elif config.range_source == "blueprint":
            self.range_provider = BlueprintRangeProvider(config)
        else:
            self.range_provider = PositionChartRangeProvider(config.range_data_path)
        self.engine_factory = engine_factory

    def _evaluate_continuation(
        self,
        state: LiveDecisionState,
        started: float,
    ) -> LiveGTOOutcome:
        """Solve a complete, replayed HU path from its true flop root."""

        from live_gto_continuation import (
            LiveGTOContinuationError,
            build_live_continuation_spec,
            flop_range_state,
        )

        try:
            ranges = self.range_provider.ranges_for(flop_range_state(state))
            spec = build_live_continuation_spec(state, ranges, self.config)
            client = self.engine_factory(
                self.config.engine_path,
                offline_only_acknowledged=False,
                owned_simulator_acknowledged=True,
                # A continuation still solves the full flop tree before
                # traversing turn/river, so it needs the flop budget.
                timeout_seconds=self.config.timeout_for(Street.FLOP),
            )
            digest = client.binary_sha256
            self.config.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with OracleCache(self.config.cache_path) as cache:
                    result = _continuation_cache_get(
                        cache,
                        spec,
                        digest=digest,
                    )
                    cache_hit = result is not None
                    if result is not None and not result.metadata.converged:
                        raise OracleValidationError(
                            "cached continuation stopped before the requested "
                            "exploitability target"
                        )
                    if result is None:
                        if self.config.flop_cache_only:
                            return LiveGTOOutcome(
                                LiveGTOStatus.CACHE_MISS,
                                "complete continuation is cache-only and this "
                                "flop path is not cached",
                                time.perf_counter() - started,
                                spec=spec,
                            )
                        result = client.solve_continuation(spec)
                        if not result.metadata.converged:
                            raise OracleValidationError(
                                "continuation solver stopped before the requested "
                                "exploitability target"
                            )
                        _continuation_cache_put(cache, spec, result)
            except sqlite3.Error as error:
                raise LiveGTOCacheError(
                    f"live continuation cache is unavailable: {error}"
                ) from error

            hero_combo = _ordered_combo(state.hero_combo)
            policy = next(
                (
                    item
                    for item in result.combo_policies
                    if item.private_combo == hero_combo
                ),
                None,
            )
            source = _postflop_source(
                cache_hit=cache_hit,
                approximate=ranges.approximate,
            )
            if policy is None:
                return LiveGTOOutcome(
                    LiveGTOStatus.UNSUPPORTED,
                    "Hero combo has zero reach after the complete recorded path",
                    time.perf_counter() - started,
                    source=source,
                    cache_hit=cache_hit,
                    spec=spec,
                    result=result,
                    approximate=ranges.approximate,
                )
            selected, roll = _stable_action(
                policy,
                f"{state.hand_id}:{spec.cache_key}:{','.join(hero_combo)}",
                self.config.mix_secret,
            )
            analysis = _format_analysis(
                policy,
                selected,
                roll,
                self.config,
                ranges,
                result,
                source,
                state,
            )
            return LiveGTOOutcome(
                LiveGTOStatus.SOLVED,
                "",
                time.perf_counter() - started,
                analysis=analysis,
                source=source,
                model="b-inary/postflop-solver",
                cache_hit=cache_hit,
                spec=spec,
                result=result,
                approximate=ranges.approximate,
            )
        except EngineResponseError as error:
            if error.code in {
                "UNREACHABLE_NODE",
                "NODE_PATH_ERROR",
                "NODE_MISMATCH",
            }:
                return LiveGTOOutcome(
                    LiveGTOStatus.UNSUPPORTED,
                    "complete public path is not reachable in the solver tree: "
                    f"{error.message}",
                    time.perf_counter() - started,
                )
            return LiveGTOOutcome(
                LiveGTOStatus.FAILED,
                str(error),
                time.perf_counter() - started,
            )
        except (
            LiveGTOContinuationError,
            LiveGTORangeError,
            PreflopHistoryError,
        ) as error:
            return LiveGTOOutcome(
                LiveGTOStatus.UNSUPPORTED,
                str(error),
                time.perf_counter() - started,
            )
        except (
            BlueprintError,
            LiveGTOCacheError,
            LiveGTOConfigurationError,
            EngineClientError,
            OracleValidationError,
            OSError,
        ) as error:
            return LiveGTOOutcome(
                LiveGTOStatus.FAILED,
                str(error),
                time.perf_counter() - started,
            )

    def evaluate(self, state: LiveDecisionState) -> LiveGTOOutcome:
        started = time.perf_counter()
        if not self.config.enabled:
            return LiveGTOOutcome(
                LiveGTOStatus.DISABLED,
                "GTO live is disabled",
                time.perf_counter() - started,
            )
        if state.street.upper() == "PREFLOP":
            policy_method = getattr(self.range_provider, "preflop_policy_for", None)
            if not callable(policy_method):
                return LiveGTOOutcome(
                    LiveGTOStatus.UNSUPPORTED,
                    "the configured range source has no preflop mixed policy",
                    time.perf_counter() - started,
                )
            try:
                bundle = policy_method(state)
                if not bundle.actions:
                    raise LiveGTORangeError("the preflop blueprint policy is empty")
                if not state.hand_id:
                    raise LiveGTORangeError(
                        "a non-empty hand id is required to realize a stable mixed policy"
                    )
                selected, roll = _stable_blueprint_action(
                    bundle,
                    (
                        f"{state.hand_id}:{bundle.stack}:{bundle.history}:"
                        f"{','.join(_ordered_combo(state.hero_combo))}"
                    ),
                    self.config.mix_secret,
                )
                analysis = _format_blueprint_analysis(
                    bundle, selected, roll, state
                )
                source = (
                    "approximate preflop blueprint cache"
                    if bundle.approximate
                    else "verified preflop blueprint cache"
                )
                return LiveGTOOutcome(
                    LiveGTOStatus.SOLVED,
                    "",
                    time.perf_counter() - started,
                    analysis=analysis,
                    source=source,
                    model="PokerStudy MonkerSolver NL v2 blueprint",
                    approximate=bundle.approximate,
                )
            except PreflopHistoryError as error:
                return LiveGTOOutcome(
                    LiveGTOStatus.UNSUPPORTED,
                    str(error),
                    time.perf_counter() - started,
                )
            except LiveGTORangeError as error:
                return LiveGTOOutcome(
                    LiveGTOStatus.UNSUPPORTED,
                    str(error),
                    time.perf_counter() - started,
                )
            except (BlueprintError, OSError) as error:
                return LiveGTOOutcome(
                    LiveGTOStatus.FAILED,
                    str(error),
                    time.perf_counter() - started,
                )
        if state.public_hand is not None:
            return self._evaluate_continuation(state, started)
        reason = _eligibility_reason(state)
        if reason:
            return LiveGTOOutcome(
                LiveGTOStatus.UNSUPPORTED,
                reason,
                time.perf_counter() - started,
            )
        try:
            ranges = self.range_provider.ranges_for(state)
            spec = build_live_spec(state, ranges, self.config)
            client = self.engine_factory(
                self.config.engine_path,
                offline_only_acknowledged=False,
                owned_simulator_acknowledged=True,
                timeout_seconds=self.config.timeout_for(spec.street),
            )
            digest = client.binary_sha256
            self.config.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with OracleCache(self.config.cache_path) as cache:
                    result = _cache_get(cache, spec, digest=digest)
                    cache_hit = result is not None
                    if result is not None and not result.metadata.converged:
                        raise OracleValidationError(
                            "cached solver result stopped before the requested "
                            "exploitability target"
                        )
                    if result is None:
                        if spec.street is Street.FLOP and self.config.flop_cache_only:
                            return LiveGTOOutcome(
                                LiveGTOStatus.CACHE_MISS,
                                "flop policy is cache-only and this node is not cached",
                                time.perf_counter() - started,
                                spec=spec,
                            )
                        result = client.solve(spec)
                        if not result.metadata.converged:
                            raise OracleValidationError(
                                "solver stopped before the requested exploitability target"
                            )
                        _cache_put(cache, spec, result)
            except sqlite3.Error as error:
                raise LiveGTOCacheError(
                    f"live solver cache is unavailable: {error}"
                ) from error
            hero_combo = _ordered_combo(state.hero_combo)
            policy = next(
                (item for item in result.combo_policies if item.private_combo == hero_combo),
                None,
            )
            if policy is None:
                source = _postflop_source(
                    cache_hit=cache_hit,
                    approximate=ranges.approximate,
                )
                return LiveGTOOutcome(
                    LiveGTOStatus.UNSUPPORTED,
                    "Hero combo is unreachable at the reconstructed solver node",
                    time.perf_counter() - started,
                    source=source,
                    cache_hit=cache_hit,
                    spec=spec,
                    result=result,
                )
            selected, roll = _stable_action(
                policy,
                f"{state.hand_id}:{spec.cache_key}:{','.join(hero_combo)}",
                self.config.mix_secret,
            )
            source = _postflop_source(
                cache_hit=cache_hit,
                approximate=ranges.approximate,
            )
            analysis = _format_analysis(
                policy, selected, roll, self.config, ranges, result, source, state
            )
            return LiveGTOOutcome(
                LiveGTOStatus.SOLVED,
                "",
                time.perf_counter() - started,
                analysis=analysis,
                source=source,
                model="b-inary/postflop-solver",
                cache_hit=cache_hit,
                spec=spec,
                result=result,
                approximate=ranges.approximate,
            )
        except EngineResponseError as error:
            if error.code == "UNREACHABLE_NODE":
                return LiveGTOOutcome(
                    LiveGTOStatus.UNSUPPORTED,
                    f"reconstructed solver node is unreachable: {error.message}",
                    time.perf_counter() - started,
                )
            return LiveGTOOutcome(
                LiveGTOStatus.FAILED,
                str(error),
                time.perf_counter() - started,
            )
        except (LiveGTORangeError, PreflopHistoryError) as error:
            return LiveGTOOutcome(
                LiveGTOStatus.UNSUPPORTED,
                str(error),
                time.perf_counter() - started,
            )
        except (
            BlueprintError,
            LiveGTOCacheError,
            LiveGTOConfigurationError,
            EngineClientError,
            OracleValidationError,
            OSError,
        ) as error:
            return LiveGTOOutcome(
                LiveGTOStatus.FAILED,
                str(error),
                time.perf_counter() - started,
            )
