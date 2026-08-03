"""Auditable solver capability declarations.

Hardware and transport must never turn a heads-up subgame solver into a
six-player full-game solver by implication.  Every backend therefore exposes a
small manifest, and the server reports the concrete gaps that prevent the
``full_six_max_ready`` claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Any


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+/-]{0,127}$")
_PREFLOP_MODES = frozenset({"NONE", "FIXED_BLUEPRINT", "SOLVED_TREE"})
_POSTFLOP_MODES = frozenset({"NONE", "HU_SUBGAME", "MULTIWAY_TREE"})
_RANGE_MODES = frozenset(
    {"NONE", "PREFLOP_REACH_ONLY", "ACTION_CONDITIONED_ALL_STREETS"}
)
_CARD_MODELS = frozenset({"ABSTRACT_BUCKETS", "CARD_EXACT"})
_ACTION_MODELS = frozenset(
    {
        "FIXED_DISCRETE_TREE",
        "DYNAMIC_DISCRETE_TREE",
        "CONTINUOUS_NO_LIMIT",
    }
)
_UNDECLARED_TEXT = frozenset(
    {
        "none",
        "unknown",
        "n/a",
        "na",
        "not declared",
        "not configured",
    }
)
_MAX_CONVERGENCE_TARGET = Decimal("10000000")
_WIRE_KEYS = frozenset(
    {
        "backend_id",
        "backend_version",
        "preflop_mode",
        "postflop_mode",
        "max_postflop_players",
        "stateful_through_river",
        "range_conditioning",
        "folded_card_bunching",
        "card_model",
        "action_model",
        "game_profile_id",
        "abstraction_id",
        "solution_concept",
        "convergence_metric",
        "convergence_target",
        "source_license",
        "full_six_max_ready",
        "full_six_max_gaps",
    }
)


class SolverCapabilitiesError(ValueError):
    """A backend capability manifest is internally inconsistent."""


def parse_capabilities_json(payload: bytes | str) -> "SolverCapabilities":
    """Parse a capability manifest as strict JSON with unique object keys."""

    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SolverCapabilitiesError(
                "capability manifest is not valid UTF-8"
            ) from error
    elif isinstance(payload, str):
        text = payload
    else:
        raise SolverCapabilitiesError(
            "capability manifest must be UTF-8 bytes or text"
        )

    def reject_constant(value: str) -> None:
        raise SolverCapabilitiesError(
            f"non-finite JSON constant {value!r} is forbidden"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SolverCapabilitiesError(
                    f"duplicate JSON key {key!r} in capability manifest"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except SolverCapabilitiesError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise SolverCapabilitiesError(
            f"capability manifest is not strict JSON: {error}"
        ) from error
    return SolverCapabilities.from_wire(value)


def _convergence_target(value: object) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise SolverCapabilitiesError(
            "convergence_target must be an exact decimal string or Decimal"
        )
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise SolverCapabilitiesError(
            "convergence_target must be a decimal"
        ) from error
    if (
        not result.is_finite()
        or result < 0
        or result > _MAX_CONVERGENCE_TARGET
        or (result != 0 and result.as_tuple().exponent < -12)
        or len(result.as_tuple().digits) > 28
    ):
        raise SolverCapabilitiesError(
            "convergence_target exceeds decimal safety limits"
        )
    return Decimal(0) if result == 0 else result.normalize()


def _decimal_text(value: Decimal) -> str:
    normalized = _convergence_target(value)
    return "0" if normalized == 0 else format(normalized, "f")


@dataclass(frozen=True, slots=True)
class SolverCapabilities:
    backend_id: str
    backend_version: str
    preflop_mode: str
    postflop_mode: str
    max_postflop_players: int
    stateful_through_river: bool
    range_conditioning: str
    folded_card_bunching: bool
    card_model: str
    action_model: str
    game_profile_id: str
    abstraction_id: str
    solution_concept: str
    convergence_metric: str
    convergence_target: Decimal
    source_license: str

    def __post_init__(self) -> None:
        for name, value in (
            ("backend_id", self.backend_id),
            ("backend_version", self.backend_version),
        ):
            if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
                raise SolverCapabilitiesError(
                    f"{name} must be a safe non-empty identifier"
                )
        if self.preflop_mode not in _PREFLOP_MODES:
            raise SolverCapabilitiesError("preflop_mode is unsupported")
        if self.postflop_mode not in _POSTFLOP_MODES:
            raise SolverCapabilitiesError("postflop_mode is unsupported")
        if (
            isinstance(self.max_postflop_players, bool)
            or not isinstance(self.max_postflop_players, int)
            or not 0 <= self.max_postflop_players <= 9
        ):
            raise SolverCapabilitiesError(
                "max_postflop_players must be an integer 0..9"
            )
        for name, value in (
            ("stateful_through_river", self.stateful_through_river),
            ("folded_card_bunching", self.folded_card_bunching),
        ):
            if not isinstance(value, bool):
                raise SolverCapabilitiesError(f"{name} must be boolean")
        if self.range_conditioning not in _RANGE_MODES:
            raise SolverCapabilitiesError("range_conditioning is unsupported")
        if self.card_model not in _CARD_MODELS:
            raise SolverCapabilitiesError("card_model is unsupported")
        if self.action_model not in _ACTION_MODELS:
            raise SolverCapabilitiesError("action_model is unsupported")
        for name, value in (
            ("game_profile_id", self.game_profile_id),
            ("abstraction_id", self.abstraction_id),
        ):
            if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
                raise SolverCapabilitiesError(
                    f"{name} must be a safe non-empty identifier"
                )
        for name, value in (
            ("solution_concept", self.solution_concept),
            ("convergence_metric", self.convergence_metric),
            ("source_license", self.source_license),
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in value
                )
            ):
                raise SolverCapabilitiesError(f"{name} cannot be empty")
        _convergence_target(self.convergence_target)

    def full_six_max_gaps(self) -> tuple[str, ...]:
        """Return every reason this backend cannot claim full six-max coverage."""

        gaps: list[str] = []
        if self.preflop_mode != "SOLVED_TREE":
            gaps.append("preflop is not a compatible solved full-game tree")
        if self.postflop_mode != "MULTIWAY_TREE":
            gaps.append("postflop backend is not a multiway game tree")
        if self.max_postflop_players < 6:
            gaps.append("postflop backend supports fewer than six players")
        if not self.stateful_through_river:
            gaps.append("one action-conditioned tree is not retained through river")
        if self.range_conditioning != "ACTION_CONDITIONED_ALL_STREETS":
            gaps.append("ranges are not conditioned on every public action")
        if not self.folded_card_bunching:
            gaps.append("folded-card bunching is not modeled")
        if self.card_model != "CARD_EXACT":
            gaps.append("private-card model is bucketed")
        if self.action_model != "CONTINUOUS_NO_LIMIT":
            gaps.append(
                "action model is a finite discrete abstraction, not "
                "continuous no-limit betting"
            )
        if self.solution_concept.strip().lower() in _UNDECLARED_TEXT:
            gaps.append("no auditable multiplayer solution concept is declared")
        if self.convergence_metric.strip().lower() in _UNDECLARED_TEXT:
            gaps.append("no auditable convergence metric is declared")
        return tuple(gaps)

    @staticmethod
    def _state_player_profile(
        state: object,
    ) -> tuple[int, int, object | None]:
        """Return current and peak-postflop live-player counts.

        A decision that is heads-up now is not necessarily a heads-up
        postflop subgame.  If three or more players saw the flop, every later
        range depends on that earlier multiway path even after folds reduce
        the current node to two players.  Capability checks therefore use the
        maximum live-player count reached after ``DEAL_FLOP`` as well as the
        final replayed count.
        """

        public_hand = getattr(state, "public_hand", None)
        if public_hand is not None:
            # Keep this import local so the capability schema remains usable by
            # small deployment/manifest tools without importing the live
            # router and its solver dependencies.
            from gto_hand_history import replay_public_hand

            replayed = replay_public_hand(public_hand)
            live_seats = {seat.seat for seat in public_hand.seats}
            postflop_started = False
            peak_postflop_players = 0
            for event in public_hand.events:
                if event.kind == "FOLD":
                    live_seats.discard(event.actor_seat)
                elif event.kind == "DEAL_FLOP":
                    postflop_started = True
                    peak_postflop_players = len(live_seats)
                if postflop_started:
                    peak_postflop_players = max(
                        peak_postflop_players,
                        len(live_seats),
                    )
            current_players = len(replayed.live_seats)
            return (
                current_players,
                max(peak_postflop_players, current_players),
                replayed,
            )
        active_villains = getattr(state, "active_villains", None)
        if (
            isinstance(active_villains, bool)
            or not isinstance(active_villains, int)
            or not 0 <= active_villains <= 8
        ):
            raise SolverCapabilitiesError(
                "state.active_villains must be an integer 0..8"
            )
        players = active_villains + 1
        return players, players, None

    def support_gaps_for_state(self, state: object) -> tuple[str, ...]:
        """Return hard capability gaps for one concrete decision.

        These checks answer whether the backend can represent the submitted
        game at all.  They deliberately do not claim that a supported,
        abstract game is exact relative to continuous six-max NLHE; that
        stricter question is handled by :meth:`exactness_gaps_for_state`.
        """

        public_hand = getattr(state, "public_hand", None)
        replayed = None
        if public_hand is not None:
            from gto_hand_history import replay_public_hand

            replayed = replay_public_hand(public_hand)
        street = str(
            replayed.street if replayed is not None else getattr(state, "street", "")
        ).strip().upper()
        if street not in {"PREFLOP", "FLOP", "TURN", "RIVER"}:
            return ("decision street is unsupported",)
        players, peak_postflop_players, count_replay = (
            self._state_player_profile(state)
        )
        if replayed is None:
            replayed = count_replay
        gaps: list[str] = []
        if players < 2:
            gaps.append("fewer than two live players remain")
        if street == "PREFLOP":
            if self.preflop_mode == "NONE":
                gaps.append("backend does not support preflop")
            elif self.preflop_mode == "SOLVED_TREE" and public_hand is None:
                gaps.append(
                    "solved preflop routing requires a complete public_hand "
                    "transcript"
                )
            return tuple(gaps)

        if self.preflop_mode == "NONE":
            gaps.append(
                "postflop ranges cannot be derived without a preflop model"
            )
        if self.postflop_mode == "NONE":
            gaps.append("backend does not support postflop")
        if peak_postflop_players > self.max_postflop_players:
            gaps.append(
                f"backend supports at most {self.max_postflop_players} "
                "postflop players; the hand path reached "
                f"{peak_postflop_players}"
            )
        if (
            peak_postflop_players > 2
            and self.postflop_mode != "MULTIWAY_TREE"
        ):
            gaps.append(
                "backend is not a multiway postflop solver for the "
                f"{peak_postflop_players}-player path"
            )
        if players > 2 and replayed is None:
            gaps.append(
                "multiway postflop solving requires a complete public_hand "
                "transcript"
            )
        if self.stateful_through_river and replayed is None:
            gaps.append(
                "stateful postflop solving requires a complete public_hand "
                "transcript"
            )
        return tuple(gaps)

    def exactness_gaps_for_state(self, state: object) -> tuple[str, ...]:
        """Return why an outcome must be disclosed as approximate.

        The comparison target is the complete public six-max hand rather than
        merely the backend's own finite abstraction.  A backend may still
        solve that abstraction usefully, but a response cannot be labelled
        exact while any returned gap remains.
        """

        gaps = list(self.support_gaps_for_state(state))
        public_hand = getattr(state, "public_hand", None)
        replayed = None
        if public_hand is not None:
            from gto_hand_history import replay_public_hand

            replayed = replay_public_hand(public_hand)
        street = str(
            replayed.street if replayed is not None else getattr(state, "street", "")
        ).strip().upper()
        if street == "PREFLOP":
            if self.preflop_mode != "SOLVED_TREE":
                gaps.append("preflop policy is not a compatible solved tree")
        else:
            if self.range_conditioning != "ACTION_CONDITIONED_ALL_STREETS":
                gaps.append("ranges are not conditioned on every public action")
            if not self.stateful_through_river:
                gaps.append(
                    "one action-conditioned tree is not retained through river"
                )
            if replayed is None:
                gaps.append(
                    "exact postflop range conditioning requires public_hand"
                )
        if (
            replayed is not None
            and replayed.folded
            and not self.folded_card_bunching
        ):
            gaps.append("folded-card bunching is not modeled")
        if self.card_model != "CARD_EXACT":
            gaps.append("private-card model is bucketed")
        if self.action_model != "CONTINUOUS_NO_LIMIT":
            gaps.append(
                "action model is a finite discrete abstraction, not "
                "continuous no-limit betting"
            )
        if self.solution_concept.strip().lower() in _UNDECLARED_TEXT:
            gaps.append("no auditable multiplayer solution concept is declared")
        if self.convergence_metric.strip().lower() in _UNDECLARED_TEXT:
            gaps.append("no auditable convergence metric is declared")
        return tuple(dict.fromkeys(gaps))

    @property
    def full_six_max_ready(self) -> bool:
        return not self.full_six_max_gaps()

    @property
    def manifest_fingerprint(self) -> str:
        """Return the canonical SHA-256 identity of this exact declaration."""

        payload = json.dumps(
            self.to_wire(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_wire(self) -> dict[str, Any]:
        gaps = self.full_six_max_gaps()
        return {
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "preflop_mode": self.preflop_mode,
            "postflop_mode": self.postflop_mode,
            "max_postflop_players": self.max_postflop_players,
            "stateful_through_river": self.stateful_through_river,
            "range_conditioning": self.range_conditioning,
            "folded_card_bunching": self.folded_card_bunching,
            "card_model": self.card_model,
            "action_model": self.action_model,
            "game_profile_id": self.game_profile_id,
            "abstraction_id": self.abstraction_id,
            "solution_concept": self.solution_concept,
            "convergence_metric": self.convergence_metric,
            "convergence_target": _decimal_text(
                self.convergence_target
            ),
            "source_license": self.source_license,
            "full_six_max_ready": not gaps,
            "full_six_max_gaps": list(gaps),
        }

    @classmethod
    def from_wire(cls, value: Any) -> "SolverCapabilities":
        if not isinstance(value, dict):
            raise SolverCapabilitiesError("capabilities must be an object")
        actual = frozenset(value)
        if actual != _WIRE_KEYS:
            missing = sorted(_WIRE_KEYS - actual)
            unexpected = sorted(actual - _WIRE_KEYS)
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise SolverCapabilitiesError(
                "capabilities schema mismatch: " + "; ".join(details)
            )
        string_fields = (
            "backend_id",
            "backend_version",
            "preflop_mode",
            "postflop_mode",
            "range_conditioning",
            "card_model",
            "action_model",
            "game_profile_id",
            "abstraction_id",
            "solution_concept",
            "convergence_metric",
            "convergence_target",
            "source_license",
        )
        for field in string_fields:
            if not isinstance(value[field], str):
                raise SolverCapabilitiesError(f"{field} must be text")
        boolean_fields = (
            "stateful_through_river",
            "folded_card_bunching",
            "full_six_max_ready",
        )
        for field in boolean_fields:
            if not isinstance(value[field], bool):
                raise SolverCapabilitiesError(f"{field} must be boolean")
        max_players = value["max_postflop_players"]
        if isinstance(max_players, bool) or not isinstance(max_players, int):
            raise SolverCapabilitiesError(
                "max_postflop_players must be an integer"
            )
        raw_gaps = value["full_six_max_gaps"]
        if (
            not isinstance(raw_gaps, list)
            or any(not isinstance(item, str) for item in raw_gaps)
        ):
            raise SolverCapabilitiesError(
                "full_six_max_gaps must be an array of strings"
            )
        capabilities = cls(
            backend_id=value["backend_id"],
            backend_version=value["backend_version"],
            preflop_mode=value["preflop_mode"],
            postflop_mode=value["postflop_mode"],
            max_postflop_players=max_players,
            stateful_through_river=value["stateful_through_river"],
            range_conditioning=value["range_conditioning"],
            folded_card_bunching=value["folded_card_bunching"],
            card_model=value["card_model"],
            action_model=value["action_model"],
            game_profile_id=value["game_profile_id"],
            abstraction_id=value["abstraction_id"],
            solution_concept=value["solution_concept"],
            convergence_metric=value["convergence_metric"],
            convergence_target=_convergence_target(
                value["convergence_target"]
            ),
            source_license=value["source_license"],
        )
        expected = capabilities.to_wire()
        if value["full_six_max_ready"] != expected["full_six_max_ready"]:
            raise SolverCapabilitiesError(
                "full_six_max_ready contradicts the declared capabilities"
            )
        if raw_gaps != expected["full_six_max_gaps"]:
            raise SolverCapabilitiesError(
                "full_six_max_gaps contradict the declared capabilities"
            )
        return capabilities


NATIVE_ROUTER_CAPABILITIES = SolverCapabilities(
    backend_id="ranges-native-hu",
    backend_version="3",
    preflop_mode="FIXED_BLUEPRINT",
    postflop_mode="HU_SUBGAME",
    max_postflop_players=2,
    # Complete transcripts are losslessly reconstructed from the true flop
    # root and traversed through every action and public card.
    stateful_through_river=True,
    range_conditioning="ACTION_CONDITIONED_ALL_STREETS",
    folded_card_bunching=False,
    card_model="CARD_EXACT",
    action_model="DYNAMIC_DISCRETE_TREE",
    game_profile_id="ranges-native-hu-v3",
    abstraction_id="ranges-native-hu-discrete-v3",
    solution_concept="two-player zero-sum CFR approximation",
    convergence_metric="HU exploitability percent of pot",
    convergence_target=Decimal("0.5"),
    source_license="AGPL-3.0-or-later plus source blueprint terms",
)


def capabilities_for_router(router: object) -> SolverCapabilities:
    """Return an explicit manifest, accepting future backend-owned manifests."""

    candidate = getattr(router, "capabilities", None)
    if callable(candidate):
        candidate = candidate()
    if candidate is None:
        return NATIVE_ROUTER_CAPABILITIES
    if not isinstance(candidate, SolverCapabilities):
        raise SolverCapabilitiesError(
            "router.capabilities must be SolverCapabilities"
        )
    return candidate


__all__ = [
    "NATIVE_ROUTER_CAPABILITIES",
    "SolverCapabilities",
    "SolverCapabilitiesError",
    "capabilities_for_router",
    "parse_capabilities_json",
]
