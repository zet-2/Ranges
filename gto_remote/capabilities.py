"""Auditable solver capability declarations.

Hardware and transport must never turn a heads-up subgame solver into a
six-player full-game solver by implication.  Every backend therefore exposes a
small manifest, and the server reports the concrete gaps that prevent the
``full_six_max_ready`` claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+/-]{0,127}$")
_PREFLOP_MODES = frozenset({"NONE", "FIXED_BLUEPRINT", "SOLVED_TREE"})
_POSTFLOP_MODES = frozenset({"NONE", "HU_SUBGAME", "MULTIWAY_TREE"})
_RANGE_MODES = frozenset(
    {"NONE", "PREFLOP_REACH_ONLY", "ACTION_CONDITIONED_ALL_STREETS"}
)
_CARD_MODELS = frozenset({"ABSTRACT_BUCKETS", "CARD_EXACT"})
_ACTION_MODELS = frozenset({"FIXED_DISCRETE_TREE", "DYNAMIC_DISCRETE_TREE"})
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
        "convergence_metric",
        "source_license",
        "full_six_max_ready",
        "full_six_max_gaps",
    }
)


class SolverCapabilitiesError(ValueError):
    """A backend capability manifest is internally inconsistent."""


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
    convergence_metric: str
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
            ("convergence_metric", self.convergence_metric),
            ("source_license", self.source_license),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SolverCapabilitiesError(f"{name} cannot be empty")

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
        if not self.convergence_metric.strip():
            gaps.append("no convergence metric is declared")
        return tuple(gaps)

    @property
    def full_six_max_ready(self) -> bool:
        return not self.full_six_max_gaps()

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
            "convergence_metric": self.convergence_metric,
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
            "convergence_metric",
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
            convergence_metric=value["convergence_metric"],
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
    convergence_metric="HU exploitability percent of pot",
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
]
