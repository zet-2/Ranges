"""Domain model and canonical serialization for the HU jam/fold harness."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import hashlib
import json
import math
import re
from typing import Mapping, Sequence


GAME_SCHEMA_VERSION = 1
SOLUTION_SCHEMA_VERSION = 1
GAME_ID = "hu_nlhe_jam_fold"
ARTIFACT_TYPE = "hu_jam_fold_solution"
TYPE_SEMANTICS = "explicit_finite_private_types"
RAKE_ROUNDING = "floor_to_chip"
RAKE_BASIS = "awarded_pot_after_uncalled_return"
SOLUTION_CONCEPT = "measured_epsilon_nash"
CERTIFICATE_METHOD = "bilateral_full_best_response_enumeration"
CERTIFICATE_NUMERIC = "ieee754_float64"
ALGORITHM_ID = "simultaneous_regret_matching_plus_linear_average"

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
PROBABILITY_TOLERANCE = 1e-12
MANIFEST_NUMBER_TOLERANCE = 1e-10
TARGET_COMPARISON_TOLERANCE = 1e-12


class JamFoldValidationError(ValueError):
    """Raised when an input or solution violates the fail-closed contract."""


def decimal_value(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise JamFoldValidationError(
            f"{field_name} must use a decimal string or integer, not binary float"
        )
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise JamFoldValidationError(
            f"{field_name} is not a decimal number"
        ) from error
    if not parsed.is_finite():
        raise JamFoldValidationError(f"{field_name} must be finite")
    return Decimal(0) if parsed == 0 else parsed.normalize()


def decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def float_text(value: float) -> str:
    if not math.isfinite(value):
        raise JamFoldValidationError("manifest numbers must be finite")
    if abs(value) < 5e-16:
        value = 0.0
    return format(value, ".17g")


def finite_float(value: Decimal, field_name: str) -> float:
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise JamFoldValidationError(
            f"{field_name} is outside the declared float64 certificate range"
        ) from error
    if not math.isfinite(result) or (value != 0 and result == 0.0):
        raise JamFoldValidationError(
            f"{field_name} overflows or underflows the float64 certificate"
        )
    return result


def identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
        raise JamFoldValidationError(
            f"{field_name} must match {IDENTIFIER_RE.pattern!r}"
        )
    return value


def mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise JamFoldValidationError(f"{field_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise JamFoldValidationError(f"{field_name} keys must be strings")
    return value


def exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    field_name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise JamFoldValidationError(
            f"{field_name} keys mismatch; missing={missing}, extra={extra}"
        )


def positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JamFoldValidationError(f"{field_name} must be a positive integer")
    return value


def fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_chip_multiple(value: Decimal, chip_unit: Decimal) -> bool:
    try:
        return value % chip_unit == 0
    except InvalidOperation as error:
        raise JamFoldValidationError(
            "chip alignment exceeds the supported decimal arithmetic range"
        ) from error


@dataclass(frozen=True, slots=True)
class RakeProfile:
    """Exact rake semantics for one restricted-game profile."""

    rate_pct: Decimal
    cap_bb: Decimal
    chip_unit_bb: Decimal
    no_flop_no_drop: bool

    def __post_init__(self) -> None:
        rate = decimal_value(self.rate_pct, "rake rate_pct")
        cap = decimal_value(self.cap_bb, "rake cap_bb")
        unit = decimal_value(self.chip_unit_bb, "rake chip_unit_bb")
        self._validate(rate, cap, unit)
        object.__setattr__(self, "rate_pct", rate)
        object.__setattr__(self, "cap_bb", cap)
        object.__setattr__(self, "chip_unit_bb", unit)

    def _validate(self, rate: Decimal, cap: Decimal, unit: Decimal) -> None:
        if not Decimal(0) <= rate <= Decimal(100):
            raise JamFoldValidationError("rake rate_pct must be between 0 and 100")
        if cap < 0:
            raise JamFoldValidationError("rake cap_bb cannot be negative")
        if unit <= 0:
            raise JamFoldValidationError("rake chip_unit_bb must be positive")
        if rate > 0 and cap <= 0:
            raise JamFoldValidationError(
                "positive rake requires a positive explicit cap_bb"
            )
        if rate == 0 and cap != 0:
            raise JamFoldValidationError("zero rake requires cap_bb=0")
        if cap and not _is_chip_multiple(cap, unit):
            raise JamFoldValidationError("rake cap_bb must align to chip_unit_bb")
        if not isinstance(self.no_flop_no_drop, bool):
            raise JamFoldValidationError("no_flop_no_drop must be boolean")
        finite_float(rate, "rake rate_pct")
        finite_float(cap, "rake cap_bb")
        finite_float(unit, "rake chip_unit_bb")

    def amount(self, pot_bb: Decimal, *, flop_seen: bool) -> Decimal:
        pot = decimal_value(pot_bb, "rake pot")
        if pot < 0:
            raise JamFoldValidationError("rake pot cannot be negative")
        if self.rate_pct == 0 or (self.no_flop_no_drop and not flop_seen):
            return Decimal(0)
        raw = min(pot * self.rate_pct / Decimal(100), self.cap_bb)
        increments = (raw / self.chip_unit_bb).to_integral_value(
            rounding=ROUND_DOWN
        )
        return increments * self.chip_unit_bb

    def to_payload(self) -> dict[str, object]:
        return {
            "rate_pct": decimal_text(self.rate_pct),
            "cap_bb": decimal_text(self.cap_bb),
            "chip_unit_bb": decimal_text(self.chip_unit_bb),
            "no_flop_no_drop": self.no_flop_no_drop,
            "rounding": RAKE_ROUNDING,
            "basis": RAKE_BASIS,
        }


@dataclass(frozen=True, slots=True)
class JamFoldProfile:
    """One fixed stack/blind/rake game profile."""

    profile_id: str
    effective_stack_bb: Decimal
    small_blind_bb: Decimal
    big_blind_bb: Decimal
    rake: RakeProfile

    def __post_init__(self) -> None:
        profile_id = identifier(self.profile_id, "profile_id")
        stack = decimal_value(self.effective_stack_bb, "effective_stack_bb")
        small = decimal_value(self.small_blind_bb, "small_blind_bb")
        big = decimal_value(self.big_blind_bb, "big_blind_bb")
        self._validate(stack, small, big)
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "effective_stack_bb", stack)
        object.__setattr__(self, "small_blind_bb", small)
        object.__setattr__(self, "big_blind_bb", big)

    def _validate(self, stack: Decimal, small: Decimal, big: Decimal) -> None:
        if not isinstance(self.rake, RakeProfile):
            raise JamFoldValidationError("profile rake must be a RakeProfile")
        if not Decimal(0) < small < big < stack:
            raise JamFoldValidationError(
                "profile requires 0 < small blind < big blind < effective stack"
            )
        for value, name in (
            (stack, "effective_stack_bb"),
            (small, "small_blind_bb"),
            (big, "big_blind_bb"),
        ):
            finite_float(value, name)
            if not _is_chip_multiple(value, self.rake.chip_unit_bb):
                raise JamFoldValidationError(
                    f"{name} must align to rake chip_unit_bb"
                )

    def sb_fold_payoffs(self) -> tuple[Decimal, Decimal]:
        pot = self.small_blind_bb + self.big_blind_bb
        rake = self.rake.amount(pot, flop_seen=False)
        return (-self.small_blind_bb, self.small_blind_bb - rake)

    def bb_fold_payoffs(self) -> tuple[Decimal, Decimal]:
        pot = Decimal(2) * self.big_blind_bb
        rake = self.rake.amount(pot, flop_seen=False)
        return (self.big_blind_bb - rake, -self.big_blind_bb)

    def called_jam_payoffs(self, sb_equity: Decimal) -> tuple[Decimal, Decimal]:
        equity = decimal_value(sb_equity, "SB showdown equity")
        if not Decimal(0) <= equity <= Decimal(1):
            raise JamFoldValidationError("SB showdown equity must be in [0, 1]")
        gross_pot = Decimal(2) * self.effective_stack_bb
        rake = self.rake.amount(gross_pot, flop_seen=True)
        awarded = gross_pot - rake
        sb = equity * awarded - self.effective_stack_bb
        bb = (Decimal(1) - equity) * awarded - self.effective_stack_bb
        return (sb, bb)

    def to_payload(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "effective_stack_bb": decimal_text(self.effective_stack_bb),
            "small_blind_bb": decimal_text(self.small_blind_bb),
            "big_blind_bb": decimal_text(self.big_blind_bb),
            "rake": self.rake.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class PrivateTypeDeal:
    """One reachable pair in the finite private-information model."""

    sb_type: str
    bb_type: str
    weight: Decimal
    sb_showdown_equity: Decimal

    def __post_init__(self) -> None:
        sb_type = identifier(self.sb_type, "deal sb_type")
        bb_type = identifier(self.bb_type, "deal bb_type")
        weight = decimal_value(self.weight, "deal weight")
        equity = decimal_value(self.sb_showdown_equity, "deal sb_showdown_equity")
        if weight <= 0:
            raise JamFoldValidationError("deal weight must be positive")
        if not Decimal(0) <= equity <= Decimal(1):
            raise JamFoldValidationError("deal sb_showdown_equity must be in [0, 1]")
        finite_float(weight, "deal weight")
        finite_float(equity, "deal sb_showdown_equity")
        object.__setattr__(self, "sb_type", sb_type)
        object.__setattr__(self, "bb_type", bb_type)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "sb_showdown_equity", equity)

    def to_payload(self) -> dict[str, str]:
        return {
            "sb_type": self.sb_type,
            "bb_type": self.bb_type,
            "weight": decimal_text(self.weight),
            "sb_showdown_equity": decimal_text(self.sb_showdown_equity),
        }


@dataclass(frozen=True, slots=True)
class JamFoldGame:
    """Validated finite Bayesian jam/fold game."""

    profile: JamFoldProfile
    model_id: str
    model_version: str
    deals: tuple[PrivateTypeDeal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.profile, JamFoldProfile):
            raise JamFoldValidationError("game profile must be a JamFoldProfile")
        model_id = identifier(self.model_id, "private model_id")
        model_version = identifier(self.model_version, "private model_version")
        deals = tuple(self.deals)
        if not deals or any(not isinstance(deal, PrivateTypeDeal) for deal in deals):
            raise JamFoldValidationError(
                "private model requires at least one PrivateTypeDeal"
            )
        ordered = tuple(sorted(deals, key=lambda deal: (deal.sb_type, deal.bb_type)))
        pairs = [(deal.sb_type, deal.bb_type) for deal in ordered]
        if len(pairs) != len(set(pairs)):
            raise JamFoldValidationError("private model cannot repeat a type pair")
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "deals", ordered)

    @property
    def sb_types(self) -> tuple[str, ...]:
        return tuple(sorted({deal.sb_type for deal in self.deals}))

    @property
    def bb_types(self) -> tuple[str, ...]:
        return tuple(sorted({deal.bb_type for deal in self.deals}))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": GAME_SCHEMA_VERSION,
            "game": GAME_ID,
            "profile": self.profile.to_payload(),
            "private_model": {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "type_semantics": TYPE_SEMANTICS,
                "deals": [deal.to_payload() for deal in self.deals],
            },
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_payload())


@dataclass(frozen=True, slots=True)
class BehaviorStrategy:
    """Jam probabilities by SB type and call probabilities by BB type."""

    sb_jam: tuple[tuple[str, float], ...]
    bb_call: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sb_jam", self._validated(self.sb_jam, "SB jam"))
        object.__setattr__(self, "bb_call", self._validated(self.bb_call, "BB call"))

    @staticmethod
    def _validated(
        values: Sequence[tuple[str, float]],
        field_name: str,
    ) -> tuple[tuple[str, float], ...]:
        result: list[tuple[str, float]] = []
        for private_type, probability in values:
            name = identifier(private_type, f"{field_name} type")
            if isinstance(probability, bool) or not isinstance(
                probability, (int, float)
            ):
                raise JamFoldValidationError(
                    f"{field_name} probability must be numeric"
                )
            number = float(probability)
            if not math.isfinite(number) or not (
                -PROBABILITY_TOLERANCE
                <= number
                <= 1 + PROBABILITY_TOLERANCE
            ):
                raise JamFoldValidationError(
                    f"{field_name} probability must be finite and in [0, 1]"
                )
            result.append((name, min(1.0, max(0.0, number))))
        ordered = tuple(sorted(result))
        names = [name for name, _ in ordered]
        if not ordered or len(names) != len(set(names)):
            raise JamFoldValidationError(
                f"{field_name} types must be non-empty and unique"
            )
        return ordered

    def validate_for(self, game: JamFoldGame) -> None:
        if tuple(name for name, _ in self.sb_jam) != game.sb_types:
            raise JamFoldValidationError("strategy SB types do not match the game")
        if tuple(name for name, _ in self.bb_call) != game.bb_types:
            raise JamFoldValidationError("strategy BB types do not match the game")

    def to_payload(self) -> dict[str, object]:
        return {
            "sb": [
                {
                    "type": private_type,
                    "fold": float_text(1.0 - jam),
                    "jam": float_text(jam),
                }
                for private_type, jam in self.sb_jam
            ],
            "bb_vs_jam": [
                {
                    "type": private_type,
                    "fold": float_text(1.0 - call),
                    "call": float_text(call),
                }
                for private_type, call in self.bb_call
            ],
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_payload())


@dataclass(frozen=True, slots=True)
class BestResponseReport:
    """Exact unilateral-deviation measurement for a supplied strategy."""

    sb_utility_bb: float
    bb_utility_bb: float
    expected_rake_bb: float
    sb_best_response_utility_bb: float
    bb_best_response_utility_bb: float
    sb_deviation_gain_bb: float
    bb_deviation_gain_bb: float
    epsilon_bb: float


@dataclass(frozen=True, slots=True)
class JamFoldSolution:
    """One measured solution candidate and its audit metadata."""

    game: JamFoldGame
    strategy: BehaviorStrategy
    report: BestResponseReport
    target_epsilon_bb: float
    iterations: int
    min_iterations: int
    check_every: int
    converged: bool

    def _claim_payload(self) -> dict[str, object]:
        return {
            "solution_concept": SOLUTION_CONCEPT,
            "certificate_method": CERTIFICATE_METHOD,
            "certificate_numeric": CERTIFICATE_NUMERIC,
            "game_scope": "restricted_hu_jam_fold",
            "full_hunl": False,
            "rake_aware": True,
            "target_epsilon_bb": float_text(self.target_epsilon_bb),
            "achieved_epsilon_bb": float_text(self.report.epsilon_bb),
            "sb_deviation_gain_bb": float_text(
                self.report.sb_deviation_gain_bb
            ),
            "bb_deviation_gain_bb": float_text(
                self.report.bb_deviation_gain_bb
            ),
            "target_reached": self.converged,
        }

    def _utilities_payload(self) -> dict[str, str]:
        return {
            "sb_utility_bb": float_text(self.report.sb_utility_bb),
            "bb_utility_bb": float_text(self.report.bb_utility_bb),
            "expected_rake_bb": float_text(self.report.expected_rake_bb),
            "sb_best_response_utility_bb": float_text(
                self.report.sb_best_response_utility_bb
            ),
            "bb_best_response_utility_bb": float_text(
                self.report.bb_best_response_utility_bb
            ),
        }

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": SOLUTION_SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "game_fingerprint": self.game.fingerprint,
            "profile": self.game.profile.to_payload(),
            "private_model": {
                "model_id": self.game.model_id,
                "model_version": self.game.model_version,
            },
            "claim": self._claim_payload(),
            "solve": {
                "algorithm": ALGORITHM_ID,
                "iterations": self.iterations,
                "min_iterations": self.min_iterations,
                "check_every": self.check_every,
            },
            "strategy": self.strategy.to_payload(),
            "strategy_fingerprint": self.strategy.fingerprint,
            "utilities": self._utilities_payload(),
        }

