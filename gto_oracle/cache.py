"""Transactional SQLite cache for reproducible local solver results."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import re
import sqlite3
from types import TracebackType

from .continuation import (
    ConditionalCombo,
    ConditionalRange,
    ContinuationResult,
    ContinuationSpec,
    validate_continuation_result,
)
from .models import (
    Action,
    ActionKind,
    ActionValue,
    ComboPolicy,
    OracleValidationError,
    Position,
    SolveResult,
    SolveSpec,
    SolverMetadata,
    validate_result_for_spec,
)
from .serialization import canonical_json


_SCHEMA = """
CREATE TABLE IF NOT EXISTS oracle_results (
    spec_key TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    solver_name TEXT NOT NULL,
    solver_version TEXT NOT NULL,
    iterations INTEGER NOT NULL,
    elapsed_seconds TEXT NOT NULL,
    exploitability TEXT NOT NULL,
    converged INTEGER NOT NULL CHECK (converged IN (0, 1)),
    metadata_extra_json TEXT NOT NULL,
    cached_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS oracle_combo_policies (
    spec_key TEXT NOT NULL,
    combo_cards TEXT NOT NULL,
    reach_weight TEXT NOT NULL,
    equity TEXT NOT NULL,
    PRIMARY KEY (spec_key, combo_cards),
    FOREIGN KEY (spec_key) REFERENCES oracle_results(spec_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS oracle_action_values (
    spec_key TEXT NOT NULL,
    combo_cards TEXT NOT NULL,
    action_kind TEXT NOT NULL,
    action_amount TEXT NOT NULL,
    frequency TEXT NOT NULL,
    ev TEXT NOT NULL,
    PRIMARY KEY (spec_key, combo_cards, action_kind, action_amount),
    FOREIGN KEY (spec_key, combo_cards)
        REFERENCES oracle_combo_policies(spec_key, combo_cards) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS oracle_conditional_ranges (
    spec_key TEXT NOT NULL,
    position TEXT NOT NULL,
    combo_cards TEXT NOT NULL,
    input_range_weight TEXT NOT NULL,
    path_weight TEXT NOT NULL,
    joint_compatible_weight TEXT NOT NULL,
    conditional_reach_weight TEXT NOT NULL,
    PRIMARY KEY (spec_key, position, combo_cards),
    FOREIGN KEY (spec_key) REFERENCES oracle_results(spec_key) ON DELETE CASCADE
);
"""


def _decimal_text(value: Decimal | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


class OracleCache:
    """Small idempotent cache for immutable solve specs and node policies."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(_SCHEMA)

    def __enter__(self) -> "OracleCache":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._connection.close()

    def put(self, spec: SolveSpec, result: SolveResult) -> None:
        """Atomically insert or replace a result; repeated writes are idempotent."""

        if not isinstance(spec, SolveSpec):
            raise OracleValidationError("spec must be a SolveSpec")
        if not isinstance(result, SolveResult):
            raise OracleValidationError("result must be a SolveResult")
        validate_result_for_spec(spec, result)
        spec_json = spec.canonical_json
        metadata = result.metadata
        extra_json = canonical_json(dict(metadata.extra))
        try:
            # Reserve the writer before the collision check so the validation,
            # metadata upsert, and child-row replacement form one transaction.
            self._connection.execute("BEGIN IMMEDIATE")
            existing = self._connection.execute(
                "SELECT spec_json FROM oracle_results WHERE spec_key = ?",
                (spec.cache_key,),
            ).fetchone()
            if existing is not None and existing["spec_json"] != spec_json:
                raise OracleValidationError(
                    "cache key collision with different canonical spec"
                )
            self._connection.execute(
                """
                INSERT INTO oracle_results (
                    spec_key, spec_json, solver_name, solver_version, iterations,
                    elapsed_seconds, exploitability, converged, metadata_extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(spec_key) DO UPDATE SET
                    spec_json = excluded.spec_json,
                    solver_name = excluded.solver_name,
                    solver_version = excluded.solver_version,
                    iterations = excluded.iterations,
                    elapsed_seconds = excluded.elapsed_seconds,
                    exploitability = excluded.exploitability,
                    converged = excluded.converged,
                    metadata_extra_json = excluded.metadata_extra_json
                """,
                (
                    result.spec_key,
                    spec_json,
                    metadata.solver_name,
                    metadata.solver_version,
                    metadata.iterations,
                    _decimal_text(metadata.elapsed_seconds),
                    _decimal_text(metadata.exploitability),
                    int(metadata.converged),
                    extra_json,
                ),
            )
            self._connection.execute(
                "DELETE FROM oracle_combo_policies WHERE spec_key = ?",
                (result.spec_key,),
            )
            self._connection.execute(
                "DELETE FROM oracle_conditional_ranges WHERE spec_key = ?",
                (result.spec_key,),
            )
            self._connection.executemany(
                """
                INSERT INTO oracle_combo_policies (
                    spec_key, combo_cards, reach_weight, equity
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        result.spec_key,
                        canonical_json(policy.private_combo),
                        _decimal_text(policy.reach_weight),
                        _decimal_text(policy.equity),
                    )
                    for policy in result.combo_policies
                ],
            )
            self._connection.executemany(
                """
                INSERT INTO oracle_action_values (
                    spec_key, combo_cards, action_kind, action_amount, frequency, ev
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        result.spec_key,
                        canonical_json(policy.private_combo),
                        value.action.kind.value,
                        _decimal_text(value.action.amount),
                        _decimal_text(value.frequency),
                        _decimal_text(value.ev),
                    )
                    for policy in result.combo_policies
                    for value in policy.action_values
                ],
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def put_continuation(
        self,
        spec: ContinuationSpec,
        result: ContinuationResult,
    ) -> None:
        """Atomically cache a cross-street policy and both conditional ranges."""

        if not isinstance(spec, ContinuationSpec):
            raise OracleValidationError("spec must be a ContinuationSpec")
        if not isinstance(result, ContinuationResult):
            raise OracleValidationError("result must be a ContinuationResult")
        validate_continuation_result(spec, result)
        spec_json = spec.canonical_json
        metadata = result.metadata
        extra_json = canonical_json(dict(metadata.extra))
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            existing = self._connection.execute(
                "SELECT spec_json FROM oracle_results WHERE spec_key = ?",
                (spec.cache_key,),
            ).fetchone()
            if existing is not None and existing["spec_json"] != spec_json:
                raise OracleValidationError(
                    "cache key collision with different canonical spec"
                )
            self._connection.execute(
                """
                INSERT INTO oracle_results (
                    spec_key, spec_json, solver_name, solver_version, iterations,
                    elapsed_seconds, exploitability, converged, metadata_extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(spec_key) DO UPDATE SET
                    spec_json = excluded.spec_json,
                    solver_name = excluded.solver_name,
                    solver_version = excluded.solver_version,
                    iterations = excluded.iterations,
                    elapsed_seconds = excluded.elapsed_seconds,
                    exploitability = excluded.exploitability,
                    converged = excluded.converged,
                    metadata_extra_json = excluded.metadata_extra_json
                """,
                (
                    result.spec_key,
                    spec_json,
                    metadata.solver_name,
                    metadata.solver_version,
                    metadata.iterations,
                    _decimal_text(metadata.elapsed_seconds),
                    _decimal_text(metadata.exploitability),
                    int(metadata.converged),
                    extra_json,
                ),
            )
            self._connection.execute(
                "DELETE FROM oracle_combo_policies WHERE spec_key = ?",
                (result.spec_key,),
            )
            self._connection.execute(
                "DELETE FROM oracle_conditional_ranges WHERE spec_key = ?",
                (result.spec_key,),
            )
            self._connection.executemany(
                """
                INSERT INTO oracle_combo_policies (
                    spec_key, combo_cards, reach_weight, equity
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        result.spec_key,
                        canonical_json(policy.private_combo),
                        _decimal_text(policy.reach_weight),
                        _decimal_text(policy.equity),
                    )
                    for policy in result.combo_policies
                ],
            )
            self._connection.executemany(
                """
                INSERT INTO oracle_action_values (
                    spec_key, combo_cards, action_kind, action_amount, frequency, ev
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        result.spec_key,
                        canonical_json(policy.private_combo),
                        value.action.kind.value,
                        _decimal_text(value.action.amount),
                        _decimal_text(value.frequency),
                        _decimal_text(value.ev),
                    )
                    for policy in result.combo_policies
                    for value in policy.action_values
                ],
            )
            self._connection.executemany(
                """
                INSERT INTO oracle_conditional_ranges (
                    spec_key, position, combo_cards, input_range_weight,
                    path_weight, joint_compatible_weight,
                    conditional_reach_weight
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        result.spec_key,
                        conditional_range.position.value,
                        canonical_json(combo.cards),
                        _decimal_text(combo.input_range_weight),
                        _decimal_text(combo.path_weight),
                        _decimal_text(combo.joint_compatible_weight),
                        _decimal_text(combo.conditional_reach_weight),
                    )
                    for conditional_range in result.conditional_ranges
                    for combo in conditional_range.combos
                ],
            )
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def get(
        self,
        spec: SolveSpec,
        *,
        expected_binary_sha256: str | None = None,
        expected_execution_context: str | None = None,
    ) -> SolveResult | None:
        """Return a validated cached result, optionally pinned to provenance.

        When ``expected_binary_sha256`` is provided, entries made by an older
        or different executable (including legacy entries without a digest)
        are cache misses rather than trusted results. Execution contexts are
        likewise never relabelled across offline and owned-simulator use.
        """

        if not isinstance(spec, SolveSpec):
            raise OracleValidationError("spec must be a SolveSpec")
        if expected_binary_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", expected_binary_sha256
        ):
            raise OracleValidationError(
                "expected_binary_sha256 must be a lowercase SHA-256 digest"
            )
        if expected_execution_context not in {None, "offline", "owned_simulator"}:
            raise OracleValidationError(
                "expected_execution_context must be offline or owned_simulator"
            )
        row = self._connection.execute(
            "SELECT * FROM oracle_results WHERE spec_key = ?",
            (spec.cache_key,),
        ).fetchone()
        if row is None:
            return None
        if row["spec_json"] != spec.canonical_json:
            raise OracleValidationError("cached canonical spec does not match its key")

        extra_data = json.loads(row["metadata_extra_json"])
        if not isinstance(extra_data, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in extra_data.items()
        ):
            raise OracleValidationError("cached metadata_extra_json is malformed")
        metadata = SolverMetadata(
            solver_name=row["solver_name"],
            solver_version=row["solver_version"],
            iterations=row["iterations"],
            elapsed_seconds=Decimal(row["elapsed_seconds"]),
            exploitability=Decimal(row["exploitability"]),
            converged=bool(row["converged"]),
            extra=tuple(extra_data.items()),
        )
        if expected_binary_sha256 is not None:
            cached_digest = dict(metadata.extra).get("binary_sha256")
            if cached_digest != expected_binary_sha256:
                return None
        if expected_execution_context is not None:
            cached_context = dict(metadata.extra).get("execution_context")
            if cached_context != expected_execution_context:
                return None
        policy_rows = self._connection.execute(
            """
            SELECT combo_cards, reach_weight, equity
            FROM oracle_combo_policies
            WHERE spec_key = ?
            ORDER BY combo_cards
            """,
            (spec.cache_key,),
        ).fetchall()
        action_rows = self._connection.execute(
            """
            SELECT combo_cards, action_kind, action_amount, frequency, ev
            FROM oracle_action_values
            WHERE spec_key = ?
            ORDER BY combo_cards, action_kind, action_amount
            """,
            (spec.cache_key,),
        ).fetchall()
        actions_by_combo: dict[str, list[ActionValue]] = {
            policy_row["combo_cards"]: [] for policy_row in policy_rows
        }
        for action_row in action_rows:
            combo_identity = action_row["combo_cards"]
            if combo_identity not in actions_by_combo:
                raise OracleValidationError(
                    "cached action value references an unknown combo policy"
                )
            actions_by_combo[combo_identity].append(
                ActionValue(
                    Action(
                        ActionKind(action_row["action_kind"]),
                        int(action_row["action_amount"])
                        if action_row["action_amount"]
                        else None,
                    ),
                    Decimal(action_row["frequency"]),
                    Decimal(action_row["ev"]),
                )
            )
        combo_policies = tuple(
            ComboPolicy(
                tuple(json.loads(policy_row["combo_cards"])),
                Decimal(policy_row["reach_weight"]),
                Decimal(policy_row["equity"]),
                tuple(actions_by_combo[policy_row["combo_cards"]]),
            )
            for policy_row in policy_rows
        )
        result = SolveResult(spec.cache_key, combo_policies, metadata)
        validate_result_for_spec(spec, result)
        return result

    def get_continuation(
        self,
        spec: ContinuationSpec,
        *,
        expected_binary_sha256: str | None = None,
        expected_execution_context: str | None = None,
    ) -> ContinuationResult | None:
        """Return one fully validated cached cross-street continuation."""

        if not isinstance(spec, ContinuationSpec):
            raise OracleValidationError("spec must be a ContinuationSpec")
        if expected_binary_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}",
            expected_binary_sha256,
        ):
            raise OracleValidationError(
                "expected_binary_sha256 must be a lowercase SHA-256 digest"
            )
        if expected_execution_context not in {
            None,
            "offline",
            "owned_simulator",
        }:
            raise OracleValidationError(
                "expected_execution_context must be offline or owned_simulator"
            )
        row = self._connection.execute(
            "SELECT * FROM oracle_results WHERE spec_key = ?",
            (spec.cache_key,),
        ).fetchone()
        if row is None:
            return None
        if row["spec_json"] != spec.canonical_json:
            raise OracleValidationError(
                "cached canonical continuation does not match its key"
            )

        extra_data = json.loads(row["metadata_extra_json"])
        if not isinstance(extra_data, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in extra_data.items()
        ):
            raise OracleValidationError(
                "cached metadata_extra_json is malformed"
            )
        metadata = SolverMetadata(
            solver_name=row["solver_name"],
            solver_version=row["solver_version"],
            iterations=row["iterations"],
            elapsed_seconds=Decimal(row["elapsed_seconds"]),
            exploitability=Decimal(row["exploitability"]),
            converged=bool(row["converged"]),
            extra=tuple(extra_data.items()),
        )
        if expected_binary_sha256 is not None:
            if dict(metadata.extra).get("binary_sha256") != expected_binary_sha256:
                return None
        if expected_execution_context is not None:
            if (
                dict(metadata.extra).get("execution_context")
                != expected_execution_context
            ):
                return None

        policy_rows = self._connection.execute(
            """
            SELECT combo_cards, reach_weight, equity
            FROM oracle_combo_policies
            WHERE spec_key = ?
            ORDER BY combo_cards
            """,
            (spec.cache_key,),
        ).fetchall()
        action_rows = self._connection.execute(
            """
            SELECT combo_cards, action_kind, action_amount, frequency, ev
            FROM oracle_action_values
            WHERE spec_key = ?
            ORDER BY combo_cards, action_kind, action_amount
            """,
            (spec.cache_key,),
        ).fetchall()
        actions_by_combo: dict[str, list[ActionValue]] = {
            policy_row["combo_cards"]: [] for policy_row in policy_rows
        }
        for action_row in action_rows:
            combo_identity = action_row["combo_cards"]
            if combo_identity not in actions_by_combo:
                raise OracleValidationError(
                    "cached action value references an unknown combo policy"
                )
            actions_by_combo[combo_identity].append(
                ActionValue(
                    Action(
                        ActionKind(action_row["action_kind"]),
                        int(action_row["action_amount"])
                        if action_row["action_amount"]
                        else None,
                    ),
                    Decimal(action_row["frequency"]),
                    Decimal(action_row["ev"]),
                )
            )
        combo_policies = tuple(
            ComboPolicy(
                tuple(json.loads(policy_row["combo_cards"])),
                Decimal(policy_row["reach_weight"]),
                Decimal(policy_row["equity"]),
                tuple(actions_by_combo[policy_row["combo_cards"]]),
            )
            for policy_row in policy_rows
        )

        range_rows = self._connection.execute(
            """
            SELECT position, combo_cards, input_range_weight, path_weight,
                   joint_compatible_weight, conditional_reach_weight
            FROM oracle_conditional_ranges
            WHERE spec_key = ?
            ORDER BY position, combo_cards
            """,
            (spec.cache_key,),
        ).fetchall()
        combos_by_position: dict[Position, list[ConditionalCombo]] = {
            Position.OOP: [],
            Position.IP: [],
        }
        for range_row in range_rows:
            try:
                position = Position(range_row["position"])
            except ValueError as error:
                raise OracleValidationError(
                    "cached conditional range has an invalid position"
                ) from error
            combos_by_position[position].append(
                ConditionalCombo(
                    cards=tuple(json.loads(range_row["combo_cards"])),
                    input_range_weight=Decimal(
                        range_row["input_range_weight"]
                    ),
                    path_weight=Decimal(range_row["path_weight"]),
                    joint_compatible_weight=Decimal(
                        range_row["joint_compatible_weight"]
                    ),
                    conditional_reach_weight=Decimal(
                        range_row["conditional_reach_weight"]
                    ),
                )
            )
        conditional_ranges = tuple(
            ConditionalRange(position, tuple(combos_by_position[position]))
            for position in (Position.OOP, Position.IP)
        )
        result = ContinuationResult(
            spec.cache_key,
            combo_policies,
            conditional_ranges,  # type: ignore[arg-type]
            metadata,
        )
        validate_continuation_result(spec, result)
        return result

    def result_count(self) -> int:
        """Return the number of unique cached solve specifications."""

        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM oracle_results"
        ).fetchone()
        return int(row["count"])

    def action_value_count(self) -> int:
        """Return the number of cached per-action values."""

        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM oracle_action_values"
        ).fetchone()
        return int(row["count"])

    def combo_policy_count(self) -> int:
        """Return the number of cached private-combo policies."""

        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM oracle_combo_policies"
        ).fetchone()
        return int(row["count"])
