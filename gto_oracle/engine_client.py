"""Strict subprocess client for the local GTO engine.

The client has no live-capture, keyboard, vision, or language-model imports. It
executes one explicitly configured local binary, sends one solve request, and
accepts a result only after request, provenance, solver commit, root actions,
and per-combo vectors all match the immutable :class:`SolveSpec`. Execution is
limited to either the existing offline context or an explicitly acknowledged
user-owned simulator context; exactly one context must be selected.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import time
from typing import Any

from .continuation import (
    ConditionalCombo,
    ConditionalRange,
    ContinuationAction,
    ContinuationDeal,
    ContinuationResult,
    ContinuationSpec,
)
from .models import (
    Action,
    ActionKind,
    ActionValue,
    AllocationMode,
    ComboPolicy,
    OracleValidationError,
    PlayerRange,
    Position,
    SolveResult,
    SolveSpec,
    SolverMetadata,
    _coerce_combo_cards,
)
from .serialization import canonical_json


SCHEMA_VERSION = 1
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_STDERR_BYTES = 4 * 1024 * 1024
_PROCESS_IO_CHUNK_BYTES = 64 * 1024
_RANK_ORDER = {rank: index for index, rank in enumerate("23456789TJQKA")}


class EngineClientError(RuntimeError):
    """Base class for local engine execution and protocol failures."""


class EngineTimeoutError(EngineClientError):
    """The local solver did not finish within the configured timeout."""


class EngineProcessError(EngineClientError):
    """The configured binary could not run or exited inconsistently."""


class EngineProtocolError(EngineClientError):
    """The bridge returned malformed, mismatched, or untrusted output."""


class EngineResponseError(EngineClientError):
    """The bridge returned a well-formed explicit error response."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _binary_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as binary_handle:
            for chunk in iter(lambda: binary_handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EngineProcessError(f"failed to hash engine binary: {error}") from error
    return digest.hexdigest()


def _run_engine_process(
    binary: Path,
    request_json: str,
    *,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    """Run the engine while bounding both output streams during collection."""

    command = [str(binary)]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    request_bytes = request_json.encode("utf-8")
    request_offset = 0
    stdout = bytearray()
    stderr = bytearray()
    output_buffers = {"stdout": stdout, "stderr": stderr}
    output_limits = {
        "stdout": MAX_RESPONSE_BYTES,
        "stderr": MAX_STDERR_BYTES,
    }
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + timeout_seconds

    def close_registered(stream) -> None:
        try:
            selector.unregister(stream)
        except (KeyError, ValueError):
            pass
        try:
            stream.close()
        except OSError:
            pass

    try:
        for name, stream in (
            ("stdout", process.stdout),
            ("stderr", process.stderr),
        ):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        os.set_blocking(process.stdin.fileno(), False)
        if request_bytes:
            # Most requests fit in the pipe immediately. Write eagerly so a
            # short-lived bridge cannot sit blocked on stdin while a platform
            # selector delays its first writable notification. Large requests
            # fall back to the normal non-blocking selector path.
            try:
                while request_offset < len(request_bytes):
                    written = os.write(
                        process.stdin.fileno(),
                        request_bytes[request_offset:],
                    )
                    if written <= 0:
                        break
                    request_offset += written
            except BlockingIOError:
                pass
            except BrokenPipeError:
                close_registered(process.stdin)
            if request_offset == len(request_bytes):
                close_registered(process.stdin)
            elif not process.stdin.closed:
                selector.register(
                    process.stdin,
                    selectors.EVENT_WRITE,
                    "stdin",
                )
        else:
            process.stdin.close()

        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            events = selector.select(remaining)
            if not events:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            for key, _ in events:
                stream = key.fileobj
                name = key.data
                if name == "stdin":
                    try:
                        written = os.write(
                            stream.fileno(),
                            request_bytes[request_offset:],
                        )
                    except BrokenPipeError:
                        close_registered(stream)
                        continue
                    request_offset += written
                    if request_offset == len(request_bytes):
                        close_registered(stream)
                    continue

                try:
                    chunk = os.read(stream.fileno(), _PROCESS_IO_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    close_registered(stream)
                    continue
                output = output_buffers[name]
                output.extend(chunk)
                limit = output_limits[name]
                if len(output) > limit:
                    raise EngineProcessError(
                        f"engine {name} exceeded the {limit}-byte safety limit "
                        "during collection"
                    )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        returncode = process.wait(timeout=remaining)
        try:
            stdout_text = bytes(stdout).decode("utf-8")
        except UnicodeDecodeError as error:
            raise EngineProtocolError("engine stdout is not valid UTF-8") from error
        stderr_text = bytes(stderr).decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout_text,
            stderr=stderr_text,
        )
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        selector.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _wire_number(value: Decimal) -> float:
    # These fields are f64 in the strict Rust schema. Preserve that JSON type
    # even for integral values so echoed provenance compares without 0 vs 0.0
    # ambiguity.
    converted = float(value)
    if not math.isfinite(converted):
        raise OracleValidationError("Decimal value overflows the engine f64 protocol")
    if value != 0 and converted == 0.0:
        raise OracleValidationError("Decimal value underflows the engine f64 protocol")
    return converted


def _engine_combo_text(cards: tuple[str, str]) -> str:
    first, second = cards
    if _RANK_ORDER[first[0]] < _RANK_ORDER[second[0]]:
        first, second = second, first
    return first + second


def render_weighted_range(player_range: PlayerRange) -> str:
    """Render every combo and weight explicitly in upstream Pio syntax."""

    if not isinstance(player_range, PlayerRange):
        raise OracleValidationError("player_range must be a PlayerRange")
    return ",".join(
        f"{_engine_combo_text(combo.cards)}:{_decimal_text(combo.weight)}"
        for combo in player_range.combos
    )


def _bet_sizes_request(spec: SolveSpec) -> dict[str, object]:
    sizing = spec.parameters.bet_sizes

    def street(value):
        return {
            "oop": {
                "bet": value.oop.bet,
                "raise": value.oop.raise_sizes,
            },
            "ip": {
                "bet": value.ip.bet,
                "raise": value.ip.raise_sizes,
            },
        }

    return {
        "flop": street(sizing.flop),
        "turn": street(sizing.turn),
        "river": street(sizing.river),
    }


def _operation_for_spec(spec: SolveSpec) -> str:
    """Return the legacy root operation or descendant-node operation."""

    if (
        not spec.tree.action_history
        and spec.acting_player is Position.OOP
        and spec.tree.facing_bet == 0
    ):
        return "solve_root"
    return "solve_node"


def _validate_root_range_compatibility(spec: SolveSpec) -> None:
    """Reject requested root combos that have no legal opponent pairing."""

    for player_range, opponent_range in (
        (spec.oop_range, spec.ip_range),
        (spec.ip_range, spec.oop_range),
    ):
        impossible = [
            combo.cards
            for combo in player_range.combos
            if not any(
                set(combo.cards).isdisjoint(opponent.cards)
                for opponent in opponent_range.combos
            )
        ]
        if impossible:
            rendered = ", ".join(_engine_combo_text(cards) for cards in impossible)
            raise OracleValidationError(
                f"{player_range.position.value} root range contains "
                f"blocker-impossible combo(s) with zero compatible opponent "
                f"mass: {rendered}"
            )


def _validate_engine_spec(spec: SolveSpec) -> str:
    if not isinstance(spec, SolveSpec):
        raise OracleValidationError("spec must be a SolveSpec")
    if spec.parameters.bet_sizes.flop_donk_sizes is not None:
        raise OracleValidationError("flop donk sizes must be explicit None")
    operation = _operation_for_spec(spec)
    if operation == "solve_root":
        _validate_root_range_compatibility(spec)
    if operation == "solve_node" and not spec.tree.action_history:
        raise OracleValidationError(
            "a non-root engine node requires a non-empty same-street action history"
        )
    return operation


def _wire_action(action: Action) -> dict[str, object]:
    return {"kind": action.kind.value, "amount": action.amount}


def _execution_context(
    offline_only_acknowledged: bool,
    owned_simulator_acknowledged: bool,
) -> str:
    """Validate the mutually exclusive execution acknowledgements."""

    if not isinstance(offline_only_acknowledged, bool):
        raise OracleValidationError("offline_only_acknowledged must be boolean")
    if not isinstance(owned_simulator_acknowledged, bool):
        raise OracleValidationError("owned_simulator_acknowledged must be boolean")
    if offline_only_acknowledged == owned_simulator_acknowledged:
        raise OracleValidationError(
            "exactly one of offline_only_acknowledged and "
            "owned_simulator_acknowledged must be true"
        )
    return "offline" if offline_only_acknowledged else "owned_simulator"


def build_engine_request(
    spec: SolveSpec,
    *,
    offline_only_acknowledged: bool = True,
    owned_simulator_acknowledged: bool = False,
) -> dict[str, object]:
    """Build the strict schema-v1 request represented by ``spec``."""

    operation = _validate_engine_spec(spec)
    _execution_context(
        offline_only_acknowledged,
        owned_simulator_acknowledged,
    )
    parameters = spec.parameters
    request = {
        "schema_version": SCHEMA_VERSION,
        "id": spec.cache_key,
        "operation": operation,
        "offline_only_acknowledged": offline_only_acknowledged,
        "chip_scale": parameters.chip_scale,
        "chip_unit": parameters.chip_unit,
        "street": spec.street.value,
        "board": list(spec.board),
        "oop_range": render_weighted_range(spec.oop_range),
        "ip_range": render_weighted_range(spec.ip_range),
        "starting_pot": spec.tree.pot,
        "effective_stack": spec.tree.effective_stack,
        "bet_sizes": _bet_sizes_request(spec),
        "rake": {
            "rate_pct": _wire_number(spec.tree.rake_rate_pct),
            "cap": float(spec.tree.rake_cap),
        },
        "tree_options": {
            "add_allin_threshold": _wire_number(parameters.add_allin_threshold),
            "force_allin_threshold": _wire_number(parameters.force_allin_threshold),
            "merging_threshold": _wire_number(parameters.merging_threshold),
            "turn_donk_sizes": parameters.bet_sizes.turn_donk_sizes,
            "river_donk_sizes": parameters.bet_sizes.river_donk_sizes,
        },
        "target_exploitability_pct": _wire_number(
            parameters.target_exploitability_pct
        ),
        "max_iterations": parameters.max_iterations,
        "allocation_mode": parameters.allocation_mode.value,
    }
    # Preserve the original offline schema on the wire. The new field is
    # optional and appears only for an explicitly selected owned simulator.
    if owned_simulator_acknowledged:
        request["owned_simulator_acknowledged"] = True
    if operation == "solve_node":
        request.update(
            {
                "action_history": [
                    _wire_action(action) for action in spec.tree.action_history
                ],
                "expected_current_player": spec.acting_player.value,
                "expected_facing_bet": spec.tree.facing_bet,
                "expected_node_actions": [
                    _wire_action(action) for action in spec.tree.modeled_actions
                ],
            }
        )
    return request


def _wire_continuation_step(
    step: ContinuationAction | ContinuationDeal,
) -> dict[str, object]:
    if isinstance(step, ContinuationAction):
        return {
            "type": "action",
            "action": _wire_action(step.action),
        }
    if isinstance(step, ContinuationDeal):
        return {
            "type": "deal",
            "card": step.card,
        }
    raise OracleValidationError("unsupported continuation path step")


def build_continuation_request(
    spec: ContinuationSpec,
    *,
    offline_only_acknowledged: bool = True,
    owned_simulator_acknowledged: bool = False,
) -> dict[str, object]:
    """Build the strict cross-street request represented by ``spec``."""

    if not isinstance(spec, ContinuationSpec):
        raise OracleValidationError(
            "continuation engine request requires ContinuationSpec"
        )
    if spec.parameters.bet_sizes.flop_donk_sizes is not None:
        raise OracleValidationError("flop donk sizes must be explicit None")
    _validate_root_range_compatibility(spec)  # type: ignore[arg-type]
    _execution_context(
        offline_only_acknowledged,
        owned_simulator_acknowledged,
    )
    parameters = spec.parameters
    request: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "id": spec.cache_key,
        "operation": "solve_path",
        "offline_only_acknowledged": offline_only_acknowledged,
        "chip_scale": parameters.chip_scale,
        "chip_unit": parameters.chip_unit,
        "street": "FLOP",
        "board": list(spec.flop),
        "oop_range": render_weighted_range(spec.oop_range),
        "ip_range": render_weighted_range(spec.ip_range),
        "starting_pot": spec.starting_pot,
        "effective_stack": spec.effective_stack,
        "bet_sizes": _bet_sizes_request(spec),  # type: ignore[arg-type]
        "rake": {
            "rate_pct": _wire_number(spec.rake_rate_pct),
            "cap": float(spec.rake_cap),
        },
        "tree_options": {
            "add_allin_threshold": _wire_number(
                parameters.add_allin_threshold
            ),
            "force_allin_threshold": _wire_number(
                parameters.force_allin_threshold
            ),
            "merging_threshold": _wire_number(parameters.merging_threshold),
            "turn_donk_sizes": parameters.bet_sizes.turn_donk_sizes,
            "river_donk_sizes": parameters.bet_sizes.river_donk_sizes,
        },
        "target_exploitability_pct": _wire_number(
            parameters.target_exploitability_pct
        ),
        "max_iterations": parameters.max_iterations,
        "allocation_mode": parameters.allocation_mode.value,
        "path_history": [
            _wire_continuation_step(step) for step in spec.path
        ],
        "expected_board": list(spec.current_board),
        "expected_total_invested": list(spec.expected_total_invested),
        "expected_current_player": spec.acting_player.value,
        "expected_facing_bet": spec.facing_bet,
        "expected_node_actions": [
            _wire_action(action) for action in spec.modeled_actions
        ],
    }
    if owned_simulator_acknowledged:
        request["owned_simulator_acknowledged"] = True
    return request


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EngineProtocolError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads_strict(payload: str) -> Any:
    if not isinstance(payload, str) or not payload.strip():
        raise EngineProtocolError("engine stdout is empty")
    if len(payload.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise EngineProtocolError("engine response exceeds the 64 MiB safety limit")
    try:
        def reject_constant(value: str):
            raise EngineProtocolError(f"non-finite JSON constant {value!r} is forbidden")

        return json.loads(
            payload,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=reject_constant,
            object_pairs_hook=_unique_object,
        )
    except EngineProtocolError:
        raise
    except (json.JSONDecodeError, InvalidOperation, ValueError) as error:
        raise EngineProtocolError(f"engine stdout is not strict JSON: {error}") from error


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EngineProtocolError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise EngineProtocolError(f"{field} must be an array")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise EngineProtocolError(f"{field} must be a string")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise EngineProtocolError(f"{field} must be boolean")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EngineProtocolError(f"{field} must be an integer")
    return value


def _number(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise EngineProtocolError(f"{field} must be a JSON number")
    result = Decimal(value)
    if not result.is_finite():
        raise EngineProtocolError(f"{field} must be finite")
    return result


def _f32_tolerance(*values: Decimal) -> Decimal:
    scale = max((abs(value) for value in values), default=Decimal(0))
    return Decimal("2e-5") * max(Decimal(1), scale)


def _require_f32_close(
    actual: Decimal,
    expected: Decimal,
    field: str,
) -> None:
    if abs(actual - expected) > _f32_tolerance(actual, expected):
        raise EngineProtocolError(
            f"{field} is inconsistent: expected approximately {expected}, got {actual}"
        )


def _exact_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise EngineProtocolError(f"{field} schema mismatch: {'; '.join(details)}")


def _roundtrip_request(request: dict[str, object]) -> dict[str, Any]:
    return _loads_strict(
        json.dumps(
            request,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


_EFFECTIVE_REQUEST_KEYS = {
    "chip_scale",
    "chip_unit",
    "street",
    "board",
    "oop_range",
    "ip_range",
    "starting_pot",
    "effective_stack",
    "bet_sizes",
    "rake",
    "tree_options",
    "target_exploitability_pct",
    "max_iterations",
    "allocation_mode",
}

_NODE_EFFECTIVE_REQUEST_KEYS = _EFFECTIVE_REQUEST_KEYS | {
    "action_history",
    "expected_current_player",
    "expected_facing_bet",
    "expected_node_actions",
}


def _effective_request_keys(spec: SolveSpec) -> set[str]:
    return (
        _NODE_EFFECTIVE_REQUEST_KEYS
        if _operation_for_spec(spec) == "solve_node"
        else _EFFECTIVE_REQUEST_KEYS
    )


def _expected_effective_request(spec: SolveSpec) -> dict[str, Any]:
    # Execution acknowledgements are provenance, not mathematical solve inputs.
    request = _roundtrip_request(build_engine_request(spec))
    return {key: request[key] for key in _effective_request_keys(spec)}


def _validate_provenance(
    spec: SolveSpec,
    response: dict[str, Any],
    *,
    expected_offline_only_acknowledged: bool,
    expected_owned_simulator_acknowledged: bool,
) -> dict[str, Any]:
    _execution_context(
        expected_offline_only_acknowledged,
        expected_owned_simulator_acknowledged,
    )
    provenance = _object(response["provenance"], "provenance")
    provenance_keys = {
        "solver",
        "offline_only_acknowledged",
        "effective_request",
    }
    if expected_owned_simulator_acknowledged:
        provenance_keys.add("owned_simulator_acknowledged")
    _exact_keys(
        provenance,
        provenance_keys,
        "provenance",
    )
    offline_only_acknowledged = _boolean(
        provenance["offline_only_acknowledged"],
        "provenance.offline_only_acknowledged",
    )
    owned_simulator_acknowledged = (
        _boolean(
            provenance["owned_simulator_acknowledged"],
            "provenance.owned_simulator_acknowledged",
        )
        if "owned_simulator_acknowledged" in provenance
        else False
    )
    if (
        offline_only_acknowledged != expected_offline_only_acknowledged
        or owned_simulator_acknowledged
        != expected_owned_simulator_acknowledged
    ):
        raise EngineProtocolError(
            "engine execution-context provenance differs from the request"
        )
    if offline_only_acknowledged == owned_simulator_acknowledged:
        raise EngineProtocolError(
            "engine provenance must acknowledge exactly one execution context"
        )

    solver = _object(provenance["solver"], "provenance.solver")
    _exact_keys(
        solver,
        {
            "name",
            "algorithm",
            "commit",
            "abstraction",
            "allocation_mode",
            "memory_hard_limit_bytes",
        },
        "provenance.solver",
    )
    solver_name = _string(solver["name"], "provenance.solver.name")
    solver_commit = _string(solver["commit"], "provenance.solver.commit")
    allocation_mode = _string(
        solver["allocation_mode"], "provenance.solver.allocation_mode"
    )
    if solver_name != spec.parameters.solver_name:
        raise EngineProtocolError("solver name does not match pinned SolveParameters")
    if solver_commit != spec.parameters.solver_commit:
        raise EngineProtocolError("solver commit does not match pinned SolveParameters")
    if allocation_mode != spec.parameters.allocation_mode.value:
        raise EngineProtocolError("solver allocation mode does not match the request")
    _integer(
        solver["memory_hard_limit_bytes"],
        "provenance.solver.memory_hard_limit_bytes",
    )
    _string(solver["algorithm"], "provenance.solver.algorithm")
    _string(solver["abstraction"], "provenance.solver.abstraction")

    effective = _object(
        provenance["effective_request"], "provenance.effective_request"
    )
    _exact_keys(
        effective,
        _effective_request_keys(spec),
        "provenance.effective_request",
    )
    expected = _expected_effective_request(spec)
    if canonical_json(effective) != canonical_json(expected):
        raise EngineProtocolError(
            "engine effective_request provenance differs from the SolveSpec request"
        )
    return solver


def _parse_action_descriptors(
    spec: SolveSpec,
    response: dict[str, Any],
    field_name: str,
) -> tuple[Action, ...]:
    return _parse_expected_action_descriptors(
        spec.tree.modeled_actions,
        response,
        field_name,
    )


def _parse_expected_action_descriptors(
    modeled_actions: tuple[Action, ...],
    response: dict[str, Any],
    field_name: str,
) -> tuple[Action, ...]:
    descriptors = _list(response[field_name], field_name)
    if not descriptors:
        raise EngineProtocolError(f"{field_name} cannot be empty")
    actions: list[Action] = []
    for expected_index, raw_descriptor in enumerate(descriptors):
        item_field = f"{field_name}[{expected_index}]"
        descriptor = _object(raw_descriptor, item_field)
        _exact_keys(
            descriptor,
            {"index", "label", "kind", "amount"},
            item_field,
        )
        index = _integer(descriptor["index"], f"{item_field}.index")
        if index != expected_index:
            raise EngineProtocolError(
                f"{field_name} indices must be contiguous and ordered"
            )
        _string(descriptor["label"], f"{item_field}.label")
        kind_text = _string(descriptor["kind"], f"{item_field}.kind")
        try:
            kind = ActionKind(kind_text)
        except ValueError as error:
            raise EngineProtocolError(
                f"unsupported {field_name} action kind {kind_text!r}"
            ) from error
        raw_amount = descriptor["amount"]
        amount = None if raw_amount is None else _integer(
            raw_amount, f"{item_field}.amount"
        )
        try:
            actions.append(Action(kind, amount))
        except OracleValidationError as error:
            raise EngineProtocolError(f"invalid {field_name} action: {error}") from error

    if len(actions) != len(set(actions)):
        raise EngineProtocolError(f"{field_name} contains duplicates")
    if set(actions) != set(modeled_actions):
        raise EngineProtocolError(
            f"engine {field_name} do not align with the modeled actions"
        )
    return tuple(actions)


def _parse_root_actions(spec: SolveSpec, response: dict[str, Any]) -> tuple[Action, ...]:
    return _parse_action_descriptors(spec, response, "root_actions")


def _parse_combo_policies(
    spec: SolveSpec,
    response: dict[str, Any],
    root_actions: tuple[Action, ...],
) -> tuple[ComboPolicy, ...]:
    raw_players = _list(response["players"], "players")
    if len(raw_players) != 2:
        raise EngineProtocolError("players must contain exactly OOP and IP")
    players: dict[str, dict[str, Any]] = {}
    player_keys = {
        "player",
        "total_reachable_weight",
        "average_equity",
        "average_ev_units",
        "combos",
    }
    combo_keys = {
        "hand",
        "range_weight",
        "normalized_weight",
        "reach_weight",
        "equity",
        "equilibrium_ev_units",
        "root_action_frequencies",
        "root_action_evs_units",
    }
    aggregates: dict[str, tuple[Decimal, Decimal, Decimal]] = {}
    for index, raw_player in enumerate(raw_players):
        player = _object(raw_player, f"players[{index}]")
        _exact_keys(player, player_keys, f"players[{index}]")
        name = _string(player["player"], f"players[{index}].player")
        if name not in {"OOP", "IP"} or name in players:
            raise EngineProtocolError("players must contain unique OOP and IP entries")
        total_weight = _number(
            player["total_reachable_weight"],
            f"players[{index}].total_reachable_weight",
        )
        average_equity = _number(
            player["average_equity"], f"players[{index}].average_equity"
        )
        average_ev = _number(
            player["average_ev_units"], f"players[{index}].average_ev_units"
        )
        if total_weight <= 0:
            raise EngineProtocolError("player total_reachable_weight must be positive")
        if not Decimal(0) <= average_equity <= Decimal(1):
            raise EngineProtocolError("player average_equity must be between 0 and 1")
        aggregates[name] = (total_weight, average_equity, average_ev)
        players[name] = player
    if set(players) != {"OOP", "IP"}:
        raise EngineProtocolError("players must contain OOP and IP")

    def parse_common_combo(
        player_name: str,
        index: int,
        raw_combo: Any,
        expected_weights: dict[tuple[str, str], Decimal],
    ) -> tuple[
        dict[str, Any],
        tuple[str, str],
        Decimal,
        Decimal,
        Decimal,
        Decimal,
    ]:
        field = f"{player_name} combos[{index}]"
        combo = _object(raw_combo, field)
        _exact_keys(combo, combo_keys, field)
        hand = _string(combo["hand"], f"{field}.hand")
        if len(hand) != 4:
            raise EngineProtocolError(f"{field}.hand must contain exactly two cards")
        try:
            private_combo = _coerce_combo_cards(
                (hand[:2], hand[2:]), f"{field}.hand"
            )
        except OracleValidationError as error:
            raise EngineProtocolError(f"invalid {field}.hand: {error}") from error
        if private_combo not in expected_weights:
            raise EngineProtocolError(
                f"engine returned a {player_name} combo outside the request range"
            )
        range_weight = _number(combo["range_weight"], f"{field}.range_weight")
        if abs(range_weight - expected_weights[private_combo]) > Decimal("1e-6"):
            raise EngineProtocolError(
                f"{field}.range_weight differs from the request"
            )
        normalized_weight = _number(
            combo["normalized_weight"], f"{field}.normalized_weight"
        )
        reach_weight = _number(combo["reach_weight"], f"{field}.reach_weight")
        equity = _number(combo["equity"], f"{field}.equity")
        equilibrium_ev = _number(
            combo["equilibrium_ev_units"], f"{field}.equilibrium_ev_units"
        )
        if normalized_weight <= 0:
            raise EngineProtocolError(f"{field}.normalized_weight must be positive")
        if not Decimal(0) < reach_weight <= Decimal(1):
            raise EngineProtocolError(f"{field}.reach_weight must be in (0, 1]")
        if not Decimal(0) <= equity <= Decimal(1):
            raise EngineProtocolError(f"{field}.equity must be between 0 and 1")
        return combo, private_combo, normalized_weight, reach_weight, equity, equilibrium_ev

    def validate_player_aggregates(
        player_name: str,
        entries: list[
            tuple[tuple[str, str], Decimal, Decimal, Decimal, Decimal]
        ],
        expected_combos: set[tuple[str, str]],
    ) -> None:
        combos = [entry[0] for entry in entries]
        if len(combos) != len(set(combos)):
            raise EngineProtocolError(f"engine returned duplicate {player_name} combos")
        if set(combos) != expected_combos:
            raise EngineProtocolError(
                f"engine {player_name} combo coverage differs from the request range"
            )
        total_reported, average_equity, average_ev = aggregates[player_name]
        normalized_sum = sum((entry[1] for entry in entries), Decimal(0))
        reach_sum = sum((entry[2] for entry in entries), Decimal(0))
        _require_f32_close(
            total_reported,
            normalized_sum,
            f"{player_name} total_reachable_weight",
        )
        _require_f32_close(
            reach_sum,
            Decimal(1),
            f"{player_name} reach-weight sum",
        )
        for combo, normalized, reach, _, _ in entries:
            _require_f32_close(
                reach,
                normalized / total_reported,
                f"{player_name} {combo} reach_weight",
            )
        calculated_equity = sum(
            (entry[3] * entry[1] for entry in entries), Decimal(0)
        ) / normalized_sum
        calculated_ev = sum(
            (entry[4] * entry[1] for entry in entries), Decimal(0)
        ) / normalized_sum
        _require_f32_close(
            average_equity,
            calculated_equity,
            f"{player_name} average_equity",
        )
        _require_f32_close(
            average_ev,
            calculated_ev,
            f"{player_name} average_ev_units",
        )

    ip_expected_weights = {combo.cards: combo.weight for combo in spec.ip_range.combos}
    ip_entries: list[tuple[tuple[str, str], Decimal, Decimal, Decimal, Decimal]] = []
    for index, raw_combo in enumerate(_list(players["IP"]["combos"], "IP combos")):
        combo, private_combo, normalized, reach, equity, equilibrium_ev = (
            parse_common_combo("IP", index, raw_combo, ip_expected_weights)
        )
        if combo["root_action_frequencies"] is not None or combo[
            "root_action_evs_units"
        ] is not None:
            raise EngineProtocolError("IP root action vectors must be null")
        ip_entries.append(
            (private_combo, normalized, reach, equity, equilibrium_ev)
        )
    validate_player_aggregates("IP", ip_entries, set(ip_expected_weights))

    oop_expected_weights = {combo.cards: combo.weight for combo in spec.oop_range.combos}
    policies: list[ComboPolicy] = []
    oop_entries: list[tuple[tuple[str, str], Decimal, Decimal, Decimal, Decimal]] = []
    for index, raw_combo in enumerate(_list(players["OOP"]["combos"], "OOP combos")):
        combo, private_combo, normalized, reach, equity, equilibrium_ev = (
            parse_common_combo("OOP", index, raw_combo, oop_expected_weights)
        )
        frequencies = _list(
            combo["root_action_frequencies"],
            f"OOP combos[{index}].root_action_frequencies",
        )
        action_evs = _list(
            combo["root_action_evs_units"],
            f"OOP combos[{index}].root_action_evs_units",
        )
        if len(frequencies) != len(root_actions) or len(action_evs) != len(
            root_actions
        ):
            raise EngineProtocolError(
                "combo root vectors do not align with root_actions"
            )
        try:
            policy = ComboPolicy(
                private_combo=private_combo,
                reach_weight=reach,
                equity=equity,
                action_values=tuple(
                    ActionValue(
                        action,
                        _number(
                            frequencies[action_index],
                            f"OOP combos[{index}] frequency[{action_index}]",
                        ),
                        _number(
                            action_evs[action_index],
                            f"OOP combos[{index}] action_ev[{action_index}]",
                        ),
                    )
                    for action_index, action in enumerate(root_actions)
                ),
            )
        except OracleValidationError as error:
            raise EngineProtocolError(f"invalid OOP combo policy: {error}") from error
        calculated_equilibrium_ev = sum(
            (
                value.frequency * value.ev
                for value in policy.action_values
            ),
            Decimal(0),
        )
        _require_f32_close(
            equilibrium_ev,
            calculated_equilibrium_ev,
            f"OOP combos[{index}].equilibrium_ev_units",
        )
        policies.append(policy)
        oop_entries.append(
            (private_combo, normalized, reach, equity, equilibrium_ev)
        )
    validate_player_aggregates("OOP", oop_entries, set(oop_expected_weights))
    return tuple(policies)


def _parse_node_policies(
    spec: SolveSpec,
    response: dict[str, Any],
    node_actions: tuple[Action, ...],
) -> tuple[tuple[ComboPolicy, ...], Decimal]:
    """Parse positive-reach policies for the requested descendant actor."""

    total_reachable = _number(
        response["node_total_reachable_weight"],
        "node_total_reachable_weight",
    )
    if total_reachable <= 0:
        raise EngineProtocolError("node_total_reachable_weight must be positive")

    raw_policies = _list(response["policies"], "policies")
    if not raw_policies:
        raise EngineProtocolError("policies cannot be empty")
    policy_keys = {
        "hand",
        "input_range_weight",
        "path_weight",
        "joint_compatible_weight",
        "conditional_reach_weight",
        "equity",
        "equilibrium_ev_units",
        "node_action_frequencies",
        "node_action_evs_units",
    }
    acting_range = (
        spec.oop_range if spec.acting_player is Position.OOP else spec.ip_range
    )
    expected_weights = {combo.cards: combo.weight for combo in acting_range.combos}
    parsed: list[ComboPolicy] = []
    seen_combos: set[tuple[str, str]] = set()
    joint_weights: list[Decimal] = []

    for index, raw_policy in enumerate(raw_policies):
        field = f"policies[{index}]"
        policy_data = _object(raw_policy, field)
        _exact_keys(policy_data, policy_keys, field)
        hand = _string(policy_data["hand"], f"{field}.hand")
        if len(hand) != 4:
            raise EngineProtocolError(f"{field}.hand must contain exactly two cards")
        try:
            private_combo = _coerce_combo_cards(
                (hand[:2], hand[2:]), f"{field}.hand"
            )
        except OracleValidationError as error:
            raise EngineProtocolError(f"invalid {field}.hand: {error}") from error
        if private_combo in seen_combos:
            raise EngineProtocolError("policies contains duplicate private combos")
        seen_combos.add(private_combo)
        if private_combo not in expected_weights:
            raise EngineProtocolError(
                "engine returned a node policy outside the acting-player range"
            )

        input_weight = _number(
            policy_data["input_range_weight"],
            f"{field}.input_range_weight",
        )
        expected_input_weight = expected_weights[private_combo]
        _require_f32_close(
            input_weight,
            expected_input_weight,
            f"{field}.input_range_weight",
        )
        path_weight = _number(policy_data["path_weight"], f"{field}.path_weight")
        joint_weight = _number(
            policy_data["joint_compatible_weight"],
            f"{field}.joint_compatible_weight",
        )
        reach_weight = _number(
            policy_data["conditional_reach_weight"],
            f"{field}.conditional_reach_weight",
        )
        if path_weight <= 0:
            raise EngineProtocolError(f"{field}.path_weight must be positive")
        if path_weight > input_weight + _f32_tolerance(path_weight, input_weight):
            raise EngineProtocolError(
                f"{field}.path_weight cannot exceed input_range_weight"
            )
        if joint_weight <= 0:
            raise EngineProtocolError(
                f"{field}.joint_compatible_weight must be positive"
            )
        if not Decimal(0) < reach_weight <= Decimal(1):
            raise EngineProtocolError(
                f"{field}.conditional_reach_weight must be in (0, 1]"
            )
        _require_f32_close(
            reach_weight,
            joint_weight / total_reachable,
            f"{field}.conditional_reach_weight",
        )

        equity = _number(policy_data["equity"], f"{field}.equity")
        if not Decimal(0) <= equity <= Decimal(1):
            raise EngineProtocolError(f"{field}.equity must be between 0 and 1")
        equilibrium_ev = _number(
            policy_data["equilibrium_ev_units"],
            f"{field}.equilibrium_ev_units",
        )
        frequencies = _list(
            policy_data["node_action_frequencies"],
            f"{field}.node_action_frequencies",
        )
        action_evs = _list(
            policy_data["node_action_evs_units"],
            f"{field}.node_action_evs_units",
        )
        if len(frequencies) != len(node_actions) or len(action_evs) != len(
            node_actions
        ):
            raise EngineProtocolError(
                "node policy vectors do not align with node_actions"
            )
        try:
            policy = ComboPolicy(
                private_combo=private_combo,
                reach_weight=reach_weight,
                equity=equity,
                action_values=tuple(
                    ActionValue(
                        action,
                        _number(
                            frequencies[action_index],
                            f"{field}.node_action_frequencies[{action_index}]",
                        ),
                        _number(
                            action_evs[action_index],
                            f"{field}.node_action_evs_units[{action_index}]",
                        ),
                    )
                    for action_index, action in enumerate(node_actions)
                ),
            )
        except OracleValidationError as error:
            raise EngineProtocolError(f"invalid node combo policy: {error}") from error
        calculated_equilibrium_ev = sum(
            (value.frequency * value.ev for value in policy.action_values),
            Decimal(0),
        )
        _require_f32_close(
            equilibrium_ev,
            calculated_equilibrium_ev,
            f"{field}.equilibrium_ev_units",
        )
        parsed.append(policy)
        joint_weights.append(joint_weight)

    _require_f32_close(
        sum(joint_weights, Decimal(0)),
        total_reachable,
        "node_total_reachable_weight",
    )
    _require_f32_close(
        sum((policy.reach_weight for policy in parsed), Decimal(0)),
        Decimal(1),
        "conditional_reach_weight sum",
    )
    return tuple(parsed), total_reachable


def _starting_pot_for_spec(spec: SolveSpec | ContinuationSpec) -> int:
    if isinstance(spec, ContinuationSpec):
        return spec.starting_pot
    return spec.tree.pot


def _validate_convergence(
    response: dict[str, Any],
    spec: SolveSpec | ContinuationSpec,
) -> dict[str, Any]:
    convergence = _object(response["convergence"], "convergence")
    _exact_keys(
        convergence,
        {
            "iterations",
            "max_iterations",
            "target_exploitability_pct",
            "target_exploitability_units",
            "exploitability_pct_of_pot",
            "exploitability_units",
            "target_reached",
        },
        "convergence",
    )
    iterations = _integer(convergence["iterations"], "convergence.iterations")
    max_iterations = _integer(
        convergence["max_iterations"], "convergence.max_iterations"
    )
    if max_iterations != spec.parameters.max_iterations or not 0 <= iterations <= max_iterations:
        raise EngineProtocolError("convergence iteration counts do not match the request")
    target = _number(
        convergence["target_exploitability_pct"],
        "convergence.target_exploitability_pct",
    )
    _require_f32_close(
        target,
        spec.parameters.target_exploitability_pct,
        "convergence.target_exploitability_pct",
    )
    target_units = _number(
        convergence["target_exploitability_units"],
        "convergence.target_exploitability_units",
    )
    exploitability_pct = _number(
        convergence["exploitability_pct_of_pot"],
        "convergence.exploitability_pct_of_pot",
    )
    exploitability_units = _number(
        convergence["exploitability_units"],
        "convergence.exploitability_units",
    )
    if min(target_units, exploitability_pct, exploitability_units) < 0:
        raise EngineProtocolError("convergence exploitability values cannot be negative")
    expected_target_units = (
        Decimal(_starting_pot_for_spec(spec))
        * spec.parameters.target_exploitability_pct
        / Decimal(100)
    )
    _require_f32_close(
        target_units,
        expected_target_units,
        "convergence.target_exploitability_units",
    )
    expected_exploitability_pct = (
        Decimal(100)
        * exploitability_units
        / Decimal(_starting_pot_for_spec(spec))
    )
    _require_f32_close(
        exploitability_pct,
        expected_exploitability_pct,
        "convergence.exploitability_pct_of_pot",
    )
    target_reached = _boolean(
        convergence["target_reached"], "convergence.target_reached"
    )
    comparison_tolerance = _f32_tolerance(exploitability_units, target_units)
    if target_reached and exploitability_units > target_units + comparison_tolerance:
        raise EngineProtocolError(
            "convergence.target_reached is true above the exploitability target"
        )
    if not target_reached and exploitability_units < target_units - comparison_tolerance:
        raise EngineProtocolError(
            "convergence.target_reached is false below the exploitability target"
        )
    if not target_reached and iterations != max_iterations:
        raise EngineProtocolError(
            "an unconverged solve must exhaust max_iterations"
        )
    return convergence


def _validate_memory_and_timings(
    response: dict[str, Any], spec: SolveSpec | ContinuationSpec
) -> tuple[dict[str, Any], dict[str, Any]]:
    memory = _object(response["memory"], "memory")
    _exact_keys(
        memory,
        {
            "estimated_uncompressed_bytes",
            "estimated_compressed_bytes",
            "allocation_mode",
            "hard_limit_bytes",
        },
        "memory",
    )
    for key in (
        "estimated_uncompressed_bytes",
        "estimated_compressed_bytes",
        "hard_limit_bytes",
    ):
        if _integer(memory[key], f"memory.{key}") < 0:
            raise EngineProtocolError(f"memory.{key} cannot be negative")
    if _string(memory["allocation_mode"], "memory.allocation_mode") != (
        spec.parameters.allocation_mode.value
    ):
        raise EngineProtocolError("memory allocation mode differs from the request")

    timings = _object(response["timings_ms"], "timings_ms")
    _exact_keys(
        timings,
        {"tree_build", "allocation", "solve", "extraction", "total"},
        "timings_ms",
    )
    for key in ("tree_build", "allocation", "solve", "extraction", "total"):
        if _number(timings[key], f"timings_ms.{key}") < 0:
            raise EngineProtocolError(f"timings_ms.{key} cannot be negative")
    return memory, timings


_PATH_EFFECTIVE_REQUEST_KEYS = _EFFECTIVE_REQUEST_KEYS | {
    "path_history",
    "expected_board",
    "expected_total_invested",
    "expected_current_player",
    "expected_facing_bet",
    "expected_node_actions",
}


def _validate_continuation_provenance(
    spec: ContinuationSpec,
    response: dict[str, Any],
    *,
    expected_offline_only_acknowledged: bool,
    expected_owned_simulator_acknowledged: bool,
) -> dict[str, Any]:
    _execution_context(
        expected_offline_only_acknowledged,
        expected_owned_simulator_acknowledged,
    )
    provenance = _object(response["provenance"], "provenance")
    provenance_keys = {
        "solver",
        "offline_only_acknowledged",
        "effective_request",
    }
    if expected_owned_simulator_acknowledged:
        provenance_keys.add("owned_simulator_acknowledged")
    _exact_keys(provenance, provenance_keys, "provenance")
    actual_offline = _boolean(
        provenance["offline_only_acknowledged"],
        "provenance.offline_only_acknowledged",
    )
    actual_owned = (
        _boolean(
            provenance["owned_simulator_acknowledged"],
            "provenance.owned_simulator_acknowledged",
        )
        if "owned_simulator_acknowledged" in provenance
        else False
    )
    if (
        actual_offline != expected_offline_only_acknowledged
        or actual_owned != expected_owned_simulator_acknowledged
        or actual_offline == actual_owned
    ):
        raise EngineProtocolError(
            "continuation execution-context provenance differs from the request"
        )

    solver = _object(provenance["solver"], "provenance.solver")
    _exact_keys(
        solver,
        {
            "name",
            "algorithm",
            "commit",
            "abstraction",
            "allocation_mode",
            "memory_hard_limit_bytes",
        },
        "provenance.solver",
    )
    if _string(solver["name"], "provenance.solver.name") != (
        spec.parameters.solver_name
    ):
        raise EngineProtocolError(
            "continuation solver name differs from the pinned request"
        )
    if _string(solver["commit"], "provenance.solver.commit") != (
        spec.parameters.solver_commit
    ):
        raise EngineProtocolError(
            "continuation solver commit differs from the pinned request"
        )
    if _string(
        solver["allocation_mode"],
        "provenance.solver.allocation_mode",
    ) != spec.parameters.allocation_mode.value:
        raise EngineProtocolError(
            "continuation allocation mode differs from the request"
        )
    _string(solver["algorithm"], "provenance.solver.algorithm")
    _string(solver["abstraction"], "provenance.solver.abstraction")
    _integer(
        solver["memory_hard_limit_bytes"],
        "provenance.solver.memory_hard_limit_bytes",
    )

    effective = _object(
        provenance["effective_request"],
        "provenance.effective_request",
    )
    _exact_keys(
        effective,
        _PATH_EFFECTIVE_REQUEST_KEYS,
        "provenance.effective_request",
    )
    request = _roundtrip_request(
        build_continuation_request(
            spec,
            offline_only_acknowledged=expected_offline_only_acknowledged,
            owned_simulator_acknowledged=(
                expected_owned_simulator_acknowledged
            ),
        )
    )
    expected_effective = {
        key: request[key] for key in _PATH_EFFECTIVE_REQUEST_KEYS
    }
    if canonical_json(effective) != canonical_json(expected_effective):
        raise EngineProtocolError(
            "engine continuation provenance differs from the request"
        )
    return solver


def _parse_conditional_ranges(
    spec: ContinuationSpec,
    response: dict[str, Any],
) -> tuple[ConditionalRange, ConditionalRange]:
    raw_ranges = _list(response["conditional_ranges"], "conditional_ranges")
    if len(raw_ranges) != 2:
        raise EngineProtocolError(
            "conditional_ranges must contain exactly OOP and IP"
        )
    expected_weights = {
        "OOP": {combo.cards: combo.weight for combo in spec.oop_range.combos},
        "IP": {combo.cards: combo.weight for combo in spec.ip_range.combos},
    }
    parsed: list[ConditionalRange] = []
    seen_players: set[str] = set()
    range_keys = {
        "player",
        "total_joint_compatible_weight",
        "combos",
    }
    combo_keys = {
        "hand",
        "input_range_weight",
        "path_weight",
        "joint_compatible_weight",
        "conditional_reach_weight",
    }
    for range_index, raw_range in enumerate(raw_ranges):
        field = f"conditional_ranges[{range_index}]"
        range_data = _object(raw_range, field)
        _exact_keys(range_data, range_keys, field)
        player = _string(range_data["player"], f"{field}.player")
        if player not in {"OOP", "IP"} or player in seen_players:
            raise EngineProtocolError(
                "conditional_ranges players must be unique OOP and IP"
            )
        seen_players.add(player)
        total = _number(
            range_data["total_joint_compatible_weight"],
            f"{field}.total_joint_compatible_weight",
        )
        if total <= 0:
            raise EngineProtocolError(
                f"{field}.total_joint_compatible_weight must be positive"
            )
        combos: list[ConditionalCombo] = []
        joint_sum = Decimal(0)
        seen_combos: set[tuple[str, str]] = set()
        for combo_index, raw_combo in enumerate(
            _list(range_data["combos"], f"{field}.combos")
        ):
            combo_field = f"{field}.combos[{combo_index}]"
            combo_data = _object(raw_combo, combo_field)
            _exact_keys(combo_data, combo_keys, combo_field)
            hand = _string(combo_data["hand"], f"{combo_field}.hand")
            if len(hand) != 4:
                raise EngineProtocolError(
                    f"{combo_field}.hand must contain two cards"
                )
            try:
                cards = _coerce_combo_cards(
                    (hand[:2], hand[2:]),
                    f"{combo_field}.hand",
                )
            except OracleValidationError as error:
                raise EngineProtocolError(
                    f"invalid {combo_field}.hand: {error}"
                ) from error
            if cards in seen_combos:
                raise EngineProtocolError(
                    f"{field}.combos contains duplicate cards"
                )
            seen_combos.add(cards)
            if cards not in expected_weights[player]:
                raise EngineProtocolError(
                    f"{combo_field} is outside the requested {player} range"
                )
            input_weight = _number(
                combo_data["input_range_weight"],
                f"{combo_field}.input_range_weight",
            )
            _require_f32_close(
                input_weight,
                expected_weights[player][cards],
                f"{combo_field}.input_range_weight",
            )
            path_weight = _number(
                combo_data["path_weight"],
                f"{combo_field}.path_weight",
            )
            joint_weight = _number(
                combo_data["joint_compatible_weight"],
                f"{combo_field}.joint_compatible_weight",
            )
            reach_weight = _number(
                combo_data["conditional_reach_weight"],
                f"{combo_field}.conditional_reach_weight",
            )
            if path_weight <= 0 or joint_weight <= 0:
                raise EngineProtocolError(
                    f"{combo_field} path and joint weights must be positive"
                )
            if path_weight > input_weight + _f32_tolerance(
                path_weight,
                input_weight,
            ):
                raise EngineProtocolError(
                    f"{combo_field}.path_weight exceeds its input weight"
                )
            _require_f32_close(
                reach_weight,
                joint_weight / total,
                f"{combo_field}.conditional_reach_weight",
            )
            try:
                combos.append(
                    ConditionalCombo(
                        cards=cards,
                        input_range_weight=input_weight,
                        path_weight=path_weight,
                        joint_compatible_weight=joint_weight,
                        conditional_reach_weight=reach_weight,
                    )
                )
            except OracleValidationError as error:
                raise EngineProtocolError(
                    f"invalid {combo_field}: {error}"
                ) from error
            joint_sum += joint_weight
        if not combos:
            raise EngineProtocolError(f"{field}.combos cannot be empty")
        _require_f32_close(
            joint_sum,
            total,
            f"{field}.total_joint_compatible_weight",
        )
        try:
            parsed.append(
                ConditionalRange(
                    position=Position(player),
                    combos=tuple(combos),
                )
            )
        except OracleValidationError as error:
            raise EngineProtocolError(
                f"invalid {field}: {error}"
            ) from error
    if seen_players != {"OOP", "IP"}:
        raise EngineProtocolError(
            "conditional_ranges must contain OOP and IP"
        )
    return tuple(sorted(parsed, key=lambda item: item.position.value))  # type: ignore[return-value]


def parse_continuation_response(
    spec: ContinuationSpec,
    payload: str,
    *,
    expected_id: str | None = None,
    binary_sha256: str | None = None,
    offline_only_acknowledged: bool = True,
    owned_simulator_acknowledged: bool = False,
) -> ContinuationResult:
    """Strictly validate a ``solve_path`` engine response."""

    if not isinstance(spec, ContinuationSpec):
        raise OracleValidationError(
            "continuation response requires ContinuationSpec"
        )
    execution_context = _execution_context(
        offline_only_acknowledged,
        owned_simulator_acknowledged,
    )
    if binary_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}",
        binary_sha256,
    ):
        raise OracleValidationError(
            "binary_sha256 must be a lowercase SHA-256 digest"
        )
    expected_id = expected_id or spec.cache_key
    response = _object(_loads_strict(payload), "response")
    common = {"schema_version", "id", "operation", "status"}
    if not common.issubset(response):
        raise EngineProtocolError(
            "continuation response is missing envelope fields"
        )
    if _integer(response["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise EngineProtocolError("unsupported engine schema_version")
    if _string(response["id"], "id") != expected_id:
        raise EngineProtocolError(
            "continuation response id differs from the request"
        )
    if _string(response["operation"], "operation") != "solve_path":
        raise EngineProtocolError(
            "continuation response operation must be solve_path"
        )
    status = _string(response["status"], "status")
    if status == "error":
        _exact_keys(response, common | {"error"}, "error response")
        error_body = _object(response["error"], "error")
        _exact_keys(error_body, {"code", "message"}, "error")
        raise EngineResponseError(
            _string(error_body["code"], "error.code"),
            _string(error_body["message"], "error.message"),
        )
    if status != "ok":
        raise EngineProtocolError(
            f"unsupported continuation status {status!r}"
        )
    _exact_keys(
        response,
        common
        | {
            "provenance",
            "current_street",
            "current_board",
            "current_player",
            "node_actions",
            "node_total_reachable_weight",
            "policies",
            "conditional_ranges",
            "convergence",
            "memory",
            "timings_ms",
        },
        "continuation success response",
    )
    solver = _validate_continuation_provenance(
        spec,
        response,
        expected_offline_only_acknowledged=offline_only_acknowledged,
        expected_owned_simulator_acknowledged=owned_simulator_acknowledged,
    )
    expected_street = {3: "FLOP", 4: "TURN", 5: "RIVER"}[
        len(spec.current_board)
    ]
    if _string(response["current_street"], "current_street") != expected_street:
        raise EngineProtocolError(
            "continuation current_street differs from current_board"
        )
    board = _list(response["current_board"], "current_board")
    if board != list(spec.current_board) or any(
        not isinstance(card, str) for card in board
    ):
        raise EngineProtocolError(
            "continuation current_board differs from the request"
        )
    if _string(response["current_player"], "current_player") != (
        spec.acting_player.value
    ):
        raise EngineProtocolError(
            "continuation current player differs from the request"
        )
    node_actions = _parse_expected_action_descriptors(
        spec.modeled_actions,
        response,
        "node_actions",
    )
    policies, node_total = _parse_node_policies(
        spec,  # type: ignore[arg-type]
        response,
        node_actions,
    )
    ranges = _parse_conditional_ranges(spec, response)
    convergence = _validate_convergence(response, spec)
    memory, timings = _validate_memory_and_timings(response, spec)
    metadata_extra = [
        ("abstraction", _string(solver["abstraction"], "solver.abstraction")),
        ("algorithm", _string(solver["algorithm"], "solver.algorithm")),
        ("allocation_mode", spec.parameters.allocation_mode.value),
        ("chip_scale", str(spec.parameters.chip_scale)),
        ("chip_unit", spec.parameters.chip_unit),
        ("execution_context", execution_context),
        (
            "exploitability_pct_of_pot",
            _decimal_text(
                _number(
                    convergence["exploitability_pct_of_pot"],
                    "convergence.exploitability_pct_of_pot",
                )
            ),
        ),
        ("node_total_reachable_weight", _decimal_text(node_total)),
        ("protocol_operation", "solve_path"),
        ("protocol_schema", str(SCHEMA_VERSION)),
        ("solver_commit", spec.parameters.solver_commit),
        (
            "estimated_uncompressed_bytes",
            str(
                _integer(
                    memory["estimated_uncompressed_bytes"],
                    "memory.estimated_uncompressed_bytes",
                )
            ),
        ),
        (
            "estimated_compressed_bytes",
            str(
                _integer(
                    memory["estimated_compressed_bytes"],
                    "memory.estimated_compressed_bytes",
                )
            ),
        ),
        (
            "memory_hard_limit_bytes",
            str(_integer(memory["hard_limit_bytes"], "memory.hard_limit_bytes")),
        ),
    ]
    for key in ("tree_build", "allocation", "solve", "extraction", "total"):
        metadata_extra.append(
            (
                f"timing_{key}_ms",
                _decimal_text(
                    _number(timings[key], f"timings_ms.{key}")
                ),
            )
        )
    if binary_sha256 is not None:
        metadata_extra.append(("binary_sha256", binary_sha256))
    metadata = SolverMetadata(
        solver_name=_string(solver["name"], "provenance.solver.name"),
        solver_version=_string(solver["commit"], "provenance.solver.commit"),
        iterations=_integer(
            convergence["iterations"],
            "convergence.iterations",
        ),
        elapsed_seconds=_number(timings["total"], "timings_ms.total")
        / Decimal(1000),
        exploitability=_number(
            convergence["exploitability_units"],
            "convergence.exploitability_units",
        ),
        converged=_boolean(
            convergence["target_reached"],
            "convergence.target_reached",
        ),
        extra=tuple(metadata_extra),
    )
    try:
        return ContinuationResult.for_spec(
            spec,
            policies,
            ranges,
            metadata,
        )
    except OracleValidationError as error:
        raise EngineProtocolError(
            f"continuation result does not satisfy its specification: {error}"
        ) from error


def parse_engine_response(
    spec: SolveSpec,
    payload: str,
    *,
    expected_id: str | None = None,
    binary_sha256: str | None = None,
    offline_only_acknowledged: bool = True,
    owned_simulator_acknowledged: bool = False,
) -> SolveResult:
    """Strictly validate a schema-v1 engine response and build a SolveResult."""

    operation = _validate_engine_spec(spec)
    execution_context = _execution_context(
        offline_only_acknowledged,
        owned_simulator_acknowledged,
    )
    if binary_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", binary_sha256):
        raise OracleValidationError(
            "binary_sha256 must be a lowercase SHA-256 digest when supplied"
        )
    expected_id = expected_id or spec.cache_key
    response = _object(_loads_strict(payload), "response")
    common = {"schema_version", "id", "operation", "status"}
    if not common.issubset(response):
        raise EngineProtocolError("response is missing protocol envelope fields")
    if _integer(response["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise EngineProtocolError("unsupported engine schema_version")
    if _string(response["id"], "id") != expected_id:
        raise EngineProtocolError("engine response id does not match the request")
    if _string(response["operation"], "operation") != operation:
        raise EngineProtocolError(
            f"engine response operation must be {operation}"
        )
    status = _string(response["status"], "status")
    if status == "error":
        _exact_keys(response, common | {"error"}, "error response")
        error_body = _object(response["error"], "error")
        _exact_keys(error_body, {"code", "message"}, "error")
        raise EngineResponseError(
            _string(error_body["code"], "error.code"),
            _string(error_body["message"], "error.message"),
        )
    if status != "ok":
        raise EngineProtocolError(f"unsupported engine status {status!r}")

    operation_fields = (
        {"root_player", "root_actions", "players"}
        if operation == "solve_root"
        else {
            "current_player",
            "node_actions",
            "node_total_reachable_weight",
            "policies",
        }
    )
    _exact_keys(
        response,
        common
        | {
            "provenance",
            "street",
            "board",
            "convergence",
            "memory",
            "timings_ms",
        }
        | operation_fields,
        "success response",
    )
    solver = _validate_provenance(
        spec,
        response,
        expected_offline_only_acknowledged=offline_only_acknowledged,
        expected_owned_simulator_acknowledged=owned_simulator_acknowledged,
    )
    if _string(response["street"], "street") != spec.street.value:
        raise EngineProtocolError("response street differs from the request")
    board = _list(response["board"], "board")
    if board != list(spec.board) or any(not isinstance(card, str) for card in board):
        raise EngineProtocolError("response board differs from the request")
    node_total_reachable: Decimal | None = None
    if operation == "solve_root":
        if _string(response["root_player"], "root_player") != "OOP":
            raise EngineProtocolError("engine root_player must be OOP")
        root_actions = _parse_root_actions(spec, response)
        policies = _parse_combo_policies(spec, response, root_actions)
    else:
        current_player = _string(response["current_player"], "current_player")
        if current_player != spec.acting_player.value:
            raise EngineProtocolError(
                "engine current_player differs from SolveSpec.acting_player"
            )
        node_actions = _parse_action_descriptors(spec, response, "node_actions")
        policies, node_total_reachable = _parse_node_policies(
            spec,
            response,
            node_actions,
        )
    convergence = _validate_convergence(response, spec)
    memory, timings = _validate_memory_and_timings(response, spec)
    metadata_extra = [
        ("abstraction", _string(solver["abstraction"], "solver.abstraction")),
        ("algorithm", _string(solver["algorithm"], "solver.algorithm")),
        ("allocation_mode", spec.parameters.allocation_mode.value),
        ("chip_scale", str(spec.parameters.chip_scale)),
        ("chip_unit", spec.parameters.chip_unit),
        ("execution_context", execution_context),
        (
            "exploitability_pct_of_pot",
            _decimal_text(
                _number(
                    convergence["exploitability_pct_of_pot"],
                    "convergence.exploitability_pct_of_pot",
                )
            ),
        ),
        ("protocol_schema", str(SCHEMA_VERSION)),
        ("solver_commit", spec.parameters.solver_commit),
        (
            "estimated_uncompressed_bytes",
            str(
                _integer(
                    memory["estimated_uncompressed_bytes"],
                    "memory.estimated_uncompressed_bytes",
                )
            ),
        ),
        (
            "estimated_compressed_bytes",
            str(
                _integer(
                    memory["estimated_compressed_bytes"],
                    "memory.estimated_compressed_bytes",
                )
            ),
        ),
        (
            "memory_hard_limit_bytes",
            str(
                _integer(
                    memory["hard_limit_bytes"],
                    "memory.hard_limit_bytes",
                )
            ),
        ),
    ]
    for key in ("tree_build", "allocation", "solve", "extraction", "total"):
        metadata_extra.append(
            (
                f"timing_{key}_ms",
                _decimal_text(
                    _number(timings[key], f"timings_ms.{key}")
                ),
            )
        )
    if node_total_reachable is not None:
        metadata_extra.extend(
            (
                ("protocol_operation", operation),
                (
                "node_total_reachable_weight",
                _decimal_text(node_total_reachable),
                ),
            )
        )
    if binary_sha256 is not None:
        metadata_extra.append(("binary_sha256", binary_sha256))
    metadata = SolverMetadata(
        solver_name=_string(solver["name"], "provenance.solver.name"),
        solver_version=_string(solver["commit"], "provenance.solver.commit"),
        iterations=_integer(convergence["iterations"], "convergence.iterations"),
        elapsed_seconds=_number(timings["total"], "timings_ms.total")
        / Decimal(1000),
        exploitability=_number(
            convergence["exploitability_units"],
            "convergence.exploitability_units",
        ),
        converged=_boolean(
            convergence["target_reached"], "convergence.target_reached"
        ),
        extra=tuple(metadata_extra),
    )
    try:
        return SolveResult.for_spec(spec, policies, metadata)
    except OracleValidationError as error:
        raise EngineProtocolError(f"engine result does not satisfy SolveSpec: {error}") from error


class EngineClient:
    """One-shot client for an explicitly specified local engine binary."""

    def __init__(
        self,
        binary: str | Path,
        *,
        offline_only_acknowledged: bool,
        owned_simulator_acknowledged: bool = False,
        timeout_seconds: Decimal | int | str = Decimal(300),
    ) -> None:
        self.execution_context = _execution_context(
            offline_only_acknowledged,
            owned_simulator_acknowledged,
        )
        self.offline_only_acknowledged = offline_only_acknowledged
        self.owned_simulator_acknowledged = owned_simulator_acknowledged
        try:
            self.binary = Path(binary).expanduser().resolve()
        except (OSError, RuntimeError, TypeError) as error:
            raise EngineProcessError(
                f"failed to resolve engine binary path: {error}"
            ) from error
        try:
            timeout = (
                timeout_seconds
                if isinstance(timeout_seconds, Decimal)
                else Decimal(timeout_seconds)
            )
        except (InvalidOperation, TypeError, ValueError) as error:
            raise OracleValidationError("timeout_seconds must be a decimal number") from error
        if not timeout.is_finite() or timeout <= 0:
            raise OracleValidationError("timeout_seconds must be finite and positive")
        self.timeout_seconds = timeout

    @property
    def binary_sha256(self) -> str:
        """Return the current executable digest used to qualify cache hits."""

        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise EngineProcessError(
                f"engine binary is missing or not executable: {self.binary}"
            )
        return _binary_digest(self.binary)

    def solve(self, spec: SolveSpec) -> SolveResult:
        """Run one local solve and return only a fully validated result."""

        request = build_engine_request(
            spec,
            offline_only_acknowledged=self.offline_only_acknowledged,
            owned_simulator_acknowledged=self.owned_simulator_acknowledged,
        )
        binary_digest = self.binary_sha256
        request_json = json.dumps(
            request,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            completed = _run_engine_process(
                self.binary,
                request_json,
                timeout_seconds=float(self.timeout_seconds),
            )
        except subprocess.TimeoutExpired as error:
            raise EngineTimeoutError(
                f"local engine exceeded {self.timeout_seconds} seconds"
            ) from error
        except OSError as error:
            raise EngineProcessError(f"failed to execute engine: {error}") from error
        if _binary_digest(self.binary) != binary_digest:
            raise EngineProcessError("engine binary changed while the solve was running")

        try:
            result = parse_engine_response(
                spec,
                completed.stdout,
                expected_id=spec.cache_key,
                binary_sha256=binary_digest,
                offline_only_acknowledged=self.offline_only_acknowledged,
                owned_simulator_acknowledged=self.owned_simulator_acknowledged,
            )
        except EngineResponseError:
            raise
        except EngineProtocolError as error:
            stderr = completed.stderr.strip()
            suffix = f"; stderr={stderr!r}" if stderr else ""
            if completed.returncode != 0:
                raise EngineProcessError(
                    f"engine exited {completed.returncode} with invalid response: {error}{suffix}"
                ) from error
            raise
        if completed.returncode != 0:
            raise EngineProcessError(
                f"engine exited {completed.returncode} despite a success response"
            )
        return result

    def solve_continuation(
        self,
        spec: ContinuationSpec,
    ) -> ContinuationResult:
        """Solve and traverse one complete flop-to-current public path."""

        request = build_continuation_request(
            spec,
            offline_only_acknowledged=self.offline_only_acknowledged,
            owned_simulator_acknowledged=self.owned_simulator_acknowledged,
        )
        binary_digest = self.binary_sha256
        request_json = json.dumps(
            request,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            completed = _run_engine_process(
                self.binary,
                request_json,
                timeout_seconds=float(self.timeout_seconds),
            )
        except subprocess.TimeoutExpired as error:
            raise EngineTimeoutError(
                f"local continuation engine exceeded {self.timeout_seconds} seconds"
            ) from error
        except OSError as error:
            raise EngineProcessError(
                f"failed to execute continuation engine: {error}"
            ) from error
        if _binary_digest(self.binary) != binary_digest:
            raise EngineProcessError(
                "engine binary changed while the continuation solve was running"
            )

        try:
            result = parse_continuation_response(
                spec,
                completed.stdout,
                expected_id=spec.cache_key,
                binary_sha256=binary_digest,
                offline_only_acknowledged=self.offline_only_acknowledged,
                owned_simulator_acknowledged=(
                    self.owned_simulator_acknowledged
                ),
            )
        except EngineResponseError:
            raise
        except EngineProtocolError as error:
            stderr = completed.stderr.strip()
            suffix = f"; stderr={stderr!r}" if stderr else ""
            if completed.returncode != 0:
                raise EngineProcessError(
                    f"engine exited {completed.returncode} with invalid "
                    f"continuation response: {error}{suffix}"
                ) from error
            raise
        if completed.returncode != 0:
            raise EngineProcessError(
                "engine exited nonzero despite a successful continuation response"
            )
        return result
