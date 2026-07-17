"""Deterministic local GTO oracle foundation.

This package contains only deterministic data, scoring, serialization, and
SQLite cache primitives. It has no capture, keyboard, vision, or language-model
dependencies; callers must select an explicit engine execution context.
"""

from .cache import OracleCache
from .engine_client import (
    EngineClient,
    EngineClientError,
    EngineProcessError,
    EngineProtocolError,
    EngineResponseError,
    EngineTimeoutError,
    build_engine_request,
    parse_engine_response,
    render_weighted_range,
)
from .models import (
    Action,
    ActionKind,
    ActionValue,
    AllocationMode,
    BetSizingConfig,
    ComboPolicy,
    DecisionQuery,
    OracleValidationError,
    PlayerRange,
    Position,
    SolveResult,
    SolveParameters,
    SolveSpec,
    SolverMetadata,
    StreetBetSizes,
    Street,
    TreeConfig,
    UnsupportedGameError,
    WeightedCombo,
    PlayerBetSizes,
    validate_result_for_spec,
)
from .scoring import (
    ActionAssessment,
    AssessmentStatus,
    IllegalActionAssessment,
    OutOfTreeActionAssessment,
    ScoredAction,
    assess_action,
)
from .serialization import canonical_json, sha256_key

__all__ = [
    "Action",
    "ActionAssessment",
    "ActionKind",
    "ActionValue",
    "AllocationMode",
    "AssessmentStatus",
    "BetSizingConfig",
    "ComboPolicy",
    "DecisionQuery",
    "EngineClient",
    "EngineClientError",
    "EngineProcessError",
    "EngineProtocolError",
    "EngineResponseError",
    "EngineTimeoutError",
    "IllegalActionAssessment",
    "OracleCache",
    "OracleValidationError",
    "OutOfTreeActionAssessment",
    "PlayerRange",
    "PlayerBetSizes",
    "Position",
    "ScoredAction",
    "SolveResult",
    "SolveParameters",
    "SolveSpec",
    "SolverMetadata",
    "Street",
    "StreetBetSizes",
    "TreeConfig",
    "UnsupportedGameError",
    "WeightedCombo",
    "assess_action",
    "build_engine_request",
    "canonical_json",
    "sha256_key",
    "parse_engine_response",
    "render_weighted_range",
    "validate_result_for_spec",
]
