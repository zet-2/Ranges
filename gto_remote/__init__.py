"""Authenticated remote transport for the live GTO router.

The public contract lives in :mod:`gto_remote.protocol`.  Importing this
package does not start a server or initialize the solver.
"""

from .protocol import (
    PROTOCOL_SCHEMA_VERSION,
    RemoteProtocolError,
    build_evaluate_request,
    decision_fingerprint,
    decision_state_from_wire,
    decision_state_to_wire,
    encode_json,
    outcome_from_wire,
    outcome_to_wire,
    parse_evaluate_request,
)
from .client import (
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_MAX_RESPONSE_BYTES,
    RemoteGTOClient,
    RemoteGTOClientConfig,
    RemoteGTOClientError,
    RemoteGTOConfigurationError,
    RemoteGTOConnectionError,
    RemoteGTOHTTPError,
    RemoteGTORequestError,
    RemoteGTORequestTooLargeError,
    RemoteGTOResponseProtocolError,
    RemoteGTOResponseTooLargeError,
    RemoteGTOTimeoutError,
)
from .capabilities import (
    NATIVE_ROUTER_CAPABILITIES,
    SolverCapabilities,
    SolverCapabilitiesError,
    capabilities_for_router,
)
from .external_backend import (
    ExternalBackendConfig,
    ExternalBackendConfigurationError,
    ExternalBackendError,
    ExternalBackendProtocolError,
    ExternalBackendTimeoutError,
    ExternalSolverBackend,
)
from .multiway_client import RemoteMultiwayClient
from .multiway_outcome import (
    MultiwayOutcomeError,
    MultiwayPolicyAction,
    MultiwaySolveOutcome,
    MultiwaySolveProof,
)
from .multiway_protocol import (
    MultiwayDecisionState,
    MultiwayProtocolError,
)

__all__ = [
    "PROTOCOL_SCHEMA_VERSION",
    "RemoteProtocolError",
    "DEFAULT_MAX_REQUEST_BYTES",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "RemoteGTOClient",
    "RemoteGTOClientConfig",
    "RemoteGTOClientError",
    "RemoteGTOConfigurationError",
    "RemoteGTOConnectionError",
    "RemoteGTOHTTPError",
    "RemoteGTORequestError",
    "RemoteGTORequestTooLargeError",
    "RemoteGTOResponseProtocolError",
    "RemoteGTOResponseTooLargeError",
    "RemoteGTOTimeoutError",
    "NATIVE_ROUTER_CAPABILITIES",
    "SolverCapabilities",
    "SolverCapabilitiesError",
    "capabilities_for_router",
    "ExternalBackendConfig",
    "ExternalBackendConfigurationError",
    "ExternalBackendError",
    "ExternalBackendProtocolError",
    "ExternalBackendTimeoutError",
    "ExternalSolverBackend",
    "RemoteMultiwayClient",
    "MultiwayDecisionState",
    "MultiwayProtocolError",
    "MultiwayOutcomeError",
    "MultiwayPolicyAction",
    "MultiwaySolveOutcome",
    "MultiwaySolveProof",
    "build_evaluate_request",
    "decision_fingerprint",
    "decision_state_from_wire",
    "decision_state_to_wire",
    "encode_json",
    "outcome_from_wire",
    "outcome_to_wire",
    "parse_evaluate_request",
]
