// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This executable links to b-inary/postflop-solver, which is licensed under
// AGPL-3.0-or-later. See NOTICE.md before distributing or offering this
// executable over a network.

use postflop_solver::{
    card_from_str, compute_exploitability, finalize, flop_from_str, hole_to_string, solve_step,
    Action, ActionTree, BetSizeOptions, BoardState, CardConfig, DonkSizeOptions, PostFlopGame,
    TreeConfig, NOT_DEALT,
};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::ExitCode;
use std::time::Instant;

const SOLVER_COMMIT: &str = "9d1509fe5077d019825f833eed04b16d342dfda1";
const SCHEMA_VERSION: u32 = 1;
const DEFAULT_MAX_ESTIMATED_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_CONFIGURED_MEMORY_GIB: u64 = 4_096;
const MEMORY_LIMIT_ENV: &str = "GTO_ENGINE_MAX_MEMORY_GIB";
const MAX_ITERATIONS: u32 = 1_000_000;
const MAX_TEXT_FIELD_BYTES: usize = 65_536;
const MAX_CHIP_UNITS: i32 = i32::MAX / 4;
const MAX_CHIP_SCALE: u32 = i32::MAX as u32;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SolveRootRequest {
    schema_version: u32,
    id: String,
    operation: Operation,
    offline_only_acknowledged: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    owned_simulator_acknowledged: bool,
    street: Street,
    board: Vec<String>,
    oop_range: String,
    ip_range: String,
    starting_pot: i32,
    effective_stack: i32,
    chip_scale: u32,
    chip_unit: String,
    allocation_mode: AllocationMode,
    #[serde(default)]
    bet_sizes: BetSizes,
    #[serde(default)]
    rake: Rake,
    tree_options: TreeOptions,
    target_exploitability_pct: f64,
    max_iterations: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SolveNodeRequest {
    schema_version: u32,
    id: String,
    operation: Operation,
    offline_only_acknowledged: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    owned_simulator_acknowledged: bool,
    street: Street,
    board: Vec<String>,
    oop_range: String,
    ip_range: String,
    starting_pot: i32,
    effective_stack: i32,
    chip_scale: u32,
    chip_unit: String,
    allocation_mode: AllocationMode,
    #[serde(default)]
    bet_sizes: BetSizes,
    #[serde(default)]
    rake: Rake,
    tree_options: TreeOptions,
    target_exploitability_pct: f64,
    max_iterations: u32,
    action_history: Vec<WireAction>,
    expected_current_player: NodePlayer,
    expected_facing_bet: i32,
    expected_node_actions: Vec<WireAction>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SolvePathRequest {
    schema_version: u32,
    id: String,
    operation: Operation,
    offline_only_acknowledged: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    owned_simulator_acknowledged: bool,
    street: Street,
    board: Vec<String>,
    oop_range: String,
    ip_range: String,
    starting_pot: i32,
    effective_stack: i32,
    chip_scale: u32,
    chip_unit: String,
    allocation_mode: AllocationMode,
    #[serde(default)]
    bet_sizes: BetSizes,
    #[serde(default)]
    rake: Rake,
    tree_options: TreeOptions,
    target_exploitability_pct: f64,
    max_iterations: u32,
    path_history: Vec<WirePathStep>,
    expected_board: Vec<String>,
    expected_total_invested: [i32; 2],
    expected_current_player: NodePlayer,
    expected_facing_bet: i32,
    expected_node_actions: Vec<WireAction>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Operation {
    SolveRoot,
    SolveNode,
    SolvePath,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
enum Street {
    Flop,
    Turn,
    River,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
enum NodePlayer {
    Oop,
    Ip,
}

impl NodePlayer {
    fn index(self) -> usize {
        match self {
            Self::Oop => 0,
            Self::Ip => 1,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Oop => "OOP",
            Self::Ip => "IP",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum WireActionKind {
    Fold,
    Check,
    Call,
    Bet,
    Raise,
    AllIn,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct WireAction {
    kind: WireActionKind,
    #[serde(deserialize_with = "deserialize_required_nullable_i32")]
    amount: Option<i32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum WirePathStep {
    Action {
        action: WireAction,
    },
    Deal {
        card: String,
    },
}

#[derive(Debug, Clone, Copy)]
enum ResolvedPathStep {
    Action(Action),
    Deal(u8),
}

fn deserialize_required_nullable_i32<'de, D>(deserializer: D) -> Result<Option<i32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<i32>::deserialize(deserializer)
}

impl WireAction {
    fn to_solver_action(&self, field: &str, effective_stack: i32) -> Result<Action, OracleError> {
        let validate_unsized = |kind: &'static str| {
            if self.amount.is_some() {
                Err(OracleError::validation(format!(
                    "{field} {kind} action must have amount=null"
                )))
            } else {
                Ok(())
            }
        };
        let sized = |kind: &'static str| {
            let amount = self.amount.ok_or_else(|| {
                OracleError::validation(format!("{field} {kind} action requires an amount"))
            })?;
            if amount <= 0 || amount > effective_stack {
                return Err(OracleError::validation(format!(
                    "{field} {kind} amount must be in [1, {effective_stack}]"
                )));
            }
            Ok(amount)
        };

        match self.kind {
            WireActionKind::Fold => {
                validate_unsized("FOLD")?;
                Ok(Action::Fold)
            }
            WireActionKind::Check => {
                validate_unsized("CHECK")?;
                Ok(Action::Check)
            }
            WireActionKind::Call => {
                validate_unsized("CALL")?;
                Ok(Action::Call)
            }
            WireActionKind::Bet => Ok(Action::Bet(sized("BET")?)),
            WireActionKind::Raise => Ok(Action::Raise(sized("RAISE")?)),
            WireActionKind::AllIn => Ok(Action::AllIn(sized("ALL_IN")?)),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AllocationMode {
    UncompressedF32,
    CompressedI16,
}

impl AllocationMode {
    fn is_compressed(self) -> bool {
        matches!(self, Self::CompressedI16)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::UncompressedF32 => "uncompressed_f32",
            Self::CompressedI16 => "compressed_i16",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct BetSizes {
    flop: StreetBetSizes,
    turn: StreetBetSizes,
    river: StreetBetSizes,
}

impl Default for BetSizes {
    fn default() -> Self {
        Self {
            flop: StreetBetSizes::default(),
            turn: StreetBetSizes::default(),
            river: StreetBetSizes::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct StreetBetSizes {
    oop: PlayerBetSizes,
    ip: PlayerBetSizes,
}

impl Default for StreetBetSizes {
    fn default() -> Self {
        Self {
            oop: PlayerBetSizes::default(),
            ip: PlayerBetSizes::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct PlayerBetSizes {
    bet: String,
    raise: String,
}

impl Default for PlayerBetSizes {
    fn default() -> Self {
        Self {
            bet: "50%".to_string(),
            raise: "2.5x".to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct Rake {
    rate_pct: f64,
    cap: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TreeOptions {
    add_allin_threshold: f64,
    force_allin_threshold: f64,
    merging_threshold: f64,
    turn_donk_sizes: Option<String>,
    river_donk_sizes: Option<String>,
}

impl Default for Rake {
    fn default() -> Self {
        Self {
            rate_pct: 0.0,
            cap: 0.0,
        }
    }
}

#[derive(Debug, Serialize)]
struct SolveRootResponse {
    schema_version: u32,
    id: String,
    operation: &'static str,
    status: &'static str,
    provenance: Provenance,
    street: Street,
    board: Vec<String>,
    root_player: &'static str,
    root_actions: Vec<ActionDescriptor>,
    players: Vec<PlayerResult>,
    convergence: Convergence,
    memory: MemoryUsage,
    timings_ms: Timings,
}

#[derive(Debug, Serialize)]
struct SolveNodeResponse {
    schema_version: u32,
    id: String,
    operation: &'static str,
    status: &'static str,
    provenance: NodeProvenance,
    street: Street,
    board: Vec<String>,
    current_player: &'static str,
    node_actions: Vec<ActionDescriptor>,
    node_total_reachable_weight: f64,
    policies: Vec<NodePolicy>,
    convergence: Convergence,
    memory: MemoryUsage,
    timings_ms: Timings,
}

#[derive(Debug, Serialize)]
struct SolvePathResponse {
    schema_version: u32,
    id: String,
    operation: &'static str,
    status: &'static str,
    provenance: PathProvenance,
    current_street: Street,
    current_board: Vec<String>,
    current_player: &'static str,
    node_actions: Vec<ActionDescriptor>,
    node_total_reachable_weight: f64,
    policies: Vec<NodePolicy>,
    conditional_ranges: Vec<ConditionalPlayerRange>,
    convergence: Convergence,
    memory: MemoryUsage,
    timings_ms: Timings,
}

#[derive(Debug, Serialize)]
struct SolverMetadata {
    name: &'static str,
    algorithm: &'static str,
    commit: &'static str,
    abstraction: &'static str,
    allocation_mode: &'static str,
    memory_hard_limit_bytes: u64,
}

#[derive(Debug, Serialize)]
struct Provenance {
    solver: SolverMetadata,
    offline_only_acknowledged: bool,
    #[serde(skip_serializing_if = "is_false")]
    owned_simulator_acknowledged: bool,
    effective_request: EffectiveRequest,
}

#[derive(Debug, Serialize)]
struct NodeProvenance {
    solver: SolverMetadata,
    offline_only_acknowledged: bool,
    #[serde(skip_serializing_if = "is_false")]
    owned_simulator_acknowledged: bool,
    effective_request: EffectiveNodeRequest,
}

#[derive(Debug, Serialize)]
struct PathProvenance {
    solver: SolverMetadata,
    offline_only_acknowledged: bool,
    #[serde(skip_serializing_if = "is_false")]
    owned_simulator_acknowledged: bool,
    effective_request: EffectivePathRequest,
}

#[derive(Debug, Serialize)]
struct EffectiveRequest {
    street: Street,
    board: Vec<String>,
    oop_range: String,
    ip_range: String,
    starting_pot: i32,
    effective_stack: i32,
    chip_scale: u32,
    chip_unit: String,
    allocation_mode: AllocationMode,
    bet_sizes: BetSizes,
    rake: Rake,
    tree_options: TreeOptions,
    target_exploitability_pct: f64,
    max_iterations: u32,
}

#[derive(Debug, Serialize)]
struct EffectiveNodeRequest {
    street: Street,
    board: Vec<String>,
    oop_range: String,
    ip_range: String,
    starting_pot: i32,
    effective_stack: i32,
    chip_scale: u32,
    chip_unit: String,
    allocation_mode: AllocationMode,
    bet_sizes: BetSizes,
    rake: Rake,
    tree_options: TreeOptions,
    target_exploitability_pct: f64,
    max_iterations: u32,
    action_history: Vec<WireAction>,
    expected_current_player: NodePlayer,
    expected_facing_bet: i32,
    expected_node_actions: Vec<WireAction>,
}

#[derive(Debug, Serialize)]
struct EffectivePathRequest {
    street: Street,
    board: Vec<String>,
    oop_range: String,
    ip_range: String,
    starting_pot: i32,
    effective_stack: i32,
    chip_scale: u32,
    chip_unit: String,
    allocation_mode: AllocationMode,
    bet_sizes: BetSizes,
    rake: Rake,
    tree_options: TreeOptions,
    target_exploitability_pct: f64,
    max_iterations: u32,
    path_history: Vec<WirePathStep>,
    expected_board: Vec<String>,
    expected_total_invested: [i32; 2],
    expected_current_player: NodePlayer,
    expected_facing_bet: i32,
    expected_node_actions: Vec<WireAction>,
}

#[derive(Debug, Serialize)]
struct ActionDescriptor {
    index: usize,
    label: String,
    kind: &'static str,
    amount: Option<i32>,
}

#[derive(Debug, Serialize)]
struct PlayerResult {
    player: &'static str,
    total_reachable_weight: f64,
    average_equity: f64,
    average_ev_units: f64,
    combos: Vec<ComboResult>,
}

#[derive(Debug, Serialize)]
struct ComboResult {
    hand: String,
    range_weight: f64,
    normalized_weight: f64,
    reach_weight: f64,
    equity: f64,
    equilibrium_ev_units: f64,
    root_action_frequencies: Option<Vec<f64>>,
    root_action_evs_units: Option<Vec<f64>>,
}

#[derive(Debug, Serialize)]
struct NodePolicy {
    hand: String,
    input_range_weight: f64,
    path_weight: f64,
    joint_compatible_weight: f64,
    conditional_reach_weight: f64,
    equity: f64,
    equilibrium_ev_units: f64,
    node_action_frequencies: Vec<f64>,
    node_action_evs_units: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct ConditionalPlayerRange {
    player: &'static str,
    total_joint_compatible_weight: f64,
    combos: Vec<ConditionalComboWeight>,
}

#[derive(Debug, Serialize)]
struct ConditionalComboWeight {
    hand: String,
    input_range_weight: f64,
    path_weight: f64,
    joint_compatible_weight: f64,
    conditional_reach_weight: f64,
}

#[derive(Debug, Serialize)]
struct Convergence {
    iterations: u32,
    max_iterations: u32,
    target_exploitability_pct: f64,
    target_exploitability_units: f64,
    exploitability_pct_of_pot: f64,
    exploitability_units: f64,
    target_reached: bool,
}

#[derive(Debug, Serialize)]
struct MemoryUsage {
    estimated_uncompressed_bytes: u64,
    estimated_compressed_bytes: u64,
    allocation_mode: &'static str,
    hard_limit_bytes: u64,
}

#[derive(Debug, Serialize)]
struct Timings {
    tree_build: f64,
    allocation: f64,
    solve: f64,
    extraction: f64,
    total: f64,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    schema_version: u32,
    id: Option<String>,
    operation: &'static str,
    status: &'static str,
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: &'static str,
    message: String,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug)]
struct OracleError {
    code: &'static str,
    message: String,
}

impl OracleError {
    fn validation(message: impl Into<String>) -> Self {
        Self {
            code: "VALIDATION_ERROR",
            message: message.into(),
        }
    }

    fn solver(message: impl Into<String>) -> Self {
        Self {
            code: "SOLVER_ERROR",
            message: message.into(),
        }
    }
}

fn memory_limit_bytes_from_value(value: Option<&str>) -> Result<u64, OracleError> {
    let Some(value) = value else {
        return Ok(DEFAULT_MAX_ESTIMATED_MEMORY_BYTES);
    };
    let gib = value.trim().parse::<u64>().map_err(|_| {
        OracleError::validation(format!(
            "{MEMORY_LIMIT_ENV} must be a positive integer number of GiB"
        ))
    })?;
    if !(1..=MAX_CONFIGURED_MEMORY_GIB).contains(&gib) {
        return Err(OracleError::validation(format!(
            "{MEMORY_LIMIT_ENV} must be between 1 and {MAX_CONFIGURED_MEMORY_GIB}"
        )));
    }
    gib.checked_mul(1024 * 1024 * 1024).ok_or_else(|| {
        OracleError::validation(format!("{MEMORY_LIMIT_ENV} is too large"))
    })
}

fn configured_memory_limit_bytes() -> Result<u64, OracleError> {
    let raw_value = std::env::var_os(MEMORY_LIMIT_ENV);
    let value = raw_value
        .as_deref()
        .map(|raw| {
            raw.to_str().ok_or_else(|| {
                OracleError::validation(format!("{MEMORY_LIMIT_ENV} must be valid UTF-8"))
            })
        })
        .transpose()?;
    memory_limit_bytes_from_value(value)
}

fn main() -> ExitCode {
    let mut input = String::new();
    if let Err(error) = io::stdin().read_to_string(&mut input) {
        write_error(None, "IO_ERROR", format!("failed to read stdin: {error}"));
        return ExitCode::FAILURE;
    }

    if input.trim().is_empty() {
        write_error(None, "INVALID_REQUEST", "stdin must contain one JSON request".to_string());
        return ExitCode::FAILURE;
    }

    let envelope = serde_json::from_str::<serde_json::Value>(&input).ok();
    let fallback_id = envelope
        .as_ref()
        .and_then(|value| value.get("id").and_then(|id| id.as_str()).map(str::to_string));
    let operation = envelope
        .as_ref()
        .and_then(|value| value.get("operation").and_then(|operation| operation.as_str()));

    match operation {
        Some("solve_node") => return run_node_request(&input, fallback_id),
        Some("solve_path") => return run_path_request(&input, fallback_id),
        _ => {}
    }

    run_root_request(&input, fallback_id)
}

fn run_root_request(input: &str, fallback_id: Option<String>) -> ExitCode {
    let request: SolveRootRequest = match serde_json::from_str(&input) {
        Ok(request) => request,
        Err(error) => {
            write_error(
                fallback_id,
                "INVALID_REQUEST",
                format!("request does not match the strict schema: {error}"),
            );
            return ExitCode::FAILURE;
        }
    };

    let id = request.id.clone();
    let result = catch_unwind(AssertUnwindSafe(|| solve_root(request)));
    match result {
        Ok(Ok(response)) => match serde_json::to_string(&response) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                write_error(Some(id), "SERIALIZATION_ERROR", error.to_string());
                ExitCode::FAILURE
            }
        },
        Ok(Err(error)) => {
            write_error(Some(id), error.code, error.message);
            ExitCode::FAILURE
        }
        Err(_) => {
            write_error(
                Some(id),
                "INTERNAL_PANIC",
                "the solver aborted while processing the request".to_string(),
            );
            ExitCode::FAILURE
        }
    }
}

fn run_node_request(input: &str, fallback_id: Option<String>) -> ExitCode {
    let request: SolveNodeRequest = match serde_json::from_str(input) {
        Ok(request) => request,
        Err(error) => {
            write_error_for_operation(
                "solve_node",
                fallback_id,
                "INVALID_REQUEST",
                format!("request does not match the strict schema: {error}"),
            );
            return ExitCode::FAILURE;
        }
    };

    let id = request.id.clone();
    let result = catch_unwind(AssertUnwindSafe(|| solve_node(request)));
    match result {
        Ok(Ok(response)) => match serde_json::to_string(&response) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                write_error_for_operation(
                    "solve_node",
                    Some(id),
                    "SERIALIZATION_ERROR",
                    error.to_string(),
                );
                ExitCode::FAILURE
            }
        },
        Ok(Err(error)) => {
            write_error_for_operation("solve_node", Some(id), error.code, error.message);
            ExitCode::FAILURE
        }
        Err(_) => {
            write_error_for_operation(
                "solve_node",
                Some(id),
                "INTERNAL_PANIC",
                "the solver aborted while processing the request".to_string(),
            );
            ExitCode::FAILURE
        }
    }
}

fn run_path_request(input: &str, fallback_id: Option<String>) -> ExitCode {
    let request: SolvePathRequest = match serde_json::from_str(input) {
        Ok(request) => request,
        Err(error) => {
            write_error_for_operation(
                "solve_path",
                fallback_id,
                "INVALID_REQUEST",
                format!("request does not match the strict schema: {error}"),
            );
            return ExitCode::FAILURE;
        }
    };

    let id = request.id.clone();
    let result = catch_unwind(AssertUnwindSafe(|| solve_path(request)));
    match result {
        Ok(Ok(response)) => match serde_json::to_string(&response) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                write_error_for_operation(
                    "solve_path",
                    Some(id),
                    "SERIALIZATION_ERROR",
                    error.to_string(),
                );
                ExitCode::FAILURE
            }
        },
        Ok(Err(error)) => {
            write_error_for_operation("solve_path", Some(id), error.code, error.message);
            ExitCode::FAILURE
        }
        Err(_) => {
            write_error_for_operation(
                "solve_path",
                Some(id),
                "INTERNAL_PANIC",
                "the solver aborted while processing the request".to_string(),
            );
            ExitCode::FAILURE
        }
    }
}

fn write_error(id: Option<String>, code: &'static str, message: String) {
    write_error_for_operation("solve_root", id, code, message);
}

fn write_error_for_operation(
    operation: &'static str,
    id: Option<String>,
    code: &'static str,
    message: String,
) {
    let response = ErrorResponse {
        schema_version: SCHEMA_VERSION,
        id,
        operation,
        status: "error",
        error: ErrorBody { code, message },
    };
    match serde_json::to_string(&response) {
        Ok(json) => println!("{json}"),
        Err(_) if operation == "solve_node" => println!(
            "{}",
            r#"{"schema_version":1,"id":null,"operation":"solve_node","status":"error","error":{"code":"SERIALIZATION_ERROR","message":"failed to serialize error"}}"#
        ),
        Err(_) => println!(
            "{}",
            r#"{"schema_version":1,"id":null,"operation":"solve_root","status":"error","error":{"code":"SERIALIZATION_ERROR","message":"failed to serialize error"}}"#
        ),
    }
}

fn solve_root(request: SolveRootRequest) -> Result<SolveRootResponse, OracleError> {
    let total_started = Instant::now();
    validate_request(&request)?;
    let memory_limit_bytes = configured_memory_limit_bytes()?;

    let board_cards = parse_board(&request.street, &request.board)?;
    validate_range_weights("oop_range", &request.oop_range)?;
    validate_range_weights("ip_range", &request.ip_range)?;
    let oop_range = request
        .oop_range
        .parse()
        .map_err(|error: String| OracleError::validation(format!("invalid oop_range: {error}")))?;
    let ip_range = request
        .ip_range
        .parse()
        .map_err(|error: String| OracleError::validation(format!("invalid ip_range: {error}")))?;
    let parsed_sizes = parse_bet_sizes(&request.bet_sizes)?;
    let turn_donk_sizes = parse_donk_sizes(
        "turn",
        request.tree_options.turn_donk_sizes.as_deref(),
    )?;
    let river_donk_sizes = parse_donk_sizes(
        "river",
        request.tree_options.river_donk_sizes.as_deref(),
    )?;

    let tree_started = Instant::now();
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [board_cards[0], board_cards[1], board_cards[2]],
        turn: board_cards.get(3).copied().unwrap_or(NOT_DEALT),
        river: board_cards.get(4).copied().unwrap_or(NOT_DEALT),
    };
    let tree_config = TreeConfig {
        initial_state: board_state(request.street),
        starting_pot: request.starting_pot,
        effective_stack: request.effective_stack,
        rake_rate: request.rake.rate_pct / 100.0,
        rake_cap: request.rake.cap,
        flop_bet_sizes: parsed_sizes[0].clone(),
        turn_bet_sizes: parsed_sizes[1].clone(),
        river_bet_sizes: parsed_sizes[2].clone(),
        turn_donk_sizes,
        river_donk_sizes,
        add_allin_threshold: request.tree_options.add_allin_threshold,
        force_allin_threshold: request.tree_options.force_allin_threshold,
        merging_threshold: request.tree_options.merging_threshold,
    };
    let action_tree = ActionTree::new(tree_config).map_err(OracleError::solver)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|error| OracleError::solver(format!("failed to build game: {error}")))?;
    let tree_build_ms = elapsed_ms(tree_started);

    if game.private_cards(0).is_empty() || game.private_cards(1).is_empty() {
        return Err(OracleError::validation(
            "both ranges must contain at least one combo compatible with the board",
        ));
    }

    let (uncompressed_bytes, compressed_bytes) = game.memory_usage();
    let selected_memory_bytes = if request.allocation_mode.is_compressed() {
        compressed_bytes
    } else {
        uncompressed_bytes
    };
    if selected_memory_bytes > memory_limit_bytes {
        return Err(OracleError {
            code: "MEMORY_LIMIT",
            message: format!(
                "estimated {} tree size is {selected_memory_bytes} bytes; configured hard limit is {memory_limit_bytes} bytes",
                request.allocation_mode.as_str()
            ),
        });
    }

    let allocation_started = Instant::now();
    game.allocate_memory(request.allocation_mode.is_compressed());
    let allocation_ms = elapsed_ms(allocation_started);

    let target_fraction = (request.target_exploitability_pct / 100.0) as f32;
    let target_units = request.starting_pot as f32 * target_fraction;
    let solve_started = Instant::now();
    let (iterations, exploitability) = run_solver(
        &mut game,
        request.max_iterations,
        target_units,
    );
    let solve_ms = elapsed_ms(solve_started);

    let extraction_started = Instant::now();
    game.cache_normalized_weights();
    let actions = game.available_actions();
    if game.current_player() != 0 {
        return Err(OracleError::solver(
            "unexpected root player: postflop solve_root must begin with OOP",
        ));
    }
    let action_descriptors = actions
        .iter()
        .enumerate()
        .map(|(index, action)| describe_action(index, *action))
        .collect::<Vec<_>>();
    let players = vec![
        extract_player(&game, 0, Some(actions.len()))?,
        extract_player(&game, 1, None)?,
    ];
    let extraction_ms = elapsed_ms(extraction_started);

    let exploitability_pct = 100.0 * exploitability as f64 / request.starting_pot as f64;
    Ok(SolveRootResponse {
        schema_version: SCHEMA_VERSION,
        id: request.id,
        operation: "solve_root",
        status: "ok",
        provenance: Provenance {
            solver: SolverMetadata {
                name: "b-inary/postflop-solver",
                algorithm: "Discounted CFR",
                commit: SOLVER_COMMIT,
                abstraction: "card-exact; caller-supplied discrete action tree",
                allocation_mode: request.allocation_mode.as_str(),
                memory_hard_limit_bytes: memory_limit_bytes,
            },
            offline_only_acknowledged: request.offline_only_acknowledged,
            owned_simulator_acknowledged: request.owned_simulator_acknowledged,
            effective_request: EffectiveRequest {
                street: request.street,
                board: request.board.clone(),
                oop_range: request.oop_range.clone(),
                ip_range: request.ip_range.clone(),
                starting_pot: request.starting_pot,
                effective_stack: request.effective_stack,
                chip_scale: request.chip_scale,
                chip_unit: request.chip_unit.clone(),
                allocation_mode: request.allocation_mode,
                bet_sizes: request.bet_sizes.clone(),
                rake: request.rake.clone(),
                tree_options: request.tree_options.clone(),
                target_exploitability_pct: request.target_exploitability_pct,
                max_iterations: request.max_iterations,
            },
        },
        street: request.street,
        board: request.board,
        root_player: "OOP",
        root_actions: action_descriptors,
        players,
        convergence: Convergence {
            iterations,
            max_iterations: request.max_iterations,
            target_exploitability_pct: request.target_exploitability_pct,
            target_exploitability_units: target_units as f64,
            exploitability_pct_of_pot: exploitability_pct,
            exploitability_units: exploitability as f64,
            target_reached: exploitability <= target_units,
        },
        memory: MemoryUsage {
            estimated_uncompressed_bytes: uncompressed_bytes,
            estimated_compressed_bytes: compressed_bytes,
            allocation_mode: request.allocation_mode.as_str(),
            hard_limit_bytes: memory_limit_bytes,
        },
        timings_ms: Timings {
            tree_build: tree_build_ms,
            allocation: allocation_ms,
            solve: solve_ms,
            extraction: extraction_ms,
            total: elapsed_ms(total_started),
        },
    })
}

fn solve_node(request: SolveNodeRequest) -> Result<SolveNodeResponse, OracleError> {
    let total_started = Instant::now();
    let (history, expected_node_actions) = validate_node_request(&request)?;
    let memory_limit_bytes = configured_memory_limit_bytes()?;

    let board_cards = parse_board(&request.street, &request.board)?;
    validate_range_weights("oop_range", &request.oop_range)?;
    validate_range_weights("ip_range", &request.ip_range)?;
    let oop_range = request
        .oop_range
        .parse()
        .map_err(|error: String| OracleError::validation(format!("invalid oop_range: {error}")))?;
    let ip_range = request
        .ip_range
        .parse()
        .map_err(|error: String| OracleError::validation(format!("invalid ip_range: {error}")))?;
    let parsed_sizes = parse_bet_sizes(&request.bet_sizes)?;
    let turn_donk_sizes = parse_donk_sizes(
        "turn",
        request.tree_options.turn_donk_sizes.as_deref(),
    )?;
    let river_donk_sizes = parse_donk_sizes(
        "river",
        request.tree_options.river_donk_sizes.as_deref(),
    )?;

    let tree_started = Instant::now();
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [board_cards[0], board_cards[1], board_cards[2]],
        turn: board_cards.get(3).copied().unwrap_or(NOT_DEALT),
        river: board_cards.get(4).copied().unwrap_or(NOT_DEALT),
    };
    let tree_config = TreeConfig {
        initial_state: board_state(request.street),
        starting_pot: request.starting_pot,
        effective_stack: request.effective_stack,
        rake_rate: request.rake.rate_pct / 100.0,
        rake_cap: request.rake.cap,
        flop_bet_sizes: parsed_sizes[0].clone(),
        turn_bet_sizes: parsed_sizes[1].clone(),
        river_bet_sizes: parsed_sizes[2].clone(),
        turn_donk_sizes,
        river_donk_sizes,
        add_allin_threshold: request.tree_options.add_allin_threshold,
        force_allin_threshold: request.tree_options.force_allin_threshold,
        merging_threshold: request.tree_options.merging_threshold,
    };
    let action_tree = ActionTree::new(tree_config).map_err(OracleError::solver)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|error| OracleError::solver(format!("failed to build game: {error}")))?;
    let tree_build_ms = elapsed_ms(tree_started);

    if game.private_cards(0).is_empty() || game.private_cards(1).is_empty() {
        return Err(OracleError::validation(
            "both ranges must contain at least one combo compatible with the board",
        ));
    }

    let (uncompressed_bytes, compressed_bytes) = game.memory_usage();
    let selected_memory_bytes = if request.allocation_mode.is_compressed() {
        compressed_bytes
    } else {
        uncompressed_bytes
    };
    if selected_memory_bytes > memory_limit_bytes {
        return Err(OracleError {
            code: "MEMORY_LIMIT",
            message: format!(
                "estimated {} tree size is {selected_memory_bytes} bytes; configured hard limit is {memory_limit_bytes} bytes",
                request.allocation_mode.as_str()
            ),
        });
    }

    let allocation_started = Instant::now();
    game.allocate_memory(request.allocation_mode.is_compressed());
    let allocation_ms = elapsed_ms(allocation_started);

    let target_fraction = (request.target_exploitability_pct / 100.0) as f32;
    let target_units = request.starting_pot as f32 * target_fraction;
    let solve_started = Instant::now();
    let (iterations, exploitability) =
        run_solver(&mut game, request.max_iterations, target_units);
    let solve_ms = elapsed_ms(solve_started);

    let extraction_started = Instant::now();
    let input_range_weights = [game.weights(0).to_vec(), game.weights(1).to_vec()];
    traverse_same_street_history(&mut game, &history)?;

    if game.is_terminal_node() {
        return Err(OracleError {
            code: "NODE_PATH_ERROR",
            message: "action_history ends at a terminal node".to_string(),
        });
    }
    if game.is_chance_node() {
        return Err(OracleError {
            code: "NODE_PATH_ERROR",
            message: "action_history advances to another street".to_string(),
        });
    }

    let current_player = game.current_player();
    if current_player != request.expected_current_player.index() {
        return Err(OracleError {
            code: "NODE_MISMATCH",
            message: format!(
                "current player mismatch: expected {}, solver reached {}",
                request.expected_current_player.as_str(),
                player_name(current_player)
            ),
        });
    }

    let total_bet_amount = game.total_bet_amount();
    let facing_bet = total_bet_amount[current_player ^ 1]
        .checked_sub(total_bet_amount[current_player])
        .ok_or_else(|| OracleError {
            code: "NODE_MISMATCH",
            message: "current player has contributed more than the opponent".to_string(),
        })?;
    if facing_bet != request.expected_facing_bet {
        return Err(OracleError {
            code: "NODE_MISMATCH",
            message: format!(
                "facing bet mismatch: expected {}, solver reached {facing_bet}",
                request.expected_facing_bet
            ),
        });
    }

    let node_actions = game.available_actions();
    if !same_action_set(&node_actions, &expected_node_actions) {
        return Err(OracleError {
            code: "NODE_MISMATCH",
            message: format!(
                "node actions mismatch: expected {expected_node_actions:?}, solver reached {node_actions:?}"
            ),
        });
    }
    let action_descriptors = node_actions
        .iter()
        .enumerate()
        .map(|(index, action)| describe_action(index, *action))
        .collect::<Vec<_>>();

    game.cache_normalized_weights();
    let (node_total_reachable_weight, policies) = extract_node_policies(
        &game,
        current_player,
        &input_range_weights[current_player],
        node_actions.len(),
    )?;
    let extraction_ms = elapsed_ms(extraction_started);

    let exploitability_pct = 100.0 * exploitability as f64 / request.starting_pot as f64;
    Ok(SolveNodeResponse {
        schema_version: SCHEMA_VERSION,
        id: request.id,
        operation: "solve_node",
        status: "ok",
        provenance: NodeProvenance {
            solver: SolverMetadata {
                name: "b-inary/postflop-solver",
                algorithm: "Discounted CFR",
                commit: SOLVER_COMMIT,
                abstraction: "card-exact; caller-supplied discrete action tree",
                allocation_mode: request.allocation_mode.as_str(),
                memory_hard_limit_bytes: memory_limit_bytes,
            },
            offline_only_acknowledged: request.offline_only_acknowledged,
            owned_simulator_acknowledged: request.owned_simulator_acknowledged,
            effective_request: EffectiveNodeRequest {
                street: request.street,
                board: request.board.clone(),
                oop_range: request.oop_range.clone(),
                ip_range: request.ip_range.clone(),
                starting_pot: request.starting_pot,
                effective_stack: request.effective_stack,
                chip_scale: request.chip_scale,
                chip_unit: request.chip_unit.clone(),
                allocation_mode: request.allocation_mode,
                bet_sizes: request.bet_sizes.clone(),
                rake: request.rake.clone(),
                tree_options: request.tree_options.clone(),
                target_exploitability_pct: request.target_exploitability_pct,
                max_iterations: request.max_iterations,
                action_history: request.action_history.clone(),
                expected_current_player: request.expected_current_player,
                expected_facing_bet: request.expected_facing_bet,
                expected_node_actions: request.expected_node_actions.clone(),
            },
        },
        street: request.street,
        board: request.board,
        current_player: player_name(current_player),
        node_actions: action_descriptors,
        node_total_reachable_weight,
        policies,
        convergence: Convergence {
            iterations,
            max_iterations: request.max_iterations,
            target_exploitability_pct: request.target_exploitability_pct,
            target_exploitability_units: target_units as f64,
            exploitability_pct_of_pot: exploitability_pct,
            exploitability_units: exploitability as f64,
            target_reached: exploitability <= target_units,
        },
        memory: MemoryUsage {
            estimated_uncompressed_bytes: uncompressed_bytes,
            estimated_compressed_bytes: compressed_bytes,
            allocation_mode: request.allocation_mode.as_str(),
            hard_limit_bytes: memory_limit_bytes,
        },
        timings_ms: Timings {
            tree_build: tree_build_ms,
            allocation: allocation_ms,
            solve: solve_ms,
            extraction: extraction_ms,
            total: elapsed_ms(total_started),
        },
    })
}

fn solve_path(request: SolvePathRequest) -> Result<SolvePathResponse, OracleError> {
    let total_started = Instant::now();
    let (path, expected_node_actions) = validate_path_request(&request)?;
    let memory_limit_bytes = configured_memory_limit_bytes()?;

    let board_cards = parse_board(&request.street, &request.board)?;
    validate_range_weights("oop_range", &request.oop_range)?;
    validate_range_weights("ip_range", &request.ip_range)?;
    let oop_range = request
        .oop_range
        .parse()
        .map_err(|error: String| OracleError::validation(format!("invalid oop_range: {error}")))?;
    let ip_range = request
        .ip_range
        .parse()
        .map_err(|error: String| OracleError::validation(format!("invalid ip_range: {error}")))?;
    let parsed_sizes = parse_bet_sizes(&request.bet_sizes)?;
    let turn_donk_sizes = parse_donk_sizes(
        "turn",
        request.tree_options.turn_donk_sizes.as_deref(),
    )?;
    let river_donk_sizes = parse_donk_sizes(
        "river",
        request.tree_options.river_donk_sizes.as_deref(),
    )?;

    let tree_started = Instant::now();
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [board_cards[0], board_cards[1], board_cards[2]],
        // A continuation solve must not know future public cards at the flop
        // root. Turn and river are traversed explicitly as chance steps.
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: request.starting_pot,
        effective_stack: request.effective_stack,
        rake_rate: request.rake.rate_pct / 100.0,
        rake_cap: request.rake.cap,
        flop_bet_sizes: parsed_sizes[0].clone(),
        turn_bet_sizes: parsed_sizes[1].clone(),
        river_bet_sizes: parsed_sizes[2].clone(),
        turn_donk_sizes,
        river_donk_sizes,
        add_allin_threshold: request.tree_options.add_allin_threshold,
        force_allin_threshold: request.tree_options.force_allin_threshold,
        merging_threshold: request.tree_options.merging_threshold,
    };
    let mut action_tree = ActionTree::new(tree_config).map_err(OracleError::solver)?;
    add_observed_path_actions(&mut action_tree, &path)?;
    action_tree.back_to_root();
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|error| OracleError::solver(format!("failed to build game: {error}")))?;
    let tree_build_ms = elapsed_ms(tree_started);

    if game.private_cards(0).is_empty() || game.private_cards(1).is_empty() {
        return Err(OracleError::validation(
            "both ranges must contain at least one combo compatible with the flop",
        ));
    }

    let (uncompressed_bytes, compressed_bytes) = game.memory_usage();
    let selected_memory_bytes = if request.allocation_mode.is_compressed() {
        compressed_bytes
    } else {
        uncompressed_bytes
    };
    if selected_memory_bytes > memory_limit_bytes {
        return Err(OracleError {
            code: "MEMORY_LIMIT",
            message: format!(
                "estimated {} tree size is {selected_memory_bytes} bytes; configured hard limit is {memory_limit_bytes} bytes",
                request.allocation_mode.as_str()
            ),
        });
    }

    let allocation_started = Instant::now();
    game.allocate_memory(request.allocation_mode.is_compressed());
    let allocation_ms = elapsed_ms(allocation_started);

    let target_fraction = (request.target_exploitability_pct / 100.0) as f32;
    let target_units = request.starting_pot as f32 * target_fraction;
    let solve_started = Instant::now();
    let (iterations, exploitability) =
        run_solver(&mut game, request.max_iterations, target_units);
    let solve_ms = elapsed_ms(solve_started);

    let extraction_started = Instant::now();
    let input_range_weights = [game.weights(0).to_vec(), game.weights(1).to_vec()];
    traverse_full_path(&mut game, &path)?;

    if game.is_terminal_node() {
        return Err(OracleError {
            code: "NODE_PATH_ERROR",
            message: "path_history ends at a terminal node".to_string(),
        });
    }
    if game.is_chance_node() {
        return Err(OracleError {
            code: "NODE_PATH_ERROR",
            message: "path_history stops before a required board card".to_string(),
        });
    }

    let current_board = game
        .current_board()
        .into_iter()
        .map(card_to_text)
        .collect::<Vec<_>>();
    if current_board != request.expected_board {
        return Err(OracleError {
            code: "NODE_MISMATCH",
            message: format!(
                "current board mismatch: expected {:?}, solver reached {:?}",
                request.expected_board, current_board
            ),
        });
    }

    let current_player = game.current_player();
    if current_player != request.expected_current_player.index() {
        return Err(OracleError {
            code: "NODE_MISMATCH",
            message: format!(
                "current player mismatch: expected {}, solver reached {}",
                request.expected_current_player.as_str(),
                player_name(current_player)
            ),
        });
    }

    let total_bet_amount = game.total_bet_amount();
    if total_bet_amount != request.expected_total_invested {
        return Err(OracleError {
            code: "NODE_MISMATCH",
            message: format!(
                "total postflop investment mismatch: expected {:?}, solver reached {:?}",
                request.expected_total_invested, total_bet_amount
            ),
        });
    }
    let facing_bet = total_bet_amount[current_player ^ 1]
        .checked_sub(total_bet_amount[current_player])
        .ok_or_else(|| OracleError {
            code: "NODE_MISMATCH",
            message: "current player has invested more than the opponent".to_string(),
        })?;
    if facing_bet != request.expected_facing_bet {
        return Err(OracleError {
            code: "NODE_MISMATCH",
            message: format!(
                "facing bet mismatch: expected {}, solver reached {facing_bet}",
                request.expected_facing_bet
            ),
        });
    }

    let node_actions = game.available_actions();
    if !same_action_set(&node_actions, &expected_node_actions) {
        return Err(OracleError {
            code: "NODE_MISMATCH",
            message: format!(
                "node actions mismatch: expected {expected_node_actions:?}, solver reached {node_actions:?}"
            ),
        });
    }
    let action_descriptors = node_actions
        .iter()
        .enumerate()
        .map(|(index, action)| describe_action(index, *action))
        .collect::<Vec<_>>();

    game.cache_normalized_weights();
    let (node_total_reachable_weight, policies) = extract_node_policies(
        &game,
        current_player,
        &input_range_weights[current_player],
        node_actions.len(),
    )?;
    let conditional_ranges = vec![
        extract_conditional_range(&game, 0, &input_range_weights[0])?,
        extract_conditional_range(&game, 1, &input_range_weights[1])?,
    ];
    let extraction_ms = elapsed_ms(extraction_started);

    let current_street = match current_board.len() {
        3 => Street::Flop,
        4 => Street::Turn,
        5 => Street::River,
        _ => {
            return Err(OracleError::solver(
                "solver returned an impossible current-board length",
            ))
        }
    };
    let exploitability_pct = 100.0 * exploitability as f64 / request.starting_pot as f64;
    Ok(SolvePathResponse {
        schema_version: SCHEMA_VERSION,
        id: request.id,
        operation: "solve_path",
        status: "ok",
        provenance: PathProvenance {
            solver: SolverMetadata {
                name: "b-inary/postflop-solver",
                algorithm: "Discounted CFR",
                commit: SOLVER_COMMIT,
                abstraction: "card-exact; caller-supplied discrete action tree",
                allocation_mode: request.allocation_mode.as_str(),
                memory_hard_limit_bytes: memory_limit_bytes,
            },
            offline_only_acknowledged: request.offline_only_acknowledged,
            owned_simulator_acknowledged: request.owned_simulator_acknowledged,
            effective_request: EffectivePathRequest {
                street: request.street,
                board: request.board.clone(),
                oop_range: request.oop_range.clone(),
                ip_range: request.ip_range.clone(),
                starting_pot: request.starting_pot,
                effective_stack: request.effective_stack,
                chip_scale: request.chip_scale,
                chip_unit: request.chip_unit.clone(),
                allocation_mode: request.allocation_mode,
                bet_sizes: request.bet_sizes.clone(),
                rake: request.rake.clone(),
                tree_options: request.tree_options.clone(),
                target_exploitability_pct: request.target_exploitability_pct,
                max_iterations: request.max_iterations,
                path_history: request.path_history.clone(),
                expected_board: request.expected_board.clone(),
                expected_total_invested: request.expected_total_invested,
                expected_current_player: request.expected_current_player,
                expected_facing_bet: request.expected_facing_bet,
                expected_node_actions: request.expected_node_actions.clone(),
            },
        },
        current_street,
        current_board,
        current_player: player_name(current_player),
        node_actions: action_descriptors,
        node_total_reachable_weight,
        policies,
        conditional_ranges,
        convergence: Convergence {
            iterations,
            max_iterations: request.max_iterations,
            target_exploitability_pct: request.target_exploitability_pct,
            target_exploitability_units: target_units as f64,
            exploitability_pct_of_pot: exploitability_pct,
            exploitability_units: exploitability as f64,
            target_reached: exploitability <= target_units,
        },
        memory: MemoryUsage {
            estimated_uncompressed_bytes: uncompressed_bytes,
            estimated_compressed_bytes: compressed_bytes,
            allocation_mode: request.allocation_mode.as_str(),
            hard_limit_bytes: memory_limit_bytes,
        },
        timings_ms: Timings {
            tree_build: tree_build_ms,
            allocation: allocation_ms,
            solve: solve_ms,
            extraction: extraction_ms,
            total: elapsed_ms(total_started),
        },
    })
}

fn validate_path_request(
    request: &SolvePathRequest,
) -> Result<(Vec<ResolvedPathStep>, Vec<Action>), OracleError> {
    if request.operation != Operation::SolvePath {
        return Err(OracleError::validation("operation must be solve_path"));
    }
    if request.street != Street::Flop {
        return Err(OracleError::validation(
            "solve_path must begin from the FLOP root",
        ));
    }

    let root_equivalent = SolveRootRequest {
        schema_version: request.schema_version,
        id: request.id.clone(),
        operation: Operation::SolveRoot,
        offline_only_acknowledged: request.offline_only_acknowledged,
        owned_simulator_acknowledged: request.owned_simulator_acknowledged,
        street: request.street,
        board: request.board.clone(),
        oop_range: request.oop_range.clone(),
        ip_range: request.ip_range.clone(),
        starting_pot: request.starting_pot,
        effective_stack: request.effective_stack,
        chip_scale: request.chip_scale,
        chip_unit: request.chip_unit.clone(),
        allocation_mode: request.allocation_mode,
        bet_sizes: request.bet_sizes.clone(),
        rake: request.rake.clone(),
        tree_options: request.tree_options.clone(),
        target_exploitability_pct: request.target_exploitability_pct,
        max_iterations: request.max_iterations,
    };
    validate_request(&root_equivalent)?;

    if request.path_history.len() > 256 {
        return Err(OracleError::validation(
            "path_history cannot exceed 256 steps",
        ));
    }
    if !(3..=5).contains(&request.expected_board.len()) {
        return Err(OracleError::validation(
            "expected_board must contain 3, 4, or 5 cards",
        ));
    }
    let current_street = match request.expected_board.len() {
        3 => Street::Flop,
        4 => Street::Turn,
        5 => Street::River,
        _ => unreachable!(),
    };
    parse_board(&current_street, &request.expected_board)?;
    if request.expected_board[..3] != request.board {
        return Err(OracleError::validation(
            "expected_board must preserve the request flop",
        ));
    }

    let mut path = Vec::with_capacity(request.path_history.len());
    let mut dealt_cards = Vec::new();
    for (index, step) in request.path_history.iter().enumerate() {
        match step {
            WirePathStep::Action { action } => {
                path.push(ResolvedPathStep::Action(action.to_solver_action(
                    &format!("path_history[{index}].action"),
                    request.effective_stack,
                )?));
            }
            WirePathStep::Deal { card } => {
                let parsed = card_from_str(card).map_err(|error| {
                    OracleError::validation(format!(
                        "path_history[{index}].card is invalid: {error}"
                    ))
                })?;
                dealt_cards.push(card.clone());
                path.push(ResolvedPathStep::Deal(parsed));
            }
        }
    }
    if dealt_cards != request.expected_board[3..] {
        return Err(OracleError::validation(
            "path deal cards must exactly equal expected_board after the flop",
        ));
    }

    if request.expected_total_invested.iter().any(|&value| {
        value < 0 || value > request.effective_stack
    }) {
        return Err(OracleError::validation(format!(
            "expected_total_invested values must be in [0, {}]",
            request.effective_stack
        )));
    }
    let actor = request.expected_current_player.index();
    let implied_facing = request.expected_total_invested[actor ^ 1]
        .checked_sub(request.expected_total_invested[actor])
        .ok_or_else(|| {
            OracleError::validation(
                "expected current player cannot have invested more than the opponent",
            )
        })?;
    if request.expected_facing_bet != implied_facing {
        return Err(OracleError::validation(format!(
            "expected_facing_bet must equal the investment difference {implied_facing}"
        )));
    }
    if request.expected_node_actions.is_empty() {
        return Err(OracleError::validation(
            "expected_node_actions cannot be empty",
        ));
    }
    let expected_actions = request
        .expected_node_actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            action.to_solver_action(
                &format!("expected_node_actions[{index}]"),
                request.effective_stack,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut unique_actions = expected_actions.clone();
    unique_actions.sort_unstable();
    unique_actions.dedup();
    if unique_actions.len() != expected_actions.len() {
        return Err(OracleError::validation(
            "expected_node_actions cannot contain duplicates",
        ));
    }

    Ok((path, expected_actions))
}

fn add_observed_path_actions(
    tree: &mut ActionTree,
    path: &[ResolvedPathStep],
) -> Result<(), OracleError> {
    for (step, item) in path.iter().enumerate() {
        match item {
            ResolvedPathStep::Deal(_) => {
                // ActionTree intentionally abstracts chance cards. The actual
                // PostFlopGame traversal below validates the exact deal order.
            }
            ResolvedPathStep::Action(action) => {
                if tree.is_terminal_node() {
                    return Err(OracleError {
                        code: "NODE_PATH_ERROR",
                        message: format!(
                            "path_history[{step}] follows a terminal action-tree node"
                        ),
                    });
                }
                if !tree.available_actions().contains(action) {
                    if !matches!(
                        action,
                        Action::Bet(_) | Action::Raise(_) | Action::AllIn(_)
                    ) {
                        return Err(OracleError {
                            code: "NODE_PATH_ERROR",
                            message: format!(
                                "path_history[{step}] {action:?} is unavailable in the action tree"
                            ),
                        });
                    }
                    tree.add_action(*action).map_err(|error| OracleError {
                        code: "NODE_PATH_ERROR",
                        message: format!(
                            "cannot add exact observed action at path_history[{step}]: {error}"
                        ),
                    })?;
                }
                tree.play(*action).map_err(|error| OracleError {
                    code: "NODE_PATH_ERROR",
                    message: format!(
                        "cannot traverse action tree at path_history[{step}]: {error}"
                    ),
                })?;
            }
        }
    }
    Ok(())
}

fn traverse_full_path(
    game: &mut PostFlopGame,
    path: &[ResolvedPathStep],
) -> Result<(), OracleError> {
    for (step, item) in path.iter().enumerate() {
        if game.is_terminal_node() {
            return Err(OracleError {
                code: "NODE_PATH_ERROR",
                message: format!("path_history[{step}] follows a terminal node"),
            });
        }
        match item {
            ResolvedPathStep::Deal(card) => {
                if !game.is_chance_node() {
                    return Err(OracleError {
                        code: "NODE_PATH_ERROR",
                        message: format!(
                            "path_history[{step}] deals a card before the betting round closed"
                        ),
                    });
                }
                if game.possible_cards() & (1u64 << *card) == 0 {
                    return Err(OracleError {
                        code: "NODE_PATH_ERROR",
                        message: format!(
                            "path_history[{step}] deal {} is unavailable",
                            card_to_text(*card)
                        ),
                    });
                }
                game.play(*card as usize);
            }
            ResolvedPathStep::Action(requested_action) => {
                if game.is_chance_node() {
                    return Err(OracleError {
                        code: "NODE_PATH_ERROR",
                        message: format!(
                            "path_history[{step}] omits the required turn or river card"
                        ),
                    });
                }
                let available_actions = game.available_actions();
                let action_index = available_actions
                    .iter()
                    .position(|action| action == requested_action)
                    .ok_or_else(|| OracleError {
                        code: "NODE_PATH_ERROR",
                        message: format!(
                            "path_history[{step}] {requested_action:?} is unavailable; exact actions are {available_actions:?}"
                        ),
                    })?;
                game.play(action_index);
            }
        }
    }
    Ok(())
}

fn card_to_text(card: u8) -> String {
    const RANKS: &[u8; 13] = b"23456789TJQKA";
    const SUITS: &[u8; 4] = b"cdhs";
    let rank = RANKS[(card / 4) as usize] as char;
    let suit = SUITS[(card & 3) as usize] as char;
    format!("{rank}{suit}")
}

fn validate_node_request(
    request: &SolveNodeRequest,
) -> Result<(Vec<Action>, Vec<Action>), OracleError> {
    if request.operation != Operation::SolveNode {
        return Err(OracleError::validation("operation must be solve_node"));
    }

    let root_equivalent = SolveRootRequest {
        schema_version: request.schema_version,
        id: request.id.clone(),
        operation: Operation::SolveRoot,
        offline_only_acknowledged: request.offline_only_acknowledged,
        owned_simulator_acknowledged: request.owned_simulator_acknowledged,
        street: request.street,
        board: request.board.clone(),
        oop_range: request.oop_range.clone(),
        ip_range: request.ip_range.clone(),
        starting_pot: request.starting_pot,
        effective_stack: request.effective_stack,
        chip_scale: request.chip_scale,
        chip_unit: request.chip_unit.clone(),
        allocation_mode: request.allocation_mode,
        bet_sizes: request.bet_sizes.clone(),
        rake: request.rake.clone(),
        tree_options: request.tree_options.clone(),
        target_exploitability_pct: request.target_exploitability_pct,
        max_iterations: request.max_iterations,
    };
    validate_request(&root_equivalent)?;

    if request.expected_facing_bet < 0
        || request.expected_facing_bet > request.effective_stack
    {
        return Err(OracleError::validation(format!(
            "expected_facing_bet must be in [0, {}]",
            request.effective_stack
        )));
    }
    if request.expected_node_actions.is_empty() {
        return Err(OracleError::validation(
            "expected_node_actions cannot be empty",
        ));
    }

    let history = request
        .action_history
        .iter()
        .enumerate()
        .map(|(index, action)| {
            action.to_solver_action(
                &format!("action_history[{index}]"),
                request.effective_stack,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected_actions = request
        .expected_node_actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            action.to_solver_action(
                &format!("expected_node_actions[{index}]"),
                request.effective_stack,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (implied_player, implied_facing_bet) = match history.as_slice() {
        [Action::Check] => (NodePlayer::Ip, 0),
        [Action::Bet(amount)] | [Action::AllIn(amount)] => (NodePlayer::Ip, *amount),
        [Action::Check, Action::Bet(amount)]
        | [Action::Check, Action::AllIn(amount)] => (NodePlayer::Oop, *amount),
        _ => {
            return Err(OracleError {
                code: "NODE_PATH_ERROR",
                message: "action_history must be exactly [CHECK], [BET|ALL_IN], or [CHECK, BET|ALL_IN]"
                    .to_string(),
            })
        }
    };
    if request.expected_current_player != implied_player {
        return Err(OracleError {
            code: "NODE_MISMATCH",
            message: format!(
                "expected_current_player must be {} for this action_history",
                implied_player.as_str()
            ),
        });
    }
    if request.expected_facing_bet != implied_facing_bet {
        return Err(OracleError {
            code: "NODE_MISMATCH",
            message: format!(
                "expected_facing_bet must be {implied_facing_bet} for this action_history"
            ),
        });
    }

    let mut unique_actions = expected_actions.clone();
    unique_actions.sort_unstable();
    unique_actions.dedup();
    if unique_actions.len() != expected_actions.len() {
        return Err(OracleError::validation(
            "expected_node_actions cannot contain duplicates",
        ));
    }

    Ok((history, expected_actions))
}

fn traverse_same_street_history(
    game: &mut PostFlopGame,
    history: &[Action],
) -> Result<(), OracleError> {
    for (step, requested_action) in history.iter().enumerate() {
        if game.is_terminal_node() {
            return Err(OracleError {
                code: "NODE_PATH_ERROR",
                message: format!("action_history[{step}] follows a terminal node"),
            });
        }
        if game.is_chance_node() {
            return Err(OracleError {
                code: "NODE_PATH_ERROR",
                message: format!("action_history[{step}] crosses a street boundary"),
            });
        }
        let available_actions = game.available_actions();
        let action_index = available_actions
            .iter()
            .position(|action| action == requested_action)
            .ok_or_else(|| OracleError {
                code: "NODE_PATH_ERROR",
                message: format!(
                    "action_history[{step}] {requested_action:?} is unavailable; exact actions are {available_actions:?}"
                ),
            })?;
        game.play(action_index);
    }
    Ok(())
}

fn same_action_set(actual: &[Action], expected: &[Action]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    let mut actual = actual.to_vec();
    let mut expected = expected.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    actual == expected
}

fn extract_node_policies(
    game: &PostFlopGame,
    player: usize,
    input_range_weights: &[f32],
    action_count: usize,
) -> Result<(f64, Vec<NodePolicy>), OracleError> {
    let cards = game.private_cards(player);
    let path_weights = game.weights(player);
    let joint_weights = game.normalized_weights(player);
    let equity = game.equity(player);
    let equilibrium_evs = game.expected_values(player);
    let strategy = game.strategy();
    let action_evs = game.expected_values_detail(player);
    let hand_count = cards.len();

    if input_range_weights.len() != hand_count
        || path_weights.len() != hand_count
        || joint_weights.len() != hand_count
        || equity.len() != hand_count
        || equilibrium_evs.len() != hand_count
        || strategy.len() != action_count * hand_count
        || action_evs.len() != action_count * hand_count
    {
        return Err(OracleError::solver(
            "solver returned inconsistent current-node vector lengths",
        ));
    }

    let total_reachable_weight = joint_weights
        .iter()
        .filter(|weight| **weight > 0.0)
        .map(|weight| *weight as f64)
        .sum::<f64>();
    if total_reachable_weight <= 0.0 || !total_reachable_weight.is_finite() {
        return Err(OracleError {
            code: "UNREACHABLE_NODE",
            message: "the current actor has no positive-reach combos at this node".to_string(),
        });
    }

    let mut policies = Vec::new();
    for index in 0..hand_count {
        let joint_weight = joint_weights[index] as f64;
        if joint_weight <= 0.0 {
            continue;
        }
        let hand = hole_to_string(cards[index])
            .map_err(|error| OracleError::solver(format!("failed to format combo: {error}")))?;
        policies.push(NodePolicy {
            hand,
            input_range_weight: input_range_weights[index] as f64,
            path_weight: path_weights[index] as f64,
            joint_compatible_weight: joint_weight,
            conditional_reach_weight: joint_weight / total_reachable_weight,
            equity: equity[index] as f64,
            equilibrium_ev_units: equilibrium_evs[index] as f64,
            node_action_frequencies: (0..action_count)
                .map(|action| strategy[action * hand_count + index] as f64)
                .collect(),
            node_action_evs_units: (0..action_count)
                .map(|action| action_evs[action * hand_count + index] as f64)
                .collect(),
        });
    }

    Ok((total_reachable_weight, policies))
}

fn extract_conditional_range(
    game: &PostFlopGame,
    player: usize,
    input_range_weights: &[f32],
) -> Result<ConditionalPlayerRange, OracleError> {
    let cards = game.private_cards(player);
    let path_weights = game.weights(player);
    let joint_weights = game.normalized_weights(player);
    if input_range_weights.len() != cards.len()
        || path_weights.len() != cards.len()
        || joint_weights.len() != cards.len()
    {
        return Err(OracleError::solver(
            "solver returned inconsistent conditional-range vector lengths",
        ));
    }
    let total_joint_compatible_weight = joint_weights
        .iter()
        .filter(|weight| **weight > 0.0)
        .map(|weight| *weight as f64)
        .sum::<f64>();
    if total_joint_compatible_weight <= 0.0
        || !total_joint_compatible_weight.is_finite()
    {
        return Err(OracleError {
            code: "UNREACHABLE_NODE",
            message: format!(
                "{} has no positive conditional range at this node",
                player_name(player)
            ),
        });
    }

    let mut combos = Vec::new();
    for index in 0..cards.len() {
        let joint_weight = joint_weights[index] as f64;
        if joint_weight <= 0.0 {
            continue;
        }
        let hand = hole_to_string(cards[index])
            .map_err(|error| OracleError::solver(format!("failed to format combo: {error}")))?;
        combos.push(ConditionalComboWeight {
            hand,
            input_range_weight: input_range_weights[index] as f64,
            path_weight: path_weights[index] as f64,
            joint_compatible_weight: joint_weight,
            conditional_reach_weight: joint_weight / total_joint_compatible_weight,
        });
    }
    Ok(ConditionalPlayerRange {
        player: player_name(player),
        total_joint_compatible_weight,
        combos,
    })
}

fn validate_request(request: &SolveRootRequest) -> Result<(), OracleError> {
    if request.schema_version != SCHEMA_VERSION {
        return Err(OracleError {
            code: "UNSUPPORTED_SCHEMA_VERSION",
            message: format!(
                "schema_version must be {SCHEMA_VERSION}, got {}",
                request.schema_version
            ),
        });
    }
    if !matches!(request.operation, Operation::SolveRoot) {
        return Err(OracleError::validation("operation must be solve_root"));
    }
    if !request.offline_only_acknowledged && !request.owned_simulator_acknowledged {
        return Err(OracleError {
            code: "EXECUTION_CONTEXT_ACK_REQUIRED",
            message: "exactly one of offline_only_acknowledged and owned_simulator_acknowledged must be true"
                .to_string(),
        });
    }
    if request.offline_only_acknowledged && request.owned_simulator_acknowledged {
        return Err(OracleError {
            code: "EXECUTION_CONTEXT_CONFLICT",
            message: "offline_only_acknowledged and owned_simulator_acknowledged cannot both be true"
                .to_string(),
        });
    }
    if request.id.trim().is_empty() || request.id.len() > 256 {
        return Err(OracleError::validation(
            "id must contain between 1 and 256 bytes",
        ));
    }
    if request.oop_range.is_empty() || request.oop_range.len() > MAX_TEXT_FIELD_BYTES {
        return Err(OracleError::validation("oop_range has an invalid length"));
    }
    if request.ip_range.is_empty() || request.ip_range.len() > MAX_TEXT_FIELD_BYTES {
        return Err(OracleError::validation("ip_range has an invalid length"));
    }
    if request.starting_pot <= 0 || request.starting_pot > MAX_CHIP_UNITS {
        return Err(OracleError::validation(format!(
            "starting_pot must be between 1 and {MAX_CHIP_UNITS}"
        )));
    }
    if request.effective_stack <= 0 || request.effective_stack > MAX_CHIP_UNITS {
        return Err(OracleError::validation(format!(
            "effective_stack must be between 1 and {MAX_CHIP_UNITS}"
        )));
    }
    if request.chip_scale == 0 || request.chip_scale > MAX_CHIP_SCALE {
        return Err(OracleError::validation(format!(
            "chip_scale must be between 1 and {MAX_CHIP_SCALE}"
        )));
    }
    if request.chip_unit.trim().is_empty()
        || request.chip_unit.len() > 64
        || request.chip_unit.chars().any(char::is_control)
    {
        return Err(OracleError::validation(
            "chip_unit must be a non-empty, control-free label of at most 64 bytes",
        ));
    }
    if request.max_iterations == 0 || request.max_iterations > MAX_ITERATIONS {
        return Err(OracleError::validation(format!(
            "max_iterations must be between 1 and {MAX_ITERATIONS}"
        )));
    }
    if !request.target_exploitability_pct.is_finite()
        || request.target_exploitability_pct <= 0.0
        || request.target_exploitability_pct > 100.0
    {
        return Err(OracleError::validation(
            "target_exploitability_pct must be finite and in (0, 100]",
        ));
    }
    validate_f32_representable(
        "target_exploitability_pct",
        request.target_exploitability_pct,
    )?;
    validate_f32_representable(
        "target_exploitability_pct / 100",
        request.target_exploitability_pct / 100.0,
    )?;
    if !request.rake.rate_pct.is_finite()
        || request.rake.rate_pct < 0.0
        || request.rake.rate_pct > 100.0
    {
        return Err(OracleError::validation(
            "rake.rate_pct must be finite and in [0, 100]",
        ));
    }
    validate_f32_representable("rake.rate_pct", request.rake.rate_pct)?;
    validate_f32_representable("rake.rate_pct / 100", request.rake.rate_pct / 100.0)?;
    if !request.rake.cap.is_finite() || request.rake.cap < 0.0 {
        return Err(OracleError::validation(
            "rake.cap must be finite and non-negative",
        ));
    }
    validate_f32_representable("rake.cap", request.rake.cap)?;
    validate_nonnegative_finite(
        "tree_options.add_allin_threshold",
        request.tree_options.add_allin_threshold,
    )?;
    validate_nonnegative_finite(
        "tree_options.force_allin_threshold",
        request.tree_options.force_allin_threshold,
    )?;
    validate_nonnegative_finite(
        "tree_options.merging_threshold",
        request.tree_options.merging_threshold,
    )?;
    validate_f32_representable(
        "tree_options.add_allin_threshold",
        request.tree_options.add_allin_threshold,
    )?;
    validate_f32_representable(
        "tree_options.force_allin_threshold",
        request.tree_options.force_allin_threshold,
    )?;
    validate_f32_representable(
        "tree_options.merging_threshold",
        request.tree_options.merging_threshold,
    )?;
    Ok(())
}

fn validate_nonnegative_finite(field: &str, value: f64) -> Result<(), OracleError> {
    if !value.is_finite() || value < 0.0 {
        Err(OracleError::validation(format!(
            "{field} must be finite and non-negative"
        )))
    } else {
        Ok(())
    }
}

fn validate_f32_representable(field: &str, value: f64) -> Result<f32, OracleError> {
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err(OracleError::validation(format!(
            "{field} is not representable as finite f32"
        )));
    }
    if value != 0.0 && narrowed == 0.0 {
        return Err(OracleError::validation(format!(
            "{field} underflows to zero when represented as f32"
        )));
    }
    Ok(narrowed)
}

fn validate_range_weights(field: &str, range_text: &str) -> Result<(), OracleError> {
    for group in range_text.split(',') {
        let group = group.trim();
        if group.is_empty() {
            continue;
        }
        let mut parts = group.split(':');
        let _hand = parts.next();
        let Some(weight_text) = parts.next() else {
            continue;
        };
        if parts.next().is_some() {
            return Err(OracleError::validation(format!(
                "{field} contains more than one weight separator in {group:?}"
            )));
        }
        let weight_text = weight_text.trim();
        let parsed = weight_text.parse::<f64>().map_err(|_| {
            OracleError::validation(format!(
                "{field} contains an invalid weight {weight_text:?}"
            ))
        })?;
        if !parsed.is_finite() {
            return Err(OracleError::validation(format!(
                "{field} weight {weight_text:?} must be finite"
            )));
        }
        if !pio_weight_syntax(weight_text) {
            return Err(OracleError::validation(format!(
                "{field} weight {weight_text:?} does not use Pio decimal syntax"
            )));
        }
        if !(0.0..=1.0).contains(&parsed) {
            return Err(OracleError::validation(format!(
                "{field} weight {weight_text:?} must be in [0, 1]"
            )));
        }
        let lexical_nonzero = weight_text.bytes().any(|byte| matches!(byte, b'1'..=b'9'));
        if lexical_nonzero && parsed == 0.0 {
            return Err(OracleError::validation(format!(
                "{field} weight {weight_text:?} underflows to zero as f64"
            )));
        }
        let narrowed = parsed as f32;
        if !narrowed.is_finite() {
            return Err(OracleError::validation(format!(
                "{field} weight {weight_text:?} is not representable as finite f32"
            )));
        }
        if lexical_nonzero && narrowed == 0.0 {
            return Err(OracleError::validation(format!(
                "{field} weight {weight_text:?} underflows to zero as f32"
            )));
        }
    }
    Ok(())
}

fn pio_weight_syntax(value: &str) -> bool {
    if let Some(fraction) = value.strip_prefix('.') {
        return !fraction.is_empty() && fraction.bytes().all(|byte| byte.is_ascii_digit());
    }
    let mut chars = value.chars();
    if !matches!(chars.next(), Some('0' | '1')) {
        return false;
    }
    match chars.next() {
        None => true,
        Some('.') => chars.all(|character| character.is_ascii_digit()),
        Some(_) => false,
    }
}

fn parse_board(street: &Street, cards: &[String]) -> Result<Vec<u8>, OracleError> {
    let expected_len = match street {
        Street::Flop => 3,
        Street::Turn => 4,
        Street::River => 5,
    };
    if cards.len() != expected_len {
        return Err(OracleError::validation(format!(
            "{} requires exactly {expected_len} board cards",
            street_name(*street)
        )));
    }

    let mut parsed = Vec::with_capacity(cards.len());
    for card in cards {
        let parsed_card = card_from_str(card)
            .map_err(|error| OracleError::validation(format!("invalid board card {card:?}: {error}")))?;
        if parsed.contains(&parsed_card) {
            return Err(OracleError::validation(format!(
                "board card {card:?} appears more than once"
            )));
        }
        parsed.push(parsed_card);
    }

    let flop_text = cards[..3].join("");
    flop_from_str(&flop_text)
        .map_err(|error| OracleError::validation(format!("invalid flop: {error}")))?;
    Ok(parsed)
}

fn parse_bet_sizes(config: &BetSizes) -> Result<[[BetSizeOptions; 2]; 3], OracleError> {
    let parse_street = |street: &str, sizes: &StreetBetSizes| {
        Ok([
            parse_player_sizes(street, "OOP", &sizes.oop)?,
            parse_player_sizes(street, "IP", &sizes.ip)?,
        ])
    };
    Ok([
        parse_street("flop", &config.flop)?,
        parse_street("turn", &config.turn)?,
        parse_street("river", &config.river)?,
    ])
}

fn parse_player_sizes(
    street: &str,
    player: &str,
    sizes: &PlayerBetSizes,
) -> Result<BetSizeOptions, OracleError> {
    if sizes.bet.len() > 512 || sizes.raise.len() > 512 {
        return Err(OracleError::validation(format!(
            "{street} {player} sizing strings must not exceed 512 bytes"
        )));
    }
    BetSizeOptions::try_from((sizes.bet.as_str(), sizes.raise.as_str())).map_err(|error| {
        OracleError::validation(format!("invalid {street} {player} bet sizes: {error}"))
    })
}

fn parse_donk_sizes(
    street: &str,
    sizes: Option<&str>,
) -> Result<Option<DonkSizeOptions>, OracleError> {
    let Some(sizes) = sizes else {
        return Ok(None);
    };
    if sizes.trim().is_empty() {
        return Ok(Some(DonkSizeOptions::default()));
    }
    if sizes.len() > 512 {
        return Err(OracleError::validation(format!(
            "tree_options.{street}_donk_sizes must not exceed 512 bytes"
        )));
    }
    DonkSizeOptions::try_from(sizes)
        .map(Some)
        .map_err(|error| {
            OracleError::validation(format!(
                "invalid tree_options.{street}_donk_sizes: {error}"
            ))
        })
}

fn run_solver(game: &mut PostFlopGame, max_iterations: u32, target: f32) -> (u32, f32) {
    let mut iterations = 0;
    let mut exploitability = compute_exploitability(game);
    for iteration in 0..max_iterations {
        if exploitability <= target {
            break;
        }
        solve_step(game, iteration);
        iterations = iteration + 1;
        if iterations % 10 == 0 || iterations == max_iterations {
            exploitability = compute_exploitability(game);
        }
    }
    finalize(game);
    (iterations, exploitability)
}

fn extract_player(
    game: &PostFlopGame,
    player: usize,
    root_action_count: Option<usize>,
) -> Result<PlayerResult, OracleError> {
    let cards = game.private_cards(player);
    let range_weights = game.weights(player);
    let normalized_weights = game.normalized_weights(player);
    let equity = game.equity(player);
    let equilibrium_evs = game.expected_values(player);
    let strategy = root_action_count.map(|_| game.strategy());
    let action_evs = root_action_count.map(|_| game.expected_values_detail(player));
    let hand_count = cards.len();

    if range_weights.len() != hand_count
        || normalized_weights.len() != hand_count
        || equity.len() != hand_count
        || equilibrium_evs.len() != hand_count
    {
        return Err(OracleError::solver(
            "solver returned inconsistent per-combo vector lengths",
        ));
    }

    let mut combos = Vec::new();
    let mut total_weight = 0.0f64;
    let mut equity_sum = 0.0f64;
    let mut ev_sum = 0.0f64;
    for index in 0..hand_count {
        let normalized_weight = normalized_weights[index] as f64;
        if normalized_weight <= 0.0 {
            continue;
        }
        let hand = hole_to_string(cards[index])
            .map_err(|error| OracleError::solver(format!("failed to format combo: {error}")))?;
        let frequencies = root_action_count.map(|action_count| {
            (0..action_count)
                .map(|action| strategy.as_ref().unwrap()[action * hand_count + index] as f64)
                .collect()
        });
        let per_action_evs = root_action_count.map(|action_count| {
            (0..action_count)
                .map(|action| action_evs.as_ref().unwrap()[action * hand_count + index] as f64)
                .collect()
        });

        total_weight += normalized_weight;
        equity_sum += equity[index] as f64 * normalized_weight;
        ev_sum += equilibrium_evs[index] as f64 * normalized_weight;
        combos.push(ComboResult {
            hand,
            range_weight: range_weights[index] as f64,
            normalized_weight,
            reach_weight: 0.0,
            equity: equity[index] as f64,
            equilibrium_ev_units: equilibrium_evs[index] as f64,
            root_action_frequencies: frequencies,
            root_action_evs_units: per_action_evs,
        });
    }

    if total_weight <= 0.0 {
        return Err(OracleError::validation(format!(
            "{} range has no reachable combos against the opponent range",
            player_name(player)
        )));
    }

    for combo in &mut combos {
        combo.reach_weight = combo.normalized_weight / total_weight;
    }

    Ok(PlayerResult {
        player: player_name(player),
        total_reachable_weight: total_weight,
        average_equity: equity_sum / total_weight,
        average_ev_units: ev_sum / total_weight,
        combos,
    })
}

fn describe_action(index: usize, action: Action) -> ActionDescriptor {
    let (label, kind, amount) = match action {
        Action::Fold => ("Fold".to_string(), "FOLD", None),
        Action::Check => ("Check".to_string(), "CHECK", None),
        Action::Call => ("Call".to_string(), "CALL", None),
        Action::Bet(amount) => (format!("Bet {amount}"), "BET", Some(amount)),
        Action::Raise(amount) => (format!("Raise to {amount}"), "RAISE", Some(amount)),
        Action::AllIn(amount) => (format!("All-in {amount}"), "ALL_IN", Some(amount)),
        Action::None | Action::Chance(_) => (format!("{action:?}"), "INTERNAL", None),
    };
    ActionDescriptor {
        index,
        label,
        kind,
        amount,
    }
}

fn board_state(street: Street) -> BoardState {
    match street {
        Street::Flop => BoardState::Flop,
        Street::Turn => BoardState::Turn,
        Street::River => BoardState::River,
    }
}

fn street_name(street: Street) -> &'static str {
    match street {
        Street::Flop => "FLOP",
        Street::Turn => "TURN",
        Street::River => "RIVER",
    }
}

fn player_name(player: usize) -> &'static str {
    if player == 0 { "OOP" } else { "IP" }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_limit_defaults_safely_and_accepts_explicit_hardware_capacity() {
        assert_eq!(
            memory_limit_bytes_from_value(None).unwrap(),
            8 * 1024 * 1024 * 1024
        );
        assert_eq!(
            memory_limit_bytes_from_value(Some("128")).unwrap(),
            128 * 1024 * 1024 * 1024
        );
        for invalid in ["", "0", "-1", "1.5", "4097", "many"] {
            let error = memory_limit_bytes_from_value(Some(invalid)).unwrap_err();
            assert_eq!(error.code, "VALIDATION_ERROR");
            assert!(error.message.contains(MEMORY_LIMIT_ENV));
        }
    }

    fn minimal_request() -> SolveRootRequest {
        SolveRootRequest {
            schema_version: SCHEMA_VERSION,
            id: "test-river".to_string(),
            operation: Operation::SolveRoot,
            offline_only_acknowledged: true,
            owned_simulator_acknowledged: false,
            street: Street::River,
            board: vec!["Td", "9d", "6h", "Qc", "2s"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            oop_range: "AdKd".to_string(),
            ip_range: "AsKs".to_string(),
            starting_pot: 100,
            effective_stack: 100,
            chip_scale: 100,
            chip_unit: "centi-BB".to_string(),
            allocation_mode: AllocationMode::UncompressedF32,
            bet_sizes: BetSizes::default(),
            rake: Rake::default(),
            tree_options: TreeOptions {
                add_allin_threshold: 1.5,
                force_allin_threshold: 0.15,
                merging_threshold: 0.1,
                turn_donk_sizes: None,
                river_donk_sizes: Some("25%".to_string()),
            },
            target_exploitability_pct: 5.0,
            max_iterations: 10,
        }
    }

    fn wire_action(kind: WireActionKind, amount: Option<i32>) -> WireAction {
        WireAction { kind, amount }
    }

    fn minimal_node_request() -> SolveNodeRequest {
        let root = minimal_request();
        SolveNodeRequest {
            schema_version: root.schema_version,
            id: "test-river-node".to_string(),
            operation: Operation::SolveNode,
            offline_only_acknowledged: root.offline_only_acknowledged,
            owned_simulator_acknowledged: root.owned_simulator_acknowledged,
            street: root.street,
            board: root.board,
            oop_range: root.oop_range,
            ip_range: root.ip_range,
            starting_pot: root.starting_pot,
            effective_stack: root.effective_stack,
            chip_scale: root.chip_scale,
            chip_unit: root.chip_unit,
            allocation_mode: root.allocation_mode,
            bet_sizes: root.bet_sizes,
            rake: root.rake,
            tree_options: root.tree_options,
            target_exploitability_pct: root.target_exploitability_pct,
            max_iterations: root.max_iterations,
            action_history: vec![wire_action(WireActionKind::Check, None)],
            expected_current_player: NodePlayer::Ip,
            expected_facing_bet: 0,
            expected_node_actions: vec![
                wire_action(WireActionKind::Check, None),
                wire_action(WireActionKind::Bet, Some(50)),
                wire_action(WireActionKind::AllIn, Some(100)),
            ],
        }
    }

    fn minimal_path_request() -> SolvePathRequest {
        let sizes = BetSizes::default();
        SolvePathRequest {
            schema_version: SCHEMA_VERSION,
            id: "test-flop-turn-path".to_string(),
            operation: Operation::SolvePath,
            offline_only_acknowledged: true,
            owned_simulator_acknowledged: false,
            street: Street::Flop,
            board: vec!["2c", "7d", "Jh"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            oop_range: "QcQd,9c9d:0.5".to_string(),
            // AsAd must be present at the flop root and disappear only after
            // the exact As chance step is traversed.
            ip_range: "AsAd:0.5,KcKd,TcTd:0.5".to_string(),
            starting_pot: 550,
            effective_stack: 9750,
            chip_scale: 100,
            chip_unit: "centi-BB".to_string(),
            allocation_mode: AllocationMode::UncompressedF32,
            bet_sizes: sizes,
            rake: Rake {
                rate_pct: 5.0,
                cap: 50.0,
            },
            tree_options: TreeOptions {
                add_allin_threshold: 0.0,
                force_allin_threshold: 0.0,
                merging_threshold: 0.0,
                turn_donk_sizes: None,
                river_donk_sizes: None,
            },
            target_exploitability_pct: 100.0,
            max_iterations: 10,
            path_history: vec![
                WirePathStep::Action {
                    action: wire_action(WireActionKind::Check, None),
                },
                WirePathStep::Action {
                    action: wire_action(WireActionKind::Bet, Some(300)),
                },
                WirePathStep::Action {
                    action: wire_action(WireActionKind::Call, None),
                },
                WirePathStep::Deal {
                    card: "As".to_string(),
                },
            ],
            expected_board: vec!["2c", "7d", "Jh", "As"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            expected_total_invested: [300, 300],
            expected_current_player: NodePlayer::Oop,
            expected_facing_bet: 0,
            expected_node_actions: vec![
                wire_action(WireActionKind::Check, None),
                wire_action(WireActionKind::Bet, Some(575)),
            ],
        }
    }

    #[test]
    fn strict_schema_rejects_unknown_fields() {
        let json = r#"{
            "schema_version":1,"id":"x","operation":"solve_root",
            "offline_only_acknowledged":true,"street":"RIVER",
            "board":["Td","9d","6h","Qc","2s"],
            "oop_range":"AdKd","ip_range":"AsKs",
            "starting_pot":100,"effective_stack":100,
            "chip_scale":100,"chip_unit":"centi-BB",
            "allocation_mode":"uncompressed_f32",
            "tree_options":{"add_allin_threshold":1.5,
                "force_allin_threshold":0.15,"merging_threshold":0.1,
                "turn_donk_sizes":null,"river_donk_sizes":null},
            "target_exploitability_pct":5.0,"max_iterations":10,
            "surprise":true
        }"#;
        assert!(serde_json::from_str::<SolveRootRequest>(json).is_err());
    }

    #[test]
    fn solve_node_wire_actions_require_an_explicit_nullable_amount() {
        let mut value = serde_json::to_value(minimal_node_request()).unwrap();
        value["action_history"][0]
            .as_object_mut()
            .unwrap()
            .remove("amount");
        let payload = serde_json::to_string(&value).unwrap();
        let error = serde_json::from_str::<SolveNodeRequest>(&payload).unwrap_err();
        assert!(error.to_string().contains("missing field `amount`"));
    }

    #[test]
    fn solve_path_wire_steps_reject_unknown_fields() {
        let mut value = serde_json::to_value(minimal_path_request()).unwrap();
        value["path_history"][0]["surprise"] = serde_json::Value::Bool(true);
        let payload = serde_json::to_string(&value).unwrap();
        let error = serde_json::from_str::<SolvePathRequest>(&payload).unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn board_card_must_be_unique() {
        let mut request = minimal_request();
        request.board[4] = "Td".to_string();
        assert!(parse_board(&request.street, &request.board).is_err());
    }

    #[test]
    fn exactly_one_execution_context_is_mandatory() {
        let mut request = minimal_request();
        request.offline_only_acknowledged = false;
        let error = solve_root(request).unwrap_err();
        assert_eq!(error.code, "EXECUTION_CONTEXT_ACK_REQUIRED");

        let mut request = minimal_request();
        request.owned_simulator_acknowledged = true;
        let error = solve_root(request).unwrap_err();
        assert_eq!(error.code, "EXECUTION_CONTEXT_CONFLICT");
    }

    #[test]
    fn owned_simulator_context_is_echoed_without_claiming_offline_use() {
        let mut request = minimal_request();
        request.offline_only_acknowledged = false;
        request.owned_simulator_acknowledged = true;
        let response = solve_root(request).unwrap();
        assert!(!response.provenance.offline_only_acknowledged);
        assert!(response.provenance.owned_simulator_acknowledged);

        let json = serde_json::to_value(response).unwrap();
        assert_eq!(
            json["provenance"]["owned_simulator_acknowledged"],
            serde_json::Value::Bool(true)
        );
    }

    #[test]
    fn schema_version_is_exact() {
        let mut request = minimal_request();
        request.schema_version = 2;
        let error = solve_root(request).unwrap_err();
        assert_eq!(error.code, "UNSUPPORTED_SCHEMA_VERSION");
    }

    #[test]
    fn tree_thresholds_must_be_finite_and_nonnegative() {
        let mut request = minimal_request();
        request.tree_options.merging_threshold = f64::NAN;
        assert!(validate_request(&request).is_err());

        request.tree_options.merging_threshold = -0.1;
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn f64_request_values_cannot_overflow_or_underflow_f32() {
        for value in [1e300, 1e-300] {
            let mut request = minimal_request();
            request.tree_options.add_allin_threshold = value;
            let error = validate_request(&request).unwrap_err();
            assert_eq!(error.code, "VALIDATION_ERROR");
            assert!(
                error.message.contains("f32"),
                "unexpected validation error: {}",
                error.message
            );
        }

        let mut tiny_target = minimal_request();
        tiny_target.target_exploitability_pct = 1e-300;
        assert!(validate_request(&tiny_target).is_err());

        let mut huge_rake_cap = minimal_request();
        huge_rake_cap.rake.cap = 1e300;
        assert!(validate_request(&huge_rake_cap).is_err());
    }

    #[test]
    fn direct_json_rejects_nan_and_infinity() {
        let valid = serde_json::to_string(&minimal_request()).unwrap();
        let marker = "\"merging_threshold\":0.1";
        assert!(valid.contains(marker));
        for invalid in ["NaN", "Infinity", "-Infinity"] {
            let payload = valid.replace(marker, &format!("\"merging_threshold\":{invalid}"));
            assert!(
                serde_json::from_str::<SolveRootRequest>(&payload).is_err(),
                "strict JSON unexpectedly accepted {invalid}"
            );
        }
    }

    #[test]
    fn direct_range_weights_reject_nonfinite_and_underflow() {
        for invalid in ["NaN", "Infinity", "-Infinity"] {
            let error = validate_range_weights("oop_range", &format!("AsAh:{invalid}"))
                .unwrap_err();
            assert_eq!(error.code, "VALIDATION_ERROR");
        }

        let f32_underflow = format!("AsAh:0.{}1", "0".repeat(49));
        let error = validate_range_weights("oop_range", &f32_underflow).unwrap_err();
        assert!(error.message.contains("f32"));

        let f64_underflow = format!("AsAh:0.{}1", "0".repeat(399));
        let error = validate_range_weights("oop_range", &f64_underflow).unwrap_err();
        assert!(error.message.contains("f64"));

        assert!(validate_range_weights("oop_range", "AsAh:0.25,QsQh:1").is_ok());
    }

    #[test]
    fn compressed_allocation_is_selected_and_echoed() {
        let mut request = minimal_request();
        request.allocation_mode = AllocationMode::CompressedI16;
        let response = solve_root(request).unwrap();
        assert_eq!(response.memory.allocation_mode, "compressed_i16");
        assert!(response.memory.estimated_compressed_bytes > 0);
        assert_eq!(
            response.provenance.solver.allocation_mode,
            "compressed_i16"
        );
        assert_eq!(
            response.provenance.effective_request.allocation_mode,
            AllocationMode::CompressedI16
        );
    }

    #[test]
    fn donk_wire_states_map_to_inherit_empty_and_custom_trees() {
        fn turn_actions(donk: Option<DonkSizeOptions>) -> Vec<Action> {
            let sizes = BetSizeOptions::try_from(("50%", "")).unwrap();
            let config = TreeConfig {
                initial_state: BoardState::Flop,
                starting_pot: 100,
                effective_stack: 400,
                rake_rate: 0.0,
                rake_cap: 0.0,
                flop_bet_sizes: [sizes.clone(), sizes.clone()],
                turn_bet_sizes: [sizes.clone(), sizes.clone()],
                river_bet_sizes: [sizes.clone(), sizes],
                turn_donk_sizes: donk,
                river_donk_sizes: None,
                add_allin_threshold: 0.0,
                force_allin_threshold: 0.0,
                merging_threshold: 0.0,
            };
            let mut tree = ActionTree::new(config).unwrap();
            tree.play(Action::Check).unwrap();
            tree.play(Action::Bet(50)).unwrap();
            tree.play(Action::Call).unwrap();
            tree.available_actions().to_vec()
        }

        let inherited = turn_actions(parse_donk_sizes("turn", None).unwrap());
        let disabled = turn_actions(parse_donk_sizes("turn", Some("")).unwrap());
        let custom = turn_actions(parse_donk_sizes("turn", Some("25%")).unwrap());

        assert_eq!(inherited, vec![Action::Check, Action::Bet(100)]);
        assert_eq!(disabled, vec![Action::Check]);
        assert_eq!(custom, vec![Action::Check, Action::Bet(50)]);
    }

    #[test]
    fn tiny_river_solve_returns_aligned_root_vectors() {
        let response = solve_root(minimal_request()).unwrap();
        assert_eq!(response.schema_version, SCHEMA_VERSION);
        assert!(response.provenance.offline_only_acknowledged);
        assert!(!response.provenance.owned_simulator_acknowledged);
        let json = serde_json::to_value(&response).unwrap();
        assert!(json["provenance"]
            .get("owned_simulator_acknowledged")
            .is_none());
        assert_eq!(response.provenance.solver.commit, SOLVER_COMMIT);
        assert_eq!(
            response
                .provenance
                .effective_request
                .tree_options
                .river_donk_sizes
                .as_deref(),
            Some("25%")
        );
        assert_eq!(response.root_player, "OOP");
        assert!(!response.root_actions.is_empty());
        let oop = &response.players[0];
        assert_eq!(oop.combos.len(), 1);
        let combo = &oop.combos[0];
        assert!((combo.reach_weight - 1.0).abs() < 1e-12);
        assert_eq!(
            combo.root_action_frequencies.as_ref().unwrap().len(),
            response.root_actions.len()
        );
        assert_eq!(
            combo.root_action_evs_units.as_ref().unwrap().len(),
            response.root_actions.len()
        );
        let probability_sum: f64 = combo
            .root_action_frequencies
            .as_ref()
            .unwrap()
            .iter()
            .sum();
        assert!((probability_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn legacy_root_response_keeps_the_original_shape() {
        let response = serde_json::to_value(solve_root(minimal_request()).unwrap()).unwrap();
        let keys = response
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            keys,
            [
                "schema_version",
                "id",
                "operation",
                "status",
                "provenance",
                "street",
                "board",
                "root_player",
                "root_actions",
                "players",
                "convergence",
                "memory",
                "timings_ms",
            ]
            .into_iter()
            .collect()
        );
        assert_eq!(response["operation"], "solve_root");
        assert!(response.get("current_player").is_none());
        assert!(response["provenance"]["effective_request"]
            .get("action_history")
            .is_none());
    }

    #[test]
    fn solve_node_traverses_check_to_ip_and_returns_node_vectors() {
        let response = solve_node(minimal_node_request()).unwrap();
        assert_eq!(response.schema_version, SCHEMA_VERSION);
        assert_eq!(response.operation, "solve_node");
        assert_eq!(response.current_player, "IP");
        assert_eq!(response.node_actions.len(), 3);
        assert_eq!(response.node_actions[0].kind, "CHECK");
        assert_eq!(response.node_actions[1].kind, "BET");
        assert_eq!(response.node_actions[1].amount, Some(50));
        assert_eq!(response.node_actions[2].kind, "ALL_IN");
        assert_eq!(response.node_actions[2].amount, Some(100));
        assert!(response.node_total_reachable_weight > 0.0);
        assert_eq!(response.policies.len(), 1);

        let policy = &response.policies[0];
        assert_eq!(policy.hand, "AsKs");
        assert!((policy.input_range_weight - 1.0).abs() < 1e-6);
        assert!((policy.path_weight - 1.0).abs() < 1e-6);
        assert!((policy.conditional_reach_weight - 1.0).abs() < 1e-6);
        assert_eq!(
            policy.node_action_frequencies.len(),
            response.node_actions.len()
        );
        assert_eq!(
            policy.node_action_evs_units.len(),
            response.node_actions.len()
        );
        let frequency_sum = policy.node_action_frequencies.iter().sum::<f64>();
        assert!((frequency_sum - 1.0).abs() < 1e-5);

        let json = serde_json::to_value(response).unwrap();
        assert_eq!(
            json["provenance"]["effective_request"]["action_history"][0]["kind"],
            "CHECK"
        );
        assert_eq!(
            json["provenance"]["effective_request"]["expected_current_player"],
            "IP"
        );
    }

    #[test]
    fn solve_path_conditions_ranges_across_a_real_turn_card() {
        let response = solve_path(minimal_path_request()).unwrap();
        assert_eq!(response.schema_version, SCHEMA_VERSION);
        assert_eq!(response.operation, "solve_path");
        assert_eq!(response.current_street, Street::Turn);
        assert_eq!(
            response.current_board,
            vec!["2c", "7d", "Jh", "As"]
        );
        assert_eq!(response.current_player, "OOP");
        assert_eq!(response.node_actions.len(), 2);
        assert_eq!(response.node_actions[0].kind, "CHECK");
        assert_eq!(response.node_actions[1].kind, "BET");
        assert_eq!(response.node_actions[1].amount, Some(575));
        assert_eq!(response.policies.len(), 2);
        assert!(response
            .policies
            .iter()
            .any(|policy| policy.hand.contains("Qc") && policy.hand.contains("Qd")));

        let ip = response
            .conditional_ranges
            .iter()
            .find(|range| range.player == "IP")
            .expect("IP conditional range");
        assert_eq!(ip.combos.len(), 2);
        assert!(ip
            .combos
            .iter()
            .any(|combo| combo.hand.contains("Kc") && combo.hand.contains("Kd")));
        assert!(ip
            .combos
            .iter()
            .any(|combo| combo.hand.contains("Tc") && combo.hand.contains("Td")));
        assert!(!ip
            .combos
            .iter()
            .any(|combo| combo.hand.contains("As") && combo.hand.contains("Ad")));
        let reach_sum = ip
            .combos
            .iter()
            .map(|combo| combo.conditional_reach_weight)
            .sum::<f64>();
        assert!((reach_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn solve_node_rejects_non_whitelisted_history_before_solving() {
        let mut request = minimal_node_request();
        request
            .action_history
            .push(wire_action(WireActionKind::Check, None));
        let error = validate_node_request(&request).unwrap_err();
        assert_eq!(error.code, "NODE_PATH_ERROR");
        assert!(error.message.contains("action_history must be exactly"));
    }

    #[test]
    fn golden_brown_river_game_converges_to_known_mixed_strategy() {
        // Canonical one-street value/bluff game: AA always value-bets, QQ bluffs
        // one third, and the hidden IP response makes OOP indifferent at equilibrium.
        let mut request = minimal_request();
        request.id = "golden-brown-river".to_string();
        request.board = vec!["2s", "3h", "4d", "6c", "7c"]
            .into_iter()
            .map(str::to_string)
            .collect();
        request.oop_range = "AsAh,QsQh".to_string();
        request.ip_range = "KsKh".to_string();
        request.starting_pot = 20;
        request.effective_stack = 10;
        let player_sizes = PlayerBetSizes {
            bet: "50%".to_string(),
            raise: "".to_string(),
        };
        let street_sizes = StreetBetSizes {
            oop: player_sizes.clone(),
            ip: player_sizes,
        };
        request.bet_sizes = BetSizes {
            flop: street_sizes.clone(),
            turn: street_sizes.clone(),
            river: street_sizes,
        };
        request.tree_options = TreeOptions {
            add_allin_threshold: 0.0,
            force_allin_threshold: 0.0,
            merging_threshold: 0.0,
            turn_donk_sizes: None,
            river_donk_sizes: None,
        };
        request.target_exploitability_pct = 0.001;
        request.max_iterations = 20_000;

        let response = solve_root(request).unwrap();
        let bet_index = response
            .root_actions
            .iter()
            .position(|action| action.kind == "ALL_IN" && action.amount == Some(10))
            .expect("10-unit root bet");
        let oop = &response.players[0];
        let queens = oop
            .combos
            .iter()
            .find(|combo| combo.hand == "QsQh")
            .expect("QQ combo");
        let aces = oop
            .combos
            .iter()
            .find(|combo| combo.hand == "AsAh")
            .expect("AA combo");
        let queens_bet = queens.root_action_frequencies.as_ref().unwrap()[bet_index];
        let aces_bet = aces.root_action_frequencies.as_ref().unwrap()[bet_index];

        assert!(
            (queens_bet - 1.0 / 3.0).abs() <= 0.01,
            "QQ bet frequency {queens_bet} is outside explicit 1% tolerance"
        );
        assert!(
            (aces_bet - 1.0).abs() <= 1e-6,
            "AA bet frequency {aces_bet} is not pure"
        );
        assert!(response.convergence.target_reached);
        assert!(response.convergence.exploitability_pct_of_pot <= 0.001);
    }
}
