"""Tests for the transcript-first multiway-v3 protocol."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import unittest

from gto_hand_history import HandEvent, HandSeat, PublicHandHistory
from gto_remote.multiway_protocol import (
    MAX_CAPTURE_ID_BYTES,
    MAX_REQUEST_BYTES,
    MAX_REQUEST_ID_BYTES,
    MultiwayDecisionState,
    MultiwayProtocolError,
    PROTOCOL_SCHEMA_VERSION,
    build_evaluate_request,
    decision_fingerprint,
    decision_state_from_wire,
    decision_state_to_wire,
    encode_json,
    parse_evaluate_request,
)


def six_max_seats() -> tuple[HandSeat, ...]:
    # With button on seat 3, clockwise order is BTN, SB, BB, UTG, HJ, CO.
    positions = {
        0: "UTG",
        1: "HJ",
        2: "CO",
        3: "BTN",
        4: "SB",
        5: "BB",
    }
    return tuple(
        HandSeat(
            seat=seat,
            position=positions[seat],
            starting_stack_bb=Decimal("100"),
        )
        for seat in range(6)
    )


def four_way_flop_history() -> PublicHandHistory:
    """Return a genuine four-player flop with action on the small blind."""

    return PublicHandHistory(
        hand_id="four-way-flop-1",
        button_seat=3,
        small_blind_bb=Decimal("0.5"),
        big_blind_bb=Decimal("1"),
        ante_bb=Decimal("0"),
        rake_rate_pct=Decimal("5"),
        rake_cap_bb=Decimal("0.5"),
        seats=six_max_seats(),
        events=(
            HandEvent(0, "FOLD", "PREFLOP", actor_seat=0),
            HandEvent(1, "FOLD", "PREFLOP", actor_seat=1),
            HandEvent(
                2,
                "CALL",
                "PREFLOP",
                actor_seat=2,
                amount_to_bb=Decimal("1"),
            ),
            HandEvent(
                3,
                "CALL",
                "PREFLOP",
                actor_seat=3,
                amount_to_bb=Decimal("1"),
            ),
            HandEvent(
                4,
                "CALL",
                "PREFLOP",
                actor_seat=4,
                amount_to_bb=Decimal("1"),
            ),
            HandEvent(5, "CHECK", "PREFLOP", actor_seat=5),
            HandEvent(
                6,
                "DEAL_FLOP",
                "FLOP",
                cards=("Ac", "Kd", "7s"),
            ),
        ),
    )


def four_way_state(*, capture_id: str = "capture-0001") -> MultiwayDecisionState:
    return MultiwayDecisionState(
        public_hand=four_way_flop_history(),
        hero_seat=4,
        hero_combo=("Qc", "Qd"),
        capture_id=capture_id,
    )


class MultiwayProtocolRoundTripTests(unittest.TestCase):
    def test_genuine_four_way_flop_round_trip_derives_public_facts(self) -> None:
        state = four_way_state()
        replayed = state.replayed

        self.assertEqual("four-way-flop-1", state.hand_id)
        self.assertEqual("SB", state.hero_position)
        self.assertEqual("FLOP", replayed.street)
        self.assertEqual(("Ac", "Kd", "7s"), replayed.board)
        self.assertEqual(frozenset({2, 3, 4, 5}), replayed.live_seats)
        self.assertEqual(4, replayed.actor_seat)
        self.assertEqual(Decimal("4"), replayed.pot_bb)
        self.assertEqual(Decimal("99"), state.hero_stack_bb)
        self.assertEqual(Decimal("0"), state.hero_current_bet_bb)
        self.assertEqual(Decimal("0"), replayed.amount_to_call_bb)
        self.assertEqual(Decimal("1"), replayed.minimum_raise_to_bb)
        self.assertEqual(
            ("CHECK", "BET_TO", "ALL_IN_TO"),
            replayed.legal_actions,
        )

        state_wire = decision_state_to_wire(state)
        self.assertEqual(
            {"capture_id", "hero_seat", "hero_combo", "public_hand"},
            set(state_wire),
        )
        self.assertFalse(
            {
                "street",
                "board",
                "pot_bb",
                "legal_actions",
                "villain_position",
                "villain_stack_bb",
            }
            & set(state_wire)
        )
        public_wire = state_wire["public_hand"]
        self.assertEqual("0.5", public_wire["small_blind_bb"])
        self.assertEqual("100", public_wire["seats"][0]["starting_stack_bb"])
        self.assertEqual("1", public_wire["events"][2]["amount_to_bb"])

        request = build_evaluate_request("request-0001", state)
        request_id, restored, fingerprint = parse_evaluate_request(
            encode_json(request)
        )

        self.assertEqual(3, PROTOCOL_SCHEMA_VERSION)
        self.assertEqual("request-0001", request_id)
        self.assertEqual(state, restored)
        self.assertEqual(decision_fingerprint(state), fingerprint)
        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        self.assertNotEqual(
            fingerprint,
            decision_fingerprint(
                four_way_state(capture_id="capture-0002")
            ),
        )

    def test_direct_state_wire_round_trip_is_canonical(self) -> None:
        state = four_way_state()

        restored = decision_state_from_wire(decision_state_to_wire(state))

        self.assertEqual(state, restored)
        self.assertEqual(
            decision_state_to_wire(state),
            decision_state_to_wire(restored),
        )


class MultiwayProtocolContradictionTests(unittest.TestCase):
    def test_hero_must_be_the_replayed_actor(self) -> None:
        state = four_way_state()

        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "does not have action on Hero",
        ):
            replace(state, hero_seat=2)

        wire = decision_state_to_wire(state)
        wire["hero_seat"] = 5
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "does not have action on Hero",
        ):
            decision_state_from_wire(wire)

    def test_private_cards_cannot_contradict_the_public_board(self) -> None:
        state = four_way_state()

        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "public board card",
        ):
            replace(state, hero_combo=("Ac", "Qd"))
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "cannot repeat",
        ):
            replace(state, hero_combo=("Qc", "Qc"))

    def test_derived_fields_and_singular_villain_fields_are_forbidden(self) -> None:
        for field, value in (
            ("pot_bb", "999"),
            ("street", "RIVER"),
            ("villain_position", "BB"),
        ):
            with self.subTest(field=field):
                wire = decision_state_to_wire(four_way_state())
                wire[field] = value
                with self.assertRaisesRegex(
                    MultiwayProtocolError,
                    rf"unexpected {field}",
                ):
                    decision_state_from_wire(wire)

    def test_transcript_must_end_at_an_active_hero_decision(self) -> None:
        history = four_way_flop_history()
        # Before DEAL_FLOP, preflop is closed and no player may act.
        closed_round = replace(history, events=history.events[:-1])

        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "not at an active player decision",
        ):
            MultiwayDecisionState(
                public_hand=closed_round,
                hero_seat=4,
                hero_combo=("Qc", "Qd"),
                capture_id="capture-closed",
            )


class MultiwayProtocolMalformedInputTests(unittest.TestCase):
    def test_exact_request_and_state_keys_are_required(self) -> None:
        request = build_evaluate_request("request-0001", four_way_state())
        request["unexpected"] = True
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "request schema mismatch.*unexpected unexpected",
        ):
            parse_evaluate_request(encode_json(request))

        state_wire = decision_state_to_wire(four_way_state())
        del state_wire["capture_id"]
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "missing capture_id",
        ):
            decision_state_from_wire(state_wire)

    def test_duplicate_keys_wrong_version_and_non_finite_json_are_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(MultiwayProtocolError, "duplicate JSON key"):
            parse_evaluate_request(
                b'{"schema_version":3,"schema_version":3,'
                b'"request_id":"r","state":{}}'
            )

        request = build_evaluate_request("request-0001", four_way_state())
        request["schema_version"] = 2
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "unsupported schema_version 2",
        ):
            parse_evaluate_request(encode_json(request))

        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "non-finite JSON constant",
        ):
            parse_evaluate_request(
                b'{"schema_version":3,"request_id":"r",'
                b'"state":NaN}'
            )

    def test_all_chip_amounts_must_be_decimal_strings(self) -> None:
        request = build_evaluate_request("request-0001", four_way_state())
        request["state"]["public_hand"]["events"][2]["amount_to_bb"] = 1.0

        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "decimal JSON string",
        ):
            parse_evaluate_request(encode_json(request))

        request = build_evaluate_request("request-0001", four_way_state())
        request["state"]["public_hand"]["seats"][0][
            "starting_stack_bb"
        ] = 100
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "decimal JSON string",
        ):
            parse_evaluate_request(encode_json(request))

    def test_scalar_types_and_cards_are_strict(self) -> None:
        request = build_evaluate_request("request-0001", four_way_state())
        request["state"]["hero_seat"] = True
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "hero_seat must be an integer",
        ):
            parse_evaluate_request(encode_json(request))

        request = build_evaluate_request("request-0001", four_way_state())
        request["state"]["hero_combo"][0] = "1c"
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "canonical notation",
        ):
            parse_evaluate_request(encode_json(request))

    def test_identifiers_and_payload_size_are_bounded(self) -> None:
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            f"exceeds {MAX_REQUEST_ID_BYTES}",
        ):
            build_evaluate_request(
                "r" * (MAX_REQUEST_ID_BYTES + 1),
                four_way_state(),
            )
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            f"exceeds {MAX_CAPTURE_ID_BYTES}",
        ):
            four_way_state(
                capture_id="c" * (MAX_CAPTURE_ID_BYTES + 1)
            )
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            f"exceeds {MAX_REQUEST_BYTES} bytes",
        ):
            parse_evaluate_request(b" " * (MAX_REQUEST_BYTES + 1))

    def test_request_id_is_safe_ascii_for_http_headers(self) -> None:
        with self.assertRaisesRegex(MultiwayProtocolError, "safe ASCII"):
            build_evaluate_request("richiesta-é", four_way_state())

    def test_invalid_utf8_and_non_json_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(MultiwayProtocolError, "must be UTF-8"):
            parse_evaluate_request(b"\xff")
        with self.assertRaisesRegex(
            MultiwayProtocolError,
            "must be bytes or text",
        ):
            parse_evaluate_request({})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
