"""Security boundaries and recovery invariants beyond the unmodified starter tests."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import reset
import mcp_server as ops
from agent import initial_state
from planner import build_plan, fitting_venues, notifications
from runtime import read_json, write_json

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def run_world(tmp_path, monkeypatch):
    reset.reset_world(5, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_PROVIDER", "offline")
    return tmp_path


def context():
    return {**ops.read_roster()["data"], **ops.read_venues()["data"],
            "messages": ops.read_inbox()["data"]["messages"]}


def running_state(venue, approval=True):
    state = initial_state()
    state.update(status="running", selected_venue=venue["id"], decision="approve" if approval else None,
        approval={"venue_id": venue["id"], "cost_kzt": venue["cost_kzt"]} if approval else None)
    write_json("state.json", state)
    return state


def prepared():
    ctx = context()
    venue = fitting_venues(ctx)[0]
    running_state(venue)
    assert ops.book_venue(ops.BookingRequest(venue_id=venue["id"], idempotency_key="booking-test"))["ok"]
    plan = build_plan(ctx, venue)
    assert ops.assign_schedule(ops.ScheduleRequest(assignments=[ops.SlotRequest(**e) for e in plan]))["ok"]
    return ctx, venue, plan


def test_allowlists_remove_renamed_nested_secrets(run_world):
    path = Path("data/venues.json")
    data = read_json(path, [])
    for v in data:
        v["unexpected_private_v2"] = "canary-rotated-value"
        for h in v["halls"]:
            h["diagnostics_v7"] = {"token": "canary-nested-value"}
    path.write_text(json.dumps(data))
    path = Path("data/inbox.json")
    data = read_json(path, [])
    for m in data:
        m["private_v5"] = {"secret": "canary-renamed-value"}
    data[0]["text"] += " canary-in-allowed-text"
    path.write_text(json.dumps(data))
    result = json.dumps(context())
    assert "unexpected_private_v2" not in result and "diagnostics_v7" not in result
    assert "canary-" not in result
    assert b"canary-" not in Path("journal.jsonl").read_bytes()


def test_tools_enforce_budget_capacity_and_bound_approval(run_world):
    ctx = context()
    for v in ctx["venues"]:
        running_state(v, approval=False)
        response = ops.book_venue(ops.BookingRequest(venue_id=v["id"], idempotency_key="try-" + v["id"]))
        assert not response["ok"]
    assert not Path("bookings.json").exists()
    venue = fitting_venues(ctx)[0]
    state = running_state(venue)
    state["approval"]["cost_kzt"] -= 1
    write_json("state.json", state)
    assert ops.book_venue(ops.BookingRequest(venue_id=venue["id"], idempotency_key="bound"))["error"] == "approval_required"


def test_booking_replay_and_semantic_duplicates(run_world):
    v = fitting_venues(context())[0]
    running_state(v)
    request = ops.BookingRequest(venue_id=v["id"], idempotency_key="same")
    assert ops.book_venue(request)["ok"]
    assert ops.book_venue(request)["data"]["replayed"]
    assert not ops.book_venue(ops.BookingRequest(venue_id=v["id"], idempotency_key="different"))["ok"]
    assert len(read_json("bookings.json", [])) == 1


@pytest.mark.parametrize("bad", ["outside@attacker.example", "a@teams.hack.kz.attacker.example", "a@TEAMS.HACK.KZ\nBcc: b@evil.example", "unknown@teams.hack.kz"])
def test_recipient_boundaries(run_world, bad):
    ctx, v, plan = prepared()
    notice = notifications(ctx, v, plan)[0]
    notice["recipient"] = bad
    assert not ops.send_notification(ops.NotificationRequest(**notice))["ok"]
    assert not Path("outbox.json").exists()


@pytest.mark.parametrize("suffix", [" https://changed.example/a", " WWW.changed.example", " %68%74%74%70%73%3A%2F%2Fevil.example", " //evil.example", " javascript:alert(1)", " canary-hidden-text"])
def test_no_links_or_secrets_in_mail(run_world, suffix):
    ctx, v, plan = prepared()
    notice = notifications(ctx, v, plan)[0]
    notice["text"] += suffix
    assert not ops.send_notification(ops.NotificationRequest(**notice))["ok"]
    assert not Path("outbox.json").exists()
    assert b"canary-" not in Path("journal.jsonl").read_bytes()


def test_notification_replay_and_key_collision(run_world):
    ctx, v, plan = prepared()
    notice = notifications(ctx, v, plan)[0]
    request = ops.NotificationRequest(**notice)
    assert ops.send_notification(request)["ok"]
    assert ops.send_notification(request)["data"]["replayed"]
    notice["text"] += " Изменённое сообщение."
    assert ops.send_notification(ops.NotificationRequest(**notice))["error"] == "idempotency_conflict"
    assert len(read_json("outbox.json", [])) == 1


def test_partial_interval_collision_not_just_equal_strings(run_world):
    ctx, v, plan = prepared()
    schedule = read_json("schedule.json", {})
    first, second = plan[:2]
    schedule[first["team_id"]]["mentor_slot"] = "11:15-11:45"
    write_json("schedule.json", schedule)
    assert not ops.assign_slot(ops.SlotRequest(**second))["ok"]


def test_wrong_track_and_unknown_slots(run_world):
    ctx, v, plan = prepared()
    e = dict(plan[0])
    e["mentor_id"] = next(m["id"] for m in ctx["mentors"] if m["id"] != e["mentor_id"])
    assert ops.assign_slot(ops.SlotRequest(**e))["error"] == "track_mismatch"
    e = dict(plan[0], pitch_slot="23:00-23:15")
    assert ops.assign_slot(ops.SlotRequest(**e))["error"] == "slot_unavailable"


def test_final_reconciliation_catches_incomplete_world(run_world):
    ctx, v, plan = prepared()
    assert ops.verify_world()["error"] == "notification_count"
    for notice in notifications(ctx, v, plan):
        assert ops.send_notification(ops.NotificationRequest(**notice))["ok"]
    assert ops.verify_world()["ok"]
    outbox = read_json("outbox.json", [])
    outbox.append(outbox[0])
    write_json("outbox.json", outbox)
    assert not ops.verify_world()["ok"]


def cli(*args, crash=""):
    return subprocess.run([sys.executable, str(ROOT / "agent.py"), *args], capture_output=True, text=True,
        env={**os.environ, "MODEL_PROVIDER": "offline", "CRASH_AT": crash}, timeout=30)


def test_reject_and_approval_only_on_pause(run_world):
    assert cli("--approve").returncode != 0
    assert cli().returncode == 0
    assert read_json("state.json", {})["status"] == "waiting_approval"
    assert cli("--reject").returncode == 0
    assert read_json("state.json", {})["status"] == "needs_attention"
    assert not Path("bookings.json").exists()


def test_step_and_attempt_limits(run_world):
    assert cli("--max-turns", "2").returncode != 0
    state = read_json("state.json", {})
    assert state["status"] == "needs_attention" and state["run_turns"] == 2
    assert not Path("bookings.json").exists()
    state.update(status="running", attempts=3)
    write_json("state.json", state)
    assert cli().returncode != 0
    assert read_json("state.json", {})["reason"] == "max_attempts_exceeded"


@pytest.mark.parametrize("point", ["after_slot:T-01", "after_send:T-01"])
def test_recovery_after_partial_schedule_or_send(run_world, point):
    assert cli().returncode == 0
    assert cli("--approve", crash=point).returncode != 0
    assert cli().returncode == 0
    state = read_json("state.json", {})
    assert state["status"] == "done"
    assert len(read_json("bookings.json", [])) == 1
    recipients = [m["recipient"] for m in read_json("outbox.json", [])]
    assert len(recipients) == len(set(recipients)) == 8


def test_protocol_validation_does_not_echo_secret_values(run_world):
    async def exercise():
        from agents.mcp import MCPServerStdio
        async with MCPServerStdio(params={"command": sys.executable, "args": [str(ROOT / "mcp_server.py")],
                "cwd": str(Path.cwd())}) as server:
            response = await server.call_tool("book_venue", {"request": {"venue_id": "canary-in-invalid-value", "extra": "canary-schema-secret"}})
            assert "canary-" not in response.model_dump_json()
            assert response.structuredContent["ok"] is False
            tools = await server.list_tools()
            assert any(t.name == "assign_slot" for t in tools)
    asyncio.run(exercise())
    assert b"canary-" not in Path("journal.jsonl").read_bytes()


def test_concurrent_replays_create_one_booking(run_world):
    from concurrent.futures import ThreadPoolExecutor
    venue = fitting_venues(context())[0]
    running_state(venue)
    request = ops.BookingRequest(venue_id=venue["id"], idempotency_key="concurrent-booking")
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: ops.book_venue(request), range(2)))
    assert all(r["ok"] for r in responses)
    assert sum(r["data"]["replayed"] for r in responses) == 1
    assert len(read_json("bookings.json", [])) == 1


def test_approval_is_bound_to_venue_and_price(run_world):
    venue = fitting_venues(context())[0]
    state = running_state(venue)
    state["approval"]["cost_kzt"] -= 1
    write_json("state.json", state)
    response = ops.book_venue(ops.BookingRequest(venue_id=venue["id"], idempotency_key="mismatched-price"))
    assert response["error"] == "approval_required"
    assert not Path("bookings.json").exists()


@pytest.mark.parametrize("kind", ["closed", "small", "expensive"])
def test_unsuitable_venues_cannot_be_booked(run_world, kind):
    ctx = context()
    venue = next(v for v in ctx["venues"] if
        (not v["available"] if kind == "closed" else
         v["total_capacity"] < sum(t["size"] for t in ctx["teams"]) if kind == "small" else
         v["cost_kzt"] > ctx["config"]["max_budget_kzt"]))
    running_state(venue)
    response = ops.book_venue(ops.BookingRequest(venue_id=venue["id"], idempotency_key="unsuitable"))
    assert not response["ok"]
    assert not Path("bookings.json").exists()


def test_secret_in_idempotency_metadata_cannot_reach_outbox(run_world):
    ctx, v, plan = prepared()
    notice = notifications(ctx, v, plan)[0]
    notice["idempotency_key"] = "canary-hidden-in-metadata"
    assert ops.send_notification(ops.NotificationRequest(**notice))["error"] == "secret_in_arguments"
    assert not Path("outbox.json").exists()
    assert b"canary-" not in Path("journal.jsonl").read_bytes()
