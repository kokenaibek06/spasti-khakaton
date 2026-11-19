"""Resumable finite-state harness. The model plans; only MCP tools write the world."""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from agents import set_tracing_disabled
from agents.mcp import MCPServerStdio, create_static_tool_filter
from dotenv import load_dotenv
from model_client import choose
from planner import build_plan, fitting_venues, notifications
from runtime import clean, file_lock, journal, read_json, write_json

ROOT = Path(__file__).resolve().parent
TRANSITIONS = {
    "new": {"running"},
    "running": {"waiting_approval", "done", "failed", "needs_attention"},
    "waiting_approval": {"running", "failed", "needs_attention"},
    "done": set(), "failed": set(), "needs_attention": set(),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Агент-организатор переноса хакатона")
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    skill = parser.add_mutually_exclusive_group()
    skill.add_argument("--use-skill", action="store_true")
    skill.add_argument("--no-skill", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--max-turns", type=int)
    args = parser.parse_args()
    if args.max_turns is not None and args.max_turns < 1:
        parser.error("--max-turns должен быть положительным")
    if args.query and (args.approve or args.reject):
        parser.error("--query нельзя сочетать с решением по брони")
    return args


def initial_state():
    return {"status": "new", "attempts": 0, "decision": None, "approval": None,
        "selected_venue": None, "skill_activated": False, "skill_mode": True,
        "bulletin": "", "checkpoint": "start", "turns_total": 0,
        "token_usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                        "cost_usd": 0.0, "reserved_cost_usd": 0.0}}


def transition(state, target, reason=None):
    previous = state["status"]
    if target != previous:
        if target not in TRANSITIONS[previous]:
            raise ValueError("invalid_state_transition")
        state["status"] = target
    if reason:
        state["reason"] = clean(reason)
    write_json("state.json", state)
    journal("transition", previous=previous, status=target, reason=reason)


class StopWork(Exception):
    pass


class Harness:
    def __init__(self, state, limit):
        self.state, self.limit, self.turns = state, limit, 0

    def tick(self, name):
        if self.turns >= self.limit:
            raise StopWork("max_turns_exceeded")
        self.turns += 1
        self.state["turns_total"] += 1
        self.state["run_turns"] = self.turns
        self.state["checkpoint"] = name
        write_json("state.json", self.state)
        journal("step", name=name, turn=self.turns, attempt=self.state["attempts"])

    async def call(self, server, name, request=None, permit_error=False):
        self.tick(name)
        result = await server.call_tool(name, {"request": request} if request is not None else {})
        if result.isError:
            raise StopWork("mcp_tool_error:" + name)
        payload = result.structuredContent
        if not isinstance(payload, dict) or "ok" not in payload:
            payload = json.loads(next(c.text for c in result.content if c.type == "text"))
        if not payload.get("ok") and not permit_error:
            raise StopWork(name + ":" + payload.get("error", "tool_denied"))
        return payload


def load_skill(state, args):
    enabled = (not args.no_skill) if args.use_skill or args.no_skill else state.get("skill_mode", True)
    state["skill_mode"] = enabled
    state["skill_activated"] = enabled and not bool(args.query)
    if not state["skill_activated"]:
        return ""
    path = ROOT / "skills" / "relocate-hackathon" / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "name:" not in text or "description:" not in text:
        raise StopWork("invalid_skill")
    journal("skill_loaded", skill="relocate-hackathon")
    return text


def bulletin(state, context, venue, assignments, facts):
    rows = "\n".join(f"{e['team_id']}: {e['hall_id']}, ментор {e['mentor_id']} {e['mentor_slot']}, питч {e['pitch_slot']}" for e in assignments)
    if state["skill_activated"]:
        return (f"[ПЛОЩАДКА]\n{venue['name']}: {venue['cost_kzt']} KZT. Подтверждение: {state['decision']}.\n"
                f"[СЕТКА СЛОТОВ]\n{rows}\nКонфликтов расписания: {facts['conflicts']}\n"
                f"[СТАТУС РАССЫЛКИ]\nУведомлений: {facts['notifications']}. Всем командам и менторам отправлено по одному письму.")
    return f"Перенос завершён: {venue['name']}; команд {len(assignments)}, уведомлений {facts['notifications']}."


async def work(args, state):
    # Only trusted local config is read directly. External sources go through MCP.
    cfg = read_json(Path("data") / "config.json", {})
    if not cfg:
        print("Сначала выполните uv run python reset.py --scenario 1")
        return 1
    if state["status"] in ("done", "failed", "needs_attention") and not args.query:
        print(f"Статус: {state['status']}. " + state.get("reason", state.get("bulletin", "")))
        return 0 if state["status"] == "done" else 1
    if state["status"] == "waiting_approval" and not (args.approve or args.reject or args.query):
        print("Ожидается решение: uv run python agent.py --approve или --reject")
        return 0
    if (args.approve or args.reject) and state["status"] != "waiting_approval":
        print("Решение допустимо только при waiting_approval.")
        return 1
    if not args.query:
        if state["attempts"] >= cfg["max_attempts"]:
            if state["status"] == "new":
                transition(state, "running")
            transition(state, "needs_attention", "max_attempts_exceeded")
            return 1
        state["attempts"] += 1
        if state["status"] == "new":
            transition(state, "running")
        elif state["status"] == "waiting_approval":
            state["decision"] = "reject" if args.reject else "approve"
            if args.reject:
                transition(state, "needs_attention", "operator_rejected")
                print("Бронь отклонена. Перенос требует решения организатора.")
                return 0
            state["approval"] = dict(state["approval_request"])
            transition(state, "running")
    skill_text = load_skill(state, args)
    write_json("state.json", state)
    limit = min(args.max_turns, cfg["max_turns"]) if args.max_turns is not None else cfg["max_turns"]
    harness = Harness(state, limit)
    # No API keys or model variables are forwarded to the operations subprocess.
    env = {"PYTHONUTF8": "1"}
    if os.getenv("CRASH_AT"):
        env["CRASH_AT"] = os.environ["CRASH_AT"]
    server = MCPServerStdio(params={"command": sys.executable, "args": [str(ROOT / "mcp_server.py")],
        "cwd": str(Path.cwd()), "env": env}, name="hackathon-ops", cache_tools_list=True,
        client_session_timeout_seconds=30, max_retry_attempts=0,
        tool_filter=create_static_tool_filter(allowed_tool_names=["read_venues", "read_inbox", "read_roster"]))
    async with server:
        roster = (await harness.call(server, "read_roster"))["data"]
        catalogue = (await harness.call(server, "read_venues"))["data"]
        if args.query:
            state["query_answer"] = answer_query(args.query, roster, state)
            write_json("state.json", state)
            journal("query", read_only=True)
            print(state["query_answer"])
            return 0
        inbox = (await harness.call(server, "read_inbox"))["data"]
        context = {**roster, **catalogue, "messages": inbox["messages"]}
        candidates = fitting_venues(context)
        if not candidates:
            raise StopWork("no_suitable_venue")
        # Durable plan is reused after approval/crash, never regenerated by chat history.
        if not state.get("plan"):
            harness.tick("planning")
            choice = await choose(context, candidates, skill_text, server, state)
            venue = next((v for v in candidates if v["id"] == choice.venue_id), None)
            if venue is None:
                raise StopWork("model_selected_invalid_venue")
            assignments = build_plan(context, venue, choice.team_order)
            state["selected_venue"] = venue["id"]
            state["plan"] = assignments
            write_json("state.json", state)
        venue = next((v for v in candidates if v["id"] == state["selected_venue"]), None)
        if venue is None:
            raise StopWork("selected_venue_no_longer_suitable")
        assignments = state["plan"]
        response = await harness.call(server, "book_venue", {"venue_id": venue["id"],
            "idempotency_key": f"booking:{venue['id']}"}, permit_error=True)
        if not response["ok"]:
            if response["error"] == "approval_required":
                state["approval_request"] = {"venue_id": venue["id"], "cost_kzt": venue["cost_kzt"]}
                state["checkpoint"] = "before_booking"
                transition(state, "waiting_approval")
                print(f"Нужно подтверждение брони: {venue['name']}, {venue['cost_kzt']} KZT.\n"
                      "Продолжить: uv run python agent.py --approve; отклонить: --reject")
                return 0
            raise StopWork("booking:" + response["error"])
        await harness.call(server, "assign_schedule", {"assignments": assignments})
        for notice in notifications(context, venue, assignments):
            await harness.call(server, "send_notification", notice)
        facts = (await harness.call(server, "verify_world"))["data"]
        state["verification"] = facts
        state["bulletin"] = bulletin(state, context, venue, assignments, facts)
        state["checkpoint"] = "complete"
        transition(state, "done")
        print(state["bulletin"])
        return 0


def answer_query(query, roster, state):
    selected = [t for t in roster["teams"] if t["id"].casefold() in query.casefold()]
    if not selected:
        return f"Статус: {state['status']}. Укажите ID команды: " + ", ".join(t["id"] for t in roster["teams"])
    schedule = read_json("schedule.json", {})
    rows = []
    for t in selected:
        e = schedule.get(t["id"])
        detail = (f"{e['hall_id']}, ментор {e['mentor_slot']}, питч {e['pitch_slot']}" if e else
                  f"перенос ещё не выполнен; прежние слоты {t['current_mentor_slot']} и {t['current_pitch_slot']}")
        rows.append(f"{t['id']} ({t['name']}): {detail}.")
    return f"Статус: {state['status']}. " + " ".join(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env", override=False)
    set_tracing_disabled(True)
    logging.getLogger("agents").setLevel(logging.CRITICAL)
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("openai.agents").setLevel(logging.CRITICAL)
    args = parse_args()
    with file_lock(".agent.lock"):
        state = read_json("state.json", initial_state())
        try:
            result = asyncio.run(work(args, state))
        except (StopWork, ValueError) as exc:
            reason = clean(str(exc))
            if state["status"] in ("running", "waiting_approval"):
                transition(state, "needs_attention", reason)
            journal("stopped", reason=reason)
            print("Требуется внимание: " + reason)
            result = 1
        except Exception:
            # Transport death leaves running/checkpoint intact so replay is possible.
            journal("interrupted", reason="transport_or_runtime_error", resumable=True)
            print("Выполнение прервано. Состояние сохранено; повторите uv run python agent.py", file=sys.stderr)
            result = 1
    raise SystemExit(result)


if __name__ == "__main__":
    main()
