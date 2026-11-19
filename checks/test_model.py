"""Real Agents SDK against a local HTTP protocol stub, without a paid LLM call."""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import reset
from runtime import read_json

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def protocol_endpoint():
    calls = []
    mode = {"error": False, "secret_output": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            if mode["error"]:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"test unavailable","type":"server_error"}}')
                return
            content = request["input"][-1]["content"]
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content)
            context = json.loads(content)
            choice = {"venue_id": context["feasible_venues"][0]["id"],
                      "team_order": sorted(t["id"] for t in context["teams"])}
            if mode["secret_output"]:
                choice["venue_id"] = "canary-unexpected-model-output"
            response = {"id": "resp_protocol", "object": "response", "created_at": int(time.time()),
                "model": request["model"], "status": "completed", "error": None, "incomplete_details": None,
                "parallel_tool_calls": False, "tool_choice": "none", "tools": [],
                "output": [{"id": "msg_protocol", "type": "message", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "annotations": [], "text": json.dumps(choice)}]}],
                "usage": {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050,
                    "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls, mode
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_agent(folder, endpoint, *args, rates=("1", "2")):
    env = {**os.environ, "MODEL_PROVIDER": "openai", "OPENAI_API_KEY": "protocol-test-placeholder",
        "OPENAI_LOG": "debug",
        "OPENAI_MODEL": "protocol-test", "OPENAI_BASE_URL": endpoint,
        "MODEL_INPUT_USD_PER_MILLION": rates[0], "MODEL_OUTPUT_USD_PER_MILLION": rates[1], "CRASH_AT": ""}
    return subprocess.run([sys.executable, str(ROOT / "agent.py"), *args], cwd=folder,
        env=env, capture_output=True, text=True, timeout=30)


def test_sdk_structured_output_sessions_and_usage(tmp_path, protocol_endpoint):
    endpoint, calls, mode = protocol_endpoint
    reset.reset_world(5, tmp_path)
    first = run_agent(tmp_path, endpoint)
    assert first.returncode == 0, first.stdout + first.stderr
    state = read_json(tmp_path / "state.json", {})
    assert state["status"] == "waiting_approval"
    assert len(calls) == 1
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert state["token_usage"]["input_tokens"] == 1000
    assert state["token_usage"]["output_tokens"] == 50
    assert state["token_usage"]["cost_usd"] == pytest.approx(0.0011)
    assert state["token_usage"]["reserved_cost_usd"] == 0
    assert (tmp_path / "sessions.db").stat().st_size > 0
    assert b"canary-" not in (tmp_path / "sessions.db").read_bytes()
    assert "canary-" not in json.dumps(calls)
    assert run_agent(tmp_path, endpoint, "--approve").returncode == 0
    assert len(calls) == 1, "approval must reuse the durable plan"
    assert read_json(tmp_path / "state.json", {})["status"] == "done"


def test_cost_guard_stops_before_network(tmp_path, protocol_endpoint):
    endpoint, calls, mode = protocol_endpoint
    reset.reset_world(1, tmp_path)
    assert run_agent(tmp_path, endpoint, rates=("1000000", "1000000")).returncode != 0
    assert not calls
    state = read_json(tmp_path / "state.json", {})
    assert state["reason"] == "cost_budget_exceeded"
    assert not (tmp_path / "bookings.json").exists()


def test_provider_error_is_honest_and_has_no_hidden_retry(tmp_path, protocol_endpoint):
    endpoint, calls, mode = protocol_endpoint
    mode["error"] = True
    reset.reset_world(1, tmp_path)
    assert run_agent(tmp_path, endpoint).returncode != 0
    assert len(calls) == 1
    state = read_json(tmp_path / "state.json", {})
    assert state["status"] == "needs_attention"
    assert state["token_usage"]["reserved_cost_usd"] > 0
    assert not (tmp_path / "bookings.json").exists()


def test_model_generated_marker_is_redacted_before_session_storage(tmp_path, protocol_endpoint):
    endpoint, calls, mode = protocol_endpoint
    mode["secret_output"] = True
    reset.reset_world(5, tmp_path)
    process = run_agent(tmp_path, endpoint)
    assert process.returncode != 0
    assert "canary-" not in process.stdout + process.stderr
    assert "protocol-test-placeholder" not in process.stdout + process.stderr
    assert len(calls) == 1
    state = read_json(tmp_path / "state.json", {})
    assert state["status"] == "needs_attention"
    for name in ("state.json", "journal.jsonl", "sessions.db"):
        assert b"canary-" not in (tmp_path / name).read_bytes()
    assert not (tmp_path / "bookings.json").exists()


def test_blank_base_url_uses_standard_endpoint(monkeypatch):
    import asyncio
    from model_client import configuration
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "protocol-test-placeholder")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.delenv("MODEL_INPUT_USD_PER_MILLION", raising=False)
    monkeypatch.delenv("MODEL_OUTPUT_USD_PER_MILLION", raising=False)
    provider, model, client, input_rate, output_rate = configuration()
    assert str(client.base_url) == "https://api.openai.com/v1/"
    assert (input_rate, output_rate) == (0.75, 4.50)
    asyncio.run(client.close())
