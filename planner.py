"""Bounded schedule search. Names, IDs, addresses and times come from run data."""
import re
from runtime import interval, overlaps


def fitting_venues(context):
    cfg, roster = context["config"], context["teams"]
    return sorted([v for v in context["venues"] if v["available"]
        and 0 <= v["cost_kzt"] <= cfg["max_budget_kzt"]
        and v["total_capacity"] >= sum(t["size"] for t in roster)
        and set(cfg["required_tracks"]) <= {t for h in v["halls"] for t in h["tracks"]}],
        key=lambda v: (v["cost_kzt"], v["id"]))


def preferences(context):
    """Only a roster captain's own feasible time wishes affect ordering."""
    result = {}
    for t in context["teams"]:
        wishes = set()
        for msg in context["messages"]:
            if msg.get("role") == "captain" and msg.get("team_id") == t["id"] and msg.get("sender") == t["captain_email"]:
                wishes.update(re.findall(r"\d{2}:\d{2}-\d{2}:\d{2}", msg.get("text", "")))
        result[t["id"]] = wishes
    return result


def build_plan(context, venue, team_order=None):
    wishes = preferences(context)
    order = team_order or sorted(t["id"] for t in context["teams"])
    if len(order) != len(context["teams"]) or set(order) != {t["id"] for t in context["teams"]}:
        raise ValueError("invalid_team_order")
    rank = {tid: i for i, tid in enumerate(order)}
    roster = sorted(context["teams"], key=lambda t: (not bool(wishes[t["id"]]), rank[t["id"]]))
    candidates = {}
    for t in roster:
        options = []
        pitch_length = interval(t["current_pitch_slot"])[1] - interval(t["current_pitch_slot"])[0]
        for hall in venue["halls"]:
            if t["track"] not in hall["tracks"] or hall["capacity"] < t["size"]:
                continue
            for m in context["mentors"]:
                if m["track"] != t["track"]:
                    continue
                for ms in sorted(set(m["available_slots"]) & set(hall["available_slots"])):
                    for ps in sorted(hall["available_slots"]):
                        if interval(ps)[0] < interval(ms)[1] or interval(ps)[1] - interval(ps)[0] != pitch_length:
                            continue
                        options.append({"team_id": t["id"], "venue_id": venue["id"], "hall_id": hall["hall_id"],
                            "mentor_id": m["id"], "mentor_slot": ms, "pitch_slot": ps, "idempotency_key": f"slot:{t['id']}"})
        candidates[t["id"]] = sorted(options, key=lambda e: (
            -len(wishes[t["id"]] & {e["mentor_slot"], e["pitch_slot"]}), e["mentor_slot"], e["pitch_slot"], e["hall_id"]))
    nodes = 0

    def search(chosen):
        nonlocal nodes
        nodes += 1
        if nodes > 20000:
            raise ValueError("planning_limit")
        if len(chosen) == len(roster):
            return chosen
        for candidate in candidates[roster[len(chosen)]["id"]]:
            if any((candidate["hall_id"] == e["hall_id"] and any(overlaps(a, b)
                    for a in (candidate["mentor_slot"], candidate["pitch_slot"])
                    for b in (e["mentor_slot"], e["pitch_slot"])))
                    or (candidate["mentor_id"] == e["mentor_id"] and overlaps(candidate["mentor_slot"], e["mentor_slot"]))
                    for e in chosen):
                continue
            found = search(chosen + [candidate])
            if found is not None:
                return found
        return None

    result = search([])
    if result is None:
        raise ValueError("no_feasible_schedule")
    return sorted(result, key=lambda e: e["team_id"])


def notifications(context, venue, assignments):
    schedule = {e["team_id"]: e for e in assignments}
    subject = "Новое расписание хакатона"
    result = []
    for t in context["teams"]:
        e = schedule[t["id"]]
        text = (f"{t['name']}: новая площадка {venue['name']}, зал {e['hall_id']}. "
                f"Ментор {e['mentor_id']}: {e['mentor_slot']}. Питч: {e['pitch_slot']}.")
        result.append({"recipient": t["captain_email"], "subject": subject, "text": text,
                       "idempotency_key": f"notice:{t['captain_email']}"})
    for m in context["mentors"]:
        entries = [e for e in assignments if e["mentor_id"] == m["id"]]
        text = f"{m['name']}: новая площадка {venue['name']}. " + "; ".join(
            f"{e['team_id']}, зал {e['hall_id']}, менторская сессия {e['mentor_slot']}, питч {e['pitch_slot']}" for e in entries)
        result.append({"recipient": m["email"], "subject": subject, "text": text,
                       "idempotency_key": f"notice:{m['email']}"})
    return result
