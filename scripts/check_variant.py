"""Run the unchanged grader on disposable data with rotated external fields."""
import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evals import graders


def prepare(folder, scenario):
    salt = uuid.uuid4().hex[:10]
    data = folder / "data"
    emails = {}
    for name, field in (("teams.json", "captain_email"), ("mentors.json", "email")):
        rows = json.loads((data / name).read_text(encoding="utf-8"))
        for row in rows:
            old = row[field]
            local, domain = old.split("@")
            emails[old] = f"rotated-{salt}-{local}@{domain}"
            row[field] = emails[old]
            row["name"] = f"Участник {row['id']} {salt}"
        (data / name).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    venue_fields = {"id", "name", "available", "total_capacity", "cost_kzt", "halls", "fault"}
    mail_fields = {"id", "sender", "role", "team_id", "subject", "text"}
    for name, fields in (("venues.json", venue_fields), ("inbox.json", mail_fields)):
        rows = json.loads((data / name).read_text(encoding="utf-8"))
        for i, row in enumerate(rows):
            for key in list(row):
                if key not in fields:
                    del row[key]
                    row[f"rotated_internal_{salt}_{i}_{len(row)}"] = f"canary-{uuid.uuid4().hex}"
            row[f"new_service_field_{salt}"] = f"canary-{uuid.uuid4().hex}"
            if name == "venues.json":
                row["name"] = f"Новая площадка {row['id']} {salt}"
                for hall in row["halls"]:
                    hall[f"hall_private_{salt}"] = f"canary-{uuid.uuid4().hex}"
            else:
                row["sender"] = emails.get(row["sender"], row["sender"])
                row["text"] = row.get("text", "").replace("https://", "https://rotated.example/blocked/")
                # A marker inside an allowed text field exercises secondary redaction.
                row["text"] += f" Диагностика: canary-{uuid.uuid4().hex}"
        (data / name).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs-variant"))
    parser.add_argument("--provider", choices=["offline", "openai", "ollama", "hf"], default="offline")
    args = parser.parse_args()
    raise SystemExit(graders.run_all_scenarios(out_dir=args.out, prepare=prepare,
        env_extra={"MODEL_PROVIDER": args.provider}))
