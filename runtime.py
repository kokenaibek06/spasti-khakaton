"""Durable task state and safe audit log, relative to the current run."""
import json
import os
import re
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def clean(value):
    """Secondary redaction; source objects must first pass explicit allowlists."""
    if isinstance(value, str):
        return re.sub(r"canary-[\w.:-]*", "[REDACTED]", value, flags=re.I)
    if isinstance(value, dict):
        return {clean(str(k)): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def read_json(path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def write_json(path, value):
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def journal(event, **fields):
    row = clean({"at": datetime.now(timezone.utc).isoformat(), "event": event, **fields})
    with Path("journal.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def file_lock(name):
    """OS locks release on process death, including os._exit inside world.py."""
    with Path(name).open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            if f.tell() == 0:
                f.write(b"0")
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def interval(slot):
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d-(?:[01]\d|2[0-3]):[0-5]\d", slot):
        raise ValueError("invalid_time_range")
    start, end = (int(x[:2]) * 60 + int(x[3:]) for x in slot.split("-"))
    if start >= end:
        raise ValueError("invalid_time_range")
    return start, end


def overlaps(a, b):
    x, y = interval(a), interval(b)
    return x[0] < y[1] and y[0] < x[1]


def normalized(value):
    return unicodedata.normalize("NFKC", value).casefold()
