"""Verify that the protected starter files retain their original bytes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    expected = json.loads((ROOT / "starter-integrity.json").read_text())
    wrong = [name for name, digest in expected.items() if not (ROOT / name).is_file()
             or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    if wrong:
        print("Изменены защищённые файлы: " + ", ".join(wrong))
        return 1
    print(f"Целостность раздатки: PASS, {len(expected)} файлов без изменений.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
