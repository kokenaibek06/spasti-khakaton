"""Write an auditable API usage summary from saved scenario state files."""
import argparse
import json
from pathlib import Path


def summarize(folder):
    rows = []
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "reserved_cost_usd": 0.0}
    for path in sorted(folder.glob("scenario-*/state.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        usage = state["token_usage"]
        row = {"scenario": path.parent.name, "status": state["status"], "model": usage.get("model", "не вызывалась"),
               **{k: usage.get(k, 0) for k in totals}}
        rows.append(row)
        for key in totals:
            totals[key] += row[key]
    if not rows:
        raise SystemExit("Не найдены scenario-*/state.json")
    lines = ["# Фактическое использование API", "",
        "Значения взяты из state.json. Стоимость рассчитана по настроенным тарифам.",
        "Для offline вызовы и токены равны нулю. Резерв показывает неизвестную стоимость после ошибки.", "",
        "| Сценарий | Статус | Модель | Вызовы | Вход | Выход | USD | Резерв USD |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['scenario']} | {row['status']} | {row['model']} | {row['calls']} | "
            f"{row['input_tokens']} | {row['output_tokens']} | {row['cost_usd']:.8f} | {row['reserved_cost_usd']:.8f} |")
    lines += ["", f"Итого: {totals['calls']} вызовов, {totals['input_tokens']} входных и "
        f"{totals['output_tokens']} выходных токенов, {totals['cost_usd']:.8f} USD; "
        f"резерв неизвестного списания {totals['reserved_cost_usd']:.8f} USD.", ""]
    (folder / "BUDGET.md").write_text("\n".join(lines), encoding="utf-8")
    (folder / "budget.json").write_text(json.dumps({"scenarios": rows, "totals": totals},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Бюджет сохранён: {folder / 'BUDGET.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
