"""One explicit live experiment with a shared, persisted spending ledger."""

import json
from datetime import UTC, datetime
from pathlib import Path

from .core import read_jsonl, write_json
from .report import report
from .runner import execute, execute_cascade, fit_and_freeze, load_records


async def experiment(data, models, admission_path, output, budget, prior_spend, location):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "ledger.json").exists():
        raise ValueError("experiment exists; do not silently rerun or spend twice")
    cases = read_jsonl(data / "cases.jsonl")
    subsets = json.loads((data / "subsets.json").read_text())
    admission = json.loads(Path(admission_path).read_text())
    spent = prior_spend
    completed = []

    def ledger(state):
        write_json(
            output / "ledger.json",
            {
                "total_budget_usd": budget,
                "prior_spend_usd": prior_spend,
                "accounted_usd": spent,
                "remaining_usd": max(0, budget - spent),
                "state": state,
                "completed_runs": completed,
                "updated_at": datetime.now(UTC).isoformat(),
                "location": location,
            },
        )

    ledger("running")
    # Preregistered phase reservations preserve test evidence if development is unexpectedly costly.
    fractions = {
        "development": 0.23,
        "calibration": 0.17,
        "routing": 0.17,
        "test": 0.28,
        "cascade": 0.07,
        "robustness": 0.04,
        "repeat-1": 0.01,
        "repeat-2": 0.02,
        "repeat-3": 0.01,
    }
    allowance = budget - prior_spend
    carry = 0
    frozen = None
    for phase in fractions:
        phase_budget = min(budget - spent, allowance * fractions[phase] + carry)
        path = output / phase
        if phase == "cascade":
            await execute_cascade(
                [c for c in cases if c["split"] == "test"],
                models,
                frozen,
                path,
                budget=max(0, phase_budget),
                location=location,
            )
        else:
            logical = "repeat" if phase.startswith("repeat-") else phase
            selected = (
                read_jsonl(data / "probes.jsonl")
                if logical == "robustness"
                else [
                    c for c in cases if c["split"] == ("test" if logical == "repeat" else logical)
                ]
            )
            await execute(
                selected,
                models,
                path,
                phase=logical,
                budget=max(0, phase_budget),
                location=location,
                preflight=admission,
                frozen=frozen,
                repeat_ids=subsets["repeat_ids"],
                block=int(phase[-1]) if logical == "repeat" else None,
            )
        completion = path / "completion.json"
        if completion.exists():
            charge = json.loads(completion.read_text())["budget_accounted_usd"]
        else:
            _, rows = load_records(path)
            charge = sum(r["cost"].get("usd") or 0 for r in rows)
        spent += charge
        carry = max(0, phase_budget - charge)
        completed.append(str(path))
        ledger("running")
        if phase == "routing":
            frozen = fit_and_freeze(
                data,
                models,
                [output / p for p in ("development", "calibration", "routing")],
                output / "frozen.json",
                data / "probes.jsonl",
            )
        if phase == "test":
            report([path], output / "primary-results", frozen=frozen)
    report(completed, output / "results", frozen=frozen)
    ledger("completed_with_results; inspect missingness before claims")
    return {"spent_accounted_usd": spent, "report": str(output / "results/report.md")}
