"""RAGTruth-only, one-hour pilot. Includes a 60-case sanity gate."""

import argparse
import asyncio
import json
import math
import random
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from judge_bench.core import SEED, digest, read_jsonl, write_json
from judge_bench.report import report
from judge_bench.runner import (
    code_hash,
    execute,
    flat_record,
    load_models,
    load_records,
    reserve_usd,
)


def sanity(rows):
    checks = {}
    for judge in sorted({r["judge"] for r in rows}):
        selected = [flat_record(r) for r in rows if r["judge"] == judge]
        valid = [r for r in selected if r["status"] == "ok"]
        predictions = Counter(r["prediction"] for r in valid)
        correct = sum(r["prediction"] == r["gold"] for r in selected)
        reasons = []
        if len(valid) < 0.8 * len(selected):
            reasons.append("over_20_percent_invalid")
        if len(predictions) < 2:
            reasons.append("constant_or_empty_prediction")
        if correct in (0, len(selected)):
            reasons.append("zero_or_perfect_accuracy_requires_review")
        checks[judge] = {
            "n": len(selected),
            "valid": len(valid),
            "correct": correct,
            "predictions": dict(predictions),
            "stop_reasons": reasons,
        }
    return {
        "continue": bool(checks) and all(not c["stop_reasons"] for c in checks.values()),
        "judges": checks,
    }


async def pilot(args):
    if not math.isfinite(args.prior_spend_usd) or not 0 <= args.prior_spend_usd < 10:
        raise ValueError("prior spend must be finite and within the shared $10 budget")
    root = Path(args.out)
    if root.exists():
        raise ValueError("pilot exists; refusing duplicate paid execution")
    root.mkdir(parents=True)
    original = load_models(args.models)
    prior_admission = json.loads(Path(args.preflight).read_text())
    models, admission = {}, {"admission": {}}
    for name in ("jev", "luna", "deepseek-json", "glm-current"):
        profile = "P" if name == "jev" else "D"
        canary = prior_admission["admission"][name + ":" + profile]
        if not canary["admitted"] or canary["config_hash"] != digest(original[name]):
            raise ValueError("primary configuration lacks matching capability check: " + name)
        models[name] = {**original[name], "profiles": [profile]}
        admission["admission"][name + ":" + profile] = {
            **canary,
            "config_hash": digest(models[name]),
            "pilot_note": "Only execution profile roster narrowed; request body/settings unchanged.",
        }
    cases = [
        c
        for c in read_jsonl("data/cases.jsonl")
        if c["split"] == "test" and c["task"] == "ragtruth"
    ]
    groups = sorted({c["group"] for c in cases})
    random.Random(SEED).shuffle(groups)
    first_groups = set(groups[:10])
    batches = {
        "sanity": [c for c in cases if c["group"] in first_groups],
        "remainder": [c for c in cases if c["group"] not in first_groups],
    }
    assert len(cases) == 900 and len(batches["sanity"]) == 60
    frozen = {
        "code_hash": code_hash(),
        "models_hash": digest(models),
        "policies": {},
        "case_hashes": {c["id"]: digest(c) for c in cases},
        "scope": "RAGTruth QA test; no calibration or cascade.",
    }
    write_json(root / "frozen.json", frozen)
    write_json(root / "models.json", models)
    deadline_reserve = sum(max(reserve_usd(cfg, c) for c in cases) for cfg in models.values())
    allowance = min(1.0, 10.0 - args.prior_spend_usd - deadline_reserve)
    if allowance <= 0:
        raise ValueError("shared $10 budget exhausted")
    ledger = {
        "state": "running",
        "prior_accounted_usd": args.prior_spend_usd,
        "pilot_budget_usd": allowance,
        "deadline_reserve_usd": deadline_reserve,
        "deadline_seconds": 3600,
        "expected_cases": len(cases),
        "primary_profiles": {k: v["profiles"] for k, v in models.items()},
        "scope": "RAGTruth QA test: one pass with four judge configurations.",
        "excluded": [
            "ANLI",
            "DeepSeek beta",
            "old GLM",
            "LLM P",
            "calibration",
            "cascade",
            "repeats",
            "robustness",
        ],
        "sanity_rule": "Stop all for review if any judge has >20% invalid, constant/empty labels, or zero/perfect accuracy on the first 60. Never drop that judge and declare a win.",
    }
    write_json(root / "ledger.json", ledger)
    paths, spent = [], 0
    try:
        async with asyncio.timeout(3600):
            for name, batch in batches.items():
                path = root / name
                paths.append(path)
                await execute(
                    batch,
                    models,
                    path,
                    phase="test",
                    budget=max(0, allowance - spent),
                    location=args.location,
                    preflight=admission,
                    frozen=frozen,
                )
                _, rows = load_records(path)
                spent += sum(r.get("budget_accounted_usd", 0) for r in rows)
                if name == "sanity":
                    checks = sanity(rows)
                    write_json(root / "sanity-check.json", checks)
                    print(json.dumps(checks), flush=True)
                    if not checks["continue"]:
                        ledger["state"] = "stopped_for_sanity_review"
                        break
            else:
                ledger["state"] = "completed"
    except TimeoutError:
        ledger["state"] = "stopped_at_one_hour"
    finally:
        actual = [r for p in paths if (p / "manifest.json").exists() for r in load_records(p)[1]]
        ledger["pilot_accounted_usd"] = sum(r.get("budget_accounted_usd", 0) for r in actual)
        ledger["total_accounted_with_reserve_usd"] = (
            args.prior_spend_usd + ledger["pilot_accounted_usd"] + deadline_reserve
        )
        ledger["requests"] = sum(r.get("request_count", 0) for r in actual)
        ledger["complete_cohort"] = len(actual) == 3600 and all(
            r.get("request_count") for r in actual
        )
        write_json(root / "ledger.json", ledger)
    report(paths, root / "results", frozen=frozen)
    metrics_path = root / "results/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["pilot_scope"] = ledger
    if not ledger["complete_cohort"]:
        for comparison in metrics["paired_comparisons"].values():
            comparison["H1"] = "untested: pilot stopped before the full fixed cohort"
            comparison["H2"] = "untested: pilot stopped before the full fixed cohort"
    write_json(metrics_path, metrics)
    report_path = root / "results/report.md"
    warning = (
        "# RAGTruth pilot\n\n"
        "One dataset and one profile per primary model. No post-hoc calibration, cascades, "
        "LLM probability comparison, repeated runs or robustness assessment. "
        "Results apply to this dataset and measurement session.\n\n"
    )
    if not ledger["complete_cohort"]:
        warning += "**Stopped before all 900 cases: no confirmatory H1/H2 claim is valid.**\n\n"
    report_path.write_text(warning + report_path.read_text())
    ledger["report_ready"] = True
    write_json(root / "ledger.json", ledger)
    print(json.dumps(ledger), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--out", default="runs/ragtruth-pilot-1h")
    parser.add_argument(
        "--preflight", default="runs/preflight/admission.json"
    )
    parser.add_argument("--models", default="config/pilot.json")
    parser.add_argument("--location", required=True)
    parser.add_argument("--prior-spend-usd", type=float, required=True)
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env", override=False)
    asyncio.run(pilot(args))
