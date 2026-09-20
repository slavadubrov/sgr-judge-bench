"""Join independently recorded model runs without rerunning inference or dropping cases."""

import argparse
import hashlib
import json
from pathlib import Path

from scipy.stats import binomtest
from tabfact_audit import audit

from judge_bench.core import read_jsonl, write_json, write_jsonl
from judge_bench.tabfact import report
from judge_bench.uncertainty import cluster_mean_interval


def compare(runs, out):
    out = Path(out)
    if out.exists():
        raise ValueError("Refusing to overwrite comparison")
    cases, models, jobs, records, calls, sources = None, {}, [], [], [], []
    for directory in map(Path, runs):
        audit(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        cohort = json.loads((directory / "cases.json").read_text())
        if cases is not None and cohort != cases:
            raise ValueError("Cohorts differ")
        if set(models) & set(manifest["models"]):
            raise ValueError("Repeated model; repeated runs need a different analysis")
        cases = cohort
        models.update(manifest["models"])
        jobs.extend(manifest["jobs"])
        records.extend(read_jsonl(directory / "records.jsonl"))
        calls.extend(read_jsonl(directory / "calls.jsonl"))
        sources.append(
            {
                "run": str(directory),
                "manifest_sha256": hashlib.sha256(
                    (directory / "manifest.json").read_bytes()
                ).hexdigest(),
            }
        )
    out.mkdir(parents=True)
    write_json(
        out / "manifest.json",
        {
            "phase": "derived_offline_comparison",
            "models": models,
            "jobs": jobs,
            "source_runs": sources,
            "note": "Models were run in separate time blocks; no outputs selected or cases removed. This directory is a derived view, not a new live run.",
        },
    )
    write_json(out / "cases.json", cases)
    write_jsonl(out / "records.jsonl", records)
    write_jsonl(out / "calls.jsonl", calls)
    report(out)
    by_key = {(r["model"], r["arm"], r["id"]): r for r in records}
    groups = [c["source_url"] for c in cases]

    def correct(name, arm, c):
        r = by_key[name, arm, c["id"]]
        return int(r["status"] == "ok" and r["prediction"] == c["gold"])

    def gains(name):
        return [correct(name, "sgr", c) - correct(name, "direct", c) for c in cases]

    tests = {}
    for name in ("luna", "deepseek-json", "terra"):
        if name not in models:
            continue
        delta = gains(name)
        wins = delta.count(1)
        losses = delta.count(-1)
        tests[name] = {
            "gain": cluster_mean_interval(delta, groups),
            "fixed_errors": wins,
            "new_errors": losses,
            "mcnemar_exact_two_sided_p": binomtest(wins, wins + losses, 0.5).pvalue
            if wins + losses
            else 1.0,
        }
        if name != "terra" and "terra" in models:
            tests[name]["gain_minus_terra_gain"] = cluster_mean_interval(
                [a - b for a, b in zip(delta, gains("terra"), strict=True)], groups
            )
            agreement = {}
            for arm in ("direct", "sgr"):
                agreement[arm] = {
                    "same_valid_prediction": 0,
                    "both_correct": 0,
                    "same_wrong_prediction": 0,
                    "either_invalid": 0,
                    "n": len(cases),
                }
                for c in cases:
                    a, b = (by_key[m, arm, c["id"]] for m in (name, "terra"))
                    valid = a["status"] == b["status"] == "ok"
                    same = valid and a["prediction"] == b["prediction"]
                    agreement[arm]["same_valid_prediction"] += int(same)
                    agreement[arm]["both_correct"] += int(
                        valid and a["prediction"] == b["prediction"] == c["gold"]
                    )
                    agreement[arm]["same_wrong_prediction"] += int(
                        same and a["prediction"] != c["gold"]
                    )
                    agreement[arm]["either_invalid"] += int(not valid)
            tests[name]["agreement_with_terra"] = agreement
    write_json(
        out / "hypothesis.json",
        {
            "comparisons": tests,
            "limitation": "Exploratory extension on an already observed cohort; unchanged inference code and prompts. Multiple comparisons unadjusted; two or three models do not establish a general law about capability. Agreement includes shared mistakes.",
        },
    )
    return tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.runs, args.out), indent=2))
