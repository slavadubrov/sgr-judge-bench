"""Bounded direct/SGR table experiment using the existing provider transport."""

import asyncio
import json
import os
import random
import statistics
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import httpx
import jsonschema

from .adapters import HTTPJudge, parse_response, request_body
from .core import digest, dumps, read_jsonl, write_json
from .runner import code_hash
from .sgr import (
    aggregate,
    aggregate_native,
    materialize_checks,
    native_case,
    parse_stage,
    stage_request,
)
from .uncertainty import cluster_mean_interval


def append(path, value):
    with path.open("a") as f:
        f.write(dumps(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def reservation(config, body):
    # UTF-8 byte bound plus wrapper headroom; released to observed upper cost after response.
    tokens = len(dumps(body).encode()) + 2048
    rates = config["rates"]
    return (
        4
        * (
            tokens * max(v or 0 for k, v in rates.items() if k != "output")
            + (body.get("max_output_tokens") or body.get("max_tokens") or 0) * rates["output"]
        )
        / 1e6
    )


def cost_bounds(call):
    c = call["cost"]
    if c.get("usd") is not None:
        return c["usd"], c["usd"]
    if c.get("usd_bounds"):
        return tuple(c["usd_bounds"])
    return None, None


def canaries():
    raw = {
        "caption": "Staff",
        "headers": ["name", "role"],
        "rows": [["Ada", "assistant manager"], ["Ben", "manager"]],
    }
    return [
        {
            "id": f"canary:{i}",
            "channel": "synthetic",
            "gold": gold,
            "source_url": f"synthetic:{i}",
            "input": {**raw, "claim": claim},
        }
        for i, (claim, gold) in enumerate(
            [
                ("Ada is an assistant manager.", "ENTAILED"),
                ("Ada is a manager.", "REFUTED"),
                ("There are exactly two staff members.", "ENTAILED"),
            ]
        )
    ]


async def run(
    cases,
    models,
    out,
    budget,
    *,
    planner="terra",
    phase="test",
    include_hybrid=True,
    include_prompt_control=False,
):
    root = Path(out)
    if root.exists():
        raise ValueError("Refusing to overwrite a recorded run")
    if budget <= 0 or not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Need positive budget and nonempty unique cases")
    if (
        include_hybrid
        and any(m["adapter"] == "jev" for m in models.values())
        and (planner not in models or models[planner]["adapter"] == "jev")
    ):
        raise ValueError("The hybrid planner must be a configured generative model")
    root.mkdir(parents=True)
    jobs = [
        (c, name, arm)
        for c in cases
        for name, config in models.items()
        for arm in (
            ("direct", "hybrid") if config["adapter"] == "jev" else ("direct_guided", "sgr")
        )
        if include_hybrid or arm != "hybrid"
    ]
    if include_prompt_control:
        jobs.extend(
            (c, name, "direct")
            for c in cases
            for name, config in models.items()
            if config["adapter"] != "jev"
        )
    random.Random(20260925).shuffle(jobs)
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "phase": phase,
        "case_sha256": digest(cases),
        "models": models,
        "code_hash": code_hash(),
        "planner": planner,
        "budget_usd": budget,
        "timeout_s": 120,
        "concurrency_per_model": 2,
        "retries": 0,
        "repairs": 0,
        "client_location": "Not independently verified in this run",
        "include_hybrid": include_hybrid,
        "include_prompt_control": include_prompt_control,
        "policy": f"Frozen SGR v2. All failures stay in denominator. Any native Jev hybrid reuses the {planner} plan; attributed cost/latency includes it. No generated Jev findings/citations.",
        "jobs": [{"id": c["id"], "model": n, "arm": a} for c, n, a in jobs],
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "cases.json", cases)
    (root / "source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        (root / "source" / path.name).write_bytes(path.read_bytes())
    plans = {c["id"]: asyncio.get_running_loop().create_future() for c in cases}
    sems = {name: asyncio.Semaphore(2) for name in models}
    accounting = {"charged_upper_or_reserved_usd": 0.0, "inflight_usd": 0.0, "requests": 0}
    started = time.perf_counter()
    async with httpx.AsyncClient() as client:
        judges = {name: HTTPJudge(config, client, timeout=120) for name, config in models.items()}

        async def call(case, name, stage, body, parser):
            async with sems[name]:
                reserve = reservation(models[name], body)
                if (
                    sum(accounting[k] for k in ("charged_upper_or_reserved_usd", "inflight_usd"))
                    + reserve
                    > budget
                ):
                    raise ValueError("budget_exhausted")
                accounting["inflight_usd"] += reserve
                record = await judges[name].request(body, parser)
                _, upper = cost_bounds(record)
                accounting["inflight_usd"] -= reserve
                accounting["charged_upper_or_reserved_usd"] += (
                    upper if upper is not None else reserve
                )
                accounting["requests"] += record["request_count"]
                record.update(
                    call_id=f"{name}:{case['id']}:{stage}",
                    case_id=case["id"],
                    model=name,
                    stage=stage,
                )
                append(root / "calls.jsonl", record)
                write_json(root / "ledger.json", accounting)
                return record

        async def generated(case, name, stage, record, views=None):
            body, schema = stage_request(models[name], case["input"], stage, views)
            call_record = await call(
                case, name, stage, body, lambda raw: parse_stage(models[name], raw, schema)
            )
            record["calls"].append(call_record["call_id"])
            if call_record["status"] != "ok":
                raise ValueError(call_record["error"])
            return call_record["parsed"]

        async def job(case, name, arm):
            record = {
                "id": case["id"],
                "model": name,
                "arm": arm,
                "calls": [],
                "status": "error",
                "prediction": None,
            }
            begin = time.perf_counter()
            try:
                if arm == "hybrid":
                    plan = await plans[case["id"]]
                    if plan is None:
                        raise ValueError("upstream_planner_failed")
                    record["calls"].append(plan["call_id"])
                    record["checks"] = plan["checks"]
                    views = materialize_checks(case["input"], plan["checks"])
                    record["source_views"] = views
                    native = native_case(case["input"], views)
                elif models[name]["adapter"] == "jev":
                    native = native_case(
                        case["input"], detailed=models[name].get("detailed_prompt", False)
                    )
                else:
                    native = None
                if native is not None:
                    c = await call(
                        case,
                        name,
                        arm,
                        request_body(models[name], native, "P"),
                        lambda raw: parse_response(models[name], raw, native, "P"),
                    )
                    record["calls"].append(c["call_id"])
                    if c["status"] != "ok":
                        raise ValueError(c["error"])
                    record["assessment"] = c["parsed"]
                    labels = c["parsed"]["labels"]
                    record["prediction"] = (
                        labels["label"] if arm == "direct" else aggregate_native(labels)
                    )
                elif arm in ("direct", "direct_guided"):
                    record["prediction"] = (await generated(case, name, arm, record))["label"]
                else:
                    checks = (await generated(case, name, "plan", record))["checks"]
                    record["checks"] = checks
                    views = materialize_checks(case["input"], checks)
                    record["source_views"] = views
                    if name == planner:
                        plans[case["id"]].set_result(
                            {"checks": checks, "call_id": record["calls"][0]}
                        )
                    assessment = await generated(case, name, "assess", record, views)
                    record["assessment"] = assessment
                    record["prediction"] = aggregate(case["input"], checks, assessment)
                record["status"] = "ok"
            except (ValueError, KeyError, TypeError, IndexError, jsonschema.ValidationError) as exc:
                record["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
            finally:
                if name == planner and arm == "sgr" and not plans[case["id"]].done():
                    plans[case["id"]].set_result(None)
            record["wall_latency_with_queue_s"] = time.perf_counter() - begin
            append(root / "records.jsonl", record)
            print(
                f"{name} {arm} {case['id']} {record['status']} {record['prediction']}", flush=True
            )
            return record

        records = await asyncio.gather(*(job(c, n, a) for c, n, a in jobs))
    write_json(
        root / "completion.json",
        {
            **accounting,
            "records": len(records),
            "wall_s": time.perf_counter() - started,
            "finished_at": datetime.now(UTC).isoformat(),
        },
    )
    return report(root)


def report(root):
    root = Path(root)
    cases = json.loads((root / "cases.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    records = read_jsonl(root / "records.jsonl") if (root / "records.jsonl").exists() else []
    calls = (
        {c["call_id"]: c for c in read_jsonl(root / "calls.jsonl")}
        if (root / "calls.jsonl").exists()
        else {}
    )
    by_key = {(r["model"], r["arm"], r["id"]): r for r in records}
    if len(by_key) != len(records):
        raise ValueError("Duplicate evaluation record")
    groups = [c["source_url"] for c in cases]
    summary, scores, predictions = {}, {}, {}
    for name, arm in sorted({(j["model"], j["arm"]) for j in manifest["jobs"]}):
        key = f"{name}/{arm}"
        rows = [
            by_key.get(
                (name, arm, c["id"]), {"status": "not_attempted", "prediction": None, "calls": []}
            )
            for c in cases
        ]
        scores[key] = [
            int(r["status"] == "ok" and r["prediction"] == c["gold"])
            for r, c in zip(rows, cases, strict=True)
        ]
        predictions[key] = [r["prediction"] for r in rows]
        used = [calls[cid] for r in rows for cid in r["calls"]]
        bounds = [cost_bounds(c) for c in used]
        latency = [sum(calls[cid]["latency_s"] for cid in r["calls"]) for r in rows]
        summary[key] = {
            "correct": sum(scores[key]),
            "n": len(cases),
            "valid": sum(r["status"] == "ok" for r in rows),
            "accuracy": cluster_mean_interval(scores[key], groups),
            "attributed_requests": len(used),
            "known_cost_lower_usd": sum(lo for lo, _ in bounds if lo is not None),
            "known_cost_upper_usd": sum(hi for _, hi in bounds if hi is not None),
            "unknown_cost_calls": sum(lo is None for lo, _ in bounds),
            "median_service_latency_s": statistics.median(latency),
            "p95_service_latency_s": sorted(latency)[
                min(len(latency) - 1, int(0.95 * len(latency)))
            ],
            "by_channel": {
                ch: {
                    "correct": sum(
                        s for s, c in zip(scores[key], cases, strict=True) if c["channel"] == ch
                    ),
                    "n": sum(c["channel"] == ch for c in cases),
                }
                for ch in sorted({c["channel"] for c in cases})
            },
        }
    comparisons = {}
    for a, b in combinations(summary, 2):
        delta = [y - x for x, y in zip(scores[a], scores[b], strict=True)]
        comparisons[a + " -> " + b] = {
            **cluster_mean_interval(delta, groups),
            "b_wins": sum(v == 1 for v in delta),
            "a_wins": sum(v == -1 for v in delta),
            "valid_prediction_disagreements": sum(
                x is not None and y is not None and x != y
                for x, y in zip(predictions[a], predictions[b], strict=True)
            ),
        }
    value = {
        "arms": summary,
        "paired_comparisons": comparisons,
        "unique_actual_requests": len(calls),
        "missing_records": len(manifest["jobs"]) - len(records),
        "note": "All failures incorrect; one page/table per case. Paired page bootstrap, 10000 resamples; exploratory unadjusted intervals. Service latency sums requests and excludes queues.",
    }
    if any(key.endswith("/hybrid") for key in summary):
        value["note"] += " Hybrid totals include shared planning; actual requests count it once."
    write_json(root / "summary.json", value)
    lines = [
        "# TabFact direct / SGR comparison",
        "",
        value["note"],
        "",
        "| Model / mode | Correct | Valid | Cost USD bounds (known calls) | Unknown cost calls | Median service s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, s in summary.items():
        lines.append(
            f"| {key} | {s['correct']}/{s['n']} | {s['valid']}/{s['n']} | {s['known_cost_lower_usd']:.4f}–{s['known_cost_upper_usd']:.4f} | {s['unknown_cost_calls']} | {s['median_service_latency_s']:.2f} |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return value
