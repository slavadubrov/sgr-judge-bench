"""Append-only attempts, immutable plans and explicit live execution."""

import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .adapters import HTTPJudge, request_body
from .core import SEED, digest, dumps, read_jsonl, reference, spec_for, write_json, write_jsonl
from .metrics import choose_threshold, classification, fit_temperature, latency, temperature_scale


def load_models(path):
    document = json.loads(Path(path).read_text())
    models = {}
    for name, entry in document["models"].items():
        config = {**document["models"].get(entry.get("extends"), {}), **entry}
        config.pop("extends", None)
        models[name] = config
    return models


def code_hash():
    root = Path(__file__).parent
    return digest(
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*.py"))
        }
    )


def load_records(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    actual = (
        {r["attempt_id"]: r for r in read_jsonl(run / "attempts.jsonl")}
        if (run / "attempts.jsonl").exists()
        else {}
    )
    rows = []
    for plan in read_jsonl(run / "plan.jsonl"):
        rows.append(
            actual.get(
                plan["attempt_id"],
                {
                    **plan,
                    "status": "not_attempted",
                    "parsed": None,
                    "error": "run_incomplete",
                    "cost": {"usd": None},
                    "usage": {},
                    "request_count": 0,
                },
            )
        )
    return manifest, rows


def flat_record(row):
    field, gold = reference(row["case"])
    parsed = row.get("parsed") or {}
    return {
        **row,
        "case_id": row["case"]["id"],
        "gold": gold,
        "prediction": parsed.get("labels", {}).get(field),
        "probabilities": parsed.get("probabilities", {}).get(field),
    }


def canaries():
    rows = []
    for claim, gold in [
        ("The crate contains four apples.", "supported"),
        ("The crate contains six apples.", "contradicted"),
        ("The crate is green.", "insufficient_evidence"),
    ]:
        rows.append(
            {
                "id": "canary-" + gold,
                "task": "anli",
                "dataset": "canary",
                "split": "preflight",
                "group": "canary",
                "input": {"premise": "The crate contains exactly four apples.", "claim": claim},
                "gold": {"claim_relation": gold},
            }
        )
    for answer, label in [("There are four apples.", "no"), ("There are four red apples.", "yes")]:
        rows.append(
            {
                "id": "canary-" + label,
                "task": "ragtruth",
                "dataset": "canary",
                "split": "preflight",
                "group": "canary",
                "input": {
                    "question": "What is in the crate?",
                    "context": "The crate contains four apples.",
                    "answer": answer,
                },
                "gold": {"unsupported_claim_present": label},
            }
        )
    return rows


def make_plan(cases, models, phase, repeat_ids=None, block=None):
    rng = random.Random(SEED)
    rng.shuffle(cases)
    plan = []
    for case in cases:
        choices = [
            (name, profile) for name, config in models.items() for profile in config["profiles"]
        ]
        rng.shuffle(choices)
        repetitions = [1] if phase == "test" else [0]
        if phase == "repeat":
            if case["id"] not in repeat_ids:
                continue
            repetitions = {1: [2], 2: [3, 4], 3: [5]}[block]
        for repetition in repetitions:
            for name, profile in choices:
                row = {
                    "case": case,
                    "judge": name,
                    "profile": profile,
                    "phase": phase,
                    "repetition": repetition,
                    "time_block": block,
                    "config": models[name],
                    "contract_hash": digest(spec_for(case)),
                    "prompt_hash": digest(request_body(models[name], case, profile)),
                }
                row["attempt_id"] = digest(row)
                plan.append(row)
    return plan


def reserve_usd(config, case):
    # Conservative byte-based operational reserve, not measured token usage or a hard provider billing cap.
    rates = config["rates"]
    tokens = len(dumps(request_body(config, case, "P")).encode()) * 4
    return (
        (
            tokens * max(v or 0 for k, v in rates.items() if k != "output")
            + (config["max_output"] or 4096) * rates["output"]
        )
        * 4
        / 1e6
    )


async def execute(
    cases,
    models,
    output,
    *,
    phase,
    budget,
    location,
    preflight=None,
    cache_mode="off",
    block=None,
    repeat_ids=None,
    holiday_status=None,
    frozen=None,
):
    output = Path(output)
    if output.exists():
        raise ValueError("run directory already exists; immutable attempts cannot be overwritten")
    if not location.strip():
        raise ValueError("record the actual EU client/network location")
    if phase in {"test", "repeat", "robustness"} and cache_mode != "off":
        raise ValueError("result caching forbidden for timing, test and consistency")
    if phase != "preflight" and preflight is None:
        raise ValueError("capability preflight required")
    if phase in {"test", "repeat", "robustness"}:
        if (
            not frozen
            or frozen["code_hash"] != code_hash()
            or frozen["models_hash"] != digest(models)
        ):
            raise ValueError("test requires matching frozen code and model configuration")
        if any(digest(c) != frozen["case_hashes"].get(c["id"]) for c in cases):
            raise ValueError("case/input/reference drift after freeze")
    plan = make_plan(cases, models, phase, repeat_ids, block)
    manifest = {
        "version": "groundedness-v1",
        "started_at": datetime.now(UTC).isoformat(),
        "phase": phase,
        "models": models,
        "models_hash": digest(models),
        "code_hash": code_hash(),
        "plan_hash": digest(plan),
        "client_location": location,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "transport": "httpx raw REST; no provider SDK",
        "dependencies": {
            p: importlib.metadata.version(p) for p in ("httpx", "jsonschema", "numpy", "scipy")
        },
        "budget_usd": budget,
        "budget_note": "Operational stop with conservative request reserve; unexposed billing remains uncertain",
        "concurrency_per_provider": 1,
        "concurrency_global": len({c["provider"] for c in models.values()}),
        "interleave_block_cases": 10,
        "attempt_timeout_s": 30,
        "retries": 0,
        "cache_mode": cache_mode,
        "seed": SEED,
        "time_block": block,
        "holiday_status": holiday_status,
        "preflight": preflight,
        "freeze_hash": digest(frozen) if frozen else None,
        "unavailable_extensions": [
            "human full-schema references",
            "H5 matched challenge annotations",
            "DeBERTa optional baseline",
        ],
    }
    write_json(output / "manifest.json", manifest)
    write_jsonl(output / "plan.jsonl", plan)
    spent = 0
    clients = {name: httpx.AsyncClient() for name in models}
    judges = {
        name: HTTPJudge(config, clients[name], holiday_status=holiday_status)
        for name, config in models.items()
    }
    in_flight = 0

    async def worker(rows, f):
        nonlocal spent, in_flight
        for row in rows:
            name, config = row["judge"], models[row["judge"]]
            blocked = None
            if phase != "preflight":
                admission = preflight["admission"].get(name + ":" + row["profile"], {})
                if not admission.get("admitted") or admission.get("config_hash") != digest(config):
                    blocked = "not_admitted_by_preflight"
            reserve = reserve_usd(config, row["case"])
            if spent + in_flight + reserve > budget:
                blocked = "budget_reserve_exceeded"
            cache_key = digest(
                {
                    "config": config,
                    "input": row["case"]["input"],
                    "profile": row["profile"],
                    "spec": spec_for(row["case"]),
                    "code": code_hash(),
                }
            )
            cache_path = Path(".cache") / (cache_key + ".json")
            if blocked:
                result = {
                    "status": "not_attempted",
                    "error": blocked,
                    "parsed": None,
                    "request_count": 0,
                    "cost": {"usd": None},
                    "usage": {},
                }
            elif cache_mode == "reuse" and cache_path.exists():
                result = json.loads(cache_path.read_text())
                result.update(
                    application_cache_hit=True,
                    request_count=0,
                    historical_latency_s=result.get("latency_s"),
                    latency_s=None,
                )
            else:
                in_flight += reserve
                result = await judges[name].evaluate(row["case"], row["profile"])
                in_flight -= reserve
                result["application_cache_hit"] = False
                result["budget_accounted_usd"] = (
                    result["cost"].get("usd")
                    if result["cost"].get("usd") is not None
                    else max(result["cost"].get("usd_bounds", [reserve]))
                )
                if result["status"] == "ok":
                    write_json(cache_path, result)
                known = result["cost"].get("usd")
                spent += (
                    known if known is not None else max(result["cost"].get("usd_bounds", [reserve]))
                )
            f.write(dumps({**row, **result}) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"{name}:{row['profile']} {row['case']['id']} {result['status']}", flush=True)

    try:
        with (output / "attempts.jsonl").open("a") as f:
            width = 10 * sum(len(c["profiles"]) for c in models.values())
            for offset in range(0, len(plan), width):
                block_rows = plan[offset : offset + width]
                providers = sorted({row["config"]["provider"] for row in block_rows})
                random.Random(SEED + offset).shuffle(providers)
                await asyncio.gather(
                    *(
                        worker([r for r in block_rows if r["config"]["provider"] == provider], f)
                        for provider in providers
                    )
                )
    finally:
        await asyncio.gather(*(c.aclose() for c in clients.values()))
    write_json(
        output / "completion.json",
        {"finished_at": datetime.now(UTC).isoformat(), "budget_accounted_usd": spent},
    )
    if phase == "preflight":
        _, rows = load_records(output)
        admission = {}
        for name, config in models.items():
            for profile in config["profiles"]:
                selected = [r for r in rows if r["judge"] == name and r["profile"] == profile]
                # Canaries establish protocol support, not benchmark quality.
                ok = any(r["status"] == "ok" and r.get("model_returned") for r in selected)
                unexpected_reasoning = (
                    config["eligible_nonthinking"]
                    and config["adapter"] != "jev"
                    and any(
                        r.get("reasoning_content_present")
                        or (r.get("usage", {}).get("reasoning_tokens") or 0) > 0
                        for r in selected
                    )
                )
                admission[name + ":" + profile] = {
                    "admitted": bool(ok and not unexpected_reasoning),
                    "admission_rule": "At least one valid native response establishes access; canary schema failures remain recorded and are measured in the benchmark.",
                    "canary_valid": sum(r["status"] == "ok" for r in selected),
                    "canary_total": len(selected),
                    "config_hash": digest(config),
                    "returned_models": sorted({str(r.get("model_returned")) for r in selected}),
                    "unexpected_reasoning": bool(unexpected_reasoning),
                    "account_quota": "not_exposed; concurrency 1",
                    "effective_reasoning": "no exposed reasoning observed"
                    if not unexpected_reasoning
                    else "contradicts requested nonthinking",
                    "failures": [r.get("error") for r in selected if r["status"] != "ok"],
                }
        write_json(
            output / "admission.json",
            {
                "admission": admission,
                "source_run": str(output),
                "code_hash": code_hash(),
                "created_at": manifest["started_at"],
            },
        )


def fit_and_freeze(data, models, run_paths, output, probe_path=None):
    all_rows = []
    phases = {}
    for path in run_paths:
        manifest, rows = load_records(path)
        if manifest["phase"] not in {"development", "calibration", "routing"}:
            raise ValueError("only development/calibration/routing may select policy")
        if manifest["models_hash"] != digest(models) or manifest["code_hash"] != code_hash():
            raise ValueError("code/configuration changed since development; rerun before freeze")
        phases[manifest["phase"]] = str(path)
        all_rows.extend(flat_record(r) for r in rows)
    if set(phases) != {"development", "calibration", "routing"}:
        raise ValueError("all three independent phases required")
    keys = [(r["phase"], r["judge"], r["profile"], r["case_id"]) for r in all_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate development observations")
    policies = {}
    for task in ("ragtruth", "anli"):
        rows = [r for r in all_rows if r["case"]["task"] == task]
        labels = (
            ["no", "yes"]
            if task == "ragtruth"
            else sorted(["supported", "contradicted", "insufficient_evidence"])
        )
        temps = {}
        for name in models:
            fit_rows = [
                r
                for r in rows
                if r["phase"] == "calibration"
                and r["judge"] == name
                and r["profile"] == "P"
                and r["status"] == "ok"
            ]
            if fit_rows:
                temps[name] = fit_temperature(
                    [r["gold"] for r in fit_rows], [r["probabilities"] for r in fit_rows], labels
                )
        candidates = []
        for name, config in models.items():
            if config["adapter"] == "jev" or not config["eligible_nonthinking"]:
                continue
            for profile in config["profiles"]:
                dev = [
                    r
                    for r in rows
                    if r["phase"] == "development"
                    and r["judge"] == name
                    and r["profile"] == profile
                ]
                if (
                    not dev
                    or not all(r["request_count"] for r in dev)
                    or not any(r["status"] == "ok" for r in dev)
                ):
                    continue
                metrics = classification(
                    [r["gold"] for r in dev], [r["prediction"] for r in dev], labels
                )
                costs = [r["cost"].get("usd") for r in dev]
                if any(c is None for c in costs):
                    continue
                candidates.append(
                    {
                        "judge": name,
                        "profile": profile,
                        "metrics": metrics,
                        "cost": sum(costs) / len(costs),
                        "p95": latency(dev)["successful_p95"] or float("inf"),
                        "model": config["model"],
                    }
                )
        if not candidates:
            policies[task] = {
                "status": "untested:no_eligible_priced_fallback",
                "temperatures": temps,
            }
            continue
        best = max(candidates, key=lambda c: c["metrics"]["macro_f1"])
        eligible = [
            c
            for c in candidates
            if c["metrics"]["macro_f1"] >= best["metrics"]["macro_f1"] - 0.02
            and (
                task != "ragtruth"
                or c["metrics"]["false_pass_rate"] <= best["metrics"]["false_pass_rate"] + 0.01
            )
        ]
        fallback = min(eligible, key=lambda c: (c["cost"], c["p95"], c["model"], c["profile"]))
        alternatives = {}
        for front in ["jev", *[c["judge"] for c in eligible if c["judge"] != fallback["judge"]]][:]:
            if front not in temps:
                continue
            first = {
                r["case_id"]: r
                for r in rows
                if r["phase"] == "routing" and r["judge"] == front and r["profile"] == "P"
            }
            for back in candidates:
                second = {
                    r["case_id"]: r
                    for r in rows
                    if r["phase"] == "routing"
                    and r["judge"] == back["judge"]
                    and r["profile"] == back["profile"]
                }
                if set(first) != set(second) or not first:
                    continue
                a, b = [], []
                for key in sorted(first):
                    r = first[key]
                    p = (
                        temperature_scale(r["probabilities"], temps[front]["temperature"])
                        if r["probabilities"]
                        else {}
                    )
                    a.append({**r, "confidence": max(p.values(), default=0)})
                    b.append(second[key])
                selected = choose_threshold(a, b)
                alternatives[front + "->" + back["judge"] + ":" + back["profile"]] = {
                    **selected,
                    "front": front,
                    "fallback": back["judge"],
                    "fallback_profile": back["profile"],
                    "temperature": temps[front]["temperature"],
                    "timing": "routing replay only; live test required",
                }
        selected_key = "jev->" + fallback["judge"] + ":" + fallback["profile"]
        policies[task] = {
            "status": "frozen",
            "fallback": fallback,
            "temperatures": temps,
            "selected": alternatives.get(selected_key),
            "alternatives": alternatives,
        }
    cases = read_jsonl(Path(data) / "cases.jsonl")
    if probe_path:
        cases += read_jsonl(probe_path)
    frozen = {
        "contract": "groundedness-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "code_hash": code_hash(),
        "models_hash": digest(models),
        "data_manifest": json.loads((Path(data) / "manifest.json").read_text()),
        "case_hashes": {c["id"]: digest(c) for c in cases},
        "development_runs": phases,
        "policies": policies,
    }
    if Path(output).exists():
        raise ValueError("freeze already exists; new contract required for amendments")
    write_json(output, frozen)
    return frozen


async def execute_cascade(cases, models, frozen, output, *, budget, location, holiday_status=None):
    if frozen["code_hash"] != code_hash() or frozen["models_hash"] != digest(models):
        raise ValueError("frozen code/configuration drift")
    if any(digest(c) != frozen["case_hashes"].get(c["id"]) for c in cases):
        raise ValueError("frozen input drift")
    output = Path(output)
    if output.exists():
        raise ValueError("immutable run directory exists")
    plans = []
    for case in cases:
        policy = frozen["policies"][case["task"]].get("selected")
        row = {
            "case": case,
            "judge": "selected-cascade",
            "profile": "P->D_or_P",
            "phase": "cascade",
            "repetition": 0,
            "policy": policy,
            "config": {"provider": "cascade", "model": "selected-cascade"},
        }
        row["attempt_id"] = digest(row)
        plans.append(row)
    random.Random(SEED).shuffle(plans)
    write_jsonl(output / "plan.jsonl", plans)
    write_json(
        output / "manifest.json",
        {
            "phase": "cascade",
            "freeze_hash": digest(frozen),
            "models": models,
            "code_hash": code_hash(),
            "client_location": location,
            "budget_usd": budget,
            "deadline_s": 60,
            "cache_mode": "off",
            "concurrency_per_provider": 1,
            "retries": 0,
        },
    )
    clients = {name: httpx.AsyncClient() for name in models}
    judges = {
        name: HTTPJudge(c, clients[name], holiday_status=holiday_status)
        for name, c in models.items()
    }
    spent = 0
    try:
        with (output / "attempts.jsonl").open("a") as f:
            for row in plans:
                policy, case = row["policy"], row["case"]
                result = {
                    "status": "not_attempted",
                    "error": "no_qualifying_policy",
                    "parsed": None,
                    "request_count": 0,
                    "cost": {"usd": None},
                    "usage": {},
                    "stages": [],
                    "auto_decided": False,
                }
                if policy:
                    names = [policy["fallback"]] + (
                        [policy["front"]] if policy["threshold"] is not None else []
                    )
                    reserve = sum(reserve_usd(models[n], case) for n in names)
                    if spent + reserve <= budget:
                        start = time.perf_counter()
                        stages = []
                        try:
                            async with asyncio.timeout(60):
                                if policy["threshold"] is not None:
                                    first = await judges[policy["front"]].evaluate(case, "P")
                                    stages.append(first)
                                    field, _ = reference(case)
                                    if first["status"] == "ok":
                                        p = temperature_scale(
                                            first["parsed"]["probabilities"][field],
                                            policy["temperature"],
                                        )
                                        result["auto_decided"] = (
                                            max(p.values()) >= policy["threshold"]
                                        )
                                if not result["auto_decided"]:
                                    stages.append(
                                        await judges[policy["fallback"]].evaluate(
                                            case, policy["fallback_profile"]
                                        )
                                    )
                                final = stages[-1]
                                result.update(
                                    status=final["status"],
                                    error=final["error"],
                                    parsed=final["parsed"],
                                )
                        except TimeoutError:
                            result.update(
                                status="timeout", error="cascade_deadline", right_censored=True
                            )
                        result["latency_s"] = time.perf_counter() - start
                        result["stages"] = stages
                        result["request_count"] = sum(r["request_count"] for r in stages)
                        amounts = [r["cost"].get("usd") for r in stages]
                        result["cost"] = {
                            "usd": sum(amounts)
                            if amounts and all(v is not None for v in amounts)
                            else None
                        }
                        spent += (
                            result["cost"]["usd"] if result["cost"]["usd"] is not None else reserve
                        )
                    else:
                        result["error"] = "budget_reserve_exceeded"
                f.write(dumps({**row, **result}) + "\n")
                f.flush()
                os.fsync(f.fileno())
                print(case["id"], result["status"], flush=True)
    finally:
        await asyncio.gather(*(client.aclose() for client in clients.values()))
    write_json(
        output / "completion.json",
        {"finished_at": datetime.now(UTC).isoformat(), "budget_accounted_usd": spent},
    )
