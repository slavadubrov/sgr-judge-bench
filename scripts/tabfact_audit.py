"""Offline reconstruction of every table request and verdict from raw responses."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import jsonschema

from judge_bench.adapters import parse_response, request_body
from judge_bench.core import digest, read_jsonl, write_json
from judge_bench.sgr import (
    aggregate,
    aggregate_native,
    materialize_checks,
    native_case,
    parse_stage,
    stage_request,
)


def audit(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    case_by_id = {c["id"]: c for c in cases}
    assert digest(cases) == manifest["case_sha256"]
    assert (
        digest(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / "source").glob("*.py"))
            }
        )
        == manifest["code_hash"]
    )
    records = read_jsonl(root / "records.jsonl")
    calls_list = read_jsonl(root / "calls.jsonl")
    calls = {c["call_id"]: c for c in calls_list}
    assert len(calls) == len(calls_list), "Duplicate call IDs"
    observed = {(r["id"], r["model"], r["arm"]): r for r in records}
    planned = {(r["id"], r["model"], r["arm"]) for r in manifest["jobs"]}
    assert len(observed) == len(records) and set(observed) == planned, (
        "Missing/extra/duplicate evaluations"
    )
    for call in calls.values():
        case = case_by_id[call["case_id"]]
        config = manifest["models"][call["model"]]
        stage = call["stage"]
        arm = stage if stage in ("direct", "direct_guided", "hybrid") else "sgr"
        record = observed[case["id"], call["model"], arm]
        # Rebuild insertion order: JSONL sorting changes dict order, while enum lists preserve it.
        views = (
            materialize_checks(case["input"], record["checks"])
            if stage in ("assess", "hybrid")
            else None
        )
        if config["adapter"] == "jev":
            native = native_case(
                case["input"],
                views if stage == "hybrid" else None,
                detailed=config.get("detailed_prompt", False),
            )
            expected = request_body(config, native, "P")
        else:
            expected, schema = stage_request(config, case["input"], stage, views)
        assert expected == call["request"], ("Request mismatch", call["call_id"])
        if call["status"] == "ok":
            raw = json.loads(call["raw_output"])
            parsed = (
                parse_response(config, raw, native, "P")
                if config["adapter"] == "jev"
                else parse_stage(config, raw, schema)
            )
            assert parsed == call["parsed"], ("Parse mismatch", call["call_id"])
    for record in records:
        raw = case_by_id[record["id"]]["input"]
        prediction = None
        try:
            stages = [calls[cid] for cid in record["calls"]]
            if not stages or any(c["status"] != "ok" for c in stages):
                raise ValueError("missing/failed stage")
            if record["arm"] in ("direct", "direct_guided"):
                assert len(stages) == 1, "Direct arm must use exactly one call"
                parsed = stages[0]["parsed"]
                prediction = parsed["labels"]["label"] if "labels" in parsed else parsed["label"]
            else:
                if len(stages) != 2:
                    raise ValueError("incomplete pipeline")
                checks = stages[0]["parsed"]["checks"]
                assert checks == record["checks"]
                assert materialize_checks(raw, checks) == record["source_views"]
                assessment = stages[1]["parsed"]
                assert assessment == record["assessment"]
                prediction = (
                    aggregate_native(assessment["labels"])
                    if record["arm"] == "hybrid"
                    else aggregate(raw, checks, assessment)
                )
        except (ValueError, KeyError, jsonschema.ValidationError):
            assert record["status"] != "ok"
        assert prediction == record["prediction"], (
            "Verdict mismatch",
            record["id"],
            record["model"],
            record["arm"],
        )
        assert (prediction is not None) == (record["status"] == "ok")
    result = {
        "status": "PASS",
        "requests_reconstructed": len(calls),
        "evaluations_replayed": len(records),
        "source_snapshot_hash_verified": True,
        "case_hash_verified": True,
        "transport_statuses": dict(Counter(c["status"] for c in calls.values())),
        "returned_models": {
            name: dict(Counter(c["model_returned"] for c in calls.values() if c["model"] == name))
            for name in manifest["models"]
        },
        "returned_providers": {
            name: dict(
                Counter(
                    str(c.get("provider_returned")) for c in calls.values() if c["model"] == name
                )
            )
            for name in manifest["models"]
        },
    }
    write_json(root / "audit.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    print(json.dumps(audit(p.parse_args().run), indent=2))
