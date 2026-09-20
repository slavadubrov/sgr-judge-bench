"""Pack a compact offline replay; keep full HTTP evidence in a separate archive."""

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

from tabfact_audit import audit

from judge_bench.core import read_jsonl, write_json, write_jsonl
from judge_bench.demo import PRIMARY
from judge_bench.tabfact import report


def pack(root, out, dataset, jev_run):
    """Preserve raw arm IDs; publish only scoring evidence and frozen prompts."""
    root, out, dataset = Path(root), Path(out), Path(dataset)
    audited = audit(root)
    bundle = {
        name: json.loads((root / f"{name}.json").read_text())
        for name in ("cases", "manifest", "summary")
    }
    jev_root = Path(jev_run)
    jev_audit = audit(jev_root)
    if json.loads((jev_root / "cases.json").read_text()) != bundle["cases"]:
        raise ValueError("Jev and LLM cohorts differ")
    jev_manifest = json.loads((jev_root / "manifest.json").read_text())
    if not jev_manifest["models"]["jev"].get("detailed_prompt"):
        raise ValueError("Jev requires the detailed prompt")
    bundle["manifest"]["models"]["jev"] = jev_manifest["models"]["jev"]
    bundle["manifest"]["run_blocks"] = {
        "llm": bundle["manifest"]["started_at"],
        "jev": jev_manifest["started_at"],
    }
    records = [r for r in read_jsonl(root / "records.jsonl") if r["model"] != "jev"]
    records += read_jsonl(jev_root / "records.jsonl")
    bundle["manifest"].pop("client_location", None)
    for model in bundle["manifest"]["models"].values():
        model.pop("version_limit", None)
    bundle["records"] = [
        {
            k: r[k]
            for k in (
                "id",
                "model",
                "arm",
                "calls",
                "status",
                "prediction",
                "checks",
                "assessment",
                "error",
            )
            if k in r
        }
        for r in records
        if f"{r['model']}/{r['arm']}" in PRIMARY
    ]
    used = {cid for r in bundle["records"] for cid in r["calls"]}
    calls = [c for c in read_jsonl(root / "calls.jsonl") if c["model"] != "jev"]
    calls += read_jsonl(jev_root / "calls.jsonl")
    calls = [c for c in calls if c["call_id"] in used]
    bundle["manifest"]["jobs"] = [
        j for j in bundle["manifest"]["jobs"] if f"{j['model']}/{j['arm']}" in PRIMARY
    ]
    bundle["manifest"].pop("include_prompt_control", None)
    bundle["manifest"]["policy"] = (
        "Direct with detailed instructions, SGR v2, and native Jev. All failures remain incorrect."
    )
    bundle["calls"] = [
        {
            **{
                k: c[k]
                for k in (
                    "call_id",
                    "case_id",
                    "model",
                    "stage",
                    "status",
                    "request_count",
                    "latency_s",
                    "cost",
                    "model_returned",
                )
            },
            "usage": {k: v for k, v in c["usage"].items() if k != "raw"},
        }
        for c in calls
    ]
    with tempfile.TemporaryDirectory() as directory:
        replay = Path(directory)
        for name in ("cases", "manifest"):
            write_json(replay / f"{name}.json", bundle[name])
        for name in ("records", "calls"):
            write_jsonl(replay / f"{name}.jsonl", bundle[name])
        bundle["summary"] = report(replay)
    bundle["prompts"] = {
        stage: next(
            c["request"]["input"][0]["content"]
            for c in calls
            if c["model"] == "luna" and c["stage"] == stage
        )
        for stage in ("direct_guided", "plan", "assess")
    }
    bundle["prompts"]["jev"] = next(
        c["request"]["questions"]["label"]["instructions"] for c in calls if c["model"] == "jev"
    )
    bundle["hypothesis"] = {
        "contrasts": {
            f"{model}/direct_guided -> {model}/sgr": bundle["summary"]["paired_comparisons"][
                f"{model}/direct_guided -> {model}/sgr"
            ]
            for model in ("luna", "terra", "deepseek-json")
        },
        "limitation": "Exploratory paired intervals on an observed cohort; no equivalence margin or multiplicity adjustment. Direct uses detailed procedural instructions.",
    }
    bundle["provenance"] = {
        "dataset_manifest": json.loads(dataset.read_text()),
        "source_run_audits": {"llm": audited, "jev": jev_audit},
        "exported_evaluations": len(bundle["records"]),
        "exported_calls": len(bundle["calls"]),
        "source_files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "cases.json",
                "manifest.json",
                "summary.json",
                "records.jsonl",
                "calls.jsonl",
                "audit.json",
            )
        },
        "jev_source_files_sha256": {
            name: hashlib.sha256((jev_root / name).read_bytes()).hexdigest()
            for name in (
                "cases.json",
                "manifest.json",
                "records.jsonl",
                "calls.jsonl",
                "audit.json",
            )
        },
        "description": "Compact scoring replay. Raw HTTP evidence remains private. Frozen system prompts are included; DeepSeek additionally receives its JSON schema in the prompt.",
        "publication_note": "Allowlisted records and calls omit raw HTTP, request IDs, duplicate source projections, queue timings and administrative notes. The export selects the six LLM configurations and detailed native Jev from their audited execution blocks, retaining every case and failure. Predictions, findings, costs and timings are unchanged. Raw arm ID direct_guided is displayed as Direct.",
    }
    out.write_bytes(
        gzip.compress(
            json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(),
            mtime=0,
        )
    )
    print(out, out.stat().st_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("out")
    parser.add_argument("--dataset", default="config/dataset.json")
    parser.add_argument("--jev-run", required=True)
    args = parser.parse_args()
    pack(args.run, args.out, args.dataset, args.jev_run)
