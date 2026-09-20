"""Pack a compact offline replay; keep full HTTP evidence in a separate archive."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from tabfact_audit import audit

from judge_bench.core import read_jsonl


def pack_prompt_control(root, out, previous):
    """Preserve raw arm IDs; publish only scoring evidence and frozen prompts."""
    root, out, previous = Path(root), Path(out), Path(previous)
    audited = audit(root)
    historical = json.loads(gzip.decompress(previous.read_bytes()))
    bundle = {
        name: json.loads((root / f"{name}.json").read_text())
        for name in ("cases", "manifest", "summary")
    }
    if bundle["cases"] != historical["cases"]:
        raise ValueError("Cohort differs from the historical replay")
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
        for r in read_jsonl(root / "records.jsonl")
    ]
    calls = read_jsonl(root / "calls.jsonl")
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
    bundle["prompts"] = {
        stage: next(
            c["request"]["input"][0]["content"]
            for c in calls
            if c["model"] == "luna" and c["stage"] == stage
        )
        for stage in ("direct", "direct_guided", "plan", "assess")
    }
    bundle["hypothesis"] = {
        "contrasts": {
            f"{model}/direct_guided -> {model}/sgr": bundle["summary"]["paired_comparisons"][
                f"{model}/direct_guided -> {model}/sgr"
            ]
            for model in ("luna", "terra", "deepseek-json")
        },
        "limitation": "Exploratory paired intervals on an observed cohort; no equivalence margin or multiplicity adjustment. Primary Direct uses the detailed prompt; original direct remains a diagnostic control.",
    }
    bundle["provenance"] = {
        "dataset_manifest": historical["provenance"]["dataset_manifest"],
        "audits": {"prompt-control": audited},
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
        "historical_replay_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
        "description": "Compact scoring replay. Raw HTTP evidence remains private and is not the v0.1.0 release asset. Frozen system prompts are included; DeepSeek additionally receives its JSON schema in the prompt.",
        "publication_note": "Allowlisted records and calls omit raw HTTP, request IDs, duplicate source projections, queue timings and administrative notes. Predictions, findings, failures, cost, usage and service timings are unchanged. Raw arm IDs are unchanged: direct_guided is displayed as Direct, direct as Direct (short prompt).",
    }
    out.write_bytes(
        gzip.compress(
            json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(),
            mtime=0,
        )
    )
    print(out, out.stat().st_size)


def pack(root, out, evidence):
    root, out, evidence = Path(root), Path(out), Path(evidence)
    files = (
        "cases.json",
        "manifest.json",
        "hypothesis.json",
        "summary.json",
        "records.jsonl",
        "calls.jsonl",
    )
    bundle = {name[:-5]: json.loads((root / name).read_text()) for name in files[:4]}
    for model in bundle["manifest"]["models"].values():
        model.pop("version_limit", None)
    bundle["records"] = [
        {k: v for k, v in record.items() if k != "source_views"}
        for record in read_jsonl(root / "records.jsonl")
    ]
    bundle["calls"] = [
        {k: v for k, v in call.items() if k not in ("request", "raw_output", "parsed", "request_id")}
        for call in read_jsonl(root / "calls.jsonl")
    ]
    bundle["provenance"] = {
        "dataset_manifest": json.loads(Path("data/tabfact-article120-manifest.json").read_text()),
        "audits": {
            source["run"]: json.loads((Path(source["run"]) / "audit.json").read_text())
            for source in bundle["manifest"]["source_runs"]
        },
        "description": "Compact scoring replay, not full HTTP evidence. Source projections are reconstructible from bundled tables and checks. Raw requests/responses remain in the separate evidence archive.",
        "source_files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
        },
        "raw_evidence": {
            "filename": evidence.name,
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        },
    }
    manifest = bundle["provenance"]["dataset_manifest"]
    manifest["excluded_files"] = [Path(name).name for name in manifest["excluded_files"]]
    bundle["provenance"]["publication_note"] = (
        "Administrative model notes and provider request identifiers omitted; input file references reduced to basenames. Predictions, usage, timing and scoring inputs unchanged; source hashes identify the original private run files."
    )
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence")
    source.add_argument(
        "--previous", help="Historical replay for a prompt-control run on the same cohort"
    )
    args = parser.parse_args()
    if args.previous:
        pack_prompt_control(args.run, args.out, args.previous)
    else:
        pack(args.run, args.out, args.evidence)
