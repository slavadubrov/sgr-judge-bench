"""Pack a compact offline replay; keep full HTTP evidence in a separate archive."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from judge_bench.core import read_jsonl


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
        {k: v for k, v in call.items() if k not in ("request", "raw_output", "parsed")}
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
        "Administrative model notes omitted and input file references reduced to basenames. Predictions, usage, timing and scoring inputs unchanged; source hashes identify the original private run files."
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
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    pack(args.run, args.out, args.evidence)
