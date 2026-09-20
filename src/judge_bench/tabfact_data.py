"""Freeze TabFact official-test cases, excluding observed tables/pages/content."""

import csv
import hashlib
import io
import json
import random
import subprocess
from pathlib import Path

from .core import digest


def prepare(
    upstream,
    out=Path("data/holdout-v2.json"),
    excluded=("data/cases.json",),
    seed=20260921,
    simple=5,
    complex_n=7,
):
    if out.exists():
        raise ValueError("Refusing to replace frozen holdout")
    old = [c for file in excluded for c in json.loads(Path(file).read_text())]
    pages = json.loads((upstream / "data/table_to_page.json").read_text())
    test_ids = set(json.loads((upstream / "data/test_id.json").read_text()))
    forbidden_pages = {c["source_url"] for c in old}
    forbidden_tables = {c["table_id"] for c in old}
    forbidden_hashes = {digest([c["input"]["headers"], *c["input"]["rows"]]) for c in old}
    seen_pages, seen_hashes = set(forbidden_pages), set(forbidden_hashes)
    pools = {(ch, label): [] for ch in ("simple", "complex") for label in (0, 1)}
    for ch, file in [("simple", "r1_training_all.json"), ("complex", "r2_training_all.json")]:
        for tid, (claims, labels, caption) in json.loads(
            (upstream / "collected_data" / file).read_text()
        ).items():
            if tid not in test_ids or tid in forbidden_tables:
                continue
            for index, (claim, label) in enumerate(zip(claims, labels, strict=True)):
                pools[ch, label].append((tid, index, claim, caption))
    rng = random.Random(seed)
    cases = []
    for (ch, label), pool in sorted(pools.items()):
        pool.sort()
        rng.shuffle(pool)
        needed = simple if ch == "simple" else complex_n
        selected = 0
        for tid, index, claim, caption in pool:
            page = pages.get(tid, [caption, caption])[1]
            if page in seen_pages:
                continue
            file = upstream / "data/all_csv" / tid
            rows = list(csv.reader(io.StringIO(file.read_text()), delimiter="#"))
            sha = digest(rows)
            if sha in seen_hashes:
                continue
            assert rows and all(len(row) == len(rows[0]) for row in rows)
            cases.append(
                {
                    "id": f"{ch}:{tid}:{index}",
                    "table_id": tid,
                    "channel": ch,
                    "split": "test",
                    "gold": "ENTAILED" if label else "REFUTED",
                    "source_url": page,
                    "table_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "input": {
                        "caption": caption,
                        "claim": claim,
                        "headers": rows[0],
                        "rows": rows[1:],
                    },
                }
            )
            seen_pages.add(page)
            seen_hashes.add(sha)
            selected += 1
            if selected == needed:
                break
        assert selected == needed
    rng.shuffle(cases)
    assert len({c["table_id"] for c in cases}) == 2 * (simple + complex_n)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "seed": seed,
        "n": len(cases),
        "split": "official test table IDs; raw r1/r2 annotations",
        "sha256": digest(cases),
        "commit": subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        "strata": f"{simple} per simple label; {complex_n} per complex label",
        "excluded_files": list(excluded),
        "excluded": "All original development tables/pages and identical normalized table contents; one page/table per selected claim",
        "selection": "No length, content, task-type, model-output or operator-coverage filtering",
        "scope": "Unseen by development procedure; model pretraining contamination and fuzzy overlap not ruled out",
    }
    out.with_name(out.stem + "-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
