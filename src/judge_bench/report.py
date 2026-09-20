"""Machine-readable metrics, article tables, plots and blinded disagreement cases."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .core import digest, write_json, write_jsonl
from .metrics import (
    calibration,
    classification,
    consistency,
    holm,
    latency,
    paired_bootstrap,
    source_error_bound,
    temperature_scale,
)
from .runner import flat_record, load_records
from .uncertainty import probability_comparison, repeated_intervals


def report(run_paths, output, *, frozen=None, resamples=10000):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups, manifests = defaultdict(list), []
    all_rows = []
    for path in run_paths:
        manifest, rows = load_records(path)
        manifests.append({"path": str(path), "manifest": manifest})
        for row in rows:
            row = flat_record(row)
            all_rows.append(row)
            dataset = row["case"]["dataset"]
            if dataset == "canary":
                dataset += "-" + row["case"]["task"]
            key = (row["phase"], dataset, row["judge"], row["profile"])
            groups[key].append(row)
    results, tables = {}, []
    for key, planned in sorted(groups.items()):
        phase, dataset, judge, profile = key
        labels = (
            ["no", "yes"]
            if planned[0]["case"]["task"] == "ragtruth"
            else sorted(["supported", "contradicted", "insufficient_evidence"])
        )
        rows = [r for r in planned if r.get("request_count") or r.get("application_cache_hit")]
        if phase == "robustness":
            # Deliberately impossible contracts are never ordinary classification errors.
            rows = [
                r
                for r in rows
                if not r["case"].get("intentionally_misspecified")
                and not r["case"].get("human_review_required")
            ]
        metrics = (
            classification([r["gold"] for r in rows], [r["prediction"] for r in rows], labels)
            if rows
            else {"n": 0}
        )
        valid = [r for r in rows if r["status"] == "ok"]
        forecasts = [r for r in valid if r["probabilities"]]
        cal = calibration(
            [r["gold"] for r in forecasts], [r["probabilities"] for r in forecasts], labels
        )
        amounts = [r["cost"].get("usd") for r in rows]
        total = sum(amounts) if rows and all(v is not None for v in amounts) else None
        correct = sum(r["gold"] == r["prediction"] for r in rows)
        cost = {
            "known_subtotal_usd": sum(v for v in amounts if v is not None),
            "unknown_cost_attempts": sum(v is None for v in amounts),
            "total_usd": total,
            "per_1k_attempted": 1000 * total / len(rows) if total is not None else None,
            "per_1k_valid": 1000 * total / len(valid) if total is not None and valid else None,
            "per_1k_correct": 1000 * total / correct if total is not None and correct else None,
        }
        for scenario in ("all_uncached_usd", "all_cached_usd"):
            values = [r["cost"].get(scenario) for r in rows]
            cost[scenario] = sum(values) if values and all(v is not None for v in values) else None
        usage = {}
        for field in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
        ):
            values = [r.get("usage", {}).get(field) for r in rows]
            usage[field] = {
                "known_total": sum(v for v in values if v is not None),
                "unknown_n": sum(v is None for v in values),
            }
        failures = dict(Counter(r["error"] for r in rows if r["status"] != "ok"))
        slices = {}
        for attribute in ("model", "quality", "is_refusal", "is_truncated"):
            for value in sorted({str(r["case"].get("metadata", {}).get(attribute)) for r in rows}):
                selected = [
                    r for r in rows if str(r["case"].get("metadata", {}).get(attribute)) == value
                ]
                slices[attribute + ":" + value] = classification(
                    [r["gold"] for r in selected], [r["prediction"] for r in selected], labels
                )
        for bucket, lo, hi in [
            ("short", 0, 2000),
            ("medium", 2000, 8000),
            ("long", 8000, float("inf")),
        ]:
            selected = [r for r in rows if lo <= r["case"].get("input_bytes", 0) < hi]
            if selected:
                slices["bytes:" + bucket] = classification(
                    [r["gold"] for r in selected], [r["prediction"] for r in selected], labels
                )
        item = {
            "planned": len(planned),
            "attempted": len(rows),
            "complete": len(rows) == len(planned),
            "quality": metrics,
            "valid_only_quality": classification(
                [r["gold"] for r in valid], [r["prediction"] for r in valid], labels
            )
            if valid
            else None,
            "calibration_raw": cal,
            "cost": cost,
            "latency": latency(rows),
            "usage": usage,
            "failures": failures,
            "not_attempted": dict(
                Counter(
                    r.get("error")
                    for r in planned
                    if not r.get("request_count") and not r.get("application_cache_hit")
                )
            ),
            "probability_missingness_by_class": {
                label: sum(r["gold"] == label and not r["probabilities"] for r in rows)
                for label in labels
            },
            "slices": slices,
            "schema_failure_rate": sum(r["status"] == "parse_error" for r in rows) / len(rows)
            if rows
            else None,
        }
        if frozen and forecasts:
            task = forecasts[0]["case"]["task"]
            temp = frozen["policies"].get(task, {}).get("temperatures", {}).get(judge)
            if temp:
                item["calibration_posthoc"] = calibration(
                    [r["gold"] for r in forecasts],
                    [temperature_scale(r["probabilities"], temp["temperature"]) for r in forecasts],
                    labels,
                )
        if phase == "repeat":
            repeat_ids = {r["case_id"] for r in rows}
            first = [
                r
                for r in all_rows
                if r["phase"] == "test"
                and r["judge"] == judge
                and r["profile"] == profile
                and r["case_id"] in repeat_ids
            ]
            item["consistency"] = consistency(first + rows)
            item["consistency"]["source_cluster_intervals"] = repeated_intervals(
                first + rows, resamples
            )
            item["consistency"]["five_complete_live_runs"] = (
                all(c["runs"] == 5 and c["valid_runs"] == 5 for c in item["consistency"]["cases"])
                and len(item["consistency"]["cases"]) == 200
            )
        if phase == "cascade":
            automated = [r for r in rows if r.get("auto_decided")]
            item["automated"] = {
                "n": len(automated),
                "coverage": len(automated) / len(rows) if rows else None,
                "observed_error": sum(r["gold"] != r["prediction"] for r in automated)
                / len(automated)
                if automated
                else None,
                "source_bound": source_error_bound(
                    [r["gold"] != r["prediction"] for r in automated],
                    [r["case"]["group"] for r in automated],
                ),
                "by_decision": {
                    label: {
                        "n": sum(r["prediction"] == label for r in automated),
                        "errors": sum(
                            r["prediction"] == label and r["prediction"] != r["gold"]
                            for r in automated
                        ),
                    }
                    for label in labels
                },
            }
        name = "/".join(key)
        results[name] = item
        tables.append(
            {
                "phase": phase,
                "dataset": dataset,
                "judge": judge,
                "profile": profile,
                "planned": len(planned),
                "attempted": len(rows),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "false_pass_rate": metrics.get("false_pass_rate"),
                "coverage": metrics.get("valid_coverage"),
                "brier": cal.get("brier"),
                "ece15": cal.get("ece15"),
                "p50_s": item["latency"]["successful_p50"],
                "p95_s": item["latency"]["successful_p95"],
                "usd_per_1k": cost["per_1k_attempted"],
                "unknown_cost_n": cost["unknown_cost_attempts"],
            }
        )
        if cal.get("n"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            bins = [b for b in cal["bins"] if b["n"]]
            axes[0].plot([0, 1], [0, 1], "--", color="gray")
            axes[0].scatter(
                [b["confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                s=[20 + b["n"] for b in bins],
            )
            axes[0].set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Forecast confidence",
                ylabel="Observed accuracy",
                title="Reliability (size = bin count)",
            )
            axes[1].plot(
                [p["coverage"] for p in cal["risk_coverage"]],
                [p["risk"] for p in cal["risk_coverage"]],
            )
            axes[1].set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Coverage",
                ylabel="Observed error",
                title="Risk–coverage",
            )
            fig.suptitle(name)
            fig.savefig(output / (name.replace("/", "-") + ".svg"))
            fig.savefig(output / (name.replace("/", "-") + ".png"), dpi=160)
            plt.close(fig)
    comparisons = {}
    for key, a_rows in groups.items():
        phase, dataset, judge, profile = key
        if phase != "test" or dataset != "ragtruth" or judge != "jev":
            continue
        a = {r["case_id"]: r for r in a_rows}
        for other_key, b_rows in groups.items():
            if other_key[:2] != key[:2] or other_key[2] == "jev":
                continue
            b = {r["case_id"]: r for r in b_rows}
            comp_name = "jev/P-vs-" + "/".join(other_key[2:])
            if set(a) != set(b) or any(not r.get("request_count") for r in a_rows + b_rows):
                comparisons[comp_name] = {"H1": "untested: incomplete paired test cohort"}
                continue
            ids = sorted(a)
            pair = paired_bootstrap(
                [a[k]["gold"] for k in ids],
                [a[k]["prediction"] for k in ids],
                [b[k]["prediction"] for k in ids],
                [a[k]["case"]["group"] for k in ids],
                ["no", "yes"],
                resamples,
            )
            common = [k for k in ids if a[k]["probabilities"] and b[k]["probabilities"]]
            pair["paired_brier"] = probability_comparison(
                a, b, "unsupported_claim_present", resamples
            )
            pair["calibration_common_valid"] = {
                "n": len(common),
                "missing": len(ids) - len(common),
                "jev": calibration(
                    [a[k]["gold"] for k in common],
                    [a[k]["probabilities"] for k in common],
                    ["no", "yes"],
                ),
                "comparator": calibration(
                    [a[k]["gold"] for k in common],
                    [b[k]["probabilities"] for k in common],
                    ["no", "yes"],
                ),
            }
            f1, fp = pair["macro_f1"]["ci95"], pair["false_pass_rate"]["ci95"]
            pair["H1"] = (
                "supported"
                if f1[0] >= -0.02 and fp and fp[1] <= 0.01
                else "contradicted"
                if f1[1] < -0.02 or (fp and fp[0] > 0.01)
                else "inconclusive"
            )
            if not b_rows[0]["config"].get("eligible_nonthinking"):
                pair["H1"] = "diagnostic: forced-thinking comparator"
                pair["noninferiority_p"] = None
            comparisons[comp_name] = pair
    for name, pair in comparisons.items():
        if "macro_f1" not in pair:
            continue
        comparator, profile = name.split("-vs-")[1].split("/")
        front = results["test/ragtruth/jev/P"]
        back = results["test/ragtruth/" + comparator + "/" + profile]
        a_cost, b_cost = front["cost"]["per_1k_attempted"], back["cost"]["per_1k_attempted"]
        a_time, b_time = front["latency"]["all_attempt_p95"], back["latency"]["all_attempt_p95"]
        pair["H2"] = "inconclusive"
        if b_cost == 0:
            pair["H2"] = "contradicted: comparator tariff is free"
        elif a_cost is not None and b_cost and a_time is not None and b_time:
            pair["cost_ratio"] = a_cost / b_cost
            pair["p95_ratio"] = a_time / b_time
            pair["H2"] = (
                "supported at observed operating point, conditional on H1"
                if pair["H1"] == "supported"
                and pair["cost_ratio"] <= 0.70
                and pair["p95_ratio"] <= 0.70
                else "contradicted at observed operating point"
                if pair["cost_ratio"] > 0.70 or pair["p95_ratio"] > 0.70
                else "inconclusive: H1 not established"
            )
    adjusted = holm(
        {
            k: v["noninferiority_p"]
            for k, v in comparisons.items()
            if v.get("noninferiority_p") is not None and k.endswith("/D")
        }
    )
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value
        if value >= 0.05 and comparisons[name]["H1"] == "supported":
            comparisons[name]["H1"] = "inconclusive after Holm adjustment"
    # Blinded presentation preserves a separate mapping rather than revealing judge names in cases.
    cohort = defaultdict(list)
    for r in all_rows:
        if r["phase"] == "test":
            cohort[(r["case"]["dataset"], r["case_id"])].append(r)
    systems = sorted({r["judge"] + ":" + r["profile"] for r in all_rows})
    aliases = {name: "judge_" + str(i + 1) for i, name in enumerate(systems)}
    disagreements = []
    for key, rows in cohort.items():
        if len({r["prediction"] for r in rows}) > 1:
            disagreements.append(
                {
                    "case_id": key[1],
                    "input": rows[0]["case"]["input"],
                    "reference": rows[0]["case"]["gold"],
                    "label_provenance": rows[0]["case"].get("label_provenance"),
                    "group": rows[0]["case"]["group"],
                    "predictions": {
                        aliases[r["judge"] + ":" + r["profile"]]: {
                            "label": r["prediction"],
                            "probabilities": r["probabilities"],
                            "status": r["status"],
                        }
                        for r in rows
                    },
                }
            )
    disagreements.sort(key=lambda r: digest(r["case_id"]))
    write_jsonl(output / "disagreements.jsonl", disagreements)
    write_json(output / "blinding-key.json", aliases)
    robustness = []
    for key, rows in groups.items():
        if key[0] != "robustness":
            continue
        by_base = defaultdict(dict)
        for r in rows:
            by_base[r["case"].get("base_id", r["case_id"])][r["case"].get("variant", "base")] = r
        for base, variants in by_base.items():
            baseline = variants.get("base")
            for variant, row in variants.items():
                robustness.append(
                    {
                        "judge": key[2],
                        "profile": key[3],
                        "base_id": base,
                        "variant": variant,
                        "status": row["status"],
                        "label": row["prediction"],
                        "probabilities": row["probabilities"],
                        "flip": row["prediction"] != baseline["prediction"]
                        if baseline and row["prediction"] and baseline["prediction"]
                        else None,
                        "ordinary_accuracy_applicable": not row["case"].get(
                            "intentionally_misspecified"
                        )
                        and not row["case"].get("human_review_required"),
                    }
                )
    write_jsonl(output / "robustness.jsonl", robustness)
    if frozen:
        from .cascades import replay

        write_json(output / "cascade-replay.json", replay(all_rows, frozen))
    write_json(
        output / "metrics.json",
        {
            "runs": manifests,
            "results": results,
            "paired_comparisons": comparisons,
            "H5": "untested without human-verified matched challenge pairs",
            "full_schema": "untested without independent human annotations",
        },
    )
    if tables:
        with (output / "summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(tables[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(tables)
    columns = [
        "phase",
        "dataset",
        "judge",
        "profile",
        "attempted",
        "macro_f1",
        "coverage",
        "brier",
        "p95_s",
        "usd_per_1k",
    ]

    def cell(value):
        return (
            "unknown"
            if value is None
            else f"{value:.4f}"
            if isinstance(value, float)
            else str(value)
        )

    lines = [
        "# Judge benchmark results",
        "",
        "Published-rate usage estimates are not invoice-confirmed charges. Failed outputs remain incorrect in primary quality metrics. Unattempted cases are reported separately; incomplete runs cannot establish a quality comparison.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines += ["| " + " | ".join(cell(r[k]) for k in columns) + " |" for r in tables]
    lines += [
        "",
        "## Interpretation",
        "",
        "These services differ in training, capacity and serving. GLM uses reasoning; the other LLM configurations disable it. Jev probabilities are uncalibrated.",
        "",
        "See metrics.json for paired source-bootstrap intervals, confusion matrices, token usage, failures and calibration bins. Quality conclusions require complete cohorts with reference labels.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    measurable = [
        r
        for r in tables
        if r["phase"] == "test" and r["usd_per_1k"] is not None and r["macro_f1"] is not None
    ]
    if measurable:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for r in measurable:
            ax.scatter(r["usd_per_1k"], r["macro_f1"])
            ax.annotate(
                r["judge"] + "/" + r["profile"] + "/" + r["dataset"],
                (r["usd_per_1k"], r["macro_f1"]),
                fontsize=7,
            )
        ax.set(
            xlabel="USD per 1,000 attempted evaluations",
            ylabel="Failure-aware macro-F1",
            ylim=(0, 1),
            title="Measured operating points (free tariffs remain at zero)",
        )
        fig.savefig(output / "quality-cost.svg")
        plt.close(fig)
    for svg in output.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    return {
        "groups": len(results),
        "disagreement_cases": len(disagreements),
        "report": str(output / "report.md"),
    }
