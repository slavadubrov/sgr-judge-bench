"""Offline alternative-cascade replay. No simulated value is called measured latency."""

from .core import reference
from .metrics import classification, source_error_bound, temperature_scale


def replay(rows, frozen):
    outputs = {}
    for task, settings in frozen["policies"].items():
        selected = [r for r in rows if r["phase"] == "test" and r["case"]["task"] == task]
        for name, policy in settings.get("alternatives", {}).items():
            first = {
                r["case_id"]: r
                for r in selected
                if r["judge"] == policy["front"] and r["profile"] == "P"
            }
            second = {
                r["case_id"]: r
                for r in selected
                if r["judge"] == policy["fallback"] and r["profile"] == policy["fallback_profile"]
            }
            if not first or set(first) != set(second):
                outputs[task + "/" + name] = {"status": "untested: incomplete paired cohort"}
                continue
            gold, predictions, costs, automatic, source_groups = [], [], [], [], []
            for key in sorted(first):
                a, b = first[key], second[key]
                field, _ = reference(a["case"])
                p = (
                    temperature_scale(a["probabilities"], policy["temperature"])
                    if a.get("probabilities")
                    else {}
                )
                auto = (
                    policy["threshold"] is not None
                    and a["status"] == "ok"
                    and p
                    and max(p.values()) >= policy["threshold"]
                )
                chosen = a if auto else b
                gold.append(chosen["gold"])
                predictions.append(chosen["prediction"])
                needed = ([a] if policy["threshold"] is not None else []) + ([] if auto else [b])
                prices = [r["cost"].get("usd") for r in needed]
                costs.append(sum(prices) if all(v is not None for v in prices) else None)
                if auto:
                    automatic.append(chosen["gold"] != chosen["prediction"])
                    source_groups.append(chosen["case"]["group"])
            labels = (
                ["no", "yes"]
                if task == "ragtruth"
                else sorted(["supported", "contradicted", "insufficient_evidence"])
            )
            outputs[task + "/" + name] = {
                "status": "offline replay",
                "policy": policy,
                "quality": classification(gold, predictions, labels),
                "usd_per_1k": 1000 * sum(costs) / len(costs)
                if all(v is not None for v in costs)
                else None,
                "latency": "not measured; use selected live cascade for H3",
                "automated_coverage": len(automatic) / len(gold),
                "automated_error": sum(automatic) / len(automatic) if automatic else None,
                "source_bound": source_error_bound(automatic, source_groups),
            }
    return outputs
