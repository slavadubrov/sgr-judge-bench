"""Paired source-cluster intervals for probability, routing and robustness summaries."""

import numpy as np

from .core import SEED


def cluster_mean_interval(values, groups, resamples=10000):
    if not values:
        return {"estimate": None, "ci95": None, "sources": 0}
    unique = sorted(set(groups))
    totals = np.array(
        [sum(v for v, group in zip(values, groups, strict=True) if group == g) for g in unique]
    )
    counts = np.array([sum(group == g for group in groups) for g in unique])
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(resamples):
        index = rng.integers(0, len(unique), len(unique))
        samples.append(float(totals[index].sum() / counts[index].sum()))
    return {
        "estimate": float(np.mean(values)),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "sources": len(unique),
        "resamples": resamples,
    }


def probability_comparison(a, b, field, resamples=10000):
    ids = sorted(set(a) & set(b))
    valid = [i for i in ids if a[i].get("probabilities") and b[i].get("probabilities")]
    scores = []
    groups = []
    for key in valid:
        pa, pb, gold = a[key]["probabilities"], b[key]["probabilities"], a[key]["gold"]
        if field == "unsupported_claim_present":
            delta = (pa["yes"] - (gold == "yes")) ** 2 - (pb["yes"] - (gold == "yes")) ** 2
        else:
            delta = sum((pa[k] - (gold == k)) ** 2 - (pb[k] - (gold == k)) ** 2 for k in pa)
        scores.append(delta)
        groups.append(a[key]["case"]["group"])
    result = cluster_mean_interval(scores, groups, resamples)
    result.update(
        n=len(valid),
        missing=len(ids) - len(valid),
        direction="Jev minus comparator; negative favors Jev",
    )
    interval = result["ci95"]
    result["H4"] = (
        "untested"
        if interval is None
        else "supported"
        if interval[1] <= -0.01 and not result["missing"]
        else "contradicted"
        if interval[0] > -0.01 and not result["missing"]
        else "inconclusive"
    )
    return result


def repeated_intervals(rows, resamples=10000):
    from collections import Counter, defaultdict
    from itertools import combinations

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["case_id"]].append(r)
    values = {
        "modal_agreement": [],
        "any_flip": [],
        "pairwise_agreement": [],
        "probability_drift_tv": [],
    }
    groups = {key: [] for key in values}
    for observations in grouped.values():
        labels = [r.get("prediction") for r in observations]
        valid = [label for label in labels if label is not None]
        pairs = list(combinations(valid, 2))
        probabilities = [r["probabilities"] for r in observations if r.get("probabilities")]
        tv = [sum(abs(a[k] - b[k]) for k in a) / 2 for a, b in combinations(probabilities, 2)]
        measurements = {
            "modal_agreement": max(Counter(valid).values(), default=0) / len(observations),
            "any_flip": float(len(set(valid)) > 1),
            "pairwise_agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
            "probability_drift_tv": float(np.mean(tv)) if tv else None,
        }
        for key, value in measurements.items():
            if value is not None:
                values[key].append(value)
                groups[key].append(observations[0]["case"]["group"])
    return {
        key: cluster_mean_interval(value, groups[key], resamples) for key, value in values.items()
    }
