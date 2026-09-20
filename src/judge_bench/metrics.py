"""Failure-aware metrics, calibration, grouped uncertainty, and frozen routing."""

import math
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta

from .core import SEED


def divide(a, b):
    return a / b if b else None


def classification(gold, predictions, labels):
    matrix = {y: {p: 0 for p in [*labels, "invalid"]} for y in labels}
    for y, p in zip(gold, predictions, strict=True):
        matrix[y][p if p in labels else "invalid"] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[y][label] for y in labels if y != label)
        fn = sum(matrix[label].values()) - tp
        per_class[label] = {
            "precision": divide(tp, tp + fp),
            "recall": divide(tp, tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "support": sum(matrix[label].values()),
        }
    valid = [i for i, p in enumerate(predictions) if p in labels]
    correct = sum(y == p for y, p in zip(gold, predictions, strict=True))
    result = {
        "n": len(gold),
        "confusion_matrix": matrix,
        "per_class": per_class,
        "accuracy": divide(correct, len(gold)),
        "macro_f1": float(np.mean([r["f1"] for r in per_class.values()])),
        "valid_coverage": divide(len(valid), len(gold)),
        "valid_accuracy": divide(sum(gold[i] == predictions[i] for i in valid), len(valid)),
    }
    if labels == ["no", "yes"]:
        result["false_pass_rate"] = divide(matrix["yes"]["no"], sum(matrix["yes"].values()))
        result["clean_contamination"] = divide(
            matrix["yes"]["no"], sum(matrix[y]["no"] for y in labels)
        )
    return result


def calibration(gold, probabilities, labels):
    if not gold:
        return {"n": 0}
    p = np.asarray([[row[label] for label in labels] for row in probabilities], dtype=float)
    indices = np.asarray([labels.index(y) for y in gold])
    target = np.eye(len(labels))[indices]
    brier = (
        np.mean((p[:, 1] - target[:, 1]) ** 2)
        if len(labels) == 2
        else np.mean(np.sum((p - target) ** 2, axis=1))
    )
    prediction = np.argmax(p, axis=1)
    correct = prediction == indices
    confidence = np.max(p, axis=1)
    bins = []
    ece = 0
    for i in range(15):
        mask = (confidence >= i / 15) & ((confidence < (i + 1) / 15) if i < 14 else confidence <= 1)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else None
        conf = float(confidence[mask].mean()) if n else None
        ece += n / len(gold) * abs(acc - conf) if n else 0
        bins.append(
            {"lower": i / 15, "upper": (i + 1) / 15, "n": n, "accuracy": acc, "confidence": conf}
        )
    # Equal-confidence examples enter as one group; input ordering cannot improve AURC.
    curve = []
    for threshold in sorted(set(confidence), reverse=True):
        mask = confidence >= threshold
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": float(mask.mean()),
                "risk": float(1 - correct[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    aurc, previous = 0, 0
    for row in curve:
        aurc += (row["coverage"] - previous) * row["risk"]
        previous = row["coverage"]
    return {
        "n": len(gold),
        "brier": float(brier),
        "nll": float(-np.log(np.clip(p[np.arange(len(gold)), indices], 1e-6, 1)).mean()),
        "ece15": ece,
        "bins": bins,
        "risk_coverage": curve,
        "aurc": aurc,
        "coverage_at_risk": {
            str(risk): max([r["coverage"] for r in curve if r["risk"] <= risk], default=0)
            for risk in (0.01, 0.02, 0.05)
        },
    }


def temperature_scale(p, temperature):
    labels = sorted(p)
    logits = np.log(np.clip([p[k] for k in labels], 1e-6, 1)) / temperature
    values = np.exp(logits - max(logits))
    return dict(zip(labels, (values / values.sum()).tolist(), strict=True))


def fit_temperature(gold, probabilities, labels):
    if not gold:
        raise ValueError("no valid calibration forecasts")

    def loss(log_t):
        t = math.exp(log_t)
        return -np.mean(
            [
                math.log(max(1e-6, temperature_scale(p, t)[y]))
                for y, p in zip(gold, probabilities, strict=True)
            ]
        )

    fit = minimize_scalar(loss, bounds=(-5, 5), method="bounded")
    return {
        "temperature": math.exp(fit.x),
        "n": len(gold),
        "nll": float(fit.fun),
        "method": "one_scalar_NLL_clipped_log_probabilities",
    }


def latency(rows):
    valid = [
        r["latency_s"] for r in rows if r["status"] == "ok" and not r.get("application_cache_hit")
    ]
    actual = [r for r in rows if r.get("request_count", 0) and not r.get("application_cache_hit")]
    result = {
        "successful_n": len(valid),
        "attempted_n": len(actual),
        "timeout_n": sum(r["status"] == "timeout" for r in actual),
        "failure_n": sum(r["status"] != "ok" for r in actual),
    }
    for q in (0.5, 0.95):
        key = "p" + str(int(q * 100))
        result["successful_" + key] = float(np.quantile(valid, q)) if valid else None
        # Failed evaluations never count as fast successful completions. Deadline bound is conservative.
        result["all_attempt_" + key] = (
            float(np.quantile(valid, min(1, q * len(actual) / len(valid))))
            if valid and len(valid) >= math.ceil(q * len(actual))
            else None
        )
        result["all_attempt_" + key + "_unobserved"] = result["all_attempt_" + key] is None
    return result


def source_error_bound(errors, groups):
    grouped = defaultdict(list)
    for error, group in zip(errors, groups, strict=True):
        grouped[group].append(error)
    n = len(grouped)
    k = sum(any(v) for v in grouped.values())
    return {
        "groups": n,
        "groups_with_any_error": k,
        "upper_95_probability_source_has_any_error": float(beta.ppf(0.95, k + 1, n - k))
        if n > k
        else 1.0,
    }


def paired_bootstrap(gold, a, b, groups, labels, resamples=10000):
    unique = sorted(set(groups))
    if not unique:
        return {}
    indices = {g: np.array([i for i, v in enumerate(groups) if v == g]) for g in unique}

    def vector(predictions):
        result = []
        for group in unique:
            matrix = np.zeros((len(labels), len(labels) + 1), int)
            for i in indices[group]:
                matrix[
                    labels.index(gold[i]),
                    labels.index(predictions[i]) if predictions[i] in labels else len(labels),
                ] += 1
            result.append(matrix)
        return np.asarray(result)

    ma, mb = vector(a), vector(b)

    def stats(m):
        tp = np.diag(m[:, : len(labels)])
        den = m.sum(axis=1) + m[:, : len(labels)].sum(axis=0)
        f1 = np.divide(2 * tp, den, out=np.zeros(len(labels), float), where=den != 0).mean()
        fp = m[1, 0] / m[1].sum() if len(labels) == 2 and m[1].sum() else np.nan
        return np.array([f1, fp, tp.sum() / m.sum()])

    observed = stats(ma.sum(axis=0)) - stats(mb.sum(axis=0))
    rng, samples = np.random.default_rng(SEED), []
    for _ in range(resamples):
        ix = rng.integers(0, len(unique), len(unique))
        samples.append(stats(ma[ix].sum(axis=0)) - stats(mb[ix].sum(axis=0)))
    samples = np.asarray(samples)
    result = {"groups": len(unique), "resamples": resamples}
    for i, name in enumerate(("macro_f1", "false_pass_rate", "accuracy")):
        finite = samples[:, i][np.isfinite(samples[:, i])]
        result[name] = {
            "difference": float(observed[i]) if np.isfinite(observed[i]) else None,
            "ci95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None,
        }
    # One-sided centered bootstrap null tests at the preregistered NI margins.
    f1_delta, fp_delta = observed[:2]
    result["noninferiority_p"] = (
        max(
            (1 + np.sum(samples[:, 0] - f1_delta >= f1_delta + 0.02)) / (resamples + 1),
            (1 + np.sum(samples[:, 1] - fp_delta <= fp_delta - 0.01)) / (resamples + 1),
        )
        if len(labels) == 2 and np.isfinite(fp_delta)
        else None
    )
    return result


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, previous = {}, 0.0
    for rank, name in enumerate(ordered):
        previous = max(previous, min(1, (len(ordered) - rank) * pvalues[name]))
        adjusted[name] = previous
    return adjusted


def choose_threshold(rows, fallback, *, target=0.02):
    """Rows aligned by case; probabilities must already be calibrated on a separate split."""
    options = []
    base_cost = (
        sum(r["cost"]["usd"] for r in fallback)
        if all(r["cost"]["usd"] is not None for r in fallback)
        else None
    )
    for threshold in [i / 100 for i in range(50, 100)] + [0.995, 0.999]:
        auto = [r["status"] == "ok" and r.get("confidence", 0) >= threshold for r in rows]
        chosen = [r for r, use in zip(rows, auto, strict=True) if use]
        errors = sum(r["prediction"] != r["gold"] for r in chosen)
        costs = [r["cost"]["usd"] for r in rows] + [
            r["cost"]["usd"] for r, use in zip(fallback, auto, strict=True) if not use
        ]
        total = sum(costs) if all(c is not None for c in costs) else None
        row = {
            "threshold": threshold,
            "coverage": divide(len(chosen), len(rows)),
            "error": divide(errors, len(chosen)),
            "cost": total,
        }
        quality_ok, timing_ok = True, True
        if all("gold" in r and "prediction" in r for r in fallback):
            labels = sorted({r["gold"] for r in rows})
            final = [
                a["prediction"] if use else b["prediction"]
                for a, b, use in zip(rows, fallback, auto, strict=True)
            ]
            base = classification(
                [r["gold"] for r in rows], [r["prediction"] for r in fallback], labels
            )
            combined = classification([r["gold"] for r in rows], final, labels)
            quality_ok = combined["macro_f1"] >= base["macro_f1"] - 0.02
            if labels == ["no", "yes"]:
                quality_ok &= combined["false_pass_rate"] <= base["false_pass_rate"] + 0.01
        if all(r.get("latency_s") is not None for r in rows + fallback):
            projected = [
                a["latency_s"] + (0 if use else b["latency_s"])
                for a, b, use in zip(rows, fallback, auto, strict=True)
            ]
            timing_ok = np.quantile(projected, 0.95) <= 1.10 * np.quantile(
                [r["latency_s"] for r in fallback], 0.95
            )
            row["projected_p95_s"] = float(np.quantile(projected, 0.95))
        row["routing_quality_within_margin"] = bool(quality_ok)
        row["projected_latency_within_margin"] = bool(timing_ok)
        row["qualifies"] = bool(
            chosen
            and quality_ok
            and timing_ok
            and errors / len(chosen) <= target
            and len(chosen) / len(rows) >= 0.5
            and base_cost
            and total is not None
            and total <= 0.75 * base_cost
        )
        options.append(row)
    qualifying = [r for r in options if r["qualifies"]]
    return {
        "threshold": min(qualifying, key=lambda r: (-r["coverage"], r["threshold"]))["threshold"]
        if qualifying
        else None,
        "policy": "threshold" if qualifying else "fallback_only",
        "sweep": options,
    }


def consistency(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    cases, pair_matches, pair_total = [], 0, 0
    for case_id, values in grouped.items():
        labels = [r.get("prediction") for r in values]
        valid = [x for x in labels if x is not None]
        pairs = list(combinations(valid, 2))
        pair_matches += sum(a == b for a, b in pairs)
        pair_total += len(pairs)
        probabilities = [r["probabilities"] for r in values if r.get("probabilities")]
        drift = [sum(abs(a[k] - b[k]) for k in a) / 2 for a, b in combinations(probabilities, 2)]
        cases.append(
            {
                "case_id": case_id,
                "runs": len(values),
                "valid_runs": len(valid),
                "any_flip": len(set(valid)) > 1,
                "modal_agreement": divide(max(Counter(valid).values(), default=0), len(values)),
                "mean_total_variation": float(np.mean(drift)) if drift else None,
            }
        )
    return {
        "cases": cases,
        "pairwise_agreement_valid_pairs": divide(pair_matches, pair_total),
        "flip_fraction": divide(sum(c["any_flip"] for c in cases), len(cases)),
        "fingerprints": sorted({str(r.get("system_fingerprint")) for r in rows}),
    }
