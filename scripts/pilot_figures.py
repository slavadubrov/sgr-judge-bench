"""Regenerate pilot figures offline from published metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("results/pilot")
metrics = json.loads((root / "metrics.json").read_text())
names = ["jev/P", "luna/D", "deepseek-json/D", "glm-current/D"]
labels = ["Jev", "Luna", "DeepSeek JSON", "GLM (thinking)"]
rows = [metrics["results"]["test/ragtruth/" + name] for name in names]
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
for ax, values, title, unit in zip(
    axes,
    [
        [r["quality"]["macro_f1"] for r in rows],
        [r["latency"]["successful_p95"] for r in rows],
        [r["cost"]["per_1k_attempted"] for r in rows],
    ],
    ["Quality: macro-F1 ↑", "Latency: p95 ↓", "Usage-priced cost ↓"],
    ["Macro-F1", "Seconds, valid completions", "USD per 1,000 attempts"],
    strict=True,
):
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.2))
    ax.tick_params(axis="x", labelrotation=25)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "RAGTruth QA pilot · 900 responses / 150 sources · 2026-09-19\n"
    "One call per judge; Jev via OpenRouter; GLM 5/900 schema failures"
)
for suffix in ("png", "svg"):
    fig.savefig(root / ("quality-cost-latency." + suffix), dpi=180)
plt.close(fig)
comparisons = list(metrics["paired_comparisons"].items())
fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
for i, (name, pair) in enumerate(comparisons):
    stat = pair["macro_f1"]
    delta = 100 * stat["difference"]
    lo, hi = np.array(stat["ci95"]) * 100
    ax.errorbar(delta, i, xerr=[[delta - lo], [hi - delta]], fmt="o", capsize=5, color="#0072B2")
ax.axvline(0, color="gray", linestyle="--")
ax.set(
    yticks=range(len(comparisons)),
    yticklabels=[name.split("-vs-")[1] for name, _ in comparisons],
    xlabel="Jev minus comparator macro-F1, percentage points (95% paired CI)",
    title="10,000 bootstrap resamples of 150 source groups",
)
ax.spines[["top", "right"]].set_visible(False)
for suffix in ("png", "svg"):
    fig.savefig(root / ("paired-quality." + suffix), dpi=180)
plt.close(fig)
