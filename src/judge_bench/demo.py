"""Offline article demo. Reuse the benchmark scorer; never call a provider."""

import csv
import gzip
import html
import json
import tempfile
from pathlib import Path

from .core import write_json, write_jsonl
from .review import render
from .tabfact import report

RECORDED = (
    "jev/direct",
    "luna/direct_guided",
    "luna/sgr",
    "terra/direct_guided",
    "terra/sgr",
    "deepseek-json/direct_guided",
    "deepseek-json/sgr",
)

PRIMARY = tuple(key for key in RECORDED if not key.endswith("/sgr"))

LABELS = {"jev/direct": "Jev / Native decision"}
for model, name in (("luna", "Luna"), ("terra", "Terra"), ("deepseek-json", "DeepSeek Flash")):
    LABELS.update(
        {
            f"{model}/direct_guided": f"{name} / Direct",
            f"{model}/sgr": f"{name} / SGR",
        }
    )


def jev_comparisons(summary):
    """Normalize every paired estimate to Jev minus the comparator."""
    jev = summary["arms"]["jev/direct"]
    result = {}
    for key in RECORDED[1:]:
        a, b = sorted(("jev/direct", key))
        pair = summary["paired_comparisons"][a + " -> " + b]
        sign = 1 if b == "jev/direct" else -1
        other = summary["arms"][key]
        lo, hi = pair["ci95"]
        result[key] = {
            "jev_minus_comparator": sign * pair["estimate"],
            "ci95": [lo, hi] if sign == 1 else [-hi, -lo],
            "jev_only_correct": pair["b_wins" if sign == 1 else "a_wins"],
            "comparator_only_correct": pair["a_wins" if sign == 1 else "b_wins"],
            "comparator_cost_over_jev": [
                other["known_cost_lower_usd"] / jev["known_cost_upper_usd"],
                other["known_cost_upper_usd"] / jev["known_cost_lower_usd"],
            ]
            if not (other["unknown_cost_calls"] or jev["unknown_cost_calls"])
            and jev["known_cost_lower_usd"] > 0
            else None,
            "comparator_median_latency_over_jev": other["median_service_latency_s"]
            / jev["median_service_latency_s"],
        }
    return result


def build(out):
    out = Path(out)
    if out.exists():
        raise ValueError("Refusing to overwrite a demo; choose a new output directory")
    bundle = json.loads(gzip.decompress(Path(__file__).with_name("article.json.gz").read_bytes()))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ("cases", "manifest"):
            write_json(root / f"{name}.json", bundle[name])
        for name in ("records", "calls"):
            write_jsonl(root / f"{name}.jsonl", bundle[name])
        summary = report(root)
        if summary != bundle["summary"]:
            raise ValueError("Bundled replay differs from the recorded summary")
        comparisons = jev_comparisons(summary)
        intro = """<p><strong>Jev versus one-call structured-output LLM judges.</strong>
Each judge receives the same full table, caption and claim, with comparable fixed instructions to check every condition against the evidence.
Jev returns Choice probabilities; code selects the most likely label. The LLMs return one JSON label with thinking disabled.
Neither path receives a separately generated plan or a prompt written for an individual case.</p>
<h2>The dataset</h2>
<p>TabFact contains human-written claims about Wikipedia tables. The input is a table, its caption and one claim.
The answer is ENTAILED if the whole claim follows from that source, or REFUTED if it is false. There is no unknown option.
Claims may require a cell lookup, counting rows, comparing values or checking several conditions together.
The supplied reference label is used only for scoring.</p>
<p>This sample contains 120 distinct tables: 60 entailed and 60 refuted claims, with 50 simple and 70 complex cases.
The evidence is bounded and inspectable, making it useful for comparing decision APIs without requiring generated explanations.
It does not represent every kind of evaluation.</p>
<h2>Recorded comparison</h2>
<p>Jev scores 110/120, Luna 102/120, DeepSeek Flash 100/120 and Terra 112/120.
All four configurations return 120 valid decisions. Jev has the lowest measured cost and latency.
The sample was already observed; the results are exploratory, not a fresh holdout.</p>"""
        intro += "<details><summary>Inspect the frozen prompts</summary>"
        for stage in ("direct_guided", "jev"):
            prompt = bundle["prompts"][stage]
            title = {
                "direct_guided": "Direct",
                "jev": "Jev: native decision",
                "plan": "SGR: plan",
                "assess": "SGR: assess",
            }[stage]
            intro += f"<h3>{title}</h3><pre>{html.escape(prompt)}</pre>"
        intro += "</details>"
        # ponytail: reuse the existing escaped case/trace renderer; no web framework.
        render(root, out, introduction=intro, primary=PRIMARY, labels=LABELS)
    lines = [
        "# Jev versus structured-output LLM judges",
        "",
        "Recorded September 20, 2026. Offline replay of 120 cases; failures remain incorrect.",
        "",
        "| Model / mode | Correct | Valid | USD / 1,000 | Median service s |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in PRIMARY:
        s = summary["arms"][key]
        lines.append(
            f"| {LABELS[key]} | {s['correct']}/{s['n']} | {s['valid']}/{s['n']} | {s['known_cost_lower_usd'] * 1000 / s['n']:.3f}–{s['known_cost_upper_usd'] * 1000 / s['n']:.3f} | {s['median_service_latency_s']:.2f} |"
        )
    lines += [
        "",
        "## Native Jev versus each comparator",
        "",
        "Positive delta means Jev is more accurate. Intervals are exploratory paired page bootstrap (10,000 resamples), unadjusted for multiple comparisons.",
        "",
        "| Comparator | Jev accuracy delta, pp [95% CI] | Jev-only / comparator-only correct | Comparator cost / Jev | Comparator median latency / Jev |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in PRIMARY[1:]:
        p = comparisons[key]
        lo, hi = p["ci95"]
        cost = p["comparator_cost_over_jev"]
        cost_text = f"{cost[0]:.1f}–{cost[1]:.1f}×" if cost else "unknown"
        lines.append(
            f"| {LABELS[key]} | {p['jev_minus_comparator'] * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}] | {p['jev_only_correct']} / {p['comparator_only_correct']} | {cost_text} | {p['comparator_median_latency_over_jev']:.1f}× |"
        )
    limits = """## Interpretation and limits

Each displayed judge makes one call against the full table and claim. Both interfaces receive comparable detailed instructions for decomposition, evidence checks and coverage. Equal call count does not imply equal internal computation. The LLMs return labels without explanations; Jev returns probabilities without findings.

Jev gets eight more claims right than Luna and ten more than DeepSeek, and two fewer than Terra. Paired 95% bootstrap intervals for Jev minus Luna and DeepSeek are positive; the interval against Terra crosses zero. The intervals are exploratory, unadjusted for multiple comparisons, and establish neither equivalence nor a validated replacement policy.

The 120 cases were already observed. Public-data exposure during training is unknown, and reference labels are retained even where ambiguous. This is table-claim verification, not a general test of answer quality. The LLM calls were interleaved; Jev ran separately. No retries, repairs or per-model prompt tuning were used.

Costs use recorded September 20, 2026 rates and usage, not invoices. Ranges reflect cache-accounting uncertainty. Service times exclude scheduling queues. All displayed configurations returned valid decisions on this run; that does not establish production reliability.

## Reproducibility

The page and report show 480 one-call evaluations across four configurations. The unchanged replay archive and machine-readable exports retain all 840 recorded evaluations and 1,200 calls, including additional workflows outside this comparison. No predictions, failures or measured results are removed from those exports.

The compact replay omits raw HTTP and provider request identifiers while preserving predictions, usage, costs and service timings. Future model calls require explicit live mode, keys and a budget.
"""
    (out / "report.md").write_text("\n".join(lines) + "\n\n" + limits)
    write_json(out / "summary.json", summary)
    write_json(out / "jev-comparison.json", comparisons)
    write_json(out / "hypothesis.json", bundle["hypothesis"])
    write_json(out / "prompts.json", bundle["prompts"])
    write_json(out / "provenance.json", bundle["provenance"])
    write_json(out / "cases.json", bundle["cases"])
    gold = {c["id"]: c["gold"] for c in bundle["cases"]}
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "model", "arm", "gold", "prediction", "status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {k: gold[r["id"]] if k == "gold" else r[k] for k in writer.fieldnames}
            for r in bundle["records"]
        )
    (out / "TABFACT_LICENSE.txt").write_bytes(
        Path(__file__).with_name("TABFACT_LICENSE.txt").read_bytes()
    )
    page = out / "review.html"
    paired = "<h2>Native Jev: paired accuracy differences</h2><div class='scroll'><table><tr><th>Comparator</th><th>Jev − comparator, pp [95% CI]</th><th>Jev-only / comparator-only correct</th></tr>"
    for key in PRIMARY[1:]:
        value = comparisons[key]
        lo, hi = value["ci95"]
        paired += f"<tr><td>{html.escape(LABELS[key])}</td><td>{value['jev_minus_comparator'] * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]</td><td>{value['jev_only_correct']} / {value['comparator_only_correct']}</td></tr>"
    appendix = (
        paired + "</table></div><details><summary>Interpretation and evidence limits</summary>"
    )
    for paragraph in limits.strip().split("\n\n"):
        tag = "h2" if paragraph.startswith("## ") else "p"
        appendix += f"<{tag}>{html.escape(paragraph.removeprefix('## '))}</{tag}>"
    appendix += "</details>"
    page.write_text(
        page.read_text().replace(
            "<h2>Cases and recorded answers</h2>",
            appendix + "<h2>Cases and recorded answers</h2>",
        )
    )
    return str(page)
