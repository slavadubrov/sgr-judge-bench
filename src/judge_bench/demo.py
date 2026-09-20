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

PRIMARY = (
    "jev/direct",
    "luna/direct",
    "luna/sgr",
    "terra/direct",
    "terra/sgr",
    "deepseek-json/direct",
    "deepseek-json/sgr",
)


def jev_comparisons(summary):
    """Normalize every paired estimate to Jev minus the comparator."""
    jev = summary["arms"]["jev/direct"]
    result = {}
    for key in PRIMARY[1:]:
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
        intro = """<p><strong>Does a fixed reasoning workflow reduce sensitivity to model choice?</strong>
Luna improves from 102 to 114 correct; DeepSeek Flash from 102 to 113; Terra changes from 112 to 111.
The quality gap across these three models shrinks from 10 to 3 cases.</p>
<p><strong>Can native Jev replace it?</strong> Jev gets 108/120 correct at $0.083 per 1,000 evaluations and 0.34 s median service latency.
It trades observed accuracy for lower cost and latency: 6 fewer correct than Luna SGR, 5 fewer than DeepSeek SGR, 3 fewer than Terra SGR.
All six paired Jev accuracy-difference intervals include zero; this does not establish equivalence or noninferiority.</p>
<h2>What the approaches do</h2>
<p><b>Direct structured output:</b> one request returns ENTAILED or REFUTED.
<b>SGR:</b> plan checks → application projects exact source columns, retaining every row → a second request assesses a dynamic schema of checks with cell citations → code validates coverage/evidence and aggregates the verdict.
<b>Native Jev:</b> one Choice decision over the same claim and full table, scored by probability argmax. Jev does not generate an SGR plan or findings.</p>
<h2>Primary comparison</h2>
<p>120 tables/pages, 60 entailed and 60 refuted; 50 simple and 70 complex. Frozen before these runs, excluding 168 previously observed cases/pages/tables. Luna was added after the other results were seen, with unchanged prompts and workflow.
Costs use recorded September 20, 2026 rates and observed usage, not invoices. Ranges reflect cache accounting uncertainty.
Latency includes every request in a pipeline and excludes queues; providers/routes differ and Luna ran in a later session.</p>"""
        # ponytail: reuse the existing escaped case/trace renderer; no web framework.
        render(root, out, introduction=intro, primary=PRIMARY)
    lines = [
        "# Groundedness judge benchmark: TabFact",
        "",
        "Recorded September 20, 2026. Offline replay of 120 cases; failures remain incorrect.",
        "",
        "| Model / mode | Correct | Valid | USD / 1,000 | Median service s |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in PRIMARY:
        s = summary["arms"][key]
        lines.append(
            f"| {key} | {s['correct']}/{s['n']} | {s['valid']}/{s['n']} | {s['known_cost_lower_usd'] * 1000 / s['n']:.3f}–{s['known_cost_upper_usd'] * 1000 / s['n']:.3f} | {s['median_service_latency_s']:.2f} |"
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
    for key, p in comparisons.items():
        lo, hi = p["ci95"]
        cost = p["comparator_cost_over_jev"]
        cost_text = f"{cost[0]:.1f}–{cost[1]:.1f}×" if cost else "unknown"
        lines.append(
            f"| {key} | {p['jev_minus_comparator'] * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}] | {p['jev_only_correct']} / {p['comparator_only_correct']} | {cost_text} | {p['comparator_median_latency_over_jev']:.1f}× |"
        )
    limits = """## Interpretation and limits

The frozen SGR workflow improves Luna by 12/120 and DeepSeek Flash by 11/120; Terra changes by −1/120. This supports reduced sensitivity to model choice on this sample. It does not establish a universal relationship between model size and SGR gains. The intervention combines decomposition, a second model call, source projection, and deterministic validation; it does not isolate schemas from additional inference. Agreement with Terra includes shared errors.

Jev is the cheapest and fastest measured primary arm. It is a candidate for applications accepting the observed quality tradeoff, not a demonstrated equivalent replacement for SGR. Every paired Jev accuracy interval includes zero, but no noninferiority margin was preregistered. No replacement cascade was evaluated.

This is binary table-claim verification, not a general RAG groundedness score. TabFact includes counting and arithmetic, which the recorded Jev compiler documentation cautions against; it is a reasoning stress test. Keep the historical RAGTruth comparison separate. Labels are the original TabFact labels, including suspected ambiguities; no post-hoc relabeling. The cohort is now observed and was not held out from model pretraining. Luna was added after observing other models. One session per model is insufficient for production latency or reliability guarantees.

Luna/Terra use strict provider-enforced JSON Schema via OpenRouter with reasoning disabled; DeepSeek uses JSON object mode with thinking disabled and local schema validation. Invalid pipelines count as errors. Costs are recorded-rate estimates, not invoices; latency excludes queues. Cost and latency ratios are descriptive, without uncertainty intervals.

## Retained negative controls and diagnostics

GLM direct: 115/120 correct, 119 valid. GLM SGR: 26/120 correct, 27 valid; 93 failures (67 schema, 25 malformed JSON, 1 truncation). Forced thinking and different structured-output support confound capability comparisons; thinking has not been established as the cause.

Terra-planned Jev hybrid: 55/120 correct, 58 valid, including the planner cost. The native coverage gate rejected 62 cases, disproportionately refuted claims. Bypassing that gate gives a post-hoc diagnostic result of 112/120, not a validated benchmark score. This hybrid is not native Jev independently performing SGR. All diagnostic records remain in the replay and case browser.

## Evidence

`summary.json` is recomputed from bundled predictions, cost records and service timings; `jev-comparison.json` contains paired deltas, discordant counts and ratios. `hypothesis.json` preserves the recorded model-transfer analysis. `predictions.csv` contains all 1,200 evaluations, including diagnostic failures. The HTML exposes full tables, plans, assessments, error messages and call IDs.

The compact bundle omits raw HTTP requests/responses and duplicate source projections. It is sufficient to reproduce scoring, not the independent raw-response audit. `provenance.json` identifies the separate full evidence archive by SHA-256 and hashes its source files. Both original live runs passed raw replay audits before bundling. Future model runs require explicit `--live`, API keys and a budget.
"""
    (out / "report.md").write_text("\n".join(lines) + "\n\n" + limits)
    write_json(out / "summary.json", summary)
    write_json(out / "jev-comparison.json", comparisons)
    write_json(out / "hypothesis.json", bundle["hypothesis"])
    write_json(out / "provenance.json", bundle["provenance"])
    write_json(out / "cases.json", bundle["cases"])
    gold = {c["id"]: c["gold"] for c in bundle["cases"]}
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "model", "arm", "gold", "prediction", "status"]
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
    for key, value in comparisons.items():
        lo, hi = value["ci95"]
        paired += f"<tr><td>{html.escape(key)}</td><td>{value['jev_minus_comparator'] * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]</td><td>{value['jev_only_correct']} / {value['comparator_only_correct']}</td></tr>"
    appendix = (
        paired
        + "</table></div><details><summary>Interpretation, negative controls and evidence limits</summary>"
    )
    for paragraph in limits.strip().split("\n\n"):
        tag = "h2" if paragraph.startswith("## ") else "p"
        appendix += f"<{tag}>{html.escape(paragraph.removeprefix('## '))}</{tag}>"
    appendix += "</details>"
    page.write_text(
        page.read_text().replace(
            "<h2>Cases and intermediate outputs</h2>",
            appendix + "<h2>Cases and intermediate outputs</h2>",
        )
    )
    return str(page)
