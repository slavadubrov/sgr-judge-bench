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
    "luna/direct_guided",
    "luna/sgr",
    "terra/direct_guided",
    "terra/sgr",
    "deepseek-json/direct_guided",
    "deepseek-json/sgr",
)

LABELS = {"jev/direct": "Jev / Native decision"}
for model, name in (("luna", "Luna"), ("terra", "Terra"), ("deepseek-json", "DeepSeek Flash")):
    LABELS.update(
        {
            f"{model}/direct_guided": f"{name} / Direct",
            f"{model}/direct": f"{name} / Direct (short prompt)",
            f"{model}/sgr": f"{name} / SGR",
        }
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
        intro = """<p><strong>Direct structured output versus SGR, with comparable detailed instructions.</strong>
Luna scores 102 → 111; DeepSeek Flash 100 → 114; Terra 112 → 114, out of 120.
The observed quality range narrows from 12 to 3 cases. The paired advantage is clearest for DeepSeek; Luna and Terra remain uncertain.</p>
<p><strong>How does native Jev compare?</strong> Jev gets 107/120 correct at $0.083 per 1,000 evaluations and 0.34 s median service latency.
It is cheaper and faster than the primary LLM arms, with 4 fewer correct answers than Luna SGR and 7 fewer than Terra or DeepSeek SGR.
Terra SGR's paired accuracy advantage over Jev excludes zero; the other primary Jev intervals include or touch zero. No equivalence or replacement margin was validated.</p>
<h2>What the approaches do</h2>
<p><b>Direct:</b> detailed instructions ask for decomposition, evidence checks and a coverage audit; one request returns only ENTAILED or REFUTED.
<b>SGR:</b> plan checks → application projects exact source columns, retaining every row → a second request assesses a dynamic schema of checks with cell citations → code validates coverage/evidence and aggregates the verdict.
<b>Native Jev:</b> one Choice decision over the same claim and full table, scored by probability argmax. Jev does not generate an SGR plan or findings.</p>
<p>The instructions are comparable, not identical, and inference work is not equalized. SGR also generates intermediate tokens and makes a second call. The original shorter Direct prompt remains a diagnostic control.</p>
<h2>Primary comparison</h2>
<p>120 tables/pages, 60 entailed and 60 refuted; 50 simple and 70 complex. This is a fresh shuffled run on the already observed cohort, not a new holdout. All primary arms and the shorter-prompt controls ran together.
Costs use recorded September 20, 2026 rates and observed usage, not invoices. Ranges reflect cache accounting uncertainty.
Latency includes every request in a pipeline and excludes queues; provider routes and cache usage differ.</p>"""
        intro += "<details><summary>Inspect the frozen prompts</summary>"
        for stage, prompt in bundle["prompts"].items():
            title = {
                "direct": "Direct (short prompt)",
                "direct_guided": "Direct",
                "plan": "SGR: plan",
                "assess": "SGR: assess",
            }[stage]
            intro += f"<h3>{title}</h3><pre>{html.escape(prompt)}</pre>"
        intro += "</details>"
        # ponytail: reuse the existing escaped case/trace renderer; no web framework.
        render(root, out, introduction=intro, primary=PRIMARY, labels=LABELS)
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
    for key, p in comparisons.items():
        lo, hi = p["ci95"]
        cost = p["comparator_cost_over_jev"]
        cost_text = f"{cost[0]:.1f}–{cost[1]:.1f}×" if cost else "unknown"
        lines.append(
            f"| {LABELS[key]} | {p['jev_minus_comparator'] * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}] | {p['jev_only_correct']} / {p['comparator_only_correct']} | {cost_text} | {p['comparator_median_latency_over_jev']:.1f}× |"
        )
    limits = """## Interpretation and limits

Direct uses detailed procedural instructions: identify predicates and columns, check evidence and scope, and audit coverage before returning only a label. SGR carries out similar operations through two calls, source projection, intermediate findings and application validation. The prompts are comparable rather than identical; generated reasoning tokens, call count and validation are not equalized. This comparison does not isolate schemas alone.

SGR improves Luna by 9/120, DeepSeek by 14/120 and Terra by 2/120 relative to Direct. Paired 95% bootstrap gains are +7.50 pp [0.00, +15.00], +11.67 pp [+5.00, +18.33] and +1.67 pp [−2.50, +5.83], respectively. DeepSeek shows the clearest advantage. Luna's exact paired test gives p=0.0784; Terra is inconclusive. Intervals are exploratory and unadjusted. The observed range narrows from 12 to 3 correct cases, without establishing a general model-independence claim.

Jev is the cheapest and fastest primary arm. It scores 107/120, compared with 102/112/100 for Luna/Terra/DeepSeek Direct and 111/114/114 for their SGR arms. Terra SGR's accuracy advantage over Jev excludes zero in the paired interval; other primary Jev intervals include or touch zero. None establishes equivalence or noninferiority. No replacement cascade was evaluated.

This is binary table-claim verification, not general RAG groundedness. TabFact includes counting and arithmetic. Original gold labels are unchanged, including suspected ambiguities. This is a fresh shuffled run on the same observed cohort; it is not a new holdout, and pretraining contamination is not ruled out. A single run does not establish production reliability or latency.

Luna/Terra use provider-enforced strict JSON Schema with reasoning disabled. DeepSeek uses JSON object mode with thinking disabled and local validation. Five SGR pipelines fail semantic validation (three incomplete coverage, two unresolved predicates); all remain incorrect. Costs use recorded rates, not invoices; caching differs between arms and latency excludes queues.

## Retained controls and historical diagnostics

Direct (short prompt) remains in the case browser and exported predictions: Luna 105/120, Terra 113/120, DeepSeek 99/120. It shares the general table rules but omits the added procedural instructions. Its raw arm ID remains `direct`; primary Direct retains raw ID `direct_guided`. Display names do not relabel the underlying evidence.

Earlier runs are preserved separately in the repository's results/historical-article.json.gz and results/tabfact and results/tabfact-luna reports. Historical GLM direct/SGR scores are 115/26; GLM SGR had 93 format failures. The historical Terra-planned Jev hybrid scores 55/120 with 58 valid; bypassing its coverage gate gives 112/120 only as a post-hoc diagnostic. These arms were not rerun here and are not mixed into the current primary comparison.

## Evidence

`summary.json` is recomputed from all 1,200 recorded evaluations and 1,560 calls. `jev-comparison.json` and `hypothesis.json` contain paired contrasts. `predictions.csv` retains the original arm IDs and every failure; `prompts.json` exposes the frozen Direct, short Direct, SGR planning and SGR assessment system prompts. DeepSeek additionally receives its JSON schema in the system message.

The compact replay omits raw HTTP, provider request identifiers and duplicate source projections, while preserving findings, scores, usage, costs and service timings. `provenance.json` contains source hashes and the successful reconstruction audit. Raw HTTP evidence remains private; the v0.1.0 release archive belongs to the earlier experiment. Future model calls require explicit `--live`, keys and a budget.
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
    for key, value in comparisons.items():
        lo, hi = value["ci95"]
        paired += f"<tr><td>{html.escape(LABELS[key])}</td><td>{value['jev_minus_comparator'] * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]</td><td>{value['jev_only_correct']} / {value['comparator_only_correct']}</td></tr>"
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
