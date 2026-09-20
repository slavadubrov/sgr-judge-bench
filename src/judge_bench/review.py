"""Small self-contained review page from saved evidence; no web dependencies."""

import html
import json
from pathlib import Path

from .core import read_jsonl


def render(root, out, *, introduction="", primary=None, labels=None):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((root / "cases.json").read_text())
    rows = read_jsonl(root / "records.jsonl")
    summary = json.loads((root / "summary.json").read_text())
    note = summary["note"]
    if not any(k.endswith("/hybrid") for k in summary["arms"]):
        note = note.replace(
            "Hybrid attributed totals include shared Terra planning; actual requests count it once. ",
            "",
        )
    esc = html.escape
    labels = labels or {}

    def label(key):
        return esc(labels.get(key, key))

    def pretty(value):
        return esc(json.dumps(value, ensure_ascii=False, indent=2))

    parts = [
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Groundedness judge benchmark — TabFact</title><style>
body{font:16px/1.5 system-ui;margin:32px auto;max-width:1180px;padding:0 20px;color:#19232e;background:#f8fafc}
h1{font-size:30px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border:1px solid #cbd5e1;text-align:left}
th{background:#e2e8f0}details{margin:14px 0;padding:12px;border:1px solid #cbd5e1;background:white;border-radius:8px}
summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f1f5f9;padding:12px}.ok{color:#11663b}.bad{color:#a32326}.scroll{overflow:auto}a{color:#155fa0}
</style><h1>Groundedness judge benchmark</h1><p>TabFact · recorded September 20, 2026 · all failures count as errors · frozen SGR v2.</p>""",
        introduction,
        "<p>"
        + esc(note)
        + "</p><table><tr><th>Model / arm</th><th>Correct</th><th>Valid</th><th>USD / 1,000 (known)</th><th>Unknown cost calls</th><th>Median service latency</th></tr>",
    ]
    for key, s in ((k, summary["arms"][k]) for k in (primary or summary["arms"])):
        parts.append(
            f"<tr><td>{label(key)}</td><td>{s['correct']}/{s['n']}</td><td>{s['valid']}/{s['n']}</td><td>${s['known_cost_lower_usd'] * 1000 / s['n']:.3f}–{s['known_cost_upper_usd'] * 1000 / s['n']:.3f}</td><td>{s['unknown_cost_calls']}</td><td>{s['median_service_latency_s']:.2f} s</td></tr>"
        )
    parts.append(
        "</table><h2>Cases and intermediate outputs</h2><p>Open any case to inspect its full source and each recorded plan/assessment. All cases are retained; order matches the frozen dataset.</p>"
    )
    parts.append(
        '<label for="search">Search claims or case IDs</label> <input id="search" type="search"> <span id="count" aria-live="polite"></span>'
    )
    for i, c in enumerate(cases, 1):
        records = sorted(
            (r for r in rows if r["id"] == c["id"]), key=lambda r: (r["model"], r["arm"])
        )
        parts.append(
            f'<details class="case" data-search="{esc(c["id"] + " " + c["input"]["claim"])}"><summary>{i}. {esc(c["input"]["claim"])} · gold {c["gold"]}</summary><p>{esc(c["id"])} · {esc(c["channel"])}</p><p>{esc(c["input"]["caption"])}</p><div class="scroll"><table><tr>'
        )
        parts.extend(
            "<th>" + esc(h) + "</th>"
            for h in ["row", *[f"c{i}: {h}" for i, h in enumerate(c["input"]["headers"])]]
        )
        parts.append("</tr>")
        for n, row in enumerate(c["input"]["rows"]):
            parts.append(
                "<tr>" + "".join("<td>" + esc(str(v)) + "</td>" for v in [f"r{n}", *row]) + "</tr>"
            )
        parts.append(
            "</table></div><table><tr><th>Model / mode</th><th>Prediction</th><th>Status</th></tr>"
        )
        for r in records:
            color = "ok" if r["status"] == "ok" and r["prediction"] == c["gold"] else "bad"
            parts.append(
                f'<tr class="{color}"><td>{label(r["model"] + "/" + r["arm"])}</td><td>{esc(str(r["prediction"]))}</td><td>{esc(r["status"])}</td></tr>'
            )
        parts.append("</table>")
        for r in records:
            evidence = {k: r[k] for k in ("checks", "assessment", "error", "calls") if k in r}
            parts.append(
                f"<details><summary>{label(r['model'] + '/' + r['arm'])} trace</summary><pre>{pretty(evidence)}</pre></details>"
            )
        parts.append("</details>")
    parts.append("""<script>
const search = document.getElementById('search');
search.addEventListener('input', () => {
  let visible = 0;
  for (const item of document.querySelectorAll('.case')) {
    item.hidden = !item.dataset.search.toLowerCase().includes(search.value.toLowerCase());
    if (!item.hidden) visible++;
  }
  document.getElementById('count').textContent = visible + ' cases';
});
</script></html>""")
    (out / "review.html").write_text("\n".join(parts))
