# SGR judge benchmark

Does schema-guided reasoning make an LLM judge less sensitive to model choice? Can a native Jev decision replace that judge at lower cost and latency?

This demo compares **Luna, Terra and DeepSeek Flash with direct structured output and a fixed SGR workflow**, against **native Jev**, on the same 120 TabFact table claims. It ships recorded predictions, intermediate checks, failures, usage and timing metadata. The default demo runs offline, without keys or model calls.

## Run the demo

Requires Python 3.11+ and uv:

```sh
uv sync --frozen
uv run --frozen judge-bench demo --out work/article-demo
```

Open `work/article-demo/review.html` in a browser. It is a self-contained page with the comparison, paired uncertainty intervals, and all 120 searchable claims with source tables and recorded intermediate outputs. The directory also contains `report.md`, `summary.json`, `jev-comparison.json`, `hypothesis.json`, `predictions.csv`, `cases.json` and provenance. Choose a new output directory when regenerating; the command refuses to overwrite prior output.

The installed Python package includes the compact replay, so the command also works outside the checkout. It recomputes scores with the same scorer used for live runs and checks them against the recorded summary. Full raw HTTP evidence is separate from this compact replay; see **Evidence** below.

## Recorded comparison

September 20, 2026; 120 different tables/pages, 60 ENTAILED and 60 REFUTED. All failures count as incorrect.

| Model | Approach | Correct / 120 | Valid / 120 | Estimated USD / 1,000 | Median service latency |
|---|---|---:|---:|---:|---:|
| Jev 1.13 | Native decision | 108 | 120 | $0.083 | 0.34 s |
| GPT-5.6 Luna | Direct structured output | 102 | 120 | $0.264–0.285 | 1.12 s |
| GPT-5.6 Luna | SGR | **114** | 118 | $0.820–0.856 | 3.00 s |
| GPT-5.6 Terra | Direct structured output | 112 | 120 | $2.640–2.854 | 1.02 s |
| GPT-5.6 Terra | SGR | 111 | 119 | $7.705–8.093 | 3.42 s |
| DeepSeek Flash | Direct structured output | 102 | 120 | $0.136 | 0.72 s |
| DeepSeek Flash | SGR | 113 | 117 | $0.599 | 2.34 s |

The fixed SGR workflow raises Luna by 12 correct answers and DeepSeek by 11, while Terra changes by −1. The quality range across the three models narrows from 10 to 3 correct answers. Luna's SGR gain is +10.0 percentage points, with a paired 95% bootstrap interval of +4.2 to +15.8 pp; DeepSeek's is +9.2 pp [+2.5, +15.8]. These exploratory results support the model-transfer hypothesis on this sample. They do not establish a general law about model size or prove that schemas alone caused the gain.

Jev is the cheapest and fastest primary arm. Against Luna SGR, it costs about 10× less and has about 9× lower median service latency, with **6 fewer correct answers**. Against DeepSeek SGR, the corresponding ratios are about 7× and 7×, with **5 fewer correct answers**. All paired Jev accuracy-difference intervals include zero; that is not evidence of equivalence or noninferiority. No quality margin or replacement cascade was validated.

Costs use recorded usage and dated tariffs, not invoices. Ranges reflect cache-accounting uncertainty; no primary-arm cost calls are unknown. Service latency sums the calls needed by a pipeline and excludes queues. Ratios are descriptive: provider routes and time blocks differ, and Luna ran after the other models.

## Structured output versus SGR

Direct structured output receives the full indexed table and returns one final label. The schema constrains that answer's format.

SGR v2 controls a sequence of operations in the application:

1. The model proposes 1–8 conjunctive checks and the source columns needed for each.
2. Code projects those exact columns from the original table, retaining every row and cell address.
3. A second request assesses a schema generated from those checks, returning findings, cell citations, per-check status and a coverage audit.
4. Code validates coverage and evidence addresses, then computes the final verdict. The second model response has no global final-label field.

Incomplete coverage, unresolved checks, malformed output and invalid citations fail the evaluation. Schema validation does not prove that the plan preserves every quantifier or that a finding is true. The intervention includes a second call, decomposition, projection and validation; this experiment does not isolate their individual effects.

Luna and Terra use provider-enforced strict JSON Schema with reasoning disabled. DeepSeek uses JSON object mode, thinking disabled, and local schema validation. Native Jev receives the same full claim/table through its Choice decision API, with probability argmax selecting the label. It does not generate an SGR plan or findings.

## Failure boundaries remain visible

The demo retains these three diagnostic arms and all their case traces:

- **GLM direct: 115/120; GLM SGR: 26/120.** The SGR pipeline has 93 format failures. Mandatory thinking and different output-format support complicate comparison; thinking has not been established as the cause.
- **Terra-planned Jev hybrid: 55/120, 58 valid.** Its native coverage gate rejects 62 cases and appears to confuse truth with checklist coverage. Planner cost is included. Bypassing this gate produces 112/120 only in a post-hoc diagnostic ablation; it is not a validated score for improved Jev SGR.

TabFact stresses table reasoning, including counting/arithmetic that the recorded Jev compiler documentation cautions against. These scores are not general RAG groundedness scores. The [historical RAGTruth comparison](results/pilot/README.md), with 900 answers from 150 source groups, remains a separate practical baseline; its one-call JSON judges were not SGR. [Historical methodology and replay commands](docs/methodology.md) remain available.

## Evidence and reproducibility

The sample was frozen from official TabFact test tables at upstream commit `2ab782ba42b5808076ac91fec846473aa5315a79`, excluding 168 previously observed cases/pages/tables. It contains 50 simple and 70 complex claims. Labels are unchanged, including suspected ambiguities. Development-unseen does not mean absent from model pretraining; this cohort is now observed. Luna was an exploratory extension after the other results were seen, using the same frozen inference code and prompts.

- [Frozen protocol, selection and failure analysis](docs/tabfact-sgr.md).
- [Initial TabFact results and audit](results/tabfact/README.md).
- [Luna extension and model-transfer statistics](results/tabfact-luna/README.md).
- `src/judge_bench/article.json.gz`: compact, packaged replay of all 1,200 evaluations and metadata for 1,596 actual calls. It omits raw requests/responses and duplicate projections. Model options, dated rates, source hashes, audit summaries and dataset provenance are included.
- Generated `provenance.json`: SHA-256 identity of the separate [evidence.tar.gz release asset](https://github.com/slavadubrov/sgr-judge-bench/releases/download/v0.1.0/evidence.tar.gz) and source files. Both live runs passed reconstruction audits of requests and verdicts. The compact replay alone does not reproduce that raw audit; the raw archive is not bundled in Git.

With the full raw runs restored, the existing tools independently audit and recompute them:

```sh
uv run python scripts/tabfact_audit.py runs/tabfact-article120
uv run python scripts/tabfact_audit.py runs/tabfact-luna120
uv run python scripts/tabfact_compare.py --help
```

The source is [TabFact / Table-Fact-Checking](https://github.com/wenhuchen/Table-Fact-Checking). The bundled excerpt retains source URLs and the upstream [MIT license](src/judge_bench/TABFACT_LICENSE.txt). Benchmark code is MIT licensed separately.

## Intentional live runs

`config/article.json` contains the four primary models with the recorded settings. Recheck model availability and tariffs before new spending. Keys stay in `.env`; `.env`, `runs/`, `data/` and `work/` are ignored. The following explicitly opt into API calls and budgets:

```sh
uv run judge-bench --models config/article.json tabfact-run --live --canaries \
  --exclude-hybrid --out runs/article-preflight --budget-usd 3
uv run judge-bench --models config/article.json tabfact-run --live \
  --cases work/article-demo/cases.json --exclude-hybrid \
  --out runs/article-repeat --budget-usd 5
```

This repeats the observed cohort; it is not a new holdout. Omit `--exclude-hybrid` only to intentionally include the historical Terra-planned hybrid. Prompts and deterministic aggregation live in `src/judge_bench/sgr.py`; provider transport is shared with the historical benchmark.

## Development

```sh
uv run --frozen ruff check .
uv run --frozen python scripts/check_public_artifacts.py
uv run --frozen python -m unittest discover -s tests -q
```

Tests run offline, including the bundled replay and a mocked full SGR/native experiment. No new dependencies or web service are needed for the demo.
