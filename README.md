# SGR judge benchmark

Two comparisons on the same 120 TabFact claims: **Direct structured output versus SGR with comparable detailed instructions**, and **native Jev versus those LLM judges**. Models are Luna, Terra and DeepSeek Flash. The demo runs offline and exposes predictions, prompts, intermediate checks, failures, costs and service timings.

## Run the demo

Requires Python 3.11+ and uv:

```sh
uv sync --frozen
uv run --frozen judge-bench demo --out work/article-demo
```

Open `work/article-demo/review.html`. It contains the primary comparison, paired Jev differences, frozen prompts, and all 120 searchable claims with source tables and recorded traces. The directory also contains `report.md`, `summary.json`, `jev-comparison.json`, `hypothesis.json`, `prompts.json`, `predictions.csv`, `cases.json` and provenance. Choose a new output directory; existing output is never overwritten.

The packaged replay works outside the checkout and recomputes scores with the live-run scorer. It reproduces scoring, not the separate raw-HTTP audit.

## Recorded comparison

September 20, 2026. All arms below ran together in a shuffled run on the same already observed cohort: 120 distinct tables/pages, 60 ENTAILED and 60 REFUTED. Failures count as incorrect.

| Model | Approach | Correct / 120 | Valid / 120 | Estimated USD / 1,000 | Median service latency |
|---|---|---:|---:|---:|---:|
| Jev 1.13 | Native decision | 107 | 120 | $0.083 | 0.34 s |
| GPT-5.6 Luna | Direct | 102 | 120 | $0.326–0.340 | 1.08 s |
| GPT-5.6 Luna | SGR | **111** | 118 | $0.828–0.864 | 2.98 s |
| GPT-5.6 Terra | Direct | 112 | 120 | $3.256–3.404 | 1.10 s |
| GPT-5.6 Terra | SGR | **114** | 119 | $7.715–8.103 | 3.37 s |
| DeepSeek Flash | Direct | 100 | 120 | $0.169 | 0.74 s |
| DeepSeek Flash | SGR | **114** | 118 | $0.359 | 2.44 s |

**Direct uses the detailed prompt throughout this primary comparison.** It requests decomposition, evidence checks and a coverage audit, then returns only a final label. The original shorter prompt is retained as a diagnostic, not used as the primary baseline.

SGR gains 9 correct answers for Luna, 2 for Terra and 14 for DeepSeek. Paired 95% bootstrap accuracy gains are +7.50 pp [0.00, +15.00], +1.67 pp [−2.50, +5.83] and +11.67 pp [+5.00, +18.33], respectively. DeepSeek shows the clearest advantage. Luna's exact paired test gives p=0.0784; Terra is inconclusive. The observed score range narrows from 12 to 3 cases, but this does not establish a general relationship between model capability and SGR gains. These exploratory intervals are unadjusted for multiple comparisons.

Jev is cheaper and faster than all primary LLM arms. It scores 5 and 7 cases above Luna and DeepSeek Direct, and 5 below Terra Direct. Against SGR, Jev scores 4 fewer than Luna and 7 fewer than Terra or DeepSeek. Terra SGR's paired advantage over Jev excludes zero; other primary Jev intervals include or touch zero. This is not evidence of equivalence or noninferiority, and no replacement cascade was validated. The demo reports every paired Jev contrast and cost/latency ratio.

Costs use recorded usage and dated tariffs, not invoices. Cache-accounting uncertainty produces ranges; all calls have bounded costs. Cache usage differs between arms. Service latency sums required calls and excludes queues. Ratios describe this run, not guaranteed deployment performance.

## Direct structured output versus SGR

Both approaches share table interpretation rules for counts, comparisons, negation, ties, units and scope. Both receive detailed procedural instructions. [The frozen prompts](src/judge_bench/sgr.py) are also inspectable inside the demo and exported as `prompts.json`.

| Operation | Direct | SGR v2 |
|---|---|---|
| Decompose and check claim coverage | Requested in the prompt | Explicit model-generated checklist of 1–8 predicates |
| Inspect relevant columns and evidence | Requested against the full indexed table | Code projects selected columns, retaining every row and cell address |
| Assess predicates | Requested before deciding | Second call returns findings, citations, statuses and coverage audit |
| Produce final decision | One label-only structured response | Code validates evidence/coverage and aggregates statuses |

The instructions are comparable, not identical. SGR additionally generates intermediate tokens, makes another call, projects evidence and validates the result. This is a workflow comparison, not an equal-compute experiment or proof that schemas alone cause the difference. Label-only instructions do not prove that internal checks occurred. A single response with generated reasoning before the label would be a separate control.

Luna and Terra use provider-enforced strict JSON Schema with reasoning disabled. DeepSeek uses JSON object mode, thinking disabled, and local validation. Five SGR pipelines fail on incomplete coverage or unresolved predicates; they remain incorrect. Schema validation cannot prove that the plan preserves the claim or that its findings are true.

Native Jev receives the same full claim/table through its Choice decision API; probability argmax selects the label. It uses the common table rules and does not generate an SGR plan or findings. Its prompt and native interface are a separate practical baseline, not part of the within-model prompt control.

## Preserved controls and earlier evidence

The current replay retains **Direct (short prompt)**: Luna 105/120, Terra 113/120 and DeepSeek 99/120. All 360 control predictions remain in the case browser and CSV. [Control results and paired tests](results/prompt-control/README.md) compare both prompts. Raw identifiers remain unchanged: `direct_guided` is displayed as Direct, while `direct` is displayed as Direct (short prompt) for LLMs. Jev's `direct` identifier means its native decision.

Earlier results are preserved separately, without mixing time blocks into the primary table:

- [Initial TabFact comparison](results/tabfact/README.md) and [Luna extension](results/tabfact-luna/README.md).
- [Historical compact replay](results/historical-article.json.gz), preserving all scores and GLM/Jev-hybrid traces; provider request identifiers are omitted with the transformation recorded in provenance. GLM direct/SGR scored 115/26, with 93 SGR format failures. The Terra-planned Jev hybrid scored 55/120 with 58 valid; bypassing its coverage gate gave 112/120 only as a post-hoc diagnostic.
- [Historical RAGTruth comparison](results/pilot/README.md), covering 900 answers from 150 source groups. Its one-call JSON judges were not SGR.

TabFact includes counting and arithmetic that the recorded Jev compiler documentation cautions against. These are table-reasoning results, not general RAG groundedness scores.

## Evidence and reproducibility

The original sample was frozen from official TabFact test tables at upstream commit `2ab782ba42b5808076ac91fec846473aa5315a79`, excluding 168 previously observed cases/pages/tables. It contains 50 simple and 70 complex claims. Gold labels are unchanged, including suspected ambiguities. The cohort is now observed, and absence from model pretraining is not established. The detailed-prompt comparison is exploratory and uses one fresh shuffled run, without retries, repairs or per-model prompt tuning.

- [Current comparison protocol](docs/prompt-control.md) and [original sampling/workflow protocol](docs/tabfact-sgr.md).
- `src/judge_bench/article.json.gz`: current compact replay of 1,200 evaluations and 1,560 calls, including the short-prompt controls. Predictions, findings, failures, model options, dated tariffs, usage, service timings and frozen system prompts are included.
- `provenance.json`: source hashes, dataset provenance and the successful reconstruction audit for all 1,560 requests and 1,200 verdicts. Raw HTTP, request identifiers and duplicate projections are omitted; full current HTTP evidence remains private.
- The [v0.1.0 evidence archive](https://github.com/slavadubrov/sgr-judge-bench/releases/download/v0.1.0/evidence.tar.gz) belongs to the **historical** experiment, not the current comparison.

With the current private raw run available, audit and rebuild its replay:

```sh
uv run python scripts/tabfact_audit.py "$PRIVATE_RUN_DIR"
uv run python scripts/tabfact_bundle.py "$PRIVATE_RUN_DIR" work/rebuilt.json.gz \
  --previous results/historical-article.json.gz
```

The source is [TabFact / Table-Fact-Checking](https://github.com/wenhuchen/Table-Fact-Checking). The bundled excerpt retains source URLs and the upstream [MIT license](src/judge_bench/TABFACT_LICENSE.txt). Benchmark code is MIT licensed separately.

## Intentional live runs

`config/article.json` contains the four primary models and recorded settings. Recheck availability and tariffs before spending. Keys stay in `.env` or the environment. Set each output variable to a new private directory outside the repository; raw responses can contain provider metadata.

```sh
uv run judge-bench --models config/article.json tabfact-run --live --canaries \
  --exclude-hybrid --out "$PRIVATE_PREFLIGHT_DIR" --budget-usd 1
uv run judge-bench --models config/article.json tabfact-run --live \
  --cases work/article-demo/cases.json --exclude-hybrid \
  --out "$PRIVATE_RUN_DIR" --budget-usd 5
```

The default LLM arms are detailed Direct and SGR. Add `--include-short-direct` to retain the original prompt as an extra control; `--include-prompt-control` remains an alias for reproducing the recorded three-arm run. Omit `--exclude-hybrid` only to intentionally include the historical Terra-planned hybrid. These commands repeat the observed cohort, not a new holdout.

## Development

```sh
uv run --frozen ruff check .
uv run --frozen python scripts/check_public_artifacts.py
uv run --frozen python -m unittest discover -s tests -q
```

Tests run offline, including replay and mocked direct/SGR/native execution. No new dependencies or web service are needed.
