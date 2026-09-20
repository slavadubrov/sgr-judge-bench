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

September 20, 2026. The LLM arms were interleaved; Jev ran separately on the same frozen cohort: 120 distinct tables/pages, 60 ENTAILED and 60 REFUTED. Failures count as incorrect.

| Model | Approach | Correct / 120 | Valid / 120 | Estimated USD / 1,000 | Median service latency |
|---|---|---:|---:|---:|---:|
| Jev 1.13 | Native decision | 110 | 120 | $0.093 | 0.35 s |
| GPT-5.6 Luna | Direct | 102 | 120 | $0.326–0.340 | 1.08 s |
| GPT-5.6 Luna | SGR | **111** | 118 | $0.828–0.864 | 2.98 s |
| GPT-5.6 Terra | Direct | 112 | 120 | $3.256–3.404 | 1.10 s |
| GPT-5.6 Terra | SGR | **114** | 119 | $7.715–8.103 | 3.37 s |
| DeepSeek Flash | Direct | 100 | 120 | $0.169 | 0.74 s |
| DeepSeek Flash | SGR | **114** | 118 | $0.359 | 2.44 s |

**Direct uses the detailed prompt throughout this primary comparison.** It requests decomposition, evidence checks and a coverage audit, then returns only a final label.

SGR gains 9 correct answers for Luna, 2 for Terra and 14 for DeepSeek. Paired 95% bootstrap accuracy gains are +7.50 pp [0.00, +15.00], +1.67 pp [−2.50, +5.83] and +11.67 pp [+5.00, +18.33], respectively. DeepSeek shows the clearest advantage. Luna's exact paired test gives p=0.0784; Terra is inconclusive. The observed score range narrows from 12 to 3 cases, but this does not establish a general relationship between model capability and SGR gains. These exploratory intervals are unadjusted for multiple comparisons.

Jev is cheaper and faster than all primary LLM arms. It scores 8 and 10 cases above Luna and DeepSeek Direct, and 2 below Terra Direct. Against SGR, Jev scores 1 fewer than Luna and 4 fewer than Terra or DeepSeek. All paired SGR-versus-Jev accuracy intervals include zero. This is not evidence of equivalence or noninferiority, and no replacement cascade was validated. The demo reports every paired Jev contrast and cost/latency ratio.

Costs use recorded usage and dated tariffs, not invoices. Cache-accounting uncertainty produces ranges; all calls have bounded costs. Cache usage differs between arms. Service latency sums required calls and excludes queues. Ratios describe this run, not guaranteed deployment performance.

## Three examples from the evaluated dataset

Each TabFact item contains a table, a caption and a claim. The judge must return ENTAILED or REFUTED using that source alone. These examples were selected after evaluation to illustrate two Jev successes and one failure; they are not a representative subsample. Claims and cell values below come from the frozen replay. Tables show only the relevant rows and columns; models received the full tables. Search the case IDs in the demo to inspect every original cell and SGR trace.

### A. Comparing participation across tournaments

Caption: **1993 in brazilian football**. Case: `complex:2-15009679-7.html.csv:2`.

> santos did not qualify for as many tournaments as cruzeiro did

The two relevant team rows are transposed here for readability:

| Tournament | santos | cruzeiro |
|---|---|---|
| copa libertadores 1993 | did not qualify | did not qualify |
| supercopa sudamericana 1993 | round of 16 | quarterfinals |
| copa conmebol 1993 | did not qualify | did not qualify |
| recopa sudamericana 1993 | n / a | runner - up |
| intercontinental cup 1993 | n / a | n / a |

Santos participated in one listed tournament and Cruzeiro in two. Gold: **ENTAILED**. Jev, Terra Direct and all three SGR arms are correct; Luna Direct and DeepSeek Direct return REFUTED.

### B. Preserving both constraints on the matching row

Caption: **vcu rams men 's basketball**. Case: `simple:2-14609295-5.html.csv:7`.

> l 72 - 86 results has a seed less than 12 and a year thats larger than 1996

| year | seed | results |
|---|---|---|
| 1980 | 12 | l 72 - 86 |

The matching row has seed 12, not less than 12, and year 1980, not later than 1996. Gold: **REFUTED**. Jev agrees with Terra Direct and all SGR arms; Luna Direct and DeepSeek Direct incorrectly accept the claim.

### C. A tie that Jev gets wrong

Caption: **1992 open championship**. Case: `complex:2-18122130-4.html.csv:6`.

> ian woosnam placed higher than craig parry and gordon brand , jnr

| place | player | score |
|---|---|---|
| t3 | gordon brand , jnr | 65 |
| t3 | ian woosnam | 65 |
| t9 | craig parry | 67 |

Woosnam ranks above Parry but ties Brand. Gold: **REFUTED**. Jev returns ENTAILED; every Direct and SGR LLM arm correctly rejects the claim. This illustrates a Jev error, without attributing an unobserved reasoning process to the native decision.

### Recorded answers

Direct means the detailed-prompt baseline. Every answer in this table is a valid recorded prediction; bold marks agreement with the unchanged dataset label.

| Model / approach | A: tournaments | B: row constraints | C: tied ranking |
|---|---|---|---|
| Dataset label | ENTAILED | REFUTED | REFUTED |
| Jev / Native | **ENTAILED** | **REFUTED** | ENTAILED |
| Luna / Direct | REFUTED | ENTAILED | **REFUTED** |
| Luna / SGR | **ENTAILED** | **REFUTED** | **REFUTED** |
| Terra / Direct | **ENTAILED** | **REFUTED** | **REFUTED** |
| Terra / SGR | **ENTAILED** | **REFUTED** | **REFUTED** |
| DeepSeek Flash / Direct | REFUTED | ENTAILED | **REFUTED** |
| DeepSeek Flash / SGR | **ENTAILED** | **REFUTED** | **REFUTED** |

Across **all 120 cases**, Jev's 91.7% sits between Luna/DeepSeek Direct (85.0%/83.3%) and Terra Direct (93.3%). It trails the SGR arms by 1–4 correct cases, or 0.8–3.3 percentage points, while costing less and responding faster. That is the observed tradeoff on this cohort; the examples do not establish equivalence or a generally negligible quality gap. [All recorded predictions](results/current/predictions.csv) remain available, including the counterexamples.

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

Native Jev receives the same full claim/table through its Choice decision API; probability argmax selects the label. It receives the same detailed decomposition, evidence-checking and coverage instructions as Direct, alongside the common table rules. It does not generate an SGR plan or findings. Its prompt and native interface are a separate practical baseline, not part of the within-model prompt control.

## Evidence and reproducibility

The original sample was frozen from official TabFact test tables at upstream commit `2ab782ba42b5808076ac91fec846473aa5315a79`, excluding 168 previously observed cases/pages/tables. It contains 50 simple and 70 complex claims. Gold labels are unchanged, including suspected ambiguities. The cohort is now observed, and absence from model pretraining is not established. The detailed-prompt comparison is exploratory and uses interleaved LLM arms and a separate Jev execution block, without retries, repairs or per-model prompt tuning. Latency comparisons are descriptive; execution time and provider routes are not controlled.

- [Dataset and comparison protocol](docs/tabfact-sgr.md).
- `src/judge_bench/article.json.gz`: current compact replay of 840 evaluations and 1,200 calls across the seven displayed configurations. Predictions, findings, failures, model options, dated tariffs, usage, service timings and frozen system prompts are included.
- `provenance.json`: source hashes, dataset provenance and the successful source-run reconstruction audit and exported evaluation/call counts. Raw HTTP, request identifiers and duplicate projections are omitted; full current HTTP evidence remains private.

With the private LLM and Jev execution blocks available, audit and rebuild the replay:

```sh
uv run python scripts/tabfact_audit.py "$PRIVATE_RUN_DIR"
uv run python scripts/tabfact_audit.py "$PRIVATE_JEV_RUN_DIR"
uv run python scripts/tabfact_bundle.py "$PRIVATE_RUN_DIR" work/rebuilt.json.gz \
  --jev-run "$PRIVATE_JEV_RUN_DIR"
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

The default LLM arms are detailed Direct and SGR. These commands repeat the observed cohort, not a new holdout.

## Development

```sh
uv run --frozen ruff check .
uv run --frozen python scripts/check_public_artifacts.py
uv run --frozen python -m unittest discover -s tests -q
```

Tests run offline, including replay and mocked direct/SGR/native execution. No new dependencies or web service are needed.
