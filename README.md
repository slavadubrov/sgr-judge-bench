# Jev versus structured-output LLM judges

An offline companion to **Does bounded semantic evaluation need text generation?** Compare Jev with GPT-5.6 Luna, DeepSeek Flash and GPT-5.6 Terra on the same 120 table claims. Each judge makes **one call**, receives the full source and comparable checking instructions, and produces one of the same two decisions.

Jev returns probabilities through its Choice interface; application code selects the most likely label. The LLMs return a JSON label with reasoning or thinking disabled. Neither path receives a separately generated plan or a prompt written for an individual case.

## Run the demo

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --frozen
uv run --frozen judge-bench demo --out work/article-demo
```

Open `work/article-demo/review.html`. The page shows the four judges, paired accuracy comparisons, their frozen instructions and all 120 searchable claims with full source tables and recorded answers. It runs without API keys, network requests or paid inference. Choose a new output directory; existing output is never overwritten.

## What the judges evaluate

[TabFact](https://github.com/wenhuchen/Table-Fact-Checking) contains human-written claims about Wikipedia tables. Each case supplies the table's caption, columns and rows, plus one claim. The judge checks the claim using only that evidence and chooses:

- **ENTAILED:** the entire claim follows from the table and caption.
- **REFUTED:** the claim is false according to that source.

There is no unknown class. The reference label is used for scoring and never sent to the model. Claims can require a lookup, counting rows, comparing values or checking several conditions together. This gives both interfaces bounded, inspectable evidence without requiring generated explanations. It does not test every kind of evaluation, such as checking a long answer against several documents.

For example, an actual claim says Ian Woosnam placed higher than both Craig Parry and Gordon Brand, Jnr. The relevant rows show Woosnam and Brand tied third, with Parry tied ninth. The claim is **REFUTED** because Woosnam did not place higher than Brand. Jev accepts it with probability 0.80; all three LLM judges reject it. The demo includes the complete table and all predictions, including mistakes.

## Recorded comparison

Recorded September 20, 2026: 120 distinct tables, with 60 ENTAILED and 60 REFUTED claims; 50 simple and 70 complex cases. The LLM calls were interleaved, while Jev ran separately. There were no retries or output repairs.

| Judge | Correct / 120 | Valid / 120 | Estimated USD / 1,000 | Median / p95 service seconds |
|---|---:|---:|---:|---:|
| Jev 1.13 | 110 | 120 | $0.093 | 0.35 / 0.51 |
| GPT-5.6 Luna | 102 | 120 | $0.326–0.340 | 1.08 / 1.50 |
| DeepSeek Flash | 100 | 120 | $0.169 | 0.74 / 1.04 |
| GPT-5.6 Terra | 112 | 120 | $3.256–3.404 | 1.10 / 1.35 |

Jev has eight more correct answers than Luna and ten more than DeepSeek, with two fewer than Terra. The exploratory paired 95% bootstrap intervals for Jev's accuracy difference are +0.83 to +12.50 percentage points against Luna, +1.67 to +15.00 against DeepSeek, and −6.67 to +3.33 against Terra. These use 10,000 table-level resamples without adjustment for multiple comparisons. The Terra interval establishes neither superiority nor equivalence.

Jev is about 2.1× faster and 1.8× cheaper than DeepSeek on this run, but the absolute saving is only about $77 per million evaluations. Quality, latency needs and integration costs still determine whether switching is worthwhile. No replacement cascade was evaluated.

### Model settings

| Judge | Recorded identity | Output settings |
|---|---|---|
| Jev | Requested `typesafe/jev-1.13`; returned `typesafe/jev-1.13-20260917` | Native Choice through OpenRouter |
| Luna | `openai/gpt-5.6-luna` | OpenAI provider through OpenRouter, no fallback; strict JSON Schema; reasoning `none` |
| DeepSeek Flash | `deepseek-flash`, documented as DeepSeek-V4.1-Flash | First-party API; JSON-object mode and local validation; thinking disabled |
| Terra | `openai/gpt-5.6-terra` | OpenAI provider through OpenRouter, no fallback; strict JSON Schema; reasoning `none` |

The prompts share the task, interpretation rules and intended checking procedure, with wording adapted to each API. Equal call count does not mean equal internal computation. The LLM aliases do not pin immutable weights.

### Limits

The cohort had already been observed, so these are exploratory results rather than a fresh holdout. Public-data exposure during training is unknown. Reference labels remain unchanged, including suspected ambiguities. All four configurations returned valid responses, but 120 successful calls do not establish production reliability.

Costs apply recorded tariffs to observed usage, not invoices. Ranges reflect cache-accounting uncertainty. Service latency excludes queues; execution time and provider routes differ. These measurements describe this run, not guaranteed deployment performance.

## Evidence and reproducibility

The demo recomputes scores with the same scorer used for live runs. Its page and report focus on 480 evaluations from the four one-call configurations. The bundled archive and machine-readable exports preserve the complete experiment, including additional workflows outside this comparison: 840 evaluations and 1,200 calls. No predictions, failures or measured results have been removed from those exports.

The output includes `report.md`, `summary.json`, `jev-comparison.json`, `prompts.json`, `predictions.csv`, `cases.json` and provenance. The records allow scores, estimated costs and service times to be recomputed. The compact archive omits raw HTTP and provider request identifiers; the offline replay reproduces scoring, not the complete HTTP-level audit.

The dataset excerpt retains source URLs and the upstream [MIT license](src/judge_bench/TABFACT_LICENSE.txt). Benchmark code is MIT licensed separately. Inspect `judge-bench --help` and the implementation before any new live run; live mode requires explicit opt-in, API keys and a budget.

## Development

```sh
uv run --frozen ruff check .
uv run --frozen python scripts/check_public_artifacts.py
uv run --frozen python -m unittest discover -s tests -q
```

Tests and the packaged demo run offline. The existing experimental implementations remain available in the source.
