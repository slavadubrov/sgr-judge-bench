# TabFact SGR protocol

This document records the original experiment. The [current comparison](prompt-control.md)
uses detailed Direct as the primary baseline and preserves the original shorter
prompt as a diagnostic. Historical results below retain their original meanings.

The SGR v2 workflow was frozen before the September 20, 2026 evaluation. The initial run compared Terra, DeepSeek Flash, GLM Flash and native Jev. Luna was added afterward on the same cohort with identical prompts and workflow; this extension is exploratory, not a new holdout.

## Scope

TabFact verifies binary claims against tables. Counts, comparisons, conjunctions and source binding exercise the intermediate operations in SGR. It is a reasoning stress test, not a general RAG hallucination benchmark. The recorded Jev compiler documentation cautions against counting and arithmetic, so the historical [RAGTruth comparison](../results/pilot/README.md) remains a separate practical baseline.

## Frozen workflow

- **Direct LLM:** full indexed table → final ENTAILED/REFUTED JSON.
- **SGR v2:** model generates 1–8 conjunctive predicates and column IDs → application selects exact columns across all rows → model assesses those views with cell citations and a coverage audit → application aggregates. There is no global model verdict in the final stage.
- **Native Jev:** Choice between ENTAILED and REFUTED, selected by maximum probability, using the existing lexical tie rule. This is a native decision, not generated JSON.
- **Experimental Jev hybrid:** reuse the case's Terra plan → identical source projection → native coverage Noul and per-predicate three-way Choice decisions → application aggregation. Noul uses a 0.5 threshold. Jev does not generate the plan, findings or citations.

The frozen implementation is in `src/judge_bench/sgr.py`; each original run includes a source snapshot and its hash. The shared Terra planning request counts toward the hybrid's attributed cost and latency, but only once in the actual-request ledger. Failed planning fails both dependent pipelines; Terra assessment failure does not block Jev.

Incomplete coverage or any unresolved predicate fails the example, including REFUTED + UNRESOLVED. Schema validation checks shape and source addresses; it cannot prove the truth of findings, exact claim equivalence, or consistent same-row binding. Correct final labels do not certify faithful intermediate reasoning.

## Models and execution

`config/article.json` contains the primary comparison; `config/tabfact.json` preserves the initial four-model configuration. Luna and Terra use strict JSON Schema and reasoning disabled through OpenRouter, restricted to the OpenAI provider with fallbacks disabled and required parameter support. Temperature is omitted; sampling defaults are provider-controlled.

DeepSeek Flash uses thinking disabled and JSON object mode. GLM-5.3-Flash requires thinking, set to low, with JSON object mode. Both receive the schema in the system prompt and undergo local validation; neither supplies the same provider-enforced strict-schema guarantee. GLM is not a non-thinking control.

Generative output caps are 7,168 direct, 3,072 planning and 4,096 assessment tokens. GLM reasoning shares that cap. There are no retries, JSON repairs, selective reruns or model-specific prompt tuning. Timeout is 120 seconds, with two simultaneous calls per model. Latency includes API/network overhead; provider routes and time blocks differ.

## Dataset and scoring

The sample uses official TabFact test table IDs at upstream commit `2ab782ba42b5808076ac91fec846473aa5315a79`. Seed 20260925 selects 120 distinct pages/tables: 25 examples per label from the simple channel and 35 per label from the complex channel. Both labels have 60 examples.

Selection excludes all 168 prior development/observed cases by table ID, page URL and normalized table-content hash. There is no filtering by length, content, operation or model output. These examples were unseen during workflow development, but may have occurred in model pretraining. The cohort is now observed.

Canonical dataset SHA-256: `e9fdd1348fc643d2ef5f33e5b5edd568d3fc0d6e9e9e690009bbabeef53b1599`.

The primary score is correct / all 120, including format, transport, unresolved and coverage failures. Valid/all is separate. A constant-class baseline scores 50%. Paired bootstrap intervals resample pages, retaining paired model predictions. The 10,000-resample intervals and multiple comparisons are exploratory and unadjusted. Score ranges and agreement describe this run; they do not measure repeated-run variance or isolate schemas from decomposition, additional inference and projection.

Costs use exposed usage and recorded rates, with cache-accounting bounds. Missing usage remains unknown. Requests, responses, parsed stages, projections and source snapshots are retained in separate raw evidence. Public metadata excludes client location, local filesystem paths and account-administration notes. Original source hashes refer to the pre-export files; a release archive has its own checksum.

## Reproduce

```sh
uv sync --frozen
uv run judge-bench demo --out work/tabfact-demo
```

The demo makes no API calls. For new paid runs, follow the explicit `--live` commands in the [README](../README.md#intentional-live-runs). Repeating the existing cohort does not create a new holdout.

With full evidence restored, the raw audits are:

```sh
uv run python scripts/tabfact_audit.py runs/tabfact-article120
uv run python scripts/tabfact_audit.py runs/tabfact-luna120
uv run python scripts/tabfact_compare.py \
  --runs runs/tabfact-article120 runs/tabfact-luna120 --out runs/tabfact-comparison
```

## Native hybrid failure

The frozen hybrid achieves 55/120 correct, with 58 valid pipelines. Its coverage gate rejects 56/60 REFUTED and 6/60 ENTAILED claims. The gate inherited a truth-verification guide while its intended target was logical checklist completeness, independent of truth. This is consistent with confusing truth and coverage.

Bypassing that gate gives 112/120 in a post-hoc ablation. This is a diagnostic result, not a held-out score for improved Jev SGR. A revised gate would need separate development and an untouched evaluation. All original failures remain in the primary denominator.

## Recorded sources

Sources were checked on September 20, 2026; rates and model aliases can change.

- [TabFact repository and paper](https://github.com/wenhuchen/Table-Fact-Checking)
- [Jev model](https://openrouter.ai/typesafe/jev-1.13), [Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request), [compiler and limits](https://openrouter.ai/labs/jev/compile)
- [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna), [Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra), [OpenRouter Responses API](https://openrouter.ai/docs/api/reference/responses/overview)
- [GLM Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash), [Z.ai rates](https://docs.z.ai/guides/overview/pricing)
- [DeepSeek Flash and rates](https://api-docs.deepseek.com/quick_start/pricing/)
