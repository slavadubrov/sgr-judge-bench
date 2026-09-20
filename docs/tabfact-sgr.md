# TabFact comparison protocol

The article compares detailed Direct structured output, SGR v2 and native Jev on the same 120 table claims. The LLM arms were interleaved; Jev ran separately on the same frozen, already observed cohort. This is an exploratory comparison, not a fresh holdout.

## Dataset

TabFact supplies a caption, a table, a claim and an ENTAILED/REFUTED label. The sample contains 120 distinct tables/pages: 25 examples per label from the simple channel and 35 per label from the complex channel. Both labels have 60 examples.

Selection uses official test tables at upstream commit `2ab782ba42b5808076ac91fec846473aa5315a79`, seed 20260925, excluding 168 development/previously observed cases by table ID, page URL and normalized table-content hash. No filtering by length, content, operation or model output was applied. Model-pretraining contamination is not ruled out. Original labels, including suspected ambiguities, remain unchanged.

`config/dataset.json` records selection provenance and the canonical dataset hash. The demo exposes every full source table; the README includes three illustrative excerpts.

## Compared approaches

- **Direct:** full indexed table plus detailed instructions to identify predicates, check evidence and scope, and audit coverage; one request returns a final ENTAILED/REFUTED label.
- **SGR v2:** generate 1–8 conjunctive predicates and column IDs; project those columns across all rows; assess the resulting views in a second request with findings, cell citations and a coverage audit; validate and aggregate in code. The assessment has no global final-label field.
- **Native Jev:** full claim/table, common table rules and the same detailed procedural instructions as Direct through a native Choice decision, scored by probability argmax. Jev does not generate plans, findings or citations.

Direct and SGR share table-interpretation rules and comparable procedural instructions. SGR also generates intermediate tokens, makes another call and validates evidence. This comparison does not equalize inference work or isolate the causal effect of schemas. The raw `direct_guided` identifier denotes the detailed Direct shown in the article.

Incomplete coverage, unresolved predicates, malformed output and invalid citations count as failures. Validation cannot prove claim equivalence or the truth of a finding.

## Execution and scoring

`config/article.json` records the four models and their settings. Luna/Terra use strict JSON Schema with reasoning disabled through OpenRouter, pinned to the OpenAI provider without fallbacks. DeepSeek uses JSON object mode, thinking disabled and local schema validation. No per-model prompt tuning, retries or repairs are used.

Output caps are 7,168 Direct tokens, 3,072 planning tokens and 4,096 assessment tokens. Timeout is 120 seconds, with two concurrent calls per model. The runner shuffles jobs with a fixed seed.

Correct/all 120 is the primary score; failures remain incorrect. Valid/all is reported separately. Paired page bootstrap intervals use 10,000 resamples and are exploratory, without multiplicity adjustment. No equivalence or noninferiority margin is specified. Costs use dated tariffs and recorded usage, not invoices; cache uncertainty is bounded. Service latency sums required calls and excludes queues. Execution time and provider routes are not controlled between Jev and the LLM arms.

## Reproduce

```sh
uv run judge-bench demo --out work/article-demo
```

The packaged replay contains exactly seven configurations: 840 evaluations and 1,200 calls. Its export selects those configurations from the audited execution blocks without dropping cases or changing predictions, findings, failures, costs or timings. Raw HTTP remains private; source hashes and export scope are recorded in provenance. The README gives explicit live-run and reconstruction commands.
