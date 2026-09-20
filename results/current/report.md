# Groundedness judge benchmark: TabFact

Recorded September 20, 2026. Offline replay of 120 cases; failures remain incorrect.

| Model / mode | Correct | Valid | USD / 1,000 | Median service s |
|---|---:|---:|---:|---:|
| Jev / Native decision | 110/120 | 120/120 | 0.093–0.093 | 0.35 |
| Luna / Direct | 102/120 | 120/120 | 0.326–0.340 | 1.08 |
| Luna / SGR | 111/120 | 118/120 | 0.828–0.864 | 2.98 |
| Terra / Direct | 112/120 | 120/120 | 3.256–3.404 | 1.10 |
| Terra / SGR | 114/120 | 119/120 | 7.715–8.103 | 3.37 |
| DeepSeek Flash / Direct | 100/120 | 120/120 | 0.169–0.169 | 0.74 |
| DeepSeek Flash / SGR | 114/120 | 118/120 | 0.359–0.359 | 2.44 |

## Native Jev versus each comparator

Positive delta means Jev is more accurate. Intervals are exploratory paired page bootstrap (10,000 resamples), unadjusted for multiple comparisons.

| Comparator | Jev accuracy delta, pp [95% CI] | Jev-only / comparator-only correct | Comparator cost / Jev | Comparator median latency / Jev |
|---|---:|---:|---:|---:|
| Luna / Direct | +6.67 [+0.83, +12.50] | 11 / 3 | 3.5–3.7× | 3.0× |
| Luna / SGR | -0.83 [-6.67, +5.00] | 6 / 7 | 8.9–9.3× | 8.4× |
| Terra / Direct | -1.67 [-6.67, +3.33] | 3 / 5 | 35.2–36.8× | 3.1× |
| Terra / SGR | -3.33 [-8.33, +0.83] | 2 / 6 | 83.3–87.5× | 9.5× |
| DeepSeek Flash / Direct | +8.33 [+1.67, +15.00] | 14 / 4 | 1.8–1.8× | 2.1× |
| DeepSeek Flash / SGR | -3.33 [-9.17, +2.50] | 4 / 8 | 3.9–3.9× | 6.9× |

## Interpretation and limits

Direct uses detailed procedural instructions: identify predicates and columns, check evidence and scope, and audit coverage before returning only a label. SGR carries out similar operations through two calls, source projection, intermediate findings and application validation. The prompts are comparable rather than identical; generated reasoning tokens, call count and validation are not equalized. This comparison does not isolate schemas alone.

SGR improves Luna by 9/120, DeepSeek by 14/120 and Terra by 2/120 relative to Direct. Paired 95% bootstrap gains are +7.50 pp [0.00, +15.00], +11.67 pp [+5.00, +18.33] and +1.67 pp [−2.50, +5.83], respectively. DeepSeek shows the clearest advantage. Luna's exact paired test gives p=0.0784; Terra is inconclusive. Intervals are exploratory and unadjusted. The observed range narrows from 12 to 3 correct cases, without establishing a general model-independence claim.

Jev is the cheapest and fastest primary arm. It scores 110/120, compared with 102/112/100 for Luna/Terra/DeepSeek Direct and 111/114/114 for their SGR arms. All paired SGR-versus-Jev accuracy intervals include zero. None establishes equivalence or noninferiority. No replacement cascade was evaluated.

This is binary table-claim verification, not general RAG groundedness. TabFact includes counting and arithmetic. Original gold labels are unchanged, including suspected ambiguities. The LLM arms were interleaved and Jev ran separately on the same frozen cohort; it is not a new holdout, and pretraining contamination is not ruled out. A single run does not establish production reliability or latency.

Luna/Terra use provider-enforced strict JSON Schema with reasoning disabled. DeepSeek uses JSON object mode with thinking disabled and local validation. Five SGR pipelines fail semantic validation (three incomplete coverage, two unresolved predicates); all remain incorrect. Costs use recorded rates, not invoices; caching differs between arms and latency excludes queues.

## Evidence

`summary.json` is recomputed from 840 recorded evaluations and 1,200 calls. `jev-comparison.json` and `hypothesis.json` contain paired contrasts. `predictions.csv` retains the original arm IDs and every failure; `prompts.json` exposes the frozen Direct, SGR planning and SGR assessment system prompts, plus native Jev instructions. DeepSeek additionally receives its JSON schema in the system message.

The compact replay omits raw HTTP, provider request identifiers and duplicate source projections, while preserving findings, scores, usage, costs and service timings. `provenance.json` contains source hashes and the successful reconstruction audit. Raw HTTP evidence remains private. Future model calls require explicit `--live`, keys and a budget.
