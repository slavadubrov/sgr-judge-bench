# Jev versus structured-output LLM judges

Recorded September 20, 2026. Offline replay of 120 cases; failures remain incorrect.

| Model / mode | Correct | Valid | USD / 1,000 | Median service s |
|---|---:|---:|---:|---:|
| Jev / Native decision | 110/120 | 120/120 | 0.093–0.093 | 0.35 |
| Luna / Direct | 102/120 | 120/120 | 0.326–0.340 | 1.08 |
| Terra / Direct | 112/120 | 120/120 | 3.256–3.404 | 1.10 |
| DeepSeek Flash / Direct | 100/120 | 120/120 | 0.169–0.169 | 0.74 |

## Native Jev versus each comparator

Positive delta means Jev is more accurate. Intervals are exploratory paired page bootstrap (10,000 resamples), unadjusted for multiple comparisons.

| Comparator | Jev accuracy delta, pp [95% CI] | Jev-only / comparator-only correct | Comparator cost / Jev | Comparator median latency / Jev |
|---|---:|---:|---:|---:|
| Luna / Direct | +6.67 [+0.83, +12.50] | 11 / 3 | 3.5–3.7× | 3.0× |
| Terra / Direct | -1.67 [-6.67, +3.33] | 3 / 5 | 35.2–36.8× | 3.1× |
| DeepSeek Flash / Direct | +8.33 [+1.67, +15.00] | 14 / 4 | 1.8–1.8× | 2.1× |

## Interpretation and limits

Each displayed judge makes one call against the full table and claim. Both interfaces receive comparable detailed instructions for decomposition, evidence checks and coverage. Equal call count does not imply equal internal computation. The LLMs return labels without explanations; Jev returns probabilities without findings.

Jev gets eight more claims right than Luna and ten more than DeepSeek, and two fewer than Terra. Paired 95% bootstrap intervals for Jev minus Luna and DeepSeek are positive; the interval against Terra crosses zero. The intervals are exploratory, unadjusted for multiple comparisons, and establish neither equivalence nor a validated replacement policy.

The 120 cases were already observed. Public-data exposure during training is unknown, and reference labels are retained even where ambiguous. This is table-claim verification, not a general test of answer quality. The LLM calls were interleaved; Jev ran separately. No retries, repairs or per-model prompt tuning were used.

Costs use recorded September 20, 2026 rates and usage, not invoices. Ranges reflect cache-accounting uncertainty. Service times exclude scheduling queues. All displayed configurations returned valid decisions on this run; that does not establish production reliability.

## Reproducibility

The page and report show 480 one-call evaluations across four configurations. The unchanged replay archive and machine-readable exports retain all 840 recorded evaluations and 1,200 calls, including additional workflows outside this comparison. No predictions, failures or measured results are removed from those exports.

The compact replay omits raw HTTP and provider request identifiers while preserving predictions, usage, costs and service timings. Future model calls require explicit live mode, keys and a budget.
