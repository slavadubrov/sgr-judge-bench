# TabFact prompt-only control

This report preserves raw arm IDs: `direct_guided` is displayed as **Direct** in
the primary demo; LLM `direct` is **Direct (short prompt)**. Jev `direct` denotes
its native decision. No predictions or historical identifiers were relabeled.

Recorded 2026-09-20. Fresh interleaved run on the same observed 120-case cohort; no retries or repairs. All failures count as incorrect.

Adding procedural instructions to a label-only response did not reproduce the SGR
gain in this run. Direct-guided changed the number correct by −3 for Luna, −1 for
Terra and +1 for DeepSeek relative to direct. None of these prompt-only differences
is distinguishable from zero with the reported paired intervals.

SGR exceeded direct-guided by 14 cases for DeepSeek, with a paired interval excluding
zero. Luna's 9-case advantage is less certain: its bootstrap interval touches zero
and the exact paired test gives p=0.0784. Terra's 2-case difference is inconclusive.
These results argue against this particular prompt-only explanation most clearly
for DeepSeek. They do not rule out a different prompt or a one-call answer that
generates intermediate reasoning before its label.

| Model / arm | Correct / 120 | Valid / 120 | Estimated USD / 1,000 | Median service s |
|---|---:|---:|---:|---:|
| deepseek-json/direct | 99 | 120 | 0.036–0.036 | 0.80 |
| deepseek-json/direct_guided | 100 | 120 | 0.169–0.169 | 0.74 |
| deepseek-json/sgr | 114 | 118 | 0.359–0.359 | 2.44 |
| jev/direct | 107 | 120 | 0.083–0.083 | 0.34 |
| luna/direct | 105 | 120 | 0.264–0.285 | 1.14 |
| luna/direct_guided | 102 | 120 | 0.326–0.340 | 1.08 |
| luna/sgr | 111 | 118 | 0.828–0.864 | 2.98 |
| terra/direct | 113 | 120 | 2.640–2.854 | 1.05 |
| terra/direct_guided | 112 | 120 | 3.256–3.404 | 1.10 |
| terra/sgr | 114 | 119 | 7.715–8.103 | 3.37 |

## Paired contrasts

Differences are B minus A, in percentage points. Wins/losses count cases where only B/A is correct. Paired page bootstrap: 10,000 resamples. Exact two-sided McNemar p-values and intervals are exploratory, without multiplicity adjustment.

| Model | A → B | Difference [95% CI] pp | B wins / A wins | Exact p |
|---|---|---:|---:|---:|
| luna | direct → direct_guided | -2.50 [-8.33, +2.50] | 4 / 7 | 0.5488 |
| luna | direct_guided → sgr | +7.50 [+0.00, +15.00] | 15 / 6 | 0.0784 |
| terra | direct → direct_guided | -0.83 [-4.17, +1.67] | 1 / 2 | 1.0000 |
| terra | direct_guided → sgr | +1.67 [-2.50, +5.83] | 4 / 2 | 0.6875 |
| deepseek-json | direct → direct_guided | +0.83 [-3.33, +5.00] | 4 / 3 | 1.0000 |
| deepseek-json | direct_guided → sgr | +11.67 [+5.00, +18.33] | 16 / 2 | 0.0013 |

## Evidence and limitations

All 1,560 API responses passed transport and output parsing. Five SGR pipelines
failed semantic validation: three incomplete-coverage and two unresolved-predicate
cases. Luna and DeepSeek each had two failures; Terra had one. They remain incorrect.
The audit reconstructed every request and replayed all 1,200 verdicts. No nonzero
reasoning-token usage was reported. The main run took 344.5 seconds and cost an
estimated $1.980 at the upper bound; the 39-request preflight added $0.021.

The historical article numbers remain unchanged. Fresh direct and SGR results
differ from that earlier run, so all contrasts here use this run's paired controls.
Cache usage also differs between arms; costs describe this shuffled run and should
not be interpreted as a cache-controlled deployment estimate.

[Protocol](../../docs/prompt-control.md), [full summary](summary.json), [all predictions](predictions.csv), [hashes and reconstruction audit](provenance.json). Model options and dated tariff estimates match `config/article.json`. Costs are estimates, not invoices; preflight is excluded from per-arm costs.

This control changes only the system instructions relative to direct. It still produces only a label, with reasoning disabled. It tests whether those instructions suffice, not whether free-form generated reasoning can replace SGR. A remaining workflow advantage cannot distinguish the extra call, intermediate generated tokens, projection, validation and schema effects. This observed cohort and single repeat do not establish generality, stability or equivalence.
