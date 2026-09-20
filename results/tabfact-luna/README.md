# Luna extension: the weaker-baseline hypothesis gains support

Same 120 TabFact cases, same inference code hash, prompts, strict-schema setup and OpenRouter options as Terra. No per-model tuning, repairs, retries or selective exclusions. Luna was added after seeing the other model results; this is an exploratory extension, not an independent holdout.

| Model | Direct structured output | Frozen SGR v2 | Change |
|---|---:|---:|---:|
| Luna | 102/120 (85.0%) | 114/120 (95.0%) | +12 |
| DeepSeek Flash | 102/120 (85.0%) | 113/120 (94.2%) | +11 |
| Terra | 112/120 (93.3%) | 111/120 (92.5%) | -1 |

Luna SGR fixes 13 direct errors and introduces 1. Paired page-bootstrap improvement: +10.0 percentage points, 95% CI +4.17 to +15.83; exact two-sided McNemar p=0.00183. Its gain exceeds Terra's by +10.83 pp, paired CI +2.50 to +19.17 pp. These exploratory comparisons are unadjusted for multiplicity.

Agreement also increases: Luna and Terra produce the same valid prediction on 98/120 cases in direct mode and 111/120 in SGR. Shared correct decisions rise from 96 to 109; shared wrong decisions remain at 2. Invalid predictions never count as agreement. This supports reduced sensitivity to the model on this cohort, not merely convergence of aggregate scores. It does not establish equal capability, repeated-run stability or a universal model-independence guarantee.

All 360 Luna responses passed JSON/schema validation. Two SGR pipelines failed on unresolved evidence; they remain errors in the primary denominator (118 valid, 114 correct). Main-run cost is estimated at $0.1301–$0.1369, excluding a roughly $0.0013 capability probe. The run took 4.9 minutes. Raw-response replay verified 360 requests and 240 verdicts. Original runs remain unchanged.

The full [combined summary](summary.json) retains GLM (115 direct, 26 SGR) and Jev (108 direct, 55 experimental hybrid). GLM had 93 invalid SGR pipelines, and its JSON-object mode was not provider-enforced strict JSON Schema. Its mandatory thinking mode is documented, but the experiment cannot establish thinking as the cause of its format failures. The Jev hybrid's coverage design remains a known failure. Both remain relevant failure cases for deploying this workflow.

**On this TabFact cohort, the frozen workflow substantially improved two models with lower direct-answer accuracy and narrowed their gap to Terra, which showed no improvement.** The result supports the proposed mechanism, but does not isolate the schema from the extra call, decomposition and source projection, nor establish a law about weaker models across tasks.

## Reproduce

```sh
uv run judge-bench --models config/tabfact-luna.json tabfact-run --live \
  --canaries --planner luna --out runs/tabfact-luna-preflight --budget-usd 0.2
uv run judge-bench --models config/tabfact-luna.json tabfact-run --live \
  --cases data/tabfact-article120.json --planner luna --out runs/tabfact-luna120 --budget-usd 1
uv run python scripts/tabfact_compare.py \
  --runs runs/tabfact-article120 runs/tabfact-luna120 --out runs/tabfact-with-luna
uv run python scripts/tabfact_review.py runs/tabfact-with-luna work/luna-extension
```

The first two commands spend API credit; the others operate offline. Existing run directories are never overwritten. The combined directory is a derived view of separate time blocks, not another live run. [Luna replay audit](audit.json), [paired tests and agreement](hypothesis.json), [base protocol and limitations](../../docs/tabfact-sgr.md).
