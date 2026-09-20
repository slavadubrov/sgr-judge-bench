# TabFact direct / SGR comparison

All failures incorrect; one page/table per case. Paired page bootstrap, 10000 resamples; exploratory unadjusted intervals. Hybrid attributed totals include shared Terra planning; actual requests count it once. Service latency sums requests and excludes queues.

| Model / mode | Correct | Valid | Cost USD bounds (known calls) | Unknown cost calls | Median service s |
|---|---:|---:|---:|---:|---:|
| deepseek-json/direct | 102/120 | 120/120 | 0.0163–0.0163 | 0 | 0.72 |
| deepseek-json/sgr | 113/120 | 117/120 | 0.0718–0.0718 | 0 | 2.34 |
| glm-current/direct | 115/120 | 119/120 | 0.0193–0.0193 | 0 | 1.98 |
| glm-current/sgr | 26/120 | 27/120 | 0.0345–0.0345 | 0 | 4.00 |
| jev/direct | 108/120 | 120/120 | 0.0100–0.0100 | 0 | 0.34 |
| jev/hybrid | 55/120 | 58/120 | 0.4274–0.4492 | 0 | 1.80 |
| terra/direct | 112/120 | 120/120 | 0.3168–0.3425 | 0 | 1.02 |
| terra/sgr | 111/120 | 119/120 | 0.9246–0.9712 | 0 | 3.42 |

The initial experiment evaluated Terra, DeepSeek, GLM and Jev. Luna was evaluated later on the same cohort with unchanged prompts and workflow; its results are reported separately below.

DeepSeek improves by 11/120 (paired 95% bootstrap CI +2.50 to +15.83 percentage points); Terra changes by -1/120. GLM SGR suffers 93 invalid pipelines. The Jev hybrid fails its coverage gate on 62 cases; this gate confounds coverage with claim truth. A post-hoc bypass gives 112/120, but is diagnostic, not a validated improved score. The frozen primary result remains 55/120.

Jev direct versus DeepSeek SGR differs by 5/120; the CI for the latter's gain is -0.83 to +10.00 percentage points and includes zero. The highest observed score is GLM direct. These results do not establish universal SGR superiority or model independence.

Actual main-run cost across unique requests is estimated at $1.4018–$1.4741 (cache-write uncertainty), excluding access/capability probes. Eight probe responses lacked billable usage telemetry; their cost remains unknown. Hybrid rows include their shared Terra plan, so summing all row costs double-counts planning.

The raw local run is `runs/tabfact-article120`; it is ignored by Git. The offline demo provenance identifies the separate evidence archive by SHA-256. [Methodology, dataset audit and reproducible commands](../../docs/tabfact-sgr.md). [Raw-response reconstruction audit](audit.json) verified all 1,236 requests and 960 verdicts.

[Subsequent Luna extension](../tabfact-luna/README.md): unchanged workflow, Luna 102/120 direct versus 114/120 SGR. This adds support for the narrower weaker-baseline hypothesis; the original model results above are unchanged.
