# Methodology

## Task and reference labels

The measured task is groundedness: does an answer contain at least one factual assertion unsupported by, or conflicting with, its context? Missing evidence counts as unsupported even when a claim happens to be true. The exact criterion and label definitions are in [`core.py`](../src/judge_bench/core.py).

The experiment uses all 900 official RAGTruth QA test responses, grouped under 150 sources. A response is labelled `yes` when its human hallucination-span annotation is nonempty, including `implicit_true`; otherwise it is `no`. There are 160 positive and 740 negative examples. Labels and annotator explanations are never sent to the judges.

Dataset source: [RAGTruth](https://github.com/ParticleMedia/RAGTruth/tree/c103204b9ce28d6bbad859304bf30de72b8ed8fe), pinned to commit `c103204b9ce28d6bbad859304bf30de72b8ed8fe`. Download hashes and split information are in [dataset-manifest.json](../results/dataset-manifest.json). RAGTruth's terms apply to its data; this repository's MIT license covers the benchmark code.

## Judge configurations

All systems receive the same evidence, question, answer and semantic criterion. The LLMs generate a final label in a structured JSON response (`D` in the artifacts). Jev uses the native Noul decision API and supplies a probability (`P`); the fixed decision threshold is 0.5. The adapter maps the rubric to each provider's API without generating intermediate claims or explanations.

Luna uses strict JSON Schema with reasoning disabled. DeepSeek uses GA JSON object mode with thinking disabled. GLM uses JSON object mode with low-effort reasoning because the selected model requires thinking. All outputs undergo local schema validation. Jev is served through OpenRouter; the LLMs use first-party APIs. Exact options, limits and dated rates are in [models.json](../results/pilot/models.json).

The published pilot uses four profiles only. Additional adapter configurations remain in `config/models.json` for development. DeepSeek beta strict mode and older free GLM are excluded from the pilot; [development diagnostics](../results/development-sanity/README.md) describe the observed failures.

## Execution and scoring

The run took place on September 19, 2026 from a single client network. Each provider had at most one request in flight. Ten-case blocks were interleaved across providers, with a 30-second request timeout and no retries or output repair. Provider prefix caching was allowed; application result caching was off.

Before scaling, 60 cases from ten source groups selected with seed 151 passed a sanity gate. The gate stops all models if any emits a constant label, has more than 20% invalid outputs, or achieves zero or perfect accuracy. The remaining 840 cases then ran once. The cohort size and rubric were fixed before test outputs; the run did not stop when a desired significance level appeared.

Accuracy and macro-F1 count invalid outputs as errors. Confusion matrices include invalid predictions. An always-supported baseline is included because class imbalance makes accuracy misleading. Paired 95% bootstrap intervals use 10,000 resamples of source groups, keeping each source's six answers together.

Latency includes request serialization, network transit, provider serving and response validation. It excludes time waiting for a scheduling slot. Both valid-response and failure-aware all-attempt summaries are retained. Cost uses exposed token counts and the dated provider tariff, with cache categories handled separately and reasoning tokens not charged twice. Unexposed values remain unknown.

The machine-readable report also retains exploratory noninferiority calculations: `H1` uses a 0.02 macro-F1 margin and a 0.01 false-pass-rate margin, with Holm adjustment across eligible label-only comparators; `H2` additionally requires cost and p95 ratios of at most 0.70. Forced-thinking GLM is outside that non-thinking comparison family. These calculations do not establish general model equivalence. `H5` and full-schema extensions are untested.

Repeated-run stability, perturbation robustness, LLM probability calibration, calibrated thresholds and cascades were not measured in this pilot. ANLI development diagnostics are not a completed ANLI test evaluation.

## Run a new pilot

First review the [recorded prices and sources](../results/pilot/README.md#configuration-and-pricing). Model aliases and tariffs can change. Update `config/pilot.json` with independently verified current values before using it for a new run. Preserve the published result configurations as historical evidence.

```sh
uv sync --frozen
cp .env.example .env
# Add the four provider keys to .env locally.
uv run --frozen judge-bench prepare --out data
uv run --frozen judge-bench --models config/pilot.json preflight --live \
  --out runs/preflight --budget-usd 0.10 --location 'YOUR CITY, NETWORK TYPE'
```

Preparation downloads pinned public datasets but makes no model calls. Preflight tests access and output compatibility on separate canaries. Its output includes `admission.json` and `completion.json`. Inspect the failures and recorded cost before proceeding. The preparation command also downloads ANLI for the development tools; the pilot evaluates only RAGTruth.

```sh
# Replace 0.10 with actual prior spend, including preflight and any interruption reserve.
uv run --frozen python scripts/pilot.py --live \
  --models config/pilot.json --preflight runs/preflight/admission.json \
  --prior-spend-usd 0.10 --location 'YOUR CITY, NETWORK TYPE' \
  --out runs/my-pilot
```

The pilot stops after one hour or at its $1 allowance, reduced when prior spend approaches the $10 cumulative ceiling. Conservative in-flight reserves are included in budget accounting; this is an operational guard, not a provider billing guarantee. Existing output directories are refused to prevent accidental duplicate runs. Check `sanity-check.json`, `ledger.json` and `results/metrics.json`; an interrupted or incomplete cohort must be reported as incomplete.

## Published records

The [raw archive](../results/pilot/raw-run.tar.gz) contains every plan and attempt, including failures. Its [manifest](../results/pilot/raw-manifest.json) records the archive checksum. Administrative metadata was removed and client client location withheld for publication; attempt and plan JSONL files are unchanged. Historical content hashes refer to the original run files before those metadata edits. Replaying the report uses recorded predictions, labels and measurements and makes no API calls.
