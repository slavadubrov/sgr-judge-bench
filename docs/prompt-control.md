# Prompt-only control for the TabFact SGR comparison

The detailed `direct_guided` arm is the primary **Direct** baseline in the current
demo. The original `direct` arm remains **Direct (short prompt)** for LLMs. Raw IDs
are preserved so the control report and recorded evidence retain their meanings.
Jev's `direct` arm is its native decision. All current primary results come from
the same shuffled run; earlier measurements remain historical.

The original direct and SGR arms share `COMMON`, including counting, quantifier,
negation and table interpretation rules. SGR additionally receives planning and
assessment instructions. Its measured gain therefore does not isolate workflow
structure from procedural prompting.

The `direct_guided` control adds the planning and assessment procedure in ordinary
language, while retaining the direct arm's full indexed input, label-only schema,
output token limit, provider route and model options. It asks the model to identify
predicates and relevant columns, check evidence and scope, and audit coverage before
returning one label. It does not emit intermediate checks, call tools, project the
table, generate a dynamic schema or aggregate a verdict in application code.
The prompt is fixed across models and is not tuned on individual errors.

Run direct, direct_guided and SGR together for Luna, Terra and DeepSeek, plus the
unchanged native Jev baseline. The existing runner shuffles all jobs with a fixed
seed, uses two concurrent requests per model, and preserves every failure as an
incorrect answer. No retries or repairs are added. The original 120 cases and gold
labels remain unchanged. This is an exploratory test on an already observed cohort,
not a fresh holdout or a preregistered confirmatory result.

Primary contrasts are direct_guided minus direct (effect of these added instructions)
and SGR minus direct_guided (remaining workflow gap). Report per-model paired page
bootstrap intervals, discordant wins/losses, validity, estimated cost and latency.
Intervals are exploratory and unadjusted for multiple comparisons. An interval
containing zero does not establish equivalence; no equivalence margin is specified.

Reasoning remains disabled as in the original experiment. An instruction to check
internally does not prove that internal checks occurred. This control tests whether
instructions alone suffice with a label-only response. Any remaining SGR advantage
still bundles extra calls, generated intermediate tokens, source projection and
validation; it does not establish a causal benefit of schemas alone. A one-call
response containing free-form analysis plus a label would be a separate control.

```sh
uv run judge-bench demo --out work/prompt-control-cases
uv run judge-bench --models config/article.json tabfact-run --live \
  --cases work/prompt-control-cases/cases.json --exclude-hybrid \
  --include-short-direct --out "$PRIVATE_RUN_DIR" --budget-usd 5
uv run python scripts/tabfact_audit.py "$PRIVATE_RUN_DIR"
```

Set `PRIVATE_RUN_DIR` to a new private directory outside the repository. Raw provider
responses and request metadata are retained there for audit, outside publication
inputs. The CLI reads keys from the local `.env` or environment.
