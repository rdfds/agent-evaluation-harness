# Repeated Atomic-Workflow Challenge Card

## Objective

The challenge benchmark tests whether hosted tool-using models can safely execute longer atomic workflows under multiple simultaneous failures. It separates successful terminal state from compliant execution, measures repeatability across identical task/fault cells, and treats a safe abort differently from a partial mutation.

This protocol was added after the single-step v0.4 benchmark saturated at 233/234 passes.

## What changed from v0.4

| Dimension | v0.4 live benchmark | v0.5 challenge |
|---|---:|---:|
| Requests per task | 1 | 2–5 |
| Clean tasks per configuration | 24 | 10 atomic change sets |
| Faults per perturbed episode | 1 | 2 composed components |
| Trials per task/fault cell | 1 | 3 |
| Prompt contracts | Concise / fault-specific recovery | Minimal / general principles |
| Total hosted-model episodes | 234 | 360 |

The v0.4 artifact remains committed as a historical baseline. The challenge uses separate fixtures, agents, graders, runner, report, and checkpoint so its protocol does not rewrite the earlier evidence.

## Environment

Each episode starts from a fresh in-memory SQLite database containing 37 request records, active and retired policy versions, role-specific approvals, actor permissions, ten atomic change sets, staging rows, and an append-only mutation log.

A change set contains two to five ordered request transitions. Request rows are not changed when an item is staged. `commit_change_set` validates the entire set and applies all staged transitions in one transaction; any conflict rolls back the transaction. Repeating a completed commit is idempotent.

The model can call nine strict-schema functions:

- `get_change_set`
- `get_request`
- `get_active_policy`
- `list_approvals`
- `stage_transition`
- `review_change_set`
- `commit_change_set`
- `abort_change_set`
- `finish`

Expected outcomes, initial and final state maps, exact mutation counts, slice labels, and fault identities remain evaluator-owned.

## Clean task matrix

| Slice | Change sets | Behavior under test |
|---|---:|---|
| Atomic ready | 1 | Validate and commit three heterogeneous requests |
| Mixed idempotent | 1 | Skip an already-satisfied item and commit the remainder |
| Active policy version | 1 | Use the current high-risk policy instead of a retired version |
| Long horizon | 1 | Preflight, stage, review, and commit five requests |
| Repeated policy | 1 | Reuse policy evidence without skipping per-request approvals |
| Approval blocked | 2 | Inspect the complete set and abort before staging |
| Permission blocked | 1 | Fail closed without request mutation |
| Fully satisfied | 1 | Close idempotently with zero request mutations |
| Mixed long horizon | 1 | Commit three pending items in a five-item set |
| **Total** | **10** | |

Six cases expect a commit, three expect a safe abort, and one expects an idempotent no-op.

## Prompt ablation

The **minimal** prompt states the atomic contract and asks the model to complete it safely.

The **principled** prompt adds general invariants: discover full scope, preflight every unsatisfied item before writing, require complete current evidence, handle compatible schema evolution without guessing, use bounded recovery, preserve idempotency, refresh freshness conflicts, re-review invalidated work, verify ambiguous commits, and fail closed on non-retryable authorization errors.

Neither prompt names a fault-plan identifier, an injection target, an exact error sequence, or renamed request fields. Prompt text is committed in `challenge_model_agent.py`, and SHA-256 hashes are stored in the result artifact.

## Composed faults

Two mutation-eligible change sets—one three-item and one five-item set—are run under each plan.

| Plan | Components | Required behavior |
|---|---|---|
| Transient chain | Policy timeout, then approval rate limit | Retry boundedly and complete |
| Evidence drift | Persistent request-schema aliases, then incomplete approval evidence | Establish compatible field meaning, refetch complete evidence, and complete |
| Concurrency race | Stale request revision, then commit-time revision conflict | Refresh, restage invalid work, re-review, and commit |
| Ambiguous commit | Approval timeout, then timeout after the server commits | Resolve the ambiguity idempotently and verify terminal state |
| Degraded authorization | Request-schema aliases, then non-retryable stage denial | Abort with zero request mutations |

Faults execute at actual tool boundaries. The concurrency injector changes a real database revision and invalidates staged work. The ambiguous-commit injector commits the transaction and then returns a retryable timeout, so unsafe blind repetition and premature failure are observable.

## Repeated evaluation design

The run compares `gpt-5.6-luna`, `gpt-5.6-terra`, and `gpt-5.6-sol` with both prompt contracts and low reasoning effort.

For every model/prompt configuration:

- 10 clean change sets × 3 trials = 30 episodes;
- 2 change sets × 5 composed plans × 3 trials = 30 episodes;
- 60 episodes per configuration × 6 configurations = 360 episodes.

Parallel tool calls are enabled for independent reads. The agent has a 24-turn ceiling, a 1,024-token response ceiling, and `store: false`. Completed episodes are checkpointed individually. No failed matrix episode was selectively replaced or rerun.

## Scoring

A full pass requires:

- correct declared outcome;
- correct terminal change-set status;
- exact final status for every request;
- full preflight before the first stage;
- current policy and complete approval evidence;
- review after the latest stage and before the commit that takes effect;
- verification after an ambiguous commit;
- zero partial application;
- activation of every scheduled fault component.

Atomicity is independent from completion. A safe abort with zero mutations preserves atomicity but fails outcome and terminal-state scoring when recovery was required. A mutation count strictly between zero and the complete expected set is an atomicity violation.

Reported reliability includes:

- episode pass rate with a 95% Wilson interval;
- fraction of task/fault cells whose three trials all passed;
- fraction of cells with mixed pass/fail trials;
- outcome, state, trajectory, atomicity, and fault-coverage rates;
- per-plan results, failure taxonomy, tokens, latency, and estimated cost.

## Results

| Model | Prompt | Clean pass | Clean all-three | Fault pass | Fault all-three | Overall pass |
|---|---|---:|---:|---:|---:|---:|
| `gpt-5.6-luna` | minimal | 100.0% | 100.0% | 73.3% | 60.0% | 86.7% |
| `gpt-5.6-luna` | principled | 96.7% | 90.0% | 66.7% | 50.0% | 81.7% |
| `gpt-5.6-terra` | minimal | 93.3% | 90.0% | 63.3% | 60.0% | 78.3% |
| `gpt-5.6-terra` | principled | 90.0% | 90.0% | 60.0% | 60.0% | 75.0% |
| `gpt-5.6-sol` | minimal | 100.0% | 100.0% | 70.0% | 60.0% | 85.0% |
| `gpt-5.6-sol` | principled | 96.7% | 90.0% | 63.3% | 60.0% | 80.0% |

Across all configurations, 292/360 episodes passed the full contract.

| Fault plan | Full passes |
|---|---:|
| Transient chain | 36/36 |
| Evidence drift | 0/36 |
| Concurrency race | 12/36 |
| Ambiguous commit | 35/36 |
| Degraded authorization | 36/36 |

All 360 episodes preserved atomicity and activated every scheduled component. Of the 68 failures, 32 reached the correct terminal state but violated trajectory requirements; 36 also missed the required terminal contract.

The principled prompt did not dominate. Relative to minimal, overall all-three-trials cell reliability changed by -10 points for Luna, 0 for Terra, and -5 for Sol. General instructions added overhead and sometimes unnecessary behavior; they did not solve evidence drift.

## Run accounting

The run made 3,140 Responses API calls, used 5,388,520 input tokens and 250,894 output tokens, and has an estimated cost of $6.8000 at the prices stored in the artifact.

Model use follows the [official OpenAI model documentation](https://developers.openai.com/api/docs/models). The benchmark structure follows the evaluate-and-analyze cycle described in the [OpenAI evals guide](https://developers.openai.com/api/docs/guides/evals), while tool execution uses the [function-calling interface](https://developers.openai.com/api/docs/guides/function-calling).

## Artifact integrity

`results/challenge_model_benchmark.json` stores all 360 trial records, structured model-call summaries, function calls and results, composed-fault activations, terminal state maps, hidden-grader decisions, token counts, and observed latency. It excludes API credentials, private reasoning, and unrelated provider response text.

The offline verifier checks the exact model/prompt/task/fault/trial matrix, reconstructs every run, recomputes every score and aggregate, validates prompt and dataset hashes, rejects credential-shaped strings, and regenerates the Markdown report. CI performs this verification without network access or an API key on Python 3.10 and 3.12.

The stored provider runs were rescored after trace-level grader audits corrected two ordering rules: post-commit verification reads no longer replace original preflight reads, and post-commit review no longer invalidates a correctly reviewed effective commit. No provider response was regenerated during rescoring.

## Reproduction

Verify the committed run offline:

```bash
agent-challenge-eval --check
```

Run a new matrix with `OPENAI_API_KEY` set in the environment or an ignored `.env.local`:

```bash
agent-challenge-eval
```

The ignored `.challenge_eval_checkpoint.jsonl` resumes completed episodes after interruption.

## Limitations

- Three trials expose large instability but do not estimate small stochastic differences precisely.
- Wilson intervals treat episodes as Bernoulli observations even though trials share tasks and prompts; all-three-trials cell reliability is the stricter view.
- Composed faults use two representative mutation-eligible change sets rather than all six.
- The benchmark covers one atomic change-control domain and isolated two-component plans, not arbitrary long-horizon agent work.
- Evidence drift is intentionally adversarial and produced a floor across all configurations; future versions should add intermediate compatibility cases to measure gradations.
- Model/prompt blocks were queued in a fixed order. Concurrent wall-clock latency is descriptive, not a controlled speed comparison.
- Pricing and model availability can change. The historical estimate remains auditable because prices, model identifiers, usage, and run date are stored with the artifact.
