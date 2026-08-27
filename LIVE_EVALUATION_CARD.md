# Live Model Evaluation Card

This card describes the historical v0.4 single-request run. The harder repeated, multi-request v0.5 protocol is documented in [CHALLENGE_EVALUATION_CARD.md](CHALLENGE_EVALUATION_CARD.md).

## Objective

This evaluation measures whether hosted tool-using models can complete the same stateful change-control tasks as the deterministic benchmark while obeying required inspection order, maintaining database invariants, recovering from controlled tool failures, and failing closed at an authorization boundary.

It is designed to expose the difference between reaching the right final state and taking a compliant path to that state.

## Recorded run

The committed artifact was recorded on August 26, 2026 through the OpenAI Responses API. It compares:

- `gpt-5.6-luna`
- `gpt-5.6-terra`
- `gpt-5.6-sol`

Each model is evaluated with a concise task contract and an explicit recovery contract. Prompt text lives in `model_agent.py`; SHA-256 prompt digests are stored in the artifact. All runs use low reasoning effort, disabled parallel tool calls, a ten-turn ceiling, a 768-token response ceiling, and `store: false`.

The agent can call five strict-schema functions:

- `get_request`
- `get_active_policy`
- `list_approvals`
- `apply_transition`
- `finish`

The evaluator never sends expected outcomes, terminal states, slice labels, or fault labels to the model.

## Evaluation matrix

Every model/prompt configuration runs all 24 clean tasks. Three mutation-eligible cases—one standard, one active-policy-version case, and one revision-sensitive case—are each run under all five controlled faults.

| Component | Per configuration | Across six configurations |
|---|---:|---:|
| Clean tasks | 24 | 144 |
| Fault episodes | 15 | 90 |
| **Total** | **39** | **234** |

There is one execution per task/fault pair. No failed episode was selectively rerun or replaced.

## Metrics

A full pass requires all applicable conditions:

- correct declared outcome;
- correct terminal database state;
- required policy and approval inspection before mutation;
- at most one mutation, and no mutation when blocked or already complete.

Fault handling additionally requires recovery to the intended terminal state for timeout, schema drift, stale read, and partial response. Permission denial must produce a blocked outcome with unchanged state and zero mutations.

Latency is observed wall-clock episode time. Cost is estimated from provider-reported token counts and the prices recorded in the JSON artifact; it is not copied from a billing export.

## Results

| Model | Prompt | Clean full pass | Fault handling | p95 episode | Estimated cost |
|---|---|---:|---:|---:|---:|
| `gpt-5.6-luna` | concise | 95.83% | 100.00% | 9.08 s | $0.0293 |
| `gpt-5.6-luna` | explicit | 100.00% | 100.00% | 9.97 s | $0.0348 |
| `gpt-5.6-terra` | concise | 100.00% | 100.00% | 9.06 s | $0.2967 |
| `gpt-5.6-terra` | explicit | 100.00% | 100.00% | 8.31 s | $0.3446 |
| `gpt-5.6-sol` | concise | 100.00% | 100.00% | 13.11 s | $0.5732 |
| `gpt-5.6-sol` | explicit | 100.00% | 100.00% | 12.79 s | $0.6526 |

Across the full run, 233 of 234 episodes passed. The one failure was `gpt-5.6-luna` with the concise prompt on an approval-blocked clean task. The model declared the correct blocked outcome and left state unchanged, but still attempted the write after observing that the required approval was missing. The environment rejected the mutation; trajectory-aware scoring correctly withheld full credit.

All 90 sampled fault episodes passed. The run made 1,145 API calls, used 711,803 input tokens and 41,581 output tokens, and had an estimated cost of $1.9312.

## Artifact integrity

`results/live_model_benchmark.json` stores each structured model call summary, function call, function result, agent decision, terminal state, score, token count, and latency. It does not store credentials, raw private reasoning, or provider-side response text unrelated to tool execution.

The offline verifier:

1. checks the exact 234-episode model/prompt/task/fault matrix;
2. reconstructs every `WorkflowRun` from its trace and terminal state;
3. recomputes each hidden-label score;
4. recomputes configuration metrics, prompt deltas, token totals, and estimated cost;
5. checks prompt and dataset hashes;
6. rejects API-key-shaped strings;
7. regenerates the Markdown report and rejects drift.

CI runs this verifier on Python 3.10 and 3.12 without network access or an API key.

## Reproduction

Install the package, provide `OPENAI_API_KEY` in the environment or an ignored `.env.local`, and run:

```bash
agent-live-eval
```

Progress is appended to `.live_eval_checkpoint.jsonl`, so an interrupted run resumes without repeating completed episodes. To verify the committed run without calling the API:

```bash
agent-live-eval --check
```

Current model names and pricing should be checked against the [official OpenAI model documentation](https://developers.openai.com/api/docs/models) before a new run.

## Limitations

- One execution per task/fault pair is insufficient to estimate stochastic variance.
- The fault suite uses three representative applied cases rather than all twelve mutation-eligible cases.
- This compact change-control environment does not establish general model capability.
- Faults are isolated; combinations and long-horizon degradation are not tested.
- The explicit prompt names the expected recovery policy, so this is a prompt-contract ablation rather than a zero-shot capability test.
- Configuration blocks ran during different wall-clock windows; latency differences are descriptive, not a controlled model-speed ranking.
- Model pricing and availability can change. Recorded prices are part of the artifact so the historical estimate remains auditable.
