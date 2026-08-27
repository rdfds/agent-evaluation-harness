# Repeated atomic-workflow challenge

This recorded run contains **360 hosted-model episodes** across 3 models, 2 prompt contracts, and 3 trials per task/fault cell. Each configuration runs 10 clean multi-request change sets and 10 composed-fault cells.

| Model | Prompt | Clean pass (95% Wilson) | Clean all-trials | Fault pass (95% Wilson) | Fault all-trials | Trajectory | Atomicity | p95 episode | Est. cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt-5.6-luna` | minimal | 100.0% [88.6%, 100.0%] | 100.0% | 73.3% [55.5%, 85.8%] | 60.0% | 86.7% | 100.0% | 32.47s | $0.1115 |
| `gpt-5.6-luna` | principled | 96.7% [83.3%, 99.4%] | 90.0% | 66.7% [48.8%, 80.8%] | 50.0% | 81.7% | 100.0% | 29.79s | $0.1226 |
| `gpt-5.6-terra` | minimal | 93.3% [78.7%, 98.2%] | 90.0% | 63.3% [45.5%, 78.1%] | 60.0% | 78.3% | 100.0% | 28.06s | $1.1502 |
| `gpt-5.6-terra` | principled | 90.0% [74.4%, 96.5%] | 90.0% | 60.0% [42.3%, 75.4%] | 60.0% | 75.0% | 100.0% | 25.31s | $1.1510 |
| `gpt-5.6-sol` | minimal | 100.0% [88.6%, 100.0%] | 100.0% | 70.0% [52.1%, 83.3%] | 60.0% | 85.0% | 100.0% | 37.25s | $2.1299 |
| `gpt-5.6-sol` | principled | 96.7% [83.3%, 99.4%] | 90.0% | 63.3% [45.5%, 78.1%] | 60.0% | 80.0% | 100.0% | 35.49s | $2.1349 |

## Principle prompt ablation

Deltas are principled minus minimal. Accuracy and all-trials deltas favor the general recovery contract when positive; latency and cost deltas measure overhead.

| Model | Clean pass | Fault pass | Overall all-trials | p95 episode | Est. cost |
|---|---:|---:|---:|---:|---:|
| `gpt-5.6-luna` | -3.3% | -6.7% | -10.0% | -2.68s | $+0.0112 |
| `gpt-5.6-terra` | -3.3% | -3.3% | +0.0% | -2.74s | $+0.0008 |
| `gpt-5.6-sol` | -3.3% | -6.7% | -5.0% | -1.76s | $+0.0049 |

## Composed-fault reliability

Each percentage is the episode pass rate across two long change sets and three trials per change set.

| Model | Prompt | Transient chain | Evidence drift | Concurrency race | Ambiguous commit | Degraded auth |
|---|---|---:|---:|---:|---:|---:|
| `gpt-5.6-luna` | minimal | 100.0% | 0.0% | 66.7% | 100.0% | 100.0% |
| `gpt-5.6-luna` | principled | 100.0% | 0.0% | 50.0% | 83.3% | 100.0% |
| `gpt-5.6-terra` | minimal | 100.0% | 0.0% | 16.7% | 100.0% | 100.0% |
| `gpt-5.6-terra` | principled | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| `gpt-5.6-sol` | minimal | 100.0% | 0.0% | 50.0% | 100.0% | 100.0% |
| `gpt-5.6-sol` | principled | 100.0% | 0.0% | 16.7% | 100.0% | 100.0% |

## Observed failure modes

- `gpt-5.6-luna` / minimal: `change_set_state_mismatch`: 6, `outcome_mismatch`: 6, `request_state_mismatch`: 6, `trajectory_noncompliance`: 8; mixed-result cells 10.0%.
- `gpt-5.6-luna` / principled: `change_set_state_mismatch`: 7, `outcome_mismatch`: 6, `request_state_mismatch`: 6, `trajectory_noncompliance`: 11; mixed-result cells 20.0%.
- `gpt-5.6-terra` / minimal: `change_set_state_mismatch`: 5, `outcome_mismatch`: 4, `request_state_mismatch`: 4, `trajectory_noncompliance`: 13; mixed-result cells 10.0%.
- `gpt-5.6-terra` / principled: `change_set_state_mismatch`: 9, `outcome_mismatch`: 6, `request_state_mismatch`: 6, `trajectory_noncompliance`: 15; mixed-result cells 0.0%.
- `gpt-5.6-sol` / minimal: `change_set_state_mismatch`: 2, `outcome_mismatch`: 2, `request_state_mismatch`: 2, `trajectory_noncompliance`: 9; mixed-result cells 10.0%.
- `gpt-5.6-sol` / principled: `change_set_state_mismatch`: 7, `outcome_mismatch`: 6, `request_state_mismatch`: 6, `trajectory_noncompliance`: 12; mixed-result cells 10.0%.

## Run accounting

The run made **3,140 Responses API calls**, consumed **5,388,520 input tokens** and **250,894 output tokens**, and has an estimated cost of **$6.8000** at the recorded prices.

The JSON artifact retains every trial, structured tool call and result, composed-fault activation, terminal request map, hidden-grader decision, token count, and observed latency. It excludes credentials and private reasoning content.

## Interpretation limits

Three trials reveal gross instability but do not provide a high-powered stochastic estimate. Wilson intervals treat episodes as Bernoulli observations even though trials share tasks and prompts, so cell-level all-trials consistency is the stricter reliability view. The benchmark covers one atomic change-control domain, and concurrent wall-clock execution makes latency descriptive rather than a controlled speed ranking.
