# Live model workflow benchmark

This recorded run compares **3 OpenAI models** under **2 prompt contracts** across **234 model-backed episodes**. Each configuration sees 24 clean tasks and 15 controlled fault episodes.

| Model | Prompt | Clean full pass | Clean terminal state | Clean trajectory | Fault handling | Recoverable faults | Fail closed | p95 episode | Est. cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpt-5.6-luna` | concise | 95.83% | 100.00% | 95.83% | 100.00% | 100.00% | 100.00% | 9.08s | $0.0293 |
| `gpt-5.6-luna` | explicit | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 9.97s | $0.0348 |
| `gpt-5.6-terra` | concise | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 9.06s | $0.2967 |
| `gpt-5.6-terra` | explicit | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 8.31s | $0.3446 |
| `gpt-5.6-sol` | concise | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 13.11s | $0.5732 |
| `gpt-5.6-sol` | explicit | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 12.79s | $0.6526 |

## Prompt ablation

Deltas are explicit minus concise; positive accuracy deltas favor the recovery contract, while positive latency and cost deltas indicate overhead.

| Model | Clean full pass | Fault handling | p95 episode | Est. cost |
|---|---:|---:|---:|---:|
| `gpt-5.6-luna` | +4.17% | +0.00% | +0.89s | $+0.0055 |
| `gpt-5.6-terra` | +0.00% | +0.00% | -0.75s | $+0.0479 |
| `gpt-5.6-sol` | +0.00% | +0.00% | -0.32s | $+0.0794 |

## Fault matrix

| Model | Prompt | Timeout | Schema drift | Stale read | Partial response | Permission denied |
|---|---|---:|---:|---:|---:|---:|
| `gpt-5.6-luna` | concise | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| `gpt-5.6-luna` | explicit | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| `gpt-5.6-terra` | concise | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| `gpt-5.6-terra` | explicit | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| `gpt-5.6-sol` | concise | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| `gpt-5.6-sol` | explicit | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

## Run accounting

The run made **1,145 Responses API calls**, consumed **711,803 input tokens** and **41,581 output tokens**, and has an estimated API cost of **$1.9312** at the recorded model prices.

The committed JSON retains every structured tool call, tool result, terminal state, hidden-grader decision, provider token count, and observed latency. It excludes API credentials and private reasoning content.

## Interpretation limits

This is one run per task/fault pair in a compact change-control domain, not a general model leaderboard. Differences smaller than one episode should be treated as directional. Latency is wall-clock client latency from one network location, and cost is estimated from token accounting rather than a billing export.
