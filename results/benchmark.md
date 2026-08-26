# Public benchmark results

The committed fixture contains **80 cases** and the default run executes **800 candidate evaluations** across 10 repeats.

| Metric | Legacy router | Evidence-aware candidate | Delta |
|---|---:|---:|---:|
| Pass rate | 26.25% | 97.50% | +71.25% |
| Route accuracy | 60.00% | 97.50% | +37.50% |
| Answer accuracy | 26.25% | 97.50% | +71.25% |
| Evidence support | 32.81% | 96.88% | +64.07% |
| Abstention accuracy | 0.00% | 100.00% | +100.00% |
| Unsupported-answer rate | 20.00% | 0.00% | -20.00% |

## Reliability and operations

- Candidate pass-rate 95% Wilson interval: `[0.9617, 0.9838]`
- Repeated-run pass-rate standard deviation: `0.000000`
- Observed local p95 harness latency: `0.1814 ms`
- Reference estimated cost per case: `$0.00000828`
- Candidate release gates: `PASS`

## Mutation testing

The gate suite detected **4/4** seeded regressions covering route swaps, stale answers, missing evidence, and disabled abstention.

| Seeded regression | Detected | Failed gates |
|---|---:|---|
| `route_swap` | yes | pass_rate, route_accuracy |
| `stale_answer` | yes | pass_rate, answer_accuracy |
| `drop_evidence` | yes | pass_rate, evidence_support_rate, unsupported_answer_rate |
| `disable_abstain` | yes | pass_rate, route_accuracy, answer_accuracy, abstention_accuracy, unsupported_answer_rate |

## Interpretation

These results validate the harness and the committed evidence-aware routing policy on the public fixture. They are not a claim about a hosted foundation model or an undisclosed production dataset. Runtime latency varies by machine and is excluded from golden-artifact equality checks.
