# Stateful workflow reliability results

The benchmark runs **24 clean stateful tasks** and **60 fault episodes per agent** across 5 controlled tool-boundary perturbations.

| Metric | One-shot baseline | Recovery-aware candidate | Delta |
|---|---:|---:|---:|
| Clean full-pass rate | 16.67% | 100.00% | +83.33% |
| Clean terminal-state accuracy | 100.00% | 100.00% | +0.00% |
| Clean trajectory compliance | 16.67% | 100.00% | +83.33% |
| Recoverable-fault terminal recovery | 50.00% | 100.00% | +50.00% |
| Full fault-handling rate | 0.00% | 100.00% | +100.00% |
| Permission fault fail-closed rate | 100.00% | 100.00% | +0.00% |

The one-shot baseline reaches the correct clean terminal state but skips required policy and approval checks. Full-pass scoring prevents that shortcut from being counted as compliant behavior.

## Controlled fault matrix

| Fault | Baseline full pass | Candidate full pass | Candidate terminal handling |
|---|---:|---:|---:|
| `timeout_once` | 0.00% | 100.00% | 100.00% |
| `schema_drift` | 0.00% | 100.00% | 100.00% |
| `stale_read` | 0.00% | 100.00% | 100.00% |
| `partial_response` | 0.00% | 100.00% | 100.00% |
| `permission_denied` | 0.00% | 100.00% | 100.00% |

## Replay and counterfactual localization

Strict replay reproduced **60/60** candidate fault episodes byte-for-byte at the structured trace level.
The first divergent tool result localized **60/60** activated candidate faults to the injected tool boundary without consulting the fault label.

| Agent | Probes | Fault activated | Strict replay | Tool localization |
|---|---:|---:|---:|---:|
| One-shot baseline | 60 | 36 | 100.00% | 100.00% |
| Recovery-aware candidate | 60 | 60 | 100.00% | 100.00% |

## Interpretation

These results validate deterministic orchestration, fault recovery, state grading, replay, and localization against the committed SQLite fixture. They do not measure a hosted language model. Model-backed adapters should retain the same environment and graders while adding prompt/model identifiers, sampled repetitions, observed tokens, latency, and cost.
