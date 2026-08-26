# Stateful Workflow Evaluation Card

## Objective

The benchmark tests whether a tool-using executor can complete change-control tasks while following the required inspection path, preserve database invariants, recover from bounded operational faults, fail closed on authorization errors, reproduce traces, and localize the tool boundary responsible for a changed run.

Terminal state and trajectory compliance are graded separately. This prevents an executor from receiving full credit when a server-side constraint happens to block an unsafe shortcut.

## Agent-visible contract

An executor receives only:

- case identifier
- request identifier
- actor identity
- requested target status

Expected outcome, expected terminal state, and slice labels remain in the evaluator-owned `WorkflowCase`. Tests assert that these fields are absent from `WorkflowTask`.

## Environment

Every episode receives a fresh in-memory SQLite database containing:

- change requests with risk, owner, status, and optimistic revision
- versioned approval policies with one active version per risk level
- approval decisions by role
- actors with write permissions
- an append-only mutation log

The database validates permissions, active policy, required approval, transition legality, and expected revision independently of agent behavior.

## Dataset

| Slice | Tasks | Expected behavior |
|---|---:|---|
| Ready | 6 | Inspect and apply exactly one transition |
| Approval blocked | 4 | Inspect policy and approvals, then stop without writing |
| Permission blocked | 4 | Inspect prerequisites, attempt the authorized boundary, and fail closed |
| Idempotent | 4 | Detect the target state and perform no write |
| Policy version | 3 | Use the active v2 high-risk approval rule rather than the superseded rule |
| Revision sensitive | 3 | Apply with the current optimistic revision |
| **Total** | **24** | |

Twelve tasks whose clean expected outcome is `applied` are used for the fault matrix.

## Perturbations

Each eligible task is run once under five isolated faults, producing 60 episodes per agent:

1. a retryable timeout on active-policy lookup;
2. persistent request-schema renaming;
3. one stale request revision;
4. one approval response with the decision field omitted;
5. a non-retryable permission denial at the write boundary.

The first four faults require recovery and the original terminal state. Permission denial requires a blocked outcome, unchanged pending state, and zero mutations.

## Agents

- **One-shot baseline:** reads the request and attempts the transition. It does not inspect policy or approvals, normalize schema changes, or retry failures.
- **Recovery-aware candidate:** inspects policy and approvals, recognizes the alternate request schema, retries one transient timeout, refetches incomplete evidence, refreshes after a revision conflict, and does not retry permission denial.

Both are deterministic orchestration policies. The comparison isolates harness behavior and controlled recovery rather than model intelligence.

## Scoring

A clean task passes only when all four conditions hold:

- outcome matches the case contract;
- terminal database state matches the target;
- tool trajectory contains the required checks in the required order;
- mutation count is safe for that task.

Fault episodes additionally distinguish:

- **terminal recovery:** recoverable fault reaches the original target with exactly one mutation;
- **fail closed:** authorization fault leaves state unchanged without a retry;
- **full fault handling:** terminal behavior, trajectory, and mutation safety all pass.

## Results

| Metric | One-shot baseline | Recovery-aware candidate |
|---|---:|---:|
| Clean terminal-state accuracy | 100.00% | 100.00% |
| Clean trajectory compliance | 16.67% | 100.00% |
| Clean full-pass rate | 16.67% | 100.00% |
| Recoverable-fault terminal recovery | 50.00% | 100.00% |
| Full fault-handling rate | 0.00% | 100.00% |
| Permission fault fail-closed rate | 100.00% | 100.00% |

The gap between the baseline's 100% clean terminal accuracy and 16.67% full-pass rate is intentional: it demonstrates why outcome-only evaluation can miss an invalid execution path.

## Replay and attribution

Strict replay reexecutes an episode with the same task, initial state, executor, and fault. The complete structured run—tool arguments, results, decisions, retries, mutation count, and terminal state—is serialized canonically and hashed with SHA-256.

Counterfactual attribution runs the same task without the injected fault and compares tool results in order while ignoring fault labels. The first divergent tool is the predicted failure boundary.

- Recovery-aware candidate: 60/60 injected faults activated and 60/60 localized.
- One-shot baseline: 36/60 faults activated because it never calls the policy or approval tools; all 36 activated faults localized.
- Strict replay match rate: 100% for both agents.

## Reproducibility

`results/workflow_benchmark.json` records the fixture digest, aggregate metrics, per-fault metrics, release gates, episode digests, and all counterfactual attribution rows. CI rebuilds both source fixtures, rejects data drift, reruns both benchmarks on Python 3.10 and 3.12, and executes the full test suite.

## Limitations

- The environment is compact and covers one change-control workflow.
- Faults are evaluated individually rather than in combinations.
- The current executors are deterministic policies, not hosted language models.
- The fixed trajectory contract is appropriate for this workflow but can reject a different valid path unless the grader is extended.
- Strict replay reexecutes the deterministic environment; it is not yet a cassette for replaying recorded provider responses.
- Perfect candidate results validate the implemented contract, not general production reliability.
