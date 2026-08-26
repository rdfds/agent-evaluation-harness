# Evaluation Card

## Objective

The benchmark tests whether a tool-using agent selects the contracted data path, returns the expected normalized answer, cites the required evidence, abstains when the fixture cannot support a response, and remains stable across repeated runs.

## Evaluated systems

- **Legacy keyword router:** selects structured queries from a small quantitative cue list, otherwise retrieves using a single canonical alias, and has no abstention path.
- **Evidence-aware router v2:** compares document and structured-intent matches, gives explicit document cues precedence, and abstains on out-of-scope or unsupported requests.

Both systems use the same committed knowledge fixture. The comparison isolates routing and evidence behavior rather than data access.

## Dataset composition

| Slice | Cases | Expected route |
|---|---:|---|
| Retrieval | 32 | `retrieval` |
| Structured query | 24 | `structured_query` |
| Abstention | 16 | `abstain` |
| Challenge | 8 | mixed |
| **Total** | **80** | |

Cases contain an ID, query, expected route, expected answer, required evidence IDs, slice, and difficulty. The default run repeats all cases 10 times, producing 800 candidate evaluations and 800 baseline evaluations.

## Metrics

- **Route accuracy:** selected route equals the case contract.
- **Answer accuracy:** normalized answer exactly matches the expected answer.
- **Evidence support:** every required evidence ID is present in the response.
- **Abstention accuracy:** missing-evidence cases return the abstention route and answer.
- **Unsupported-answer rate:** the agent answers an abstention case or produces an answer without required evidence.
- **Pass rate:** route, answer, evidence, and support checks all pass.
- **Repeated-run reliability:** minimum, maximum, and standard deviation of per-run pass rate.
- **Operations:** p50/p95 harness latency and reference token-cost accounting.

Wilson 95% intervals are reported for pass rate, route accuracy, and answer accuracy.

## Results

| Metric | Legacy | Candidate |
|---|---:|---:|
| Pass rate | 26.25% | 97.50% |
| Route accuracy | 60.00% | 97.50% |
| Answer accuracy | 26.25% | 97.50% |
| Evidence support | 32.81% | 96.88% |
| Abstention accuracy | 0.00% | 100.00% |
| Unsupported-answer rate | 20.00% | 0.00% |

Candidate pass-rate 95% Wilson interval: **96.17%-98.38%**. Repeated-run pass-rate standard deviation: **0.000000**.

## Mutation testing

Four deterministic faults are injected one at a time:

1. swap retrieval responses to the structured-query route;
2. replace non-abstention answers with a stale result;
3. remove all returned evidence identifiers;
4. convert abstentions into unsupported guesses.

The release contract detects all four faults. This tests the tests: a green benchmark must demonstrate that its gates are capable of failing for the behaviors they claim to protect.

## Reproducibility

`results/benchmark.json` includes a SHA-256 digest of the cases and knowledge fixture. CI regenerates the fixture, checks for repository drift, re-runs the benchmark, compares deterministic fields with the committed artifact, and executes the test suite on Python 3.10 and 3.12.

Observed latency is machine-dependent and excluded from artifact equality. Quality, slice, reliability, cost-accounting, failure, gate, and mutation fields remain deterministic.

## Limitations

- The public fixture is compact and domain-specific; it does not estimate performance on arbitrary enterprise requests.
- Exact-match answers are appropriate for the normalized fixture but should be supplemented with semantic or rubric-based scoring for open-ended generation.
- The included agents are deterministic routing policies. A model-backed adapter should additionally capture provider usage, prompt/model identifiers, decoding settings, and repeated-run variance.
- Reference cost uses configurable token rates and is not a provider invoice.
- A passing aggregate result does not replace slice review or inspection of retained hard failures.

