"""Run, aggregate, and render model-backed workflow evaluations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import fmean
import threading
import time
from typing import Any, Callable, Iterable, Sequence

from .live_client import ResponsesClient
from .model_agent import ModelWorkflowAgent, PROMPT_VARIANTS, PromptVariant
from .workflow_dataset import (
    DEFAULT_WORKFLOW_CASES_PATH,
    DEFAULT_WORKFLOW_SEED_PATH,
    load_workflow_cases,
)
from .workflow_env import WorkflowEnvironment
from .workflow_eval import FAULT_SUITE, RECOVERABLE_FAULTS, score_clean_run, score_fault_run
from .workflow_models import FaultMode, WorkflowCase
from .workflow_models import WorkflowRun


DEFAULT_MODELS = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")
DEFAULT_PROMPTS: tuple[PromptVariant, ...] = ("concise", "explicit")
DEFAULT_FAULT_REQUEST_IDS = ("ready-01", "policy-v2-01", "revision-01")
MODEL_PRICING_PER_MILLION = {
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gpt-5.6-sol": {"input": 4.00, "cached_input": 0.40, "output": 20.00},
}
SECRET_PATTERN = re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}")


@dataclass(frozen=True)
class LiveEpisodeSpec:
    model: str
    prompt_variant: PromptVariant
    case_id: str
    fault: FaultMode

    @property
    def episode_id(self) -> str:
        return "::".join((self.model, self.prompt_variant, self.case_id, self.fault))


def _dataset_sha256() -> str:
    digest = hashlib.sha256()
    for path in (DEFAULT_WORKFLOW_CASES_PATH, DEFAULT_WORKFLOW_SEED_PATH):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_live_episode_specs(
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    prompts: Sequence[PromptVariant] = DEFAULT_PROMPTS,
    cases: Sequence[WorkflowCase] | None = None,
    fault_request_ids: Sequence[str] = DEFAULT_FAULT_REQUEST_IDS,
) -> list[LiveEpisodeSpec]:
    cases = list(cases or load_workflow_cases())
    known_requests = {case.request_id for case in cases}
    missing = sorted(set(fault_request_ids) - known_requests)
    if missing:
        raise ValueError(f"Unknown fault-suite request IDs: {', '.join(missing)}")
    if any(prompt not in PROMPT_VARIANTS for prompt in prompts):
        raise ValueError("Unsupported prompt variant")

    specs: list[LiveEpisodeSpec] = []
    for model in models:
        for prompt in prompts:
            specs.extend(
                LiveEpisodeSpec(model, prompt, case.case_id, "none") for case in cases
            )
            for case in cases:
                if case.request_id not in fault_request_ids:
                    continue
                specs.extend(
                    LiveEpisodeSpec(model, prompt, case.case_id, fault)
                    for fault in FAULT_SUITE
                )
    return specs


def run_live_episode(
    spec: LiveEpisodeSpec,
    *,
    client: ResponsesClient,
    cases_by_id: dict[str, WorkflowCase],
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    case = cases_by_id[spec.case_id]
    agent = ModelWorkflowAgent(
        client,
        model=spec.model,
        prompt_variant=spec.prompt_variant,
        reasoning_effort=reasoning_effort,
    )
    started = time.perf_counter()
    with WorkflowEnvironment(fault=spec.fault) as environment:
        run = agent.run(case.task(), environment)
    elapsed_ms = round((time.perf_counter() - started) * 1_000, 2)
    score = (
        score_clean_run(run, case)
        if spec.fault == "none"
        else score_fault_run(run, case)
    )
    score_payload = asdict(score)
    score_payload.pop("run")
    score_payload["failures"] = list(score.failures)
    return {
        "episode_id": spec.episode_id,
        "requested_model": spec.model,
        "prompt_variant": spec.prompt_variant,
        "reasoning_effort": reasoning_effort,
        "case_id": case.case_id,
        "request_id": case.request_id,
        "slice": case.slice_name,
        "fault": spec.fault,
        "score": score_payload,
        "run": run.to_dict(),
        "provider": agent.last_metadata,
        "episode_latency_ms": elapsed_ms,
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            records[str(record["episode_id"])] = record
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid checkpoint record on line {line_number}") from exc
    return records


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def execute_live_specs(
    specs: Sequence[LiveEpisodeSpec],
    *,
    client: ResponsesClient,
    checkpoint_path: Path,
    max_workers: int = 6,
    reasoning_effort: str = "low",
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    cases = load_workflow_cases()
    cases_by_id = {case.case_id: case for case in cases}
    checkpoint = load_checkpoint(checkpoint_path)
    requested_ids = {spec.episode_id for spec in specs}
    records = {
        episode_id: record
        for episode_id, record in checkpoint.items()
        if episode_id in requested_ids
    }
    missing = [spec for spec in specs if spec.episode_id not in records]
    total = len(specs)
    completed = len(records)
    write_lock = threading.Lock()

    if progress:
        for record in records.values():
            progress(completed, total, {**record, "resumed": True})
            break

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_live_episode,
                spec,
                client=client,
                cases_by_id=cases_by_id,
                reasoning_effort=reasoning_effort,
            ): spec
            for spec in missing
        }
        try:
            for future in as_completed(futures):
                record = future.result()
                with write_lock:
                    append_checkpoint(checkpoint_path, record)
                records[record["episode_id"]] = record
                completed += 1
                if progress:
                    progress(completed, total, record)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [records[spec.episode_id] for spec in specs]


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * percentile)
    return round(float(ordered[index]), 2)


def _estimated_cost(model: str, records: Sequence[dict[str, Any]]) -> float | None:
    prices = MODEL_PRICING_PER_MILLION.get(model)
    if not prices:
        return None
    usage = [record["provider"]["usage"] for record in records]
    input_tokens = sum(item["input_tokens"] for item in usage)
    cached_tokens = sum(item["cached_input_tokens"] for item in usage)
    output_tokens = sum(item["output_tokens"] for item in usage)
    cost = (
        (input_tokens - cached_tokens) * prices["input"]
        + cached_tokens * prices["cached_input"]
        + output_tokens * prices["output"]
    ) / 1_000_000
    return round(cost, 6)


def summarize_live_config(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    clean = [record for record in records if record["fault"] == "none"]
    faults = [record for record in records if record["fault"] != "none"]
    recoverable = [record for record in faults if record["fault"] in RECOVERABLE_FAULTS]
    permission = [record for record in faults if record["fault"] == "permission_denied"]
    usage = [record["provider"]["usage"] for record in records]
    by_fault: dict[str, Any] = {}
    for fault in FAULT_SUITE:
        subset = [record for record in faults if record["fault"] == fault]
        by_fault[fault] = {
            "episodes": len(subset),
            "pass_rate": _rate(record["score"]["passed"] for record in subset),
            "mutation_safety_rate": _rate(
                record["score"]["mutation_safe"] for record in subset
            ),
        }
    model = str(records[0]["requested_model"])
    return {
        "requested_model": model,
        "resolved_models": sorted(
            {
                resolved
                for record in records
                for resolved in record["provider"]["resolved_models"]
            }
        ),
        "prompt_variant": records[0]["prompt_variant"],
        "episodes": len(records),
        "clean": {
            "episodes": len(clean),
            "pass_rate": _rate(record["score"]["passed"] for record in clean),
            "outcome_accuracy": _rate(
                record["score"]["outcome_correct"] for record in clean
            ),
            "terminal_state_accuracy": _rate(
                record["score"]["terminal_state_correct"] for record in clean
            ),
            "trajectory_compliance_rate": _rate(
                record["score"]["trajectory_compliant"] for record in clean
            ),
            "mutation_safety_rate": _rate(
                record["score"]["mutation_safe"] for record in clean
            ),
        },
        "faults": {
            "episodes": len(faults),
            "fault_handling_rate": _rate(
                record["score"]["passed"] for record in faults
            ),
            "terminal_recovery_rate": _rate(
                record["score"]["outcome_correct"]
                and record["score"]["terminal_state_correct"]
                and record["score"]["mutation_safe"]
                for record in recoverable
            ),
            "fail_closed_rate": _rate(
                record["score"]["outcome_correct"]
                and record["score"]["terminal_state_correct"]
                and record["score"]["mutation_safe"]
                for record in permission
            ),
            "by_fault": by_fault,
        },
        "usage": {
            "response_calls": sum(record["provider"]["response_calls"] for record in records),
            "input_tokens": sum(item["input_tokens"] for item in usage),
            "cached_input_tokens": sum(item["cached_input_tokens"] for item in usage),
            "output_tokens": sum(item["output_tokens"] for item in usage),
            "reasoning_tokens": sum(item["reasoning_tokens"] for item in usage),
        },
        "latency_ms": {
            "episode_mean": round(fmean(record["episode_latency_ms"] for record in records), 2),
            "episode_p50": _percentile(
                (record["episode_latency_ms"] for record in records), 0.50
            ),
            "episode_p95": _percentile(
                (record["episode_latency_ms"] for record in records), 0.95
            ),
            "provider_total": round(
                sum(record["provider"]["provider_latency_ms"] for record in records), 2
            ),
        },
        "estimated_cost_usd": _estimated_cost(model, records),
        "failures": [
            {
                "case_id": record["case_id"],
                "fault": record["fault"],
                "failures": record["score"]["failures"],
            }
            for record in records
            if not record["score"]["passed"]
        ],
    }


def build_live_artifact(
    records: Sequence[dict[str, Any]],
    *,
    models: Sequence[str],
    prompts: Sequence[PromptVariant],
    fault_request_ids: Sequence[str] = DEFAULT_FAULT_REQUEST_IDS,
) -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    for model in models:
        for prompt in prompts:
            subset = [
                record
                for record in records
                if record["requested_model"] == model
                and record["prompt_variant"] == prompt
            ]
            if not subset:
                raise ValueError(f"No live records for {model}/{prompt}")
            configs.append(summarize_live_config(subset))

    prompt_deltas: list[dict[str, Any]] = []
    if "concise" in prompts and "explicit" in prompts:
        by_key = {
            (config["requested_model"], config["prompt_variant"]): config
            for config in configs
        }
        for model in models:
            concise = by_key[(model, "concise")]
            explicit = by_key[(model, "explicit")]
            prompt_deltas.append(
                {
                    "model": model,
                    "clean_pass_rate": round(
                        explicit["clean"]["pass_rate"] - concise["clean"]["pass_rate"],
                        4,
                    ),
                    "fault_handling_rate": round(
                        explicit["faults"]["fault_handling_rate"]
                        - concise["faults"]["fault_handling_rate"],
                        4,
                    ),
                    "episode_p95_ms": round(
                        explicit["latency_ms"]["episode_p95"]
                        - concise["latency_ms"]["episode_p95"],
                        2,
                    ),
                    "estimated_cost_usd": round(
                        (explicit["estimated_cost_usd"] or 0)
                        - (concise["estimated_cost_usd"] or 0),
                        6,
                    ),
                }
            )

    clean_cases = len({record["case_id"] for record in records if record["fault"] == "none"})
    fault_cases = len({record["case_id"] for record in records if record["fault"] != "none"})
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "models": list(models),
            "prompts": {
                prompt: {
                    "sha256": hashlib.sha256(
                        PROMPT_VARIANTS[prompt].encode("utf-8")
                    ).hexdigest()
                }
                for prompt in prompts
            },
            "reasoning_effort": records[0]["reasoning_effort"],
            "clean_cases_per_config": clean_cases,
            "fault_cases_per_config": fault_cases,
            "fault_modes": list(FAULT_SUITE),
            "fault_request_ids": list(fault_request_ids),
            "episodes_per_config": clean_cases + fault_cases * len(FAULT_SUITE),
            "total_episodes": len(records),
            "repetitions_per_case_fault": 1,
            "dataset_sha256": _dataset_sha256(),
            "pricing_usd_per_million_tokens": {
                model: MODEL_PRICING_PER_MILLION[model]
                for model in models
                if model in MODEL_PRICING_PER_MILLION
            },
        },
        "configs": configs,
        "prompt_deltas_explicit_minus_concise": prompt_deltas,
        "totals": {
            "episodes": len(records),
            "response_calls": sum(config["usage"]["response_calls"] for config in configs),
            "input_tokens": sum(config["usage"]["input_tokens"] for config in configs),
            "output_tokens": sum(config["usage"]["output_tokens"] for config in configs),
            "estimated_cost_usd": round(
                sum(config["estimated_cost_usd"] or 0 for config in configs), 6
            ),
        },
        "episodes": sorted(records, key=lambda record: record["episode_id"]),
    }


def verify_live_artifact(artifact: dict[str, Any]) -> dict[str, int]:
    """Recompute recorded scores and summaries without making provider calls."""

    protocol = artifact.get("protocol") or {}
    models = list(protocol.get("models") or [])
    prompts = list((protocol.get("prompts") or {}).keys())
    records = list(artifact.get("episodes") or [])
    if artifact.get("schema_version") != 1 or not models or not prompts or not records:
        raise ValueError("Incomplete live artifact")
    if SECRET_PATTERN.search(json.dumps(artifact)):
        raise ValueError("Potential API credential found in live artifact")

    specs = build_live_episode_specs(
        models=models,
        prompts=prompts,
        fault_request_ids=protocol.get("fault_request_ids") or DEFAULT_FAULT_REQUEST_IDS,
    )
    expected_ids = [spec.episode_id for spec in specs]
    recorded_ids = [str(record.get("episode_id")) for record in records]
    if len(recorded_ids) != len(set(recorded_ids)):
        raise ValueError("Duplicate live episode IDs")
    if set(recorded_ids) != set(expected_ids):
        raise ValueError("Live episode matrix does not match the recorded protocol")

    cases_by_id = {case.case_id: case for case in load_workflow_cases()}
    for record in records:
        case = cases_by_id.get(str(record.get("case_id")))
        if case is None:
            raise ValueError(f"Unknown case in live artifact: {record.get('case_id')}")
        run_payload = dict(record.get("run") or {})
        run_payload["trace"] = tuple(run_payload.get("trace") or [])
        try:
            run = WorkflowRun(**run_payload)
        except TypeError as exc:
            raise ValueError(f"Invalid recorded run for {record['episode_id']}") from exc
        score = (
            score_clean_run(run, case)
            if record.get("fault") == "none"
            else score_fault_run(run, case)
        )
        score_payload = asdict(score)
        score_payload.pop("run")
        score_payload["failures"] = list(score.failures)
        if score_payload != record.get("score"):
            raise ValueError(f"Recorded score drift for {record['episode_id']}")
        if record.get("fault") != run.fault:
            raise ValueError(f"Fault identity drift for {record['episode_id']}")
        if run.tool_calls != sum(
            event.get("stage") == "tool_call" for event in run.trace
        ):
            raise ValueError(f"Tool-call count drift for {record['episode_id']}")
        provider = record.get("provider") or {}
        if provider.get("requested_model") != record.get("requested_model"):
            raise ValueError(f"Model identity drift for {record['episode_id']}")
        if provider.get("prompt_variant") != record.get("prompt_variant"):
            raise ValueError(f"Prompt identity drift for {record['episode_id']}")
        if provider.get("response_calls") != sum(
            event.get("stage") == "model_call" for event in run.trace
        ):
            raise ValueError(f"Provider-call count drift for {record['episode_id']}")

    rebuilt = build_live_artifact(
        records,
        models=models,
        prompts=prompts,
        fault_request_ids=protocol.get("fault_request_ids") or DEFAULT_FAULT_REQUEST_IDS,
    )
    for key in ("protocol", "configs", "prompt_deltas_explicit_minus_concise", "totals"):
        if rebuilt[key] != artifact.get(key):
            raise ValueError(f"Live artifact {key} summary drift")
    return {
        "episodes": len(records),
        "passed": sum(bool(record["score"]["passed"]) for record in records),
        "failed": sum(not bool(record["score"]["passed"]) for record in records),
    }


def render_live_markdown(artifact: dict[str, Any]) -> str:
    protocol = artifact["protocol"]
    lines = [
        "# Live model workflow benchmark",
        "",
        (
            f"This recorded run compares **{len(protocol['models'])} OpenAI models** under "
            f"**{len(protocol['prompts'])} prompt contracts** across "
            f"**{protocol['total_episodes']} model-backed episodes**. Each configuration sees "
            f"{protocol['clean_cases_per_config']} clean tasks and "
            f"{protocol['fault_cases_per_config'] * len(protocol['fault_modes'])} controlled fault episodes."
        ),
        "",
        "| Model | Prompt | Clean full pass | Clean terminal state | Clean trajectory | Fault handling | Recoverable faults | Fail closed | p95 episode | Est. cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in artifact["configs"]:
        lines.append(
            f"| `{config['requested_model']}` | {config['prompt_variant']} | "
            f"{config['clean']['pass_rate']:.2%} | "
            f"{config['clean']['terminal_state_accuracy']:.2%} | "
            f"{config['clean']['trajectory_compliance_rate']:.2%} | "
            f"{config['faults']['fault_handling_rate']:.2%} | "
            f"{config['faults']['terminal_recovery_rate']:.2%} | "
            f"{config['faults']['fail_closed_rate']:.2%} | "
            f"{config['latency_ms']['episode_p95'] / 1_000:.2f}s | "
            f"${config['estimated_cost_usd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Prompt ablation",
            "",
            "Deltas are explicit minus concise; positive accuracy deltas favor the recovery contract, while positive latency and cost deltas indicate overhead.",
            "",
            "| Model | Clean full pass | Fault handling | p95 episode | Est. cost |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for delta in artifact["prompt_deltas_explicit_minus_concise"]:
        lines.append(
            f"| `{delta['model']}` | {delta['clean_pass_rate']:+.2%} | "
            f"{delta['fault_handling_rate']:+.2%} | "
            f"{delta['episode_p95_ms'] / 1_000:+.2f}s | "
            f"${delta['estimated_cost_usd']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fault matrix",
            "",
            "| Model | Prompt | Timeout | Schema drift | Stale read | Partial response | Permission denied |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for config in artifact["configs"]:
        by_fault = config["faults"]["by_fault"]
        lines.append(
            f"| `{config['requested_model']}` | {config['prompt_variant']} | "
            f"{by_fault['timeout_once']['pass_rate']:.2%} | "
            f"{by_fault['schema_drift']['pass_rate']:.2%} | "
            f"{by_fault['stale_read']['pass_rate']:.2%} | "
            f"{by_fault['partial_response']['pass_rate']:.2%} | "
            f"{by_fault['permission_denied']['pass_rate']:.2%} |"
        )
    totals = artifact["totals"]
    lines.extend(
        [
            "",
            "## Run accounting",
            "",
            f"The run made **{totals['response_calls']:,} Responses API calls**, consumed "
            f"**{totals['input_tokens']:,} input tokens** and **{totals['output_tokens']:,} output tokens**, "
            f"and has an estimated API cost of **${totals['estimated_cost_usd']:.4f}** at the recorded model prices.",
            "",
            "The committed JSON retains every structured tool call, tool result, terminal state, hidden-grader decision, provider token count, and observed latency. It excludes API credentials and private reasoning content.",
            "",
            "## Interpretation limits",
            "",
            "This is one run per task/fault pair in a compact change-control domain, not a general model leaderboard. Differences smaller than one episode should be treated as directional. Latency is wall-clock client latency from one network location, and cost is estimated from token accounting rather than a billing export.",
            "",
        ]
    )
    return "\n".join(lines)


def write_live_artifacts(
    artifact: dict[str, Any], json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_live_markdown(artifact), encoding="utf-8")
