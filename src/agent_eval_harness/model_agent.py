"""Responses API adapter for the stateful workflow environment."""

from __future__ import annotations

import json
from typing import Any, Literal

from .live_client import ResponseCall, ResponsesClient
from .workflow_agents import finalize_workflow_run
from .workflow_env import WorkflowEnvironment
from .workflow_models import ToolResult, WorkflowRun, WorkflowTask


PromptVariant = Literal["concise", "explicit"]


COMMON_INSTRUCTIONS = """You operate a change-control workflow through function tools.
Never invent tool results or treat a failed tool call as success. End every task by calling finish exactly once.
The finish outcome must be one of applied, blocked, no_op, or failed."""


PROMPT_VARIANTS: dict[PromptVariant, str] = {
    "concise": COMMON_INSTRUCTIONS
    + "\nComplete the requested transition safely and comply with the workflow controls exposed by the tools.",
    "explicit": COMMON_INSTRUCTIONS
    + """
First read the request. If it is already at the target state, finish no_op without writing.
Before any write, retrieve the active policy for the request risk level and list its approvals. Only continue when an approved decision exists for the active policy's required role.
Use the request revision for apply_transition. Retry a timeout once. If a revision conflict occurs, refresh the request and retry once. Refetch an approval response once when required fields are missing.
If a request response renames status to state and revision to version, accept those fields only when the request ID, risk level, and owner team are still present, and use version as the expected revision.
Do not retry permission_denied. If policy, approval, or authorization is insufficient, finish blocked without mutation.""",
}


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_request",
        "description": "Read the current request state, risk level, owner, and optimistic revision.",
        "parameters": {
            "type": "object",
            "properties": {"request_id": {"type": "string"}},
            "required": ["request_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_active_policy",
        "description": "Read the active approval policy for one risk level.",
        "parameters": {
            "type": "object",
            "properties": {"risk_level": {"type": "string"}},
            "required": ["risk_level"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "list_approvals",
        "description": "List approval roles and decisions recorded for a request.",
        "parameters": {
            "type": "object",
            "properties": {"request_id": {"type": "string"}},
            "required": ["request_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "apply_transition",
        "description": "Attempt one guarded workflow state transition using an expected revision.",
        "parameters": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string"},
                "actor": {"type": "string"},
                "target_status": {"type": "string"},
                "expected_revision": {"type": "integer"},
            },
            "required": ["request_id", "actor", "target_status", "expected_revision"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "finish",
        "description": "Finish the episode after all necessary tool work is complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": ["applied", "blocked", "no_op", "failed"],
                },
                "reason": {"type": "string"},
            },
            "required": ["outcome", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class ModelWorkflowAgent:
    max_turns = 10

    def __init__(
        self,
        client: ResponsesClient,
        *,
        model: str,
        prompt_variant: PromptVariant,
        reasoning_effort: str = "low",
    ) -> None:
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(f"Unknown prompt variant {prompt_variant!r}")
        self.client = client
        self.model = model
        self.prompt_variant = prompt_variant
        self.reasoning_effort = reasoning_effort
        self.name = f"openai__{model}__{prompt_variant}"
        self.last_metadata: dict[str, Any] = {}

    @staticmethod
    def _usage(call: ResponseCall) -> dict[str, int]:
        usage = call.payload.get("usage") or {}
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        return {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "cached_input_tokens": int(input_details.get("cached_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    @staticmethod
    def _record_tool_failure(
        environment: WorkflowEnvironment,
        name: str,
        arguments: dict[str, Any],
        result: ToolResult,
    ) -> ToolResult:
        call_id = (
            len(
                [
                    event
                    for event in environment.trace
                    if event.get("stage") == "tool_call"
                ]
            )
            + 1
        )
        environment.trace.extend(
            [
                {
                    "stage": "tool_call",
                    "call_id": call_id,
                    "tool": name,
                    "arguments": arguments,
                },
                {
                    "stage": "tool_result",
                    "call_id": call_id,
                    "tool": name,
                    "fault": "none",
                    "result": result.to_dict(),
                },
            ]
        )
        return result

    @classmethod
    def _tool_result(
        cls,
        environment: WorkflowEnvironment,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        try:
            return environment.call(name, **arguments)
        except (TypeError, ValueError):
            return cls._record_tool_failure(
                environment,
                name,
                arguments,
                ToolResult(False, {}, error="invalid_arguments", retryable=False),
            )

    def run(self, task: WorkflowTask, environment: WorkflowEnvironment) -> WorkflowRun:
        user_input = (
            "Execute this workflow task:\n"
            f"request_id={task.request_id}\n"
            f"actor={task.actor}\n"
            f"target_status={task.target_status}"
        )
        input_items: list[dict[str, Any]] = [
            {"role": "user", "content": user_input}
        ]
        aggregate_usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        latencies: list[float] = []
        resolved_models: list[str] = []
        outcome = "failed"
        finished = False

        for turn in range(1, self.max_turns + 1):
            call = self.client.create(
                {
                    "model": self.model,
                    "instructions": PROMPT_VARIANTS[self.prompt_variant],
                    "input": input_items,
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "reasoning": {"effort": self.reasoning_effort},
                    "max_output_tokens": 768,
                    "store": False,
                }
            )
            usage = self._usage(call)
            for key, value in usage.items():
                aggregate_usage[key] += value
            latencies.append(call.latency_ms)
            resolved_models.append(str(call.payload.get("model") or self.model))
            output = list(call.payload.get("output") or [])
            function_calls = [item for item in output if item.get("type") == "function_call"]
            environment.trace.append(
                {
                    "stage": "model_call",
                    "turn": turn,
                    "model": resolved_models[-1],
                    "latency_ms": round(call.latency_ms, 2),
                    "usage": usage,
                    "function_calls": [item.get("name") for item in function_calls],
                }
            )
            input_items.extend(output)
            if not function_calls:
                environment.trace.append(
                    {"stage": "agent_decision", "decision": "missing_finish_call"}
                )
                break

            tool_outputs: list[dict[str, Any]] = []
            for item in function_calls:
                name = str(item.get("name") or "")
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                if name == "finish":
                    requested = arguments.get("outcome")
                    if requested in {"applied", "blocked", "no_op", "failed"}:
                        outcome = str(requested)
                    environment.trace.append(
                        {
                            "stage": "agent_decision",
                            "decision": outcome,
                            "reason": str(arguments.get("reason") or ""),
                        }
                    )
                    finished = True
                    break
                if name not in {
                    "get_request",
                    "get_active_policy",
                    "list_approvals",
                    "apply_transition",
                }:
                    result = self._record_tool_failure(
                        environment,
                        name,
                        arguments,
                        ToolResult(False, {}, error="unknown_tool", retryable=False),
                    )
                else:
                    result = self._tool_result(environment, name, arguments)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps(result.to_dict(), sort_keys=True),
                    }
                )
            if finished:
                break
            input_items.extend(tool_outputs)

        if not finished:
            environment.trace.append(
                {"stage": "agent_decision", "decision": "failed", "reason": "turn_limit"}
            )
        self.last_metadata = {
            "requested_model": self.model,
            "resolved_models": sorted(set(resolved_models)),
            "prompt_variant": self.prompt_variant,
            "reasoning_effort": self.reasoning_effort,
            "response_calls": len(latencies),
            "provider_latency_ms": round(sum(latencies), 2),
            "usage": aggregate_usage,
        }
        return finalize_workflow_run(
            agent=self.name,
            task=task,
            environment=environment,
            outcome=outcome,
            retries=sum(
                event.get("decision") in {"retry", "refresh_after_revision_conflict", "refetch_incomplete_approvals"}
                for event in environment.trace
                if event.get("stage") == "agent_decision"
            ),
        )
