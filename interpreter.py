"""
nexari.agent.interpreter
─────────────────────────
Step 1: Parse natural language intent into a structured task definition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from nexari.llm.base import LLMClient, get_client


class TaskType(str, Enum):
    TEXT_CLASSIFICATION = "text_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    TEXT_GENERATION = "text_generation"
    SUMMARIZATION = "summarization"
    UNKNOWN = "unknown"


@dataclass
class TaskDefinition:
    raw_intent: str
    task_type: TaskType
    domain: str
    input_description: str
    output_description: str
    suggested_metric: str
    notes: str = ""


SYSTEM_PROMPT = """You are a machine learning task analyst.
Given a natural language description of an ML problem, extract a structured task definition.
Respond ONLY with valid JSON matching this schema exactly:
{
  "task_type": one of ["text_classification", "token_classification", "text_generation", "summarization", "unknown"],
  "domain": short domain label e.g. "customer support", "medical", "finance",
  "input_description": what the model receives as input,
  "output_description": what the model should produce,
  "suggested_metric": primary evaluation metric e.g. "accuracy", "f1", "rouge",
  "notes": any important constraints or observations
}
No preamble. No markdown fences. JSON only."""


def interpret(intent: str, llm: LLMClient | None = None) -> TaskDefinition:
    """Parse a natural language intent string into a TaskDefinition."""
    client = llm or get_client()
    raw = client.complete(system=SYSTEM_PROMPT, user=intent, max_tokens=512)

    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Graceful fallback — return unknown task with raw output in notes
        return TaskDefinition(
            raw_intent=intent,
            task_type=TaskType.UNKNOWN,
            domain="unknown",
            input_description="",
            output_description="",
            suggested_metric="accuracy",
            notes=f"Parse failed. Raw LLM output: {raw}",
        )

    return TaskDefinition(
        raw_intent=intent,
        task_type=TaskType(data.get("task_type", "unknown")),
        domain=data.get("domain", ""),
        input_description=data.get("input_description", ""),
        output_description=data.get("output_description", ""),
        suggested_metric=data.get("suggested_metric", "accuracy"),
        notes=data.get("notes", ""),
    )
