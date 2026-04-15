"""
nexari.session
──────────────
Saves and restores pipeline state between runs.
Eliminates redundant Bedrock calls when resuming after failures.

Session file: .nexari_session.json in the current working directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

SESSION_FILE = Path(".nexari_session.json")


@dataclass
class NexariSession:
    intent: str
    completed_steps: list[int] = field(default_factory=list)

    # Step 1 output
    task: dict | None = None

    # Step 2 output
    dataset_id: str | None = None
    candidates: list[dict] | None = None

    # Step 3 output
    backbone: dict | None = None

    # Step 4 output
    model_path: str | None = None

    # Step 5 output
    endpoint_url: str | None = None

    def save(self):
        SESSION_FILE.write_text(json.dumps(asdict(self), indent=2))

    def mark_complete(self, step: int):
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        self.save()

    def is_complete(self, step: int) -> bool:
        return step in self.completed_steps

    @classmethod
    def load(cls) -> "NexariSession | None":
        if not SESSION_FILE.exists():
            return None
        try:
            data = json.loads(SESSION_FILE.read_text())
            return cls(**data)
        except Exception:
            return None

    @classmethod
    def load_or_create(cls, intent: str) -> "NexariSession":
        existing = cls.load()
        if existing and existing.intent == intent:
            return existing
        return cls(intent=intent)

    def clear(self):
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()


def restore_task(session: NexariSession):
    """Reconstruct TaskDefinition from session data."""
    from nexari.agent.interpreter import TaskDefinition, TaskType
    d = session.task
    return TaskDefinition(
        raw_intent=d["raw_intent"],
        task_type=TaskType(d["task_type"]),
        domain=d["domain"],
        input_description=d["input_description"],
        output_description=d["output_description"],
        suggested_metric=d["suggested_metric"],
        notes=d.get("notes", ""),
    )


def restore_backbone(session: NexariSession):
    """Reconstruct BackboneSelection from session data."""
    from nexari.agent.selector import BackboneSelection
    d = session.backbone
    return BackboneSelection(
        model_id=d["model_id"],
        tokenizer_id=d["tokenizer_id"],
        rationale=d["rationale"],
        estimated_train_time_minutes=d["estimated_train_time_minutes"],
    )