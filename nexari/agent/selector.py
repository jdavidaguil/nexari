from __future__ import annotations
import json
from dataclasses import dataclass
from nexari.agent.interpreter import TaskDefinition
from nexari.llm.base import LLMClient, get_client

BACKBONE_DEFAULTS = {
    "text_classification": "distilbert-base-uncased",
    "token_classification": "distilbert-base-uncased",
    "text_generation": "gpt2",
    "summarization": "facebook/bart-base",
    "unknown": "distilbert-base-uncased",
}

SYSTEM_PROMPT = """You are a model selection expert for Hugging Face fine-tuning.
Given a task definition and chosen dataset, recommend the single best backbone model.
Optimize for: task fit > inference speed > model size (smaller is better for v0).
Respond ONLY with valid JSON:
{"model_id": "exact HuggingFace model id", "rationale": "2-3 sentence explanation", "estimated_train_time_minutes": 10, "tokenizer_id": "usually same as model_id"}
No preamble. No markdown. JSON only."""

@dataclass
class BackboneSelection:
    model_id: str
    tokenizer_id: str
    rationale: str
    estimated_train_time_minutes: int

def select_backbone(task: TaskDefinition, dataset_id: str, llm: LLMClient | None = None) -> BackboneSelection:
    client = llm or get_client()
    user_prompt = f"Task: {task.task_type.value}, Domain: {task.domain}, Dataset: {dataset_id}. Select best backbone."
    raw = client.complete(system=SYSTEM_PROMPT, user=user_prompt, max_tokens=512)
    try:
        data = json.loads(raw.strip())
        return BackboneSelection(
            model_id=data["model_id"],
            tokenizer_id=data.get("tokenizer_id", data["model_id"]),
            rationale=data.get("rationale", ""),
            estimated_train_time_minutes=int(data.get("estimated_train_time_minutes", 10)),
        )
    except (json.JSONDecodeError, KeyError):
        default = BACKBONE_DEFAULTS.get(task.task_type.value, "distilbert-base-uncased")
        return BackboneSelection(model_id=default, tokenizer_id=default, rationale="Fallback to default.", estimated_train_time_minutes=10)
