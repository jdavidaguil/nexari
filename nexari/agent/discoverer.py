from __future__ import annotations
import json
from dataclasses import dataclass
from huggingface_hub import HfApi
from nexari.agent.interpreter import TaskDefinition
from nexari.llm.base import LLMClient, get_client

@dataclass
class DatasetCandidate:
    dataset_id: str
    downloads: int
    likes: int
    tags: list
    description: str
    recommendation_rationale: str = ""
    rank: int = 0

SYSTEM_PROMPT = """You are a dataset selection expert for machine learning.
Given a task definition and candidate datasets from Hugging Face Hub, rank the top 3.
Respond ONLY with valid JSON array, no markdown fences:
[{"dataset_id": "exact id", "rank": 1, "rationale": "2-3 sentences explaining fit"}]"""

def discover(task: TaskDefinition, llm: LLMClient | None = None, limit: int = 10) -> list:
    client = llm or get_client()
    api = HfApi()
    results = list(api.list_datasets(search=task.domain, limit=limit))
    raw_candidates = [{"dataset_id": ds.id, "downloads": getattr(ds, "downloads", 0) or 0,
                       "likes": getattr(ds, "likes", 0) or 0, "tags": list(ds.tags or []),
                       "description": (ds.description or "")[:300]} for ds in results]
    if not raw_candidates:
        return []
    user_prompt = f"""Task: {task.task_type.value}, Domain: {task.domain}
Input: {task.input_description}
Output: {task.output_description}

Candidates:
{json.dumps(raw_candidates, indent=2)}

Rank the top 3 most suitable datasets for this task."""

    raw = client.complete(system=SYSTEM_PROMPT, user=user_prompt, max_tokens=1024)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        ranked = json.loads(raw)
    except json.JSONDecodeError:
        ranked = [{"dataset_id": c["dataset_id"], "rank": i+1, "rationale": "Selected by position."} for i, c in enumerate(raw_candidates[:3])]
    candidate_map = {c["dataset_id"]: c for c in raw_candidates}
    output = []
    for item in ranked[:3]:
        meta = candidate_map.get(item["dataset_id"], {})
        output.append(DatasetCandidate(dataset_id=item["dataset_id"], downloads=meta.get("downloads", 0),
            likes=meta.get("likes", 0), tags=meta.get("tags", []), description=meta.get("description", ""),
            recommendation_rationale=item.get("rationale", ""), rank=item.get("rank", 0)))
    return sorted(output, key=lambda x: x.rank)
