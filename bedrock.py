"""
nexari.llm.bedrock
──────────────────
Claude via AWS Bedrock. Default backend.
Requires: boto3, AWS credentials in env or ~/.aws/credentials
"""

from __future__ import annotations

import json
from typing import Iterator

import boto3

from nexari.config import BEDROCK_MODEL_ID, BEDROCK_REGION
from nexari.llm.base import LLMClient


class BedrockClient(LLMClient):
    def __init__(self):
        self._client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        self._model_id = BEDROCK_MODEL_ID

    def _build_body(self, system: str, user: str, max_tokens: int) -> dict:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        body = self._build_body(system, user, max_tokens)
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    def stream(self, system: str, user: str, max_tokens: int = 1024) -> Iterator[str]:
        body = self._build_body(system, user, max_tokens)
        response = self._client.invoke_model_with_response_stream(
            modelId=self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta":
                yield chunk["delta"].get("text", "")
