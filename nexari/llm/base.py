from __future__ import annotations
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...
    @abstractmethod
    def stream(self, system: str, user: str, max_tokens: int = 1024): ...

def get_client() -> LLMClient:
    from nexari.config import LLMBackend, LLM_BACKEND
    if LLM_BACKEND == LLMBackend.BEDROCK:
        from nexari.llm.bedrock import BedrockClient
        return BedrockClient()
    from nexari.llm.ollama import OllamaClient
    return OllamaClient()
