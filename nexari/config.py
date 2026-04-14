"""
nexari.config
─────────────
Central configuration. All env vars and backend selection live here.
Switch LLM backend with: NEXARI_LLM_BACKEND=bedrock|ollama
"""

from __future__ import annotations

import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class LLMBackend(str, Enum):
    BEDROCK = "bedrock"
    OLLAMA = "ollama"


# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_BACKEND = LLMBackend(os.getenv("NEXARI_LLM_BACKEND", "bedrock"))
BEDROCK_MODEL_ID = os.getenv("NEXARI_BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-5")
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
OLLAMA_MODEL = os.getenv("NEXARI_OLLAMA_MODEL", "gemma3:4b")
OLLAMA_HOST = os.getenv("NEXARI_OLLAMA_HOST", "http://localhost:11434")

# ── Hugging Face ──────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
HF_NAMESPACE = os.getenv("HF_NAMESPACE")          # your HF username/org
HF_ENDPOINT_REGION = os.getenv("HF_ENDPOINT_REGION", "us-east-1")

# ── Training defaults ─────────────────────────────────────────────────────────
DEFAULT_MAX_TRAIN_SAMPLES = int(os.getenv("NEXARI_MAX_TRAIN_SAMPLES", "5000"))
DEFAULT_EPOCHS = int(os.getenv("NEXARI_EPOCHS", "3"))
DEFAULT_BATCH_SIZE = int(os.getenv("NEXARI_BATCH_SIZE", "16"))

# ── Preview server ────────────────────────────────────────────────────────────
PREVIEW_HOST = os.getenv("NEXARI_PREVIEW_HOST", "127.0.0.1")
PREVIEW_PORT = int(os.getenv("NEXARI_PREVIEW_PORT", "7860"))
