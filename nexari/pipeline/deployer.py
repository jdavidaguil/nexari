"""
nexari.pipeline.deployer
─────────────────────────
Step 5: Push model to HF Hub and create an Inference Endpoint.
Returns the endpoint URL.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from nexari.config import HF_NAMESPACE, HF_TOKEN, HF_ENDPOINT_REGION

console = Console()


def deploy(model_path: str, task) -> str:
    """
    Push fine-tuned model to HF Hub and spin up an Inference Endpoint.
    Returns the endpoint inference URL.
    """
    from huggingface_hub import HfApi, create_inference_endpoint, get_inference_endpoint
    from huggingface_hub.utils import RepositoryNotFoundError

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set. Add it to your .env or Codespaces secrets.")
    if not HF_NAMESPACE:
        raise ValueError("HF_NAMESPACE not set. Add your HF username to .env or Codespaces secrets.")

    api = HfApi(token=HF_TOKEN)
    path = Path(model_path)

    # ── Load metadata ─────────────────────────────────────────────────────────
    metadata_file = path / "nexari_metadata.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    domain_slug = metadata.get("domain", "model").replace(" ", "-").lower()
    repo_name = f"nexari-{domain_slug}-classifier"
    repo_id = f"{HF_NAMESPACE}/{repo_name}"

    # ── Push to Hub ───────────────────────────────────────────────────────────
    console.print(f"  [dim]Pushing model to {repo_id}...[/]")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    except Exception as e:
        console.print(f"  [yellow]Repo creation note: {e}[/]")

    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["*.pt", "optimizer.pt"],  # skip raw optimizer state
    )
    console.print(f"  [green]✓[/] Model pushed to https://huggingface.co/{repo_id}")

    # ── Create Inference Endpoint ─────────────────────────────────────────────
    endpoint_name = f"nexari-{domain_slug}"
    console.print(f"  [dim]Creating inference endpoint {endpoint_name}...[/]")

    try:
        endpoint = create_inference_endpoint(
            name=endpoint_name,
            repository=repo_id,
            framework="pytorch",
            task="text-classification",
            accelerator="cpu",
            instance_size="small",
            instance_type="intel-icl",
            region=HF_ENDPOINT_REGION,
            type="public",
            token=HF_TOKEN,
        )
        endpoint.wait(timeout=300)
        url = endpoint.url
    except Exception as e:
        # Endpoint may already exist — try to fetch it
        console.print(f"  [yellow]Endpoint note: {e}[/]")
        try:
            endpoint = get_inference_endpoint(endpoint_name, token=HF_TOKEN)
            endpoint.wait(timeout=300)
            url = endpoint.url
        except Exception:
            url = f"https://api-inference.huggingface.co/models/{repo_id}"
            console.print(f"  [yellow]Using serverless inference fallback[/]")

    console.print(f"  [green]✓[/] Endpoint live at {url}")

    # ── Start preview server ──────────────────────────────────────────────────
    _start_preview(url=url, metadata=metadata)

    return url


def _start_preview(url: str, metadata: dict):
    """Launch the FastAPI preview server in the background."""
    import threading
    from nexari.preview.server import start_server
    t = threading.Thread(target=start_server, args=(url, metadata), daemon=True)
    t.start()