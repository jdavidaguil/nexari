"""
nexari.cli
──────────
Entry point. `nexari run "your intent here"`
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()


@click.group()
@click.version_option(package_name="nexari")
def main():
    """Nexari — from intent to deployed model."""
    pass


@main.command()
@click.argument("intent")
@click.option("--dry-run", is_flag=True, help="Run pipeline without deploying.")
@click.option("--skip-approval", is_flag=True, help="Auto-approve dataset selection.")
def run(intent: str, dry_run: bool, skip_approval: bool):
    """Run the full Nexari pipeline from a natural language INTENT."""
    from nexari.agent.interpreter import interpret
    from nexari.agent.discoverer import discover
    from nexari.agent.selector import select_backbone

    console.print(Panel.fit(f"[bold cyan]nexari[/] starting\n[dim]{intent}[/]"))

    # ── Step 1: Interpret ──────────────────────────────────────────────────
    console.print("\n[bold]Step 1/5[/] Interpreting intent...")
    task = interpret(intent)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[dim]Task type[/]", f"[green]{task.task_type.value}[/]")
    table.add_row("[dim]Domain[/]", task.domain)
    table.add_row("[dim]Input[/]", task.input_description)
    table.add_row("[dim]Output[/]", task.output_description)
    table.add_row("[dim]Metric[/]", task.suggested_metric)
    if task.notes:
        table.add_row("[dim]Notes[/]", task.notes)
    console.print(table)

    # ── Step 2: Discover datasets ──────────────────────────────────────────
    console.print("\n[bold]Step 2/5[/] Searching Hugging Face for datasets...")
    candidates = discover(task)

    if not candidates:
        console.print("[red]No suitable datasets found. Try rephrasing your intent.[/]")
        raise click.Abort()

    console.print("\n[bold]Candidate datasets:[/]")
    for c in candidates:
        console.print(f"\n  [cyan][{c.rank}][/] [bold]{c.dataset_id}[/]")
        console.print(f"      {c.recommendation_rationale}")
        console.print(f"      [dim]↓ {c.downloads:,} downloads · ♥ {c.likes:,} likes[/]")

    # ── Human approval ─────────────────────────────────────────────────────
    if skip_approval:
        chosen = candidates[0].dataset_id
        console.print(f"\n[dim]Auto-selected:[/] [green]{chosen}[/]")
    else:
        choice = Prompt.ask(
            "\nSelect dataset",
            choices=[str(c.rank) for c in candidates],
            default="1",
        )
        chosen = candidates[int(choice) - 1].dataset_id
        console.print(f"[green]✓[/] Selected [bold]{chosen}[/]")

    # ── Step 3: Select backbone ────────────────────────────────────────────
    console.print("\n[bold]Step 3/5[/] Selecting backbone model...")
    backbone = select_backbone(task, chosen)
    console.print(f"  [green]✓[/] [bold]{backbone.model_id}[/]")
    console.print(f"  [dim]{backbone.rationale}[/]")
    console.print(f"  [dim]Estimated training time: ~{backbone.estimated_train_time_minutes} min[/]")

    if dry_run:
        console.print("\n[yellow]Dry run complete. Steps 4-5 (train + deploy) skipped.[/]")
        return

    # ── Step 4: Train ──────────────────────────────────────────────────────
    console.print("\n[bold]Step 4/5[/] Fine-tuning model...")
    from nexari.pipeline.trainer import train
    model_path = train(task=task, dataset_id=chosen, backbone=backbone)
    console.print(f"  [green]✓[/] Model saved to [bold]{model_path}[/]")

    # ── Step 5: Deploy + Preview ───────────────────────────────────────────
    console.print("\n[bold]Step 5/5[/] Deploying preview...")
    from nexari.pipeline.deployer import deploy
    endpoint_url = deploy(model_path=model_path, task=task)

    console.print(Panel.fit(
        f"[bold green]✓ Done![/]\n\n"
        f"[dim]Endpoint:[/] {endpoint_url}\n"
        f"[dim]Preview:[/]  http://{__import__('nexari.config', fromlist=['PREVIEW_HOST']).PREVIEW_HOST}:"
        f"{__import__('nexari.config', fromlist=['PREVIEW_PORT']).PREVIEW_PORT}",
        title="nexari",
    ))


@main.command()
def config():
    """Show current nexari configuration."""
    from nexari import config as cfg
    table = Table(title="Nexari Configuration", show_header=True)
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="cyan")
    table.add_row("LLM Backend", cfg.LLM_BACKEND.value)
    table.add_row("Bedrock Model", cfg.BEDROCK_MODEL_ID)
    table.add_row("Bedrock Region", cfg.BEDROCK_REGION)
    table.add_row("Ollama Model", cfg.OLLAMA_MODEL)
    table.add_row("HF Namespace", cfg.HF_NAMESPACE or "[not set]")
    table.add_row("Preview Port", str(cfg.PREVIEW_PORT))
    console.print(table)