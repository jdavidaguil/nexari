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
@click.option("--backend", default=None, type=click.Choice(["local", "sagemaker"]), help="Training backend (default: local).")
@click.option("--resume", is_flag=True, help="Resume from last saved session, skipping completed steps.")
@click.option("--clear-session", is_flag=True, help="Clear saved session and start fresh.")
def run(intent: str, dry_run: bool, skip_approval: bool, backend: str | None, resume: bool, clear_session: bool):
    """Run the full Nexari pipeline from a natural language INTENT."""
    from nexari.agent.interpreter import interpret
    from nexari.agent.discoverer import discover
    from nexari.agent.selector import select_backbone
    from nexari.session import NexariSession, restore_task, restore_backbone
    from dataclasses import asdict

    # ── Session setup ──────────────────────────────────────────────────────────
    if clear_session:
        NexariSession(intent=intent).clear()
        console.print("[dim]Session cleared.[/]")

    session = NexariSession.load_or_create(intent) if resume else NexariSession(intent=intent)

    if resume and session.completed_steps:
        console.print(f"[dim]Resuming session. Completed steps: {session.completed_steps}[/]")

    console.print(Panel.fit(f"[bold cyan]nexari[/] starting\n[dim]{intent}[/]"))

    # ── Step 1: Interpret ──────────────────────────────────────────────────────
    if session.is_complete(1) and resume:
        console.print("\n[bold]Step 1/5[/] [dim]Interpreting intent... (cached)[/]")
        task = restore_task(session)
    else:
        console.print("\n[bold]Step 1/5[/] Interpreting intent...")
        task = interpret(intent)
        session.task = asdict(task)
        session.mark_complete(1)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[dim]Task type[/]", f"[green]{task.task_type.value}[/]")
    table.add_row("[dim]Domain[/]", task.domain)
    table.add_row("[dim]Input[/]", task.input_description)
    table.add_row("[dim]Output[/]", task.output_description)
    table.add_row("[dim]Metric[/]", task.suggested_metric)
    if task.notes:
        table.add_row("[dim]Notes[/]", task.notes)
    console.print(table)

    # ── Step 2: Discover datasets ──────────────────────────────────────────────
    if session.is_complete(2) and resume:
        console.print("\n[bold]Step 2/5[/] [dim]Dataset selection... (cached)[/]")
        chosen = session.dataset_id
        console.print(f"  [green]✓[/] [bold]{chosen}[/]")
    else:
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

        session.dataset_id = chosen
        session.mark_complete(2)

    # ── Step 3: Select backbone ────────────────────────────────────────────────
    if session.is_complete(3) and resume:
        console.print("\n[bold]Step 3/5[/] [dim]Backbone selection... (cached)[/]")
        backbone = restore_backbone(session)
        console.print(f"  [green]✓[/] [bold]{backbone.model_id}[/] [dim](cached)[/]")
    else:
        console.print("\n[bold]Step 3/5[/] Selecting backbone model...")
        backbone = select_backbone(task, chosen)
        console.print(f"  [green]✓[/] [bold]{backbone.model_id}[/]")
        console.print(f"  [dim]{backbone.rationale}[/]")
        console.print(f"  [dim]Estimated training time: ~{backbone.estimated_train_time_minutes} min[/]")
        session.backbone = asdict(backbone)
        session.mark_complete(3)

    if dry_run:
        console.print("\n[yellow]Dry run complete. Steps 4-5 (train + deploy) skipped.[/]")
        return

    # ── Step 4: Train ──────────────────────────────────────────────────────────
    if session.is_complete(4) and resume and session.model_path:
        console.print("\n[bold]Step 4/5[/] [dim]Fine-tuning... (cached)[/]")
        model_path = session.model_path
        console.print(f"  [green]✓[/] Model at [bold]{model_path}[/] [dim](cached)[/]")
    else:
        console.print("\n[bold]Step 4/5[/] Fine-tuning model...")
        from nexari.pipeline.trainer import train, TrainBackend
        model_path = train(
            task=task, dataset_id=chosen, backbone=backbone,
            backend=TrainBackend(backend) if backend else None,
        )
        console.print(f"  [green]✓[/] Model saved to [bold]{model_path}[/]")
        session.model_path = model_path
        session.mark_complete(4)

    # ── Step 5: Deploy + Preview ───────────────────────────────────────────────
    if session.is_complete(5) and resume and session.endpoint_url:
        console.print("\n[bold]Step 5/5[/] [dim]Deployment... (cached)[/]")
        endpoint_url = session.endpoint_url
        console.print(f"  [green]✓[/] Endpoint: {endpoint_url} [dim](cached)[/]")
    else:
        console.print("\n[bold]Step 5/5[/] Deploying preview...")
        from nexari.pipeline.deployer import deploy
        endpoint_url = deploy(model_path=model_path, task=task)
        session.endpoint_url = endpoint_url
        session.mark_complete(5)

    from nexari import config as cfg
    console.print(Panel.fit(
        f"[bold green]✓ Done![/]\n\n"
        f"[dim]Endpoint:[/] {endpoint_url}\n"
        f"[dim]Preview:[/]  http://{cfg.PREVIEW_HOST}:{cfg.PREVIEW_PORT}",
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


@main.command()
def session():
    """Show current session state."""
    from nexari.session import NexariSession, SESSION_FILE
    s = NexariSession.load()
    if not s:
        console.print("[dim]No active session. Run nexari run to start one.[/]")
        return
    table = Table(title=f"Session: {SESSION_FILE}", show_header=True)
    table.add_column("Step", style="dim")
    table.add_column("Status", style="cyan")
    table.add_column("Value")
    steps = {
        1: ("Interpret", s.task.get("task_type") if s.task else None),
        2: ("Dataset", s.dataset_id),
        3: ("Backbone", s.backbone.get("model_id") if s.backbone else None),
        4: ("Train", s.model_path),
        5: ("Deploy", s.endpoint_url),
    }
    for num, (name, value) in steps.items():
        status = "[green]✓[/]" if num in s.completed_steps else "[dim]pending[/]"
        table.add_row(f"{num}. {name}", status, value or "")
    console.print(table)