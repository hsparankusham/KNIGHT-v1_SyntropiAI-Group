"""KNIGHT CLI entry point."""
from __future__ import annotations

import typer

app = typer.Typer(
    name="knight",
    help="KNIGHT v1: Brain Neuroimmune Cell State Foundation Model",
    no_args_is_help=True,
)


@app.command()
def info() -> None:
    """Show KNIGHT version and system info."""
    from knight import __version__
    from knight.utils.reproducibility import get_device, count_parameters

    typer.echo(f"KNIGHT v{__version__}")
    typer.echo(f"Device: {get_device()}")

    try:
        import torch
        typer.echo(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            typer.echo(f"CUDA: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                typer.echo(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        typer.echo("PyTorch: not installed")

    try:
        import scanpy
        typer.echo(f"Scanpy: {scanpy.__version__}")
    except ImportError:
        typer.echo("Scanpy: not installed")


@app.command()
def train(
    config: str = typer.Option("config/defaults.yaml", help="Path to config file"),
    stage: str = typer.Option("pretrain", help="Training stage: pretrain, cellstate, perturbation"),
    resume: str = typer.Option(None, help="Checkpoint path to resume from"),
) -> None:
    """Run model training."""
    from pathlib import Path
    from knight.utils.io import load_config

    cfg = load_config(Path(config))
    typer.echo(f"Starting {stage} training with config: {config}")

    if stage == "pretrain":
        typer.echo("Pretraining not yet configured — run scripts/training/01_pretrain_knight_min.py")
    elif stage == "cellstate":
        typer.echo("Cell state fine-tuning — run scripts/training/02_finetune_cellstate.py")
    elif stage == "perturbation":
        typer.echo("Perturbation fine-tuning — run scripts/training/03_finetune_perturbation.py")
    else:
        typer.echo(f"Unknown stage: {stage}")
        raise typer.Exit(1)


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(..., help="Model checkpoint path"),
    benchmark: str = typer.Option("all", help="Benchmark: cellstate, harmonization, perturbation, all"),
) -> None:
    """Run evaluation benchmarks."""
    typer.echo(f"Evaluating {checkpoint} on {benchmark} benchmark(s)")
    typer.echo("Evaluation not yet configured — run scripts/evaluation/ scripts")


@app.command()
def embed(
    input_path: str = typer.Argument(..., help="Path to h5ad file"),
    output_path: str = typer.Argument(..., help="Output path for embeddings"),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    batch_size: int = typer.Option(256, help="Batch size for inference"),
) -> None:
    """Generate cell embeddings for a dataset."""
    typer.echo(f"Generating embeddings for {input_path}")
    typer.echo(f"Using checkpoint: {checkpoint}")
    typer.echo("Inference pipeline not yet configured")


if __name__ == "__main__":
    app()
