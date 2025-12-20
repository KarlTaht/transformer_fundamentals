"""Comparison logic and summary computations for training runs."""

from typing import Dict, List, Any
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training import TrainingRun
from .compute import estimate_model_params, compute_total_flops, format_params, format_flops


def compute_run_summary(run: TrainingRun) -> Dict[str, Any]:
    """
    Compute summary statistics for a single training run.

    Args:
        run: TrainingRun object

    Returns:
        Dict with summary statistics including:
        - experiment_name, model_type
        - params (estimated parameter count)
        - total_steps, total_epochs
        - best_val_loss, best_val_perplexity, best_val_step
        - final_train_loss
        - total_tflops
        - avg_throughput
        - duration_seconds
    """
    summary = {
        "experiment_name": run.experiment_name,
        "model_type": run.model_type,
        "total_steps": len(run.train_metrics),
        "total_epochs": run.train_metrics[-1].epoch + 1 if run.train_metrics else 0,
    }

    # Estimated parameters
    params = estimate_model_params(run.model_config)
    summary["params"] = params
    summary["params_formatted"] = format_params(params)

    # Best validation metrics
    if run.val_metrics:
        best_val = min(run.val_metrics, key=lambda m: m.loss)
        summary["best_val_loss"] = best_val.loss
        summary["best_val_perplexity"] = best_val.perplexity
        summary["best_val_step"] = best_val.step
    else:
        summary["best_val_loss"] = None
        summary["best_val_perplexity"] = None
        summary["best_val_step"] = None

    # Final training loss
    if run.train_metrics:
        summary["final_train_loss"] = run.train_metrics[-1].loss
    else:
        summary["final_train_loss"] = None

    # Total FLOPs
    total_tflops = compute_total_flops(run)
    summary["total_tflops"] = total_tflops
    summary["total_tflops_formatted"] = format_flops(total_tflops)

    # Average throughput (tokens/sec)
    tps_values = [m.tokens_per_sec for m in run.train_metrics
                  if m.tokens_per_sec is not None]
    if tps_values:
        summary["avg_throughput"] = sum(tps_values) / len(tps_values)
    else:
        summary["avg_throughput"] = None

    # Training duration
    if run.started_at and run.finished_at:
        try:
            start = datetime.fromisoformat(run.started_at)
            end = datetime.fromisoformat(run.finished_at)
            summary["duration_seconds"] = (end - start).total_seconds()
        except (ValueError, TypeError):
            summary["duration_seconds"] = None
    else:
        summary["duration_seconds"] = None

    return summary


def create_summary_table(runs: Dict[str, TrainingRun]) -> List[Dict[str, Any]]:
    """
    Create summary table data for all runs.

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        List of dicts suitable for display in Gradio DataFrame
    """
    rows = []
    for name, run in runs.items():
        summary = compute_run_summary(run)
        rows.append({
            "Run": name,
            "Type": summary.get("model_type", ""),
            "Params": summary.get("params_formatted", ""),
            "Steps": summary.get("total_steps", 0),
            "Best Val Loss": f"{summary['best_val_loss']:.4f}"
                            if summary.get('best_val_loss') is not None else "N/A",
            "Perplexity": f"{summary['best_val_perplexity']:.2f}"
                         if summary.get('best_val_perplexity') is not None else "N/A",
            "TFLOPs": summary.get("total_tflops_formatted", "N/A"),
            "Avg Tok/s": f"{summary['avg_throughput']:,.0f}"
                        if summary.get('avg_throughput') is not None else "N/A",
        })
    return rows


def format_config_comparison(runs: Dict[str, TrainingRun]) -> str:
    """
    Format model and training configs for side-by-side comparison.

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        Markdown-formatted string for display
    """
    if not runs:
        return "No runs selected"

    run_names = list(runs.keys())
    run_list = list(runs.values())

    lines = []

    # Model config comparison
    lines.append("## Model Configuration\n")
    lines.append("| Parameter | " + " | ".join(run_names) + " |")
    lines.append("|-----------|" + "|".join(["---"] * len(runs)) + "|")

    # Collect all model config keys
    all_model_keys = set()
    for run in run_list:
        all_model_keys.update(run.model_config.keys())

    for key in sorted(all_model_keys):
        values = [str(run.model_config.get(key, "-")) for run in run_list]
        lines.append(f"| {key} | " + " | ".join(values) + " |")

    # Training config comparison
    lines.append("\n## Training Configuration\n")
    lines.append("| Parameter | " + " | ".join(run_names) + " |")
    lines.append("|-----------|" + "|".join(["---"] * len(runs)) + "|")

    all_train_keys = set()
    for run in run_list:
        all_train_keys.update(run.train_config.keys())

    for key in sorted(all_train_keys):
        values = [str(run.train_config.get(key, "-")) for run in run_list]
        lines.append(f"| {key} | " + " | ".join(values) + " |")

    # Add estimated params row
    lines.append("\n## Computed Metrics\n")
    lines.append("| Metric | " + " | ".join(run_names) + " |")
    lines.append("|--------|" + "|".join(["---"] * len(runs)) + "|")

    params_values = [format_params(estimate_model_params(run.model_config)) for run in run_list]
    lines.append(f"| Est. Parameters | " + " | ".join(params_values) + " |")

    flops_values = [format_flops(compute_total_flops(run)) for run in run_list]
    lines.append(f"| Total Compute | " + " | ".join(flops_values) + " |")

    return "\n".join(lines)
