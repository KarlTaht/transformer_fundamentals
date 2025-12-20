"""Web-based training visualization with Gradio.

This module provides a post-hoc comparison visualizer for training runs.
Select multiple runs to compare loss curves, validation metrics,
compute efficiency (FLOPs), learning rate schedules, and throughput.

Usage:
    # Command line
    python -m visualizer

    # With public sharing link
    python -m visualizer --share

    # Python API
    from visualizer import launch, create_app
    launch()

    # Use individual plotting functions
    from visualizer import create_loss_curves, create_flops_normalized_loss
    from training import load_training_log

    run = load_training_log('assets/logs/my_experiment.json')
    fig = create_loss_curves({'my_run': run})
    fig.show()
"""

from .app import create_app, launch
from .plots import (
    create_loss_curves,
    create_validation_curves,
    create_lr_schedule_plot,
    create_throughput_plot,
    create_flops_normalized_loss,
    create_empty_figure,
)
from .data import discover_training_logs, load_runs_from_paths, get_log_choices
from .compare import compute_run_summary, create_summary_table, format_config_comparison
from .compute import estimate_model_params, compute_cumulative_flops, compute_total_flops

__all__ = [
    # Main entry points
    "create_app",
    "launch",
    # Plotting functions
    "create_loss_curves",
    "create_validation_curves",
    "create_lr_schedule_plot",
    "create_throughput_plot",
    "create_flops_normalized_loss",
    "create_empty_figure",
    # Data utilities
    "discover_training_logs",
    "load_runs_from_paths",
    "get_log_choices",
    # Comparison utilities
    "compute_run_summary",
    "create_summary_table",
    "format_config_comparison",
    # Compute utilities
    "estimate_model_params",
    "compute_cumulative_flops",
    "compute_total_flops",
]
