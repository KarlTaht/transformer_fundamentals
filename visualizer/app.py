"""Web-based training comparison visualizer with Gradio."""

import argparse
import gradio as gr
from typing import List, Tuple
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .data import get_log_choices, load_runs_from_paths
from .plots import (
    create_loss_curves,
    create_validation_curves,
    create_lr_schedule_plot,
    create_throughput_plot,
    create_flops_normalized_loss,
    create_empty_figure,
)
from .compare import create_summary_table, format_config_comparison


def update_visualizations(selected_logs: List[str]) -> Tuple:
    """
    Main callback to update all visualizations when selection changes.

    Args:
        selected_logs: List of selected log file paths

    Returns:
        Tuple of (loss_fig, val_fig, flops_fig, lr_fig, throughput_fig, summary_df, config_md)
    """
    if not selected_logs:
        empty = create_empty_figure()
        return empty, empty, empty, empty, empty, [], "No runs selected"

    runs = load_runs_from_paths(selected_logs)

    if not runs:
        empty = create_empty_figure("Failed to load runs")
        return empty, empty, empty, empty, empty, [], "Failed to load selected runs"

    # Generate all visualizations
    loss_fig = create_loss_curves(runs)
    val_fig = create_validation_curves(runs)
    flops_fig = create_flops_normalized_loss(runs)
    lr_fig = create_lr_schedule_plot(runs)
    throughput_fig = create_throughput_plot(runs)
    summary_data = create_summary_table(runs)
    config_md = format_config_comparison(runs)

    return loss_fig, val_fig, flops_fig, lr_fig, throughput_fig, summary_data, config_md


def refresh_log_choices() -> dict:
    """Refresh the dropdown choices with latest log files."""
    choices = get_log_choices()
    return gr.update(choices=choices)


def create_app() -> gr.Blocks:
    """Create the Gradio application interface."""

    with gr.Blocks(title="Training Comparison") as app:
        gr.Markdown("# Training Run Comparison")
        gr.Markdown(
            "Select 1-4 training runs to compare loss curves, compute efficiency, "
            "learning rate schedules, throughput, and configurations."
        )

        # Run Selection Row
        with gr.Row():
            with gr.Column(scale=4):
                log_dropdown = gr.Dropdown(
                    choices=get_log_choices(),
                    multiselect=True,
                    max_choices=4,
                    label="Select Training Runs (max 4)",
                    info="Choose log files from assets/logs/",
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")

        # Main Visualization Tabs
        with gr.Tabs():
            with gr.Tab("Loss"):
                loss_plot = gr.Plot(label="Training & Validation Loss")

            with gr.Tab("Validation"):
                val_plot = gr.Plot(label="Validation Loss & Perplexity")

            with gr.Tab("FLOPs"):
                flops_plot = gr.Plot(label="Loss vs Compute")

            with gr.Tab("Learning Rate"):
                lr_plot = gr.Plot(label="Learning Rate Schedule")

            with gr.Tab("Throughput"):
                throughput_plot = gr.Plot(label="Tokens/Second")

            with gr.Tab("Config"):
                config_display = gr.Markdown(label="Configuration Comparison")

        # Summary Table (always visible below tabs)
        gr.Markdown("## Summary")
        summary_table = gr.DataFrame(
            headers=["Run", "Type", "Params", "Steps", "Best Val Loss",
                    "Perplexity", "TFLOPs", "Avg Tok/s"],
            label="Run Summary",
            wrap=True,
        )

        # Event handlers
        log_dropdown.change(
            fn=update_visualizations,
            inputs=[log_dropdown],
            outputs=[loss_plot, val_plot, flops_plot, lr_plot, throughput_plot,
                    summary_table, config_display],
        )

        refresh_btn.click(
            fn=refresh_log_choices,
            outputs=[log_dropdown],
        )

    return app


def launch(share: bool = False, port: int = 7860):
    """
    Launch the Gradio application.

    Args:
        share: Whether to create a public link
        port: Port to run on (default: 7860)
    """
    app = create_app()
    app.launch(share=share, server_port=port)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Training run comparison visualizer"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)",
    )
    args = parser.parse_args()

    print("Starting Training Comparison Visualizer...")
    print(f"Log directory: {PROJECT_ROOT / 'assets' / 'logs'}")
    launch(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
