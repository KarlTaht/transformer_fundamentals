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
    create_combined_steps_plot,
    create_combined_flops_plot,
    create_lr_schedule_plot,
    create_throughput_plot,
    create_empty_figure,
)
from .compare import create_summary_table, format_config_comparison


def update_visualizations(selected_logs: List[str]) -> Tuple:
    """
    Main callback to update all visualizations when selection changes.

    Args:
        selected_logs: List of selected log file paths

    Returns:
        Tuple of (loss_steps_fig, loss_flops_fig, lr_fig, throughput_fig, summary_df, config_md)
    """
    if not selected_logs:
        empty = create_empty_figure()
        return empty, empty, empty, empty, [], "No runs selected"

    runs = load_runs_from_paths(selected_logs)

    if not runs:
        empty = create_empty_figure("Failed to load runs")
        return empty, empty, empty, empty, [], "Failed to load selected runs"

    # Generate all visualizations
    loss_steps_fig = create_combined_steps_plot(runs)
    loss_flops_fig = create_combined_flops_plot(runs)
    lr_fig = create_lr_schedule_plot(runs)
    throughput_fig = create_throughput_plot(runs)
    summary_data = create_summary_table(runs)
    config_md = format_config_comparison(runs)

    return loss_steps_fig, loss_flops_fig, lr_fig, throughput_fig, summary_data, config_md


def refresh_log_choices() -> dict:
    """Refresh the dropdown choices with latest log files."""
    choices = get_log_choices()
    return gr.update(choices=choices)


def create_app() -> gr.Blocks:
    """Create the Gradio application interface."""

    with gr.Blocks(title="Training Comparison") as app:
        gr.Markdown("# Training Run Comparison")
        gr.Markdown(
            "Select 1-4 training runs to compare. "
            "**Left column**: metrics vs training steps. "
            "**Right column**: metrics vs compute (FLOPs)."
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
                refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")

        # Two-Column Visualization Layout
        with gr.Row():
            # Left Column: Model Performance (X-axis: Steps)
            with gr.Column(scale=1):
                gr.Markdown("### Model Performance (Steps)")
                loss_steps_plot = gr.Plot(label="Loss & Perplexity vs Steps")
                lr_plot = gr.Plot(label="Learning Rate Schedule")

            # Right Column: Compute Performance (X-axis: FLOPs)
            with gr.Column(scale=1):
                gr.Markdown("### Compute Performance (FLOPs)")
                loss_flops_plot = gr.Plot(label="Loss & Perplexity vs FLOPs")
                throughput_plot = gr.Plot(label="Throughput (Tokens/sec)")

        # Config Comparison (collapsible)
        with gr.Accordion("Configuration Comparison", open=False):
            config_display = gr.Markdown(label="Configuration Comparison")

        # Summary Table (always visible)
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
            outputs=[loss_steps_plot, loss_flops_plot, lr_plot, throughput_plot,
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
