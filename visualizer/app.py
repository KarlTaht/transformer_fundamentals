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
from .compare import create_summary_table, format_config_comparison, compute_run_summary


def format_stats_html(runs) -> str:
    """
    Format key metrics as HTML cards for quick comparison.

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        HTML string with stats cards
    """
    if not runs:
        return ""

    html = '<div style="display:flex; gap:15px; flex-wrap:wrap; margin-bottom:10px;">'
    for name, run in runs.items():
        summary = compute_run_summary(run)
        final_loss = summary.get('final_train_loss')
        best_val = summary.get('best_val_loss')

        final_loss_str = f"{final_loss:.4f}" if final_loss is not None else "N/A"
        best_val_str = f"{best_val:.4f}" if best_val is not None else "N/A"

        html += f'''
        <div style="border:1px solid #ccc; padding:12px; border-radius:8px; min-width:180px; background:#f5f5f5;">
            <div style="font-weight:600; margin-bottom:8px; font-size:0.9em; color:#111 !important;">{name}</div>
            <div style="font-size:0.85em; color:#333 !important;">
                <div>Final Loss: <b style="color:#000 !important;">{final_loss_str}</b></div>
                <div>Best Val: <b style="color:#000 !important;">{best_val_str}</b></div>
                <div>Params: <b style="color:#000 !important;">{summary.get('params_formatted', 'N/A')}</b></div>
                <div>Compute: <b style="color:#000 !important;">{summary.get('total_tflops_formatted', 'N/A')}</b></div>
            </div>
        </div>
        '''
    html += '</div>'
    return html


def update_visualizations(selected_logs: List[str]) -> Tuple:
    """
    Main callback to update all visualizations when selection changes.

    Args:
        selected_logs: List of selected log file paths

    Returns:
        Tuple of (stats_html, loss_steps_fig, loss_flops_fig, lr_fig, throughput_fig, summary_df, config_md)
    """
    if not selected_logs:
        empty = create_empty_figure()
        return "", empty, empty, empty, empty, [], "No runs selected"

    runs = load_runs_from_paths(selected_logs)

    if not runs:
        empty = create_empty_figure("Failed to load runs")
        return "", empty, empty, empty, empty, [], "Failed to load selected runs"

    # Generate all visualizations
    stats_html = format_stats_html(runs)
    loss_steps_fig = create_combined_steps_plot(runs)
    loss_flops_fig = create_combined_flops_plot(runs)
    lr_fig = create_lr_schedule_plot(runs)
    throughput_fig = create_throughput_plot(runs)
    summary_data = create_summary_table(runs)
    config_md = format_config_comparison(runs)

    return stats_html, loss_steps_fig, loss_flops_fig, lr_fig, throughput_fig, summary_data, config_md


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

        # Quick Stats Row (shows key metrics for selected runs)
        stats_display = gr.HTML(label="Quick Stats")

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

        # Config Comparison (expanded by default for transparency)
        with gr.Accordion("Configuration Comparison", open=True):
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
            outputs=[stats_display, loss_steps_plot, loss_flops_plot, lr_plot,
                    throughput_plot, summary_table, config_display],
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
