"""Web-based training comparison visualizer with Gradio."""

import argparse
import gradio as gr
from typing import List, Tuple
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from .data import get_log_choices, load_runs_from_paths
from .plots import (
    create_combined_steps_plot,
    create_combined_flops_plot,
    create_lr_schedule_plot,
    create_throughput_plot,
    create_empty_figure,
    create_compute_efficiency_comparison,
    create_efficiency_curve,
    find_crossover_points,
    get_val_flops_data,
)
from .compute import compute_cumulative_flops
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


def get_compute_range(runs) -> Tuple[float, float]:
    """
    Get min/max compute (TFLOPs) across all runs.

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        Tuple of (min_tflops, max_tflops)
    """
    all_flops = []
    for run in runs.values():
        _, tflops = compute_cumulative_flops(run)
        if tflops:
            all_flops.extend(tflops)

    if not all_flops:
        return 0, 1000

    return min(all_flops), max(all_flops)


def generate_compute_insights(runs, budget_tflops: float = None) -> str:
    """
    Generate markdown insights about compute efficiency.

    Args:
        runs: Dict of run_name -> TrainingRun
        budget_tflops: Optional compute budget for comparison

    Returns:
        Markdown string with insights
    """
    if not runs:
        return ""

    insights = []

    # Get compute range
    min_tflops, max_tflops = get_compute_range(runs)
    common_budget = min(max_tflops, min(run_max for run_max in [
        get_val_flops_data(run)[0].max() if len(get_val_flops_data(run)[0]) > 0 else 0
        for run in runs.values()
    ] if run_max > 0)) if runs else None

    if budget_tflops is not None:
        common_budget = budget_tflops

    # Compare losses at common budget
    if common_budget:
        losses_at_budget = {}
        for name, run in runs.items():
            flops, losses = get_val_flops_data(run)
            if len(flops) > 0 and flops.min() <= common_budget <= flops.max():
                loss_at_budget = np.interp(common_budget, flops, losses)
                losses_at_budget[name] = loss_at_budget

        if losses_at_budget:
            best_run = min(losses_at_budget, key=losses_at_budget.get)
            insights.append(
                f"**At {common_budget:.0f} TFLOPs:** {best_run} achieves lowest "
                f"validation loss ({losses_at_budget[best_run]:.4f})"
            )

    # Find crossover points
    crossovers = find_crossover_points(runs)
    for cross in crossovers[:2]:  # Limit to first 2
        insights.append(
            f"**Crossover at loss={cross['loss']:.3f}:** {cross['larger']} becomes "
            f"more efficient than {cross['smaller']} ({cross['flops_large']:.0f} vs "
            f"{cross['flops_small']:.0f} TFLOPs)"
        )

    # Compute to reach specific loss thresholds
    target_losses = [2.0, 1.5]
    for target in target_losses:
        for name, run in runs.items():
            flops, losses = get_val_flops_data(run)
            if len(flops) > 0 and losses.min() < target < losses.max():
                # Interpolate to find FLOPs needed
                sorted_idx = np.argsort(losses)[::-1]
                flops_to_target = np.interp(target, losses[sorted_idx], flops[sorted_idx])
                insights.append(
                    f"**{name}** reaches loss={target:.1f} at {flops_to_target:.0f} TFLOPs"
                )
                break  # Only show first run that reaches this threshold

    if not insights:
        return "*Select multiple runs to see compute efficiency insights.*"

    return "\n\n".join(insights)


def update_visualizations(selected_logs: List[str], budget_tflops: float = None) -> Tuple:
    """
    Main callback to update all visualizations when selection changes.

    Args:
        selected_logs: List of selected log file paths
        budget_tflops: Optional compute budget for comparison (in TFLOPs)

    Returns:
        Tuple of outputs for all UI components
    """
    if not selected_logs:
        empty = create_empty_figure()
        return (
            "",  # stats_html
            empty, empty,  # loss_steps, lr
            empty, empty, empty,  # loss_flops, comparison, efficiency
            empty,  # throughput
            "",  # insights
            gr.update(minimum=0, maximum=1000, value=500),  # slider
            [], "No runs selected"  # summary, config
        )

    runs = load_runs_from_paths(selected_logs)

    if not runs:
        empty = create_empty_figure("Failed to load runs")
        return (
            "",
            empty, empty,
            empty, empty, empty,
            empty,
            "",
            gr.update(minimum=0, maximum=1000, value=500),
            [], "Failed to load selected runs"
        )

    # Get compute range for slider
    min_tflops, max_tflops = get_compute_range(runs)
    # Round to nice values
    slider_min = int(min_tflops // 50) * 50
    slider_max = int((max_tflops // 50) + 1) * 50
    slider_value = int((slider_min + slider_max) / 2)

    # Generate all visualizations
    stats_html = format_stats_html(runs)
    loss_steps_fig = create_combined_steps_plot(runs)
    lr_fig = create_lr_schedule_plot(runs)

    # Compute performance charts
    loss_flops_fig = create_combined_flops_plot(runs)
    comparison_fig = create_compute_efficiency_comparison(runs, budget_tflops)
    efficiency_fig = create_efficiency_curve(runs)

    throughput_fig = create_throughput_plot(runs)
    insights_md = generate_compute_insights(runs, budget_tflops)

    summary_data = create_summary_table(runs)
    config_md = format_config_comparison(runs)

    return (
        stats_html,
        loss_steps_fig, lr_fig,
        loss_flops_fig, comparison_fig, efficiency_fig,
        throughput_fig,
        insights_md,
        gr.update(minimum=slider_min, maximum=slider_max, value=slider_value),
        summary_data, config_md
    )


def update_budget_comparison(selected_logs: List[str], budget_tflops: float) -> Tuple:
    """
    Update only the budget-dependent visualizations when slider changes.

    Args:
        selected_logs: List of selected log file paths
        budget_tflops: Compute budget in TFLOPs

    Returns:
        Tuple of (comparison_fig, insights_md)
    """
    if not selected_logs:
        return create_empty_figure(), ""

    runs = load_runs_from_paths(selected_logs)
    if not runs:
        return create_empty_figure(), ""

    comparison_fig = create_compute_efficiency_comparison(runs, budget_tflops)
    insights_md = generate_compute_insights(runs, budget_tflops)

    return comparison_fig, insights_md


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
            "**Right column**: compute efficiency analysis."
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
                with gr.Tabs():
                    with gr.Tab("Loss vs Compute"):
                        loss_flops_plot = gr.Plot(label="Loss & Perplexity vs FLOPs")
                    with gr.Tab("Equal Compute Comparison"):
                        comparison_plot = gr.Plot(label="Validation Loss at Equal Compute")
                    with gr.Tab("Compute Efficiency"):
                        efficiency_plot = gr.Plot(label="Efficiency Curve")

                # Compute budget slider
                compute_budget_slider = gr.Slider(
                    minimum=0,
                    maximum=1000,
                    value=500,
                    step=10,
                    label="Compute Budget (TFLOPs)",
                    info="Highlight a specific compute budget in the comparison",
                )

                # Compute insights
                with gr.Accordion("Compute Insights", open=True):
                    insights_display = gr.Markdown(
                        value="*Select training runs to see insights.*"
                    )

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
            inputs=[log_dropdown, compute_budget_slider],
            outputs=[
                stats_display,
                loss_steps_plot, lr_plot,
                loss_flops_plot, comparison_plot, efficiency_plot,
                throughput_plot,
                insights_display,
                compute_budget_slider,
                summary_table, config_display
            ],
        )

        # Update comparison chart when budget slider changes
        compute_budget_slider.change(
            fn=update_budget_comparison,
            inputs=[log_dropdown, compute_budget_slider],
            outputs=[comparison_plot, insights_display],
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
