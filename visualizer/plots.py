"""Reusable Plotly plotting functions for training visualization."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training import TrainingRun
from .compute import compute_cumulative_flops

from typing import List


def smooth_data(values: List[float], alpha: float = 0.1) -> List[float]:
    """
    Apply exponential moving average (EMA) smoothing to reduce noise.

    Args:
        values: List of values to smooth
        alpha: Smoothing factor (0-1). Lower = smoother. Default 0.1.

    Returns:
        Smoothed values
    """
    if not values:
        return values
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def get_val_flops_data(run: "TrainingRun") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract validation loss data with corresponding FLOPs values.

    Args:
        run: TrainingRun object

    Returns:
        Tuple of (flops_array, val_loss_array) as numpy arrays
    """
    if not run.train_metrics or not run.val_metrics:
        return np.array([]), np.array([])

    steps, tflops = compute_cumulative_flops(run)
    step_to_flops = dict(zip(steps, tflops))

    val_flops = []
    val_losses = []
    for m in run.val_metrics:
        if m.step in step_to_flops:
            val_flops.append(step_to_flops[m.step])
            val_losses.append(m.loss)

    return np.array(val_flops), np.array(val_losses)


def compute_iso_loss_thresholds(
    runs: Dict[str, "TrainingRun"],
    num_lines: int = 4,
) -> List[float]:
    """
    Auto-detect iso-loss threshold values based on data range.

    Finds round values (like 1.5, 2.0, 2.5) that fall within the
    loss range of the selected runs.

    Args:
        runs: Dict of run_name -> TrainingRun
        num_lines: Maximum number of iso-loss lines to generate

    Returns:
        List of loss threshold values
    """
    all_losses = []
    for run in runs.values():
        _, val_losses = get_val_flops_data(run)
        if len(val_losses) > 0:
            all_losses.extend(val_losses.tolist())

    if not all_losses:
        return []

    min_loss = min(all_losses)
    max_loss = max(all_losses)

    # Generate candidate round values
    candidates = []
    for base in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]:
        if min_loss < base < max_loss:
            candidates.append(base)

    # If we have too many, pick evenly spaced ones
    if len(candidates) > num_lines:
        step = len(candidates) // num_lines
        candidates = candidates[::step][:num_lines]

    return candidates


def add_iso_loss_lines(
    fig: go.Figure,
    thresholds: List[float],
    secondary_y: bool = False,
) -> None:
    """
    Add horizontal dashed lines at specified loss thresholds.

    Args:
        fig: Plotly Figure to modify
        thresholds: List of loss values for horizontal lines
        secondary_y: Whether to add to secondary y-axis
    """
    for threshold in thresholds:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="rgba(128, 128, 128, 0.5)",
            line_width=1,
            annotation_text=f"{threshold:.1f}",
            annotation_position="right",
            annotation_font_size=9,
            annotation_font_color="gray",
            secondary_y=secondary_y,
        )


# Color palette for runs - each run gets a train/val color pair
# Train colors are darker/saturated, val colors are lighter variants
RUN_COLORS_TRAIN = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]  # Blue, Red, Green, Purple
RUN_COLORS_VAL = ["#7fcdff", "#ff9896", "#98df8a", "#c5b0d5"]    # Light variants
# Legacy single-color palette (for backwards compatibility)
RUN_COLORS = RUN_COLORS_TRAIN


def create_loss_curves(
    runs: Dict[str, TrainingRun],
    show_train: bool = True,
    show_val: bool = True,
) -> go.Figure:
    """
    Create overlaid loss curves for multiple training runs.

    Args:
        runs: Dict of run_name -> TrainingRun
        show_train: Include training loss (solid lines)
        show_val: Include validation loss (dashed lines with markers)

    Returns:
        Plotly Figure with loss curves
    """
    fig = go.Figure()

    for i, (name, run) in enumerate(runs.items()):
        train_color = RUN_COLORS_TRAIN[i % len(RUN_COLORS_TRAIN)]
        val_color = RUN_COLORS_VAL[i % len(RUN_COLORS_VAL)]

        # Training loss (solid line)
        if show_train and run.train_metrics:
            steps = [m.step for m in run.train_metrics]
            losses = [m.loss for m in run.train_metrics]
            fig.add_trace(go.Scatter(
                x=steps,
                y=losses,
                mode='lines',
                name=f"{name} (train)",
                line=dict(color=train_color, width=2),
                hovertemplate="Step: %{x}<br>Loss: %{y:.4f}<extra></extra>",
            ))

        # Validation loss (solid line, markers) - different color
        if show_val and run.val_metrics:
            steps = [m.step for m in run.val_metrics]
            losses = [m.loss for m in run.val_metrics]
            fig.add_trace(go.Scatter(
                x=steps,
                y=losses,
                mode='lines+markers',
                name=f"{name} (val)",
                line=dict(color=val_color, width=2),
                marker=dict(size=8),
                hovertemplate="Step: %{x}<br>Val Loss: %{y:.4f}<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text="Loss Curves", y=0.95),
        xaxis_title="Step",
        yaxis_title="Loss",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=80),
        hovermode="x unified",
        template="plotly_white",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
    )

    return fig


def create_validation_curves(runs: Dict[str, TrainingRun]) -> go.Figure:
    """
    Create validation loss and perplexity curves with dual y-axis.

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        Plotly Figure with validation metrics on dual y-axes
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i, (name, run) in enumerate(runs.items()):
        if not run.val_metrics:
            continue

        color = RUN_COLORS[i % len(RUN_COLORS)]
        steps = [m.step for m in run.val_metrics]
        losses = [m.loss for m in run.val_metrics]
        perplexities = [m.perplexity for m in run.val_metrics]

        # Loss on primary y-axis (solid line)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=losses,
                mode='lines+markers',
                name=f"{name} (loss)",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate="Step: %{x}<br>Loss: %{y:.4f}<extra></extra>",
            ),
            secondary_y=False,
        )

        # Perplexity on secondary y-axis (dotted line)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=perplexities,
                mode='lines+markers',
                name=f"{name} (ppl)",
                line=dict(color=color, dash='dot', width=2),
                marker=dict(size=6, symbol='diamond'),
                hovertemplate="Step: %{x}<br>Perplexity: %{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=dict(text="Validation Metrics", y=0.95),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=80),
        hovermode="x unified",
        template="plotly_white",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
    )
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(
        title_text="Loss",
        secondary_y=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
    )
    fig.update_yaxes(
        title_text="Perplexity",
        secondary_y=True,
        title_font=dict(color="#666"),
        tickfont=dict(color="#666"),
    )

    return fig


def create_lr_schedule_plot(runs: Dict[str, TrainingRun]) -> go.Figure:
    """
    Create learning rate schedule visualization.

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        Plotly Figure with LR schedules (log scale y-axis)
    """
    fig = go.Figure()

    for i, (name, run) in enumerate(runs.items()):
        if not run.train_metrics:
            continue

        color = RUN_COLORS[i % len(RUN_COLORS)]
        steps = [m.step for m in run.train_metrics]
        lrs = [m.learning_rate for m in run.train_metrics]

        fig.add_trace(go.Scatter(
            x=steps,
            y=lrs,
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            hovertemplate="Step: %{x}<br>LR: %{y:.2e}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Learning Rate Schedule", y=0.95),
        xaxis_title="Step",
        yaxis_title="Learning Rate",
        yaxis_type="log",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=80),
        hovermode="x unified",
        template="plotly_white",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
    )

    return fig


def create_throughput_plot(runs: Dict[str, TrainingRun]) -> go.Figure:
    """
    Create tokens/second throughput comparison with EMA smoothing.

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        Plotly Figure with throughput over training
    """
    fig = go.Figure()

    for i, (name, run) in enumerate(runs.items()):
        if not run.train_metrics:
            continue

        color = RUN_COLORS[i % len(RUN_COLORS)]

        # Filter out None values for tokens_per_sec
        data = [(m.step, m.tokens_per_sec) for m in run.train_metrics
                if m.tokens_per_sec is not None]
        if not data:
            continue

        steps, tps = zip(*data)
        # Apply EMA smoothing to reduce checkpoint noise
        tps_smoothed = smooth_data(list(tps), alpha=0.1)

        fig.add_trace(go.Scatter(
            x=list(steps),
            y=tps_smoothed,
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            hovertemplate="Step: %{x}<br>Tok/s: %{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Training Throughput", y=0.95),
        xaxis_title="Step",
        yaxis_title="Tokens/Second",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=80),
        hovermode="x unified",
        template="plotly_white",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
    )

    return fig


def create_flops_normalized_loss(runs: Dict[str, TrainingRun]) -> go.Figure:
    """
    Create loss curve with compute (FLOPs) on x-axis.

    This enables fair comparison across models of different sizes.
    A smaller model trained longer may achieve the same loss with less compute.

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        Plotly Figure with loss vs cumulative TFLOPs
    """
    fig = go.Figure()

    for i, (name, run) in enumerate(runs.items()):
        if not run.train_metrics:
            continue

        train_color = RUN_COLORS_TRAIN[i % len(RUN_COLORS_TRAIN)]
        val_color = RUN_COLORS_VAL[i % len(RUN_COLORS_VAL)]

        # Get cumulative FLOPs and corresponding losses
        steps, tflops = compute_cumulative_flops(run)
        losses = [m.loss for m in run.train_metrics]

        if not tflops:
            continue

        # Training loss vs FLOPs
        fig.add_trace(go.Scatter(
            x=tflops,
            y=losses,
            mode='lines',
            name=f"{name} (train)",
            line=dict(color=train_color, width=2),
            hovertemplate="TFLOPs: %{x:.2f}<br>Loss: %{y:.4f}<extra></extra>",
        ))

        # Also plot validation points if available
        if run.val_metrics:
            val_steps = [m.step for m in run.val_metrics]
            val_losses = [m.loss for m in run.val_metrics]

            # Find corresponding FLOPs for validation steps
            step_to_flops = dict(zip(steps, tflops))
            val_flops = []
            val_losses_filtered = []
            for step, loss in zip(val_steps, val_losses):
                if step in step_to_flops:
                    val_flops.append(step_to_flops[step])
                    val_losses_filtered.append(loss)

            if val_flops:
                fig.add_trace(go.Scatter(
                    x=val_flops,
                    y=val_losses_filtered,
                    mode='lines',
                    name=f"{name} (val)",
                    line=dict(color=val_color, width=2),
                    hovertemplate="TFLOPs: %{x:.2f}<br>Val Loss: %{y:.4f}<extra></extra>",
                ))

    fig.update_layout(
        title=dict(text="Loss vs Compute (FLOPs-Normalized)", y=0.95),
        xaxis_title="Cumulative TFLOPs",
        yaxis_title="Loss (log scale)",
        yaxis_type="log",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=80),
        hovermode="x unified",
        template="plotly_white",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
    )

    return fig


def create_combined_steps_plot(
    runs: Dict[str, TrainingRun],
    show_train: bool = True,
    show_val: bool = True,
) -> go.Figure:
    """
    Create combined loss and perplexity plot with dual y-axis.

    X-axis: Training Steps
    Left Y-axis: Loss (training + validation)
    Right Y-axis: Perplexity (validation only)

    Args:
        runs: Dict of run_name -> TrainingRun
        show_train: Include training loss (solid lines)
        show_val: Include validation loss and perplexity (dashed/dotted)

    Returns:
        Plotly Figure with loss and perplexity curves
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i, (name, run) in enumerate(runs.items()):
        train_color = RUN_COLORS_TRAIN[i % len(RUN_COLORS_TRAIN)]
        val_color = RUN_COLORS_VAL[i % len(RUN_COLORS_VAL)]

        # Training loss (solid line, primary y-axis)
        if show_train and run.train_metrics:
            steps = [m.step for m in run.train_metrics]
            losses = [m.loss for m in run.train_metrics]
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=losses,
                    mode='lines',
                    name=f"{name} (train)",
                    line=dict(color=train_color, width=2),
                    hovertemplate="Step: %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
                ),
                secondary_y=False,
            )

        # Validation loss (solid line, primary y-axis) - different color
        if show_val and run.val_metrics:
            val_steps = [m.step for m in run.val_metrics]
            val_losses = [m.loss for m in run.val_metrics]
            fig.add_trace(
                go.Scatter(
                    x=val_steps,
                    y=val_losses,
                    mode='lines',
                    name=f"{name} (val)",
                    line=dict(color=val_color, width=2),
                    hovertemplate="Step: %{x}<br>Val Loss: %{y:.4f}<extra></extra>",
                ),
                secondary_y=False,
            )

            # Perplexity (dashed line, secondary y-axis) - same color as val
            perplexities = [m.perplexity for m in run.val_metrics]
            fig.add_trace(
                go.Scatter(
                    x=val_steps,
                    y=perplexities,
                    mode='lines',
                    name=f"{name} (ppl)",
                    line=dict(color=val_color, dash='dash', width=2),
                    hovertemplate="Step: %{x}<br>Perplexity: %{y:.2f}<extra></extra>",
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title=dict(text="Loss & Perplexity vs Training Steps", y=0.95),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=80),
        hovermode="x unified",
        template="plotly_white",
        height=400,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
    )
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(
        title_text="Loss (log scale)",
        type="log",
        secondary_y=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
    )
    fig.update_yaxes(
        title_text="Perplexity",
        secondary_y=True,
        title_font=dict(color="#666"),
        tickfont=dict(color="#666"),
    )

    return fig


def create_combined_flops_plot(runs: Dict[str, TrainingRun]) -> go.Figure:
    """
    Create loss vs cumulative FLOPs plot.

    Enables fair comparison across different model sizes by normalizing
    by compute instead of training steps.

    X-axis: Cumulative TFLOPs
    Y-axis: Loss (training + validation)

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        Plotly Figure with loss vs compute
    """
    fig = go.Figure()

    for i, (name, run) in enumerate(runs.items()):
        if not run.train_metrics:
            continue

        train_color = RUN_COLORS_TRAIN[i % len(RUN_COLORS_TRAIN)]
        val_color = RUN_COLORS_VAL[i % len(RUN_COLORS_VAL)]
        steps, tflops = compute_cumulative_flops(run)
        losses = [m.loss for m in run.train_metrics]

        if not tflops:
            continue

        # Training loss vs FLOPs (solid line)
        fig.add_trace(
            go.Scatter(
                x=tflops,
                y=losses,
                mode='lines',
                name=f"{name} (train)",
                line=dict(color=train_color, width=2),
                hovertemplate="TFLOPs: %{x:.2f}<br>Loss: %{y:.4f}<extra></extra>",
            ),
        )

        # Validation loss vs FLOPs
        if run.val_metrics:
            val_steps = [m.step for m in run.val_metrics]
            val_losses = [m.loss for m in run.val_metrics]

            step_to_flops = dict(zip(steps, tflops))
            val_flops, val_losses_f = [], []
            for step, loss in zip(val_steps, val_losses):
                if step in step_to_flops:
                    val_flops.append(step_to_flops[step])
                    val_losses_f.append(loss)

            if val_flops:
                fig.add_trace(
                    go.Scatter(
                        x=val_flops,
                        y=val_losses_f,
                        mode='lines',
                        name=f"{name} (val)",
                        line=dict(color=val_color, width=2),
                        hovertemplate="TFLOPs: %{x:.2f}<br>Val Loss: %{y:.4f}<extra></extra>",
                    ),
                )

    fig.update_layout(
        title=dict(text="Loss vs Compute (FLOPs)", y=0.95),
        xaxis_title="Cumulative TFLOPs",
        yaxis_title="Loss (log scale)",
        yaxis_type="log",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=100),
        hovermode="x unified",
        template="plotly_white",
        height=420,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
    )

    # Add auto-detected iso-loss reference lines
    thresholds = compute_iso_loss_thresholds(runs)
    for threshold in thresholds:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="rgba(128, 128, 128, 0.5)",
            line_width=1,
            annotation_text=f"{threshold:.1f}",
            annotation_position="right",
            annotation_font_size=9,
            annotation_font_color="gray",
        )

    return fig


def create_compute_efficiency_comparison(
    runs: Dict[str, TrainingRun],
    budget_tflops: Optional[float] = None,
) -> go.Figure:
    """
    Compare validation loss at equal compute budgets.

    Interpolates all runs to common FLOPs points for fair comparison.
    Optionally highlights a specific compute budget with annotations.

    Args:
        runs: Dict of run_name -> TrainingRun
        budget_tflops: Optional compute budget to highlight (in TFLOPs)

    Returns:
        Plotly Figure with interpolated loss curves
    """
    fig = go.Figure()

    # Collect data from all runs
    run_data = {}
    for name, run in runs.items():
        flops, losses = get_val_flops_data(run)
        if len(flops) > 0:
            run_data[name] = (flops, losses)

    if not run_data:
        return create_empty_figure("No validation data available")

    # Find common compute range (overlap region)
    min_flops = max(flops.min() for flops, _ in run_data.values())
    max_flops = min(flops.max() for flops, _ in run_data.values())

    if min_flops >= max_flops:
        # No overlap - show full range with note
        min_flops = min(flops.min() for flops, _ in run_data.values())
        max_flops = max(flops.max() for flops, _ in run_data.values())

    # Create common FLOPs grid for interpolation
    common_flops = np.linspace(min_flops, max_flops, 100)

    for i, (name, (flops, losses)) in enumerate(run_data.items()):
        color = RUN_COLORS[i % len(RUN_COLORS)]

        # Interpolate to common FLOPs points
        interpolated_loss = np.interp(
            common_flops, flops, losses,
            left=np.nan, right=np.nan
        )

        fig.add_trace(go.Scatter(
            x=common_flops,
            y=interpolated_loss,
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            hovertemplate="TFLOPs: %{x:.1f}<br>Val Loss: %{y:.4f}<extra></extra>",
        ))

    # Add vertical lines at round compute budgets
    for budget in [100, 200, 300, 400, 500, 600, 700, 800]:
        if min_flops < budget < max_flops:
            fig.add_vline(
                x=budget,
                line_dash="dash",
                line_color="rgba(128,128,128,0.3)",
                annotation_text=f"{budget} TFLOPs",
                annotation_position="top",
                annotation_font_size=9,
            )

    # Highlight selected budget with loss annotations
    if budget_tflops is not None and min_flops <= budget_tflops <= max_flops:
        fig.add_vline(
            x=budget_tflops,
            line_color="green",
            line_width=2,
        )
        # Add annotations for each run's loss at this budget
        for i, (name, (flops, losses)) in enumerate(run_data.items()):
            loss_at_budget = np.interp(budget_tflops, flops, losses)
            if not np.isnan(loss_at_budget):
                fig.add_annotation(
                    x=budget_tflops,
                    y=loss_at_budget,
                    text=f"{name}: {loss_at_budget:.4f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=40 + i * 20,
                    ay=-20 - i * 15,
                )

    fig.update_layout(
        title=dict(text="Validation Loss at Equal Compute Budgets", y=0.95),
        xaxis_title="Cumulative TFLOPs",
        yaxis_title="Validation Loss (log scale)",
        yaxis_type="log",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=100),
        hovermode="x unified",
        template="plotly_white",
        height=420,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
    )

    return fig


def create_efficiency_curve(runs: Dict[str, TrainingRun]) -> go.Figure:
    """
    Plot compute efficiency: loss improvement per FLOP.

    Shows -dLoss/dFLOPs to visualize diminishing returns as training progresses.
    Higher values indicate more efficient learning (more loss reduction per FLOP).

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        Plotly Figure with efficiency curves
    """
    fig = go.Figure()

    for i, (name, run) in enumerate(runs.items()):
        flops, losses = get_val_flops_data(run)
        if len(flops) < 3:
            continue

        color = RUN_COLORS[i % len(RUN_COLORS)]

        # Calculate derivative: -d(loss) / d(flops)
        # Use validation loss for cleaner signal
        dloss = np.diff(losses)
        dflops = np.diff(flops)

        # Avoid division by zero
        dflops = np.where(dflops == 0, 1e-10, dflops)
        efficiency = -dloss / dflops  # Negative because loss decreases

        # Clip negative values (where loss increased)
        efficiency = np.clip(efficiency, 1e-10, None)

        # Apply smoothing
        efficiency_smoothed = smooth_data(efficiency.tolist(), alpha=0.3)

        # X values are midpoints between FLOPs samples
        x_vals = (flops[:-1] + flops[1:]) / 2

        fig.add_trace(go.Scatter(
            x=x_vals.tolist(),
            y=efficiency_smoothed,
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            hovertemplate="TFLOPs: %{x:.1f}<br>Efficiency: %{y:.2e}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Compute Efficiency (Loss Improvement per TFLOP)", y=0.95),
        xaxis_title="Cumulative TFLOPs",
        yaxis_title="-dLoss/dTFLOPs (log scale)",
        yaxis_type="log",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=100),
        hovermode="x unified",
        template="plotly_white",
        height=420,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
    )

    return fig


def find_crossover_points(
    runs: Dict[str, TrainingRun],
) -> List[Dict[str, Any]]:
    """
    Find where larger model becomes more compute-efficient.

    Compares pairs of runs to find loss values where the larger model
    achieves the same loss with less compute.

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        List of dicts with crossover info: {loss, flops, smaller, larger}
    """
    if len(runs) < 2:
        return []

    # Collect data and sort by parameter count
    run_info = []
    for name, run in runs.items():
        flops, losses = get_val_flops_data(run)
        if len(flops) == 0:
            continue

        # Get parameter count from config
        params = run.model_config.get('d_model', 0) * run.model_config.get('n_blocks', 0)
        run_info.append({
            'name': name,
            'params': params,
            'flops': flops,
            'losses': losses,
        })

    if len(run_info) < 2:
        return []

    # Sort by params (smallest first)
    run_info.sort(key=lambda x: x['params'])

    crossovers = []

    # Compare consecutive pairs
    for j in range(len(run_info) - 1):
        small = run_info[j]
        large = run_info[j + 1]

        # Find common loss range
        min_loss = max(small['losses'].min(), large['losses'].min())
        max_loss = min(small['losses'].max(), large['losses'].max())

        if min_loss >= max_loss:
            continue

        # Create common loss grid
        common_losses = np.linspace(min_loss, max_loss, 50)

        # Interpolate FLOPs required to reach each loss
        # Note: losses decrease, so we reverse for interp
        small_sorted_idx = np.argsort(small['losses'])[::-1]
        large_sorted_idx = np.argsort(large['losses'])[::-1]

        flops_small = np.interp(
            common_losses,
            small['losses'][small_sorted_idx],
            small['flops'][small_sorted_idx],
        )
        flops_large = np.interp(
            common_losses,
            large['losses'][large_sorted_idx],
            large['flops'][large_sorted_idx],
        )

        # Find where larger model becomes more efficient (uses less FLOPs)
        diff = flops_small - flops_large
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        for idx in sign_changes:
            crossovers.append({
                'loss': common_losses[idx],
                'flops_small': flops_small[idx],
                'flops_large': flops_large[idx],
                'smaller': small['name'],
                'larger': large['name'],
            })

    return crossovers


def create_empty_figure(title: str = "No data selected") -> go.Figure:
    """Create an empty placeholder figure."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        annotations=[
            dict(
                text="Select training runs to visualize",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray"),
            )
        ],
    )
    return fig
