"""Reusable Plotly plotting functions for training visualization."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict
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
    Create loss and perplexity vs cumulative FLOPs plot.

    Enables fair comparison across different model sizes by normalizing
    by compute instead of training steps.

    X-axis: Cumulative TFLOPs
    Left Y-axis: Loss (training + validation)
    Right Y-axis: Perplexity (validation only)

    Args:
        runs: Dict of run_name -> TrainingRun

    Returns:
        Plotly Figure with loss and perplexity vs compute
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

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
            secondary_y=False,
        )

        # Validation loss and perplexity vs FLOPs
        if run.val_metrics:
            val_steps = [m.step for m in run.val_metrics]
            val_losses = [m.loss for m in run.val_metrics]
            perplexities = [m.perplexity for m in run.val_metrics]

            step_to_flops = dict(zip(steps, tflops))
            val_flops, val_losses_f, val_ppl_f = [], [], []
            for step, loss, ppl in zip(val_steps, val_losses, perplexities):
                if step in step_to_flops:
                    val_flops.append(step_to_flops[step])
                    val_losses_f.append(loss)
                    val_ppl_f.append(ppl)

            if val_flops:
                # Val loss (line) - different color
                fig.add_trace(
                    go.Scatter(
                        x=val_flops,
                        y=val_losses_f,
                        mode='lines',
                        name=f"{name} (val)",
                        line=dict(color=val_color, width=2),
                        hovertemplate="TFLOPs: %{x:.2f}<br>Val Loss: %{y:.4f}<extra></extra>",
                    ),
                    secondary_y=False,
                )

                # Perplexity (dashed line, secondary y-axis) - same as val
                fig.add_trace(
                    go.Scatter(
                        x=val_flops,
                        y=val_ppl_f,
                        mode='lines',
                        name=f"{name} (ppl)",
                        line=dict(color=val_color, width=2, dash='dash'),
                        hovertemplate="TFLOPs: %{x:.2f}<br>Perplexity: %{y:.2f}<extra></extra>",
                    ),
                    secondary_y=True,
                )

    fig.update_layout(
        title=dict(text="Loss & Perplexity vs Compute (FLOPs)", y=0.95),
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
    fig.update_xaxes(title_text="Cumulative TFLOPs")
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
