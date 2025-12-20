"""FLOPs estimation and compute metrics for training visualization."""

from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training import TrainingRun


def estimate_model_params(model_config: Dict[str, Any], vocab_size: int = 50257) -> int:
    """
    Estimate parameter count from model configuration.

    Components:
    - Token embeddings: vocab_size * d_model
    - Position embeddings: max_seq_len * d_model
    - Per transformer block:
        - Attention: 4 * d_model^2 (Q, K, V, O projections)
        - FFN: 2 * d_model * d_ffn
        - Layer norms: 4 * d_model (2 norms per block, each with weight + bias)
    - Final layer norm: 2 * d_model
    - Output projection: vocab_size * d_model (often tied with embeddings)

    Args:
        model_config: Dict with d_model, n_heads, n_blocks, d_ffn, max_seq_len
        vocab_size: Vocabulary size (default: GPT-2's 50257)

    Returns:
        Estimated parameter count
    """
    d_model = model_config.get('d_model', 256)
    n_blocks = model_config.get('n_blocks', 6)
    d_ffn = model_config.get('d_ffn', d_model * 4)
    max_seq_len = model_config.get('max_seq_len', 256)

    # Embeddings
    token_emb = vocab_size * d_model
    pos_emb = max_seq_len * d_model

    # Per-block parameters
    attention_params = 4 * d_model * d_model  # Q, K, V, O projections
    ffn_params = 2 * d_model * d_ffn  # Up and down projections
    layer_norm_params = 4 * d_model  # 2 layer norms * (weight + bias)
    block_params = attention_params + ffn_params + layer_norm_params

    # All blocks
    total_block_params = n_blocks * block_params

    # Final layer norm
    final_ln = 2 * d_model

    # Output projection (often tied with token embeddings, so we don't count it separately)
    # If untied: output_proj = vocab_size * d_model

    total = token_emb + pos_emb + total_block_params + final_ln

    return total


def compute_tokens_per_step(train_config: Dict[str, Any], model_config: Dict[str, Any]) -> int:
    """
    Compute tokens processed per training step.

    Args:
        train_config: Training configuration with batch_size
        model_config: Model configuration with max_seq_len

    Returns:
        Tokens per step
    """
    batch_size = train_config.get('batch_size', 8)
    max_seq_len = model_config.get('max_seq_len', 256)
    return batch_size * max_seq_len


def compute_cumulative_flops(run: TrainingRun) -> Tuple[List[int], List[float]]:
    """
    Compute cumulative FLOPs at each training step.

    Uses the Chinchilla approximation:
        FLOPs ≈ 6 * N * D

    Where:
        N = number of parameters
        D = number of tokens processed

    The factor of 6 accounts for:
        - ~2N FLOPs for forward pass
        - ~4N FLOPs for backward pass (gradients + optimizer)

    Args:
        run: TrainingRun object with model_config, train_config, and train_metrics

    Returns:
        Tuple of (steps, flops_in_tflops) - cumulative TFLOPs at each step
    """
    if not run.train_metrics:
        return [], []

    # Estimate parameters
    params = estimate_model_params(run.model_config)

    # Tokens per step
    tokens_per_step = compute_tokens_per_step(run.train_config, run.model_config)

    steps = []
    cumulative_flops = []

    for metric in run.train_metrics:
        step = metric.step
        cumulative_tokens = step * tokens_per_step
        flops = 6 * params * cumulative_tokens
        tflops = flops / 1e12  # Convert to TFLOPs

        steps.append(step)
        cumulative_flops.append(tflops)

    return steps, cumulative_flops


def compute_total_flops(run: TrainingRun) -> float:
    """
    Compute total FLOPs for a training run.

    Args:
        run: TrainingRun object

    Returns:
        Total TFLOPs for the training run
    """
    if not run.train_metrics:
        return 0.0

    params = estimate_model_params(run.model_config)
    tokens_per_step = compute_tokens_per_step(run.train_config, run.model_config)
    total_steps = len(run.train_metrics)
    total_tokens = total_steps * tokens_per_step
    total_flops = 6 * params * total_tokens

    return total_flops / 1e12  # TFLOPs


def format_params(params: int) -> str:
    """Format parameter count for display."""
    if params >= 1e9:
        return f"{params / 1e9:.1f}B"
    elif params >= 1e6:
        return f"{params / 1e6:.1f}M"
    elif params >= 1e3:
        return f"{params / 1e3:.1f}K"
    return str(params)


def format_flops(tflops: float) -> str:
    """Format TFLOPs for display."""
    if tflops >= 1000:
        return f"{tflops / 1000:.2f} PFLOPs"
    elif tflops >= 1:
        return f"{tflops:.2f} TFLOPs"
    else:
        return f"{tflops * 1000:.2f} GFLOPs"
