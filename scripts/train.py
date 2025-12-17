#!/usr/bin/env python3
"""Unified training script for transformer models.

Supports both TorchTransformer (PyTorch autograd) and CustomTransformer (manual backprop).

Features:
- Multi-dataset support via dataset registry
- Custom tokenizer support (BPE, GPT-2, etc.)
- bfloat16 training with mixed precision
- Checkpoint saving with training resumption
- Learning rate scheduling (warmup, cosine/linear decay)
- JSON-compatible logging for visualization

Usage:
    # Train TorchTransformer on TinyStories
    python train.py --config configs/torch_tinystories.yaml --model-type torch

    # Train CustomTransformer with manual backprop
    python train.py --config configs/custom_tinystories.yaml --model-type custom

    # Resume from checkpoint
    python train.py --config configs/torch_tinystories.yaml --model-type torch --resume

    # Resume from specific checkpoint
    python train.py --config configs/torch_tinystories.yaml --model-type torch --resume checkpoints/best.pt
"""

import sys
from pathlib import Path
import math
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
from tqdm import tqdm
import argparse
import time
from typing import Optional, Dict, Any
from functools import partial

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import TorchTransformer, CustomTransformerWrapper
from tools import load_training_data, get_datasets_dir, get_models_dir
from training import CheckpointManager, Evaluator, TrainingLogger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerFast:
    """Load tokenizer from various asset locations.

    Args:
        tokenizer_name: Either 'gpt2' or path to custom tokenizer

    Returns:
        Loaded tokenizer with pad_token set
    """
    if tokenizer_name == 'gpt2':
        print("  Using GPT-2 tokenizer (vocab_size=50257)")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # Search paths for custom tokenizers
    search_paths = [
        get_models_dir() / 'tokenizers' / tokenizer_name,
        get_datasets_dir() / 'tokenizers' / tokenizer_name,
        get_models_dir() / tokenizer_name,
        Path(tokenizer_name),  # Absolute/relative path fallback
    ]

    tokenizer_path = None
    for path in search_paths:
        if path.exists():
            tokenizer_path = path
            break

    if tokenizer_path is None:
        raise FileNotFoundError(
            f"Tokenizer '{tokenizer_name}' not found. Searched:\n" +
            "\n".join(f"  - {p}" for p in search_paths)
        )

    print(f"  Loading tokenizer from: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})

    print(f"  Vocab size: {len(tokenizer)}")
    return tokenizer


def dynamic_pad_collate(batch: list, pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collate function that pads to max length in batch.

    Reduces wasted compute on padding by trimming to longest actual sequence.
    """
    lengths = []
    for item in batch:
        input_ids = item['input_ids']
        non_pad_mask = input_ids != pad_token_id
        if non_pad_mask.any():
            last_non_pad = non_pad_mask.nonzero()[-1].item() + 1
        else:
            last_non_pad = 1
        lengths.append(last_non_pad)

    max_len = max(lengths)

    input_ids = torch.stack([item['input_ids'][:max_len] for item in batch])
    labels = torch.stack([item['labels'][:max_len] for item in batch])

    return {'input_ids': input_ids, 'labels': labels}


def compute_learning_rate(
    global_step: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
    lr_decay: Optional[str],
) -> float:
    """Compute learning rate with warmup and optional decay."""
    if global_step < warmup_steps:
        return base_lr * (global_step / warmup_steps) if warmup_steps > 0 else base_lr

    if lr_decay == 'cosine':
        decay_steps = total_steps - warmup_steps
        decay_progress = (global_step - warmup_steps) / decay_steps if decay_steps > 0 else 1.0
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * decay_progress))

    if lr_decay == 'linear':
        decay_steps = total_steps - warmup_steps
        decay_progress = (global_step - warmup_steps) / decay_steps if decay_steps > 0 else 1.0
        return base_lr - (base_lr - min_lr) * decay_progress

    return base_lr


def train_torch_model(
    model: TorchTransformer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: dict,
    checkpoint_manager: CheckpointManager,
    logger: TrainingLogger,
    evaluator: Evaluator,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
    start_epoch: int = 0,
    global_step: int = 0,
    best_val_loss: float = float('inf'),
) -> None:
    """Train TorchTransformer using PyTorch autograd."""
    train_config = config['training']
    num_epochs = train_config['num_epochs']
    base_lr = train_config['learning_rate']
    min_lr = train_config.get('min_learning_rate', base_lr * 0.1)
    lr_decay = train_config.get('lr_decay', None)
    max_grad_norm = train_config.get('max_grad_norm', 1.0)
    eval_every = train_config.get('eval_every', 500)

    total_steps = num_epochs * len(train_loader)
    warmup_ratio = train_config.get('warmup_ratio', 0.0)
    warmup_steps = train_config.get('warmup_steps', int(total_steps * warmup_ratio))

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=train_config.get('weight_decay', 0.01),
        betas=(0.9, 0.95),
    )

    # Restore optimizer state if resuming
    if global_step > 0:
        latest = checkpoint_manager.get_latest()
        if latest:
            checkpoint_manager.load(optimizer=optimizer)

    # Parse dtype
    dtype_str = config['model'].get('dtype', 'bfloat16')
    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    model_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    use_amp = model_dtype == torch.bfloat16 and device.type == 'cuda'

    pad_token_id = tokenizer.pad_token_id
    total_tokens = 0

    print(f"\nTraining TorchTransformer for epochs {start_epoch + 1} to {num_epochs}...")
    if warmup_steps > 0:
        print(f"  Warmup: {warmup_steps} steps")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")

        epoch_losses = []
        epoch_start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        model.train()

        for step, batch in enumerate(progress_bar):
            step_start_time = time.time()

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Mask padding tokens
            if pad_token_id is not None:
                labels = labels.clone()
                labels[labels == pad_token_id] = -100

            # Compute learning rate
            learning_rate = compute_learning_rate(
                global_step, base_lr, min_lr, warmup_steps, total_steps, lr_decay
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # Forward pass
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda', dtype=model_dtype):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs['loss']
            else:
                outputs = model(input_ids, labels=labels)
                loss = outputs['loss']

            # Backward and step
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Metrics
            batch_tokens = input_ids.numel()
            total_tokens += batch_tokens
            batch_time = time.time() - step_start_time
            tokens_per_sec = batch_tokens / batch_time

            global_step += 1
            epoch_losses.append(loss.item())

            # Log step
            logger.log_step(
                step=global_step,
                epoch=epoch,
                loss=loss.item(),
                learning_rate=learning_rate,
                grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                tokens_processed=total_tokens,
            )

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{learning_rate:.2e}",
                'ktok/s': f"{tokens_per_sec / 1e3:.1f}",
            })

            # Periodic evaluation
            if global_step % eval_every == 0 and val_loader:
                eval_result = evaluator.evaluate(val_loader, max_batches=50)
                logger.log_validation(global_step, epoch, eval_result.loss, eval_result.perplexity)

        # End of epoch
        epoch_time = time.time() - epoch_start_time
        train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')

        print(f"\nEpoch {epoch + 1} complete in {epoch_time:.1f}s")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Tokens processed: {total_tokens / 1e6:.1f}M")

        # Full validation
        val_loss = float('inf')
        if val_loader:
            print("  Running full evaluation...")
            eval_result = evaluator.evaluate(val_loader)
            val_loss = eval_result.loss
            print(f"  Val loss: {val_loss:.4f}")
            print(f"  Val perplexity: {eval_result.perplexity:.2f}")
            logger.log_validation(global_step, epoch, val_loss, eval_result.perplexity)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print("  New best val loss!")

        checkpoint_manager.save(
            epoch=epoch + 1,
            global_step=global_step,
            train_config=train_config,
            learning_rate=learning_rate,
            optimizer=optimizer,
            metrics={'train_loss': train_loss, 'val_loss': val_loss},
            is_best=is_best,
        )

    logger.finish()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


def train_custom_model(
    model: CustomTransformerWrapper,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: dict,
    checkpoint_manager: CheckpointManager,
    logger: TrainingLogger,
    evaluator: Evaluator,
    start_epoch: int = 0,
    global_step: int = 0,
    best_val_loss: float = float('inf'),
) -> None:
    """Train CustomTransformer using manual backprop via wrapper."""
    train_config = config['training']
    num_epochs = train_config['num_epochs']
    base_lr = train_config['learning_rate']
    min_lr = train_config.get('min_learning_rate', base_lr * 0.1)
    lr_decay = train_config.get('lr_decay', None)
    max_grad_norm = train_config.get('max_grad_norm', 1.0)
    eval_every = train_config.get('eval_every', 500)
    max_nan_count = train_config.get('max_nan_count', 10)

    total_steps = num_epochs * len(train_loader)
    warmup_ratio = train_config.get('warmup_ratio', 0.0)
    warmup_steps = train_config.get('warmup_steps', int(total_steps * warmup_ratio))

    total_tokens = 0
    nan_count = 0

    print(f"\nTraining CustomTransformer for epochs {start_epoch + 1} to {num_epochs}...")
    if warmup_steps > 0:
        print(f"  Warmup: {warmup_steps} steps")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")

        epoch_losses = []
        epoch_start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

        for step, batch in enumerate(progress_bar):
            step_start_time = time.time()

            input_ids = batch['input_ids']
            labels = batch['labels']

            # Compute learning rate
            learning_rate = compute_learning_rate(
                global_step, base_lr, min_lr, warmup_steps, total_steps, lr_decay
            )

            # Training step via wrapper
            result = model.train_step(
                input_ids,
                labels,
                learning_rate,
                max_grad_norm=max_grad_norm,
            )

            # Handle NaN detection
            if result.get('status') in ['nan_detected', 'nan_gradient']:
                nan_count += 1
                print(f"\n  NaN detected at step {global_step}. Count: {nan_count}/{max_nan_count}")
                if nan_count >= max_nan_count:
                    print("  Too many NaNs. Halting training.")
                    logger.finish()
                    return
                continue

            # Metrics
            batch_tokens = input_ids.numel()
            total_tokens += batch_tokens
            batch_time = time.time() - step_start_time
            tokens_per_sec = batch_tokens / batch_time

            global_step += 1
            epoch_losses.append(result['loss'])

            # Log step
            logger.log_step(
                step=global_step,
                epoch=epoch,
                loss=result['loss'],
                learning_rate=learning_rate,
                grad_norm=result.get('grad_norm'),
                tokens_processed=total_tokens,
            )

            progress_bar.set_postfix({
                'loss': f"{result['loss']:.4f}",
                'grad': f"{result.get('grad_norm', 0):.2f}",
            })

            # Periodic evaluation
            if global_step % eval_every == 0 and val_loader:
                eval_result = evaluator.evaluate(val_loader, max_batches=50)
                logger.log_validation(global_step, epoch, eval_result.loss, eval_result.perplexity)

        # End of epoch
        epoch_time = time.time() - epoch_start_time
        train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')

        print(f"\nEpoch {epoch + 1} complete in {epoch_time:.1f}s")
        print(f"  Train loss: {train_loss:.4f}")

        # Full validation
        val_loss = float('inf')
        if val_loader:
            print("  Running full evaluation...")
            eval_result = evaluator.evaluate(val_loader)
            val_loss = eval_result.loss
            print(f"  Val loss: {val_loss:.4f}")
            print(f"  Val perplexity: {eval_result.perplexity:.2f}")
            logger.log_validation(global_step, epoch, val_loss, eval_result.perplexity)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print("  New best val loss!")

        checkpoint_manager.save(
            epoch=epoch + 1,
            global_step=global_step,
            train_config=train_config,
            learning_rate=learning_rate,
            metrics={'train_loss': train_loss, 'val_loss': val_loss},
            is_best=is_best,
        )

    logger.finish()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"  NaN occurrences: {nan_count}")


def main():
    parser = argparse.ArgumentParser(description='Train transformer models')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--model-type', type=str, choices=['torch', 'custom'], required=True,
                       help='Model type: torch (TorchTransformer) or custom (CustomTransformer)')
    parser.add_argument('--resume', nargs='?', const='latest', default=None,
                       help='Resume from checkpoint. Use without value for latest, or specify path.')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    experiment_name = config.get('experiment_name', f'{args.model_type}_experiment')

    print("="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Model Type: {args.model_type}")
    print("="*60)
    print("\nConfiguration:")
    print(yaml.dump(config, default_flow_style=False))

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer_name = config.get('data', {}).get('tokenizer', 'gpt2')
    tokenizer = load_tokenizer(tokenizer_name)

    # Load data
    print(f"\nLoading dataset: {config['data']['dataset']}")
    train_loader, val_loader = load_training_data(
        config['data']['dataset'],
        tokenizer,
        max_length=config['data']['max_length'],
        batch_size=config['training']['batch_size'],
        subset_size=config['data'].get('subset_size'),
        val_subset_size=config['data'].get('val_subset_size'),
    )

    # Apply dynamic padding collate
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=partial(dynamic_pad_collate, pad_token_id=tokenizer.pad_token_id),
    )
    if val_loader:
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=partial(dynamic_pad_collate, pad_token_id=tokenizer.pad_token_id),
        )

    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")

    # Parse dtype
    dtype_str = config['model'].get('dtype', 'bfloat16')
    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    model_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # Create model based on type
    print(f"\nInitializing {args.model_type} model...")
    model_config = {
        'vocab_size': len(tokenizer),
        'd_model': config['model']['d_model'],
        'n_heads': config['model']['n_heads'],
        'n_blocks': config['model']['n_blocks'],
        'd_ffn': config['model']['d_ffn'],
        'max_seq_len': config['model']['max_seq_len'],
    }

    if args.model_type == 'torch':
        model = TorchTransformer(model_config)
        model = model.to(device=device, dtype=model_dtype)

        # Compile model for faster training (PyTorch 2.0+)
        if hasattr(torch, 'compile') and device.type == 'cuda':
            print("  Compiling model with torch.compile()...")
            model = torch.compile(model)

        info = model.get_model_info()
    else:
        model = CustomTransformerWrapper(
            vocab_size=len(tokenizer),
            max_seq_len=config['model']['max_seq_len'],
            n_blocks=config['model']['n_blocks'],
            n_heads=config['model']['n_heads'],
            d_model=config['model']['d_model'],
            d_ffn=config['model']['d_ffn'],
            dtype=model_dtype,
            pad_token_id=tokenizer.pad_token_id,
        )
        info = {'parameters': model.count_parameters(), 'parameters_millions': model.count_parameters() / 1e6}

    print(f"  Parameters: {info['parameters']:,} ({info.get('parameters_millions', info['parameters']/1e6):.1f}M)")
    print(f"  Dtype: {model_dtype}")

    # Setup checkpoint manager
    checkpoint_dir = PROJECT_ROOT / 'assets' / 'models' / experiment_name
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        model=model,
        experiment_name=experiment_name,
        max_checkpoints=5,
    )

    # Setup logger
    log_dir = PROJECT_ROOT / 'assets' / 'logs'
    logger = TrainingLogger(
        log_dir=str(log_dir),
        experiment_name=experiment_name,
        model_type=args.model_type,
        model_config=config['model'],
        train_config=config['training'],
    )

    # Setup evaluator
    evaluator = Evaluator(
        model=model,
        device=str(device),
        vocab_size=len(tokenizer),
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint_path = None if args.resume == 'latest' else args.resume
        resume_state = checkpoint_manager.load(checkpoint_path=checkpoint_path)
        if resume_state:
            start_epoch = resume_state['epoch']
            global_step = resume_state['global_step']
            best_val_loss = resume_state['metrics'].get('val_loss', float('inf'))
            print(f"\nResumed from epoch {start_epoch}, step {global_step}")
            print(f"  Best val loss: {best_val_loss:.4f}")

    # Train based on model type
    if args.model_type == 'torch':
        train_torch_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            evaluator=evaluator,
            tokenizer=tokenizer,
            device=device,
            start_epoch=start_epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
        )
    else:
        train_custom_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            evaluator=evaluator,
            start_epoch=start_epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
        )


if __name__ == '__main__':
    main()
