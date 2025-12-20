#!/usr/bin/env python3
"""Validate trained models with evaluation metrics and interactive chat mode.

Supports both TorchTransformer and CustomTransformer models.

Usage:
    # Evaluate on validation set
    python validate.py --checkpoint assets/models/test_torch/best.pt --config configs/test_torch.yaml --eval

    # Interactive chat mode
    python validate.py --checkpoint assets/models/test_torch/best.pt --config configs/test_torch.yaml --chat

    # Single prompt generation
    python validate.py --checkpoint assets/models/test_torch/best.pt --config configs/test_torch.yaml --prompt "Once upon a time"

    # CustomTransformer
    python validate.py --checkpoint assets/models/test_custom/best.pt --config configs/test_custom.yaml --model-type custom --chat
"""

import argparse
import torch
import yaml
from pathlib import Path
import sys
from typing import Optional, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import TorchTransformer, CustomTransformerWrapper
from tools import load_training_data, get_models_dir, get_datasets_dir
from training import Evaluator
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast


def load_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerFast:
    """Load tokenizer from various asset locations."""
    if tokenizer_name == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    search_paths = [
        get_models_dir() / 'tokenizers' / tokenizer_name,
        get_datasets_dir() / 'tokenizers' / tokenizer_name,
        get_models_dir() / tokenizer_name,
        Path(tokenizer_name),
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

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '<pad>'
    return tokenizer


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    model_type: str,
    device: torch.device,
    tokenizer_override: Optional[str] = None,
) -> Tuple[Any, PreTrainedTokenizerFast, dict]:
    """Load model and tokenizer from checkpoint + config.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        config_path: Path to YAML config file
        model_type: 'torch' or 'custom'
        device: torch device
        tokenizer_override: Optional tokenizer name to override config

    Returns:
        Tuple of (model, tokenizer, checkpoint_dict)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get tokenizer
    tokenizer_name = tokenizer_override or config['data'].get('tokenizer', 'gpt2')
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = load_tokenizer(tokenizer_name)

    # Parse dtype
    dtype_str = config['model'].get('dtype', 'bfloat16')
    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    model_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # Build model config
    model_config = {
        'vocab_size': len(tokenizer),
        'd_model': config['model']['d_model'],
        'n_heads': config['model']['n_heads'],
        'n_blocks': config['model']['n_blocks'],
        'd_ffn': config['model']['d_ffn'],
        'max_seq_len': config['model']['max_seq_len'],
    }

    # Create model based on type
    if model_type == 'torch':
        model = TorchTransformer(model_config)

        # Handle state dict from torch.compile() wrapped models
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("  Detected torch.compile() checkpoint, stripping '_orig_mod.' prefix...")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model = model.to(device=device, dtype=model_dtype)
        model.eval()
    else:
        # CustomTransformer
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

        # Load weights
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model_weights')
        if state_dict:
            model.load_state_dict(state_dict)

    return model, tokenizer, checkpoint


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = 'cuda',
    model_type: str = 'torch',
) -> str:
    """Generate text continuation from a prompt using autoregressive sampling."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    if model_type == 'torch':
        input_ids = input_ids.to(device)
        generated = input_ids

        max_seq_len = getattr(model, 'max_seq_len', 256)

        for _ in range(max_length):
            outputs = model(generated)
            logits = outputs['logits']

            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
            if generated.size(1) >= max_seq_len:
                break

        return tokenizer.decode(generated[0], skip_special_tokens=True)
    else:
        # CustomTransformer - use generate method if available
        if hasattr(model, 'generate'):
            return model.generate(
                prompt,
                tokenizer,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
            )
        else:
            # Manual generation for custom model
            generated = input_ids.numpy()
            import numpy as np

            max_seq_len = getattr(model.model, 'max_seq_len', 256) if hasattr(model, 'model') else 256

            for _ in range(max_length):
                logits = model.model.forward(generated)
                next_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    top_k_vals = np.partition(next_logits[0], -top_k)[-top_k]
                    next_logits[next_logits < top_k_vals.min()] = float('-inf')

                exp_logits = np.exp(next_logits - next_logits.max())
                probs = exp_logits / exp_logits.sum()
                next_token = np.random.choice(len(probs[0]), p=probs[0])

                generated = np.concatenate([generated, [[next_token]]], axis=1)

                if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
                    break
                if generated.shape[1] >= max_seq_len:
                    break

            return tokenizer.decode(generated[0], skip_special_tokens=True)


def evaluate_model(
    model,
    tokenizer,
    config: dict,
    device: torch.device,
    model_type: str,
    max_batches: Optional[int] = None,
):
    """Evaluate model on validation set."""
    print("\nLoading validation data...")

    _, val_loader = load_training_data(
        config['data']['dataset'],
        tokenizer,
        max_length=config['data']['max_length'],
        batch_size=config['training'].get('batch_size', 8),
        subset_size=config['data'].get('subset_size'),
        val_subset_size=config['data'].get('val_subset_size'),
    )

    if val_loader is None:
        print("No validation data available")
        return

    print(f"Evaluating on {len(val_loader)} validation batches...")

    evaluator = Evaluator(
        model=model,
        device=str(device),
        vocab_size=len(tokenizer),
    )

    result = evaluator.evaluate(val_loader, max_batches=max_batches)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Loss:       {result.loss:.4f}")
    print(f"  Perplexity: {result.perplexity:.2f}")
    print(f"  Tokens:     {result.num_tokens:,}")
    print(f"  Batches:    {result.num_batches}")
    print("=" * 50)


def chat_mode(
    model,
    tokenizer,
    device,
    model_type: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
):
    """Interactive chat loop."""
    print("\n" + "=" * 50)
    print("CHAT MODE")
    print("=" * 50)
    print("Type your prompt and press Enter.")
    print("Commands:")
    print("  quit/exit/q  - Exit chat")
    print("  temp X       - Set temperature (e.g., temp 0.5)")
    print("  topk X       - Set top-k (e.g., topk 40)")
    print(f"\nSettings: max_length={max_length}, temperature={temperature}, top_k={top_k}")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input("You: ").strip()

            if prompt.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            if prompt.lower().startswith('temp '):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"Temperature set to {temperature}")
                except (IndexError, ValueError):
                    print("Usage: temp 0.5")
                continue
            if prompt.lower().startswith('topk '):
                try:
                    top_k = int(prompt.split()[1])
                    print(f"Top-k set to {top_k}")
                except (IndexError, ValueError):
                    print("Usage: topk 40")
                continue
            if not prompt:
                continue

            response = generate_response(
                model, tokenizer, prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                device=str(device),
                model_type=model_type,
            )
            print(f"Model: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate trained transformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on validation set
  python validate.py --checkpoint assets/models/test_torch/best.pt --config configs/test_torch.yaml --eval

  # Interactive chat
  python validate.py --checkpoint assets/models/test_torch/best.pt --config configs/test_torch.yaml --chat

  # Single generation
  python validate.py --checkpoint assets/models/test_torch/best.pt --config configs/test_torch.yaml --prompt "Once upon a time"

  # CustomTransformer
  python validate.py --checkpoint assets/models/test_custom/best.pt --config configs/test_custom.yaml --model-type custom --chat
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.pt)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--model-type', type=str, choices=['torch', 'custom'], default='torch',
                        help='Model type: torch or custom (default: torch)')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Override tokenizer name')

    # Modes
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate on validation set')
    parser.add_argument('--chat', action='store_true',
                        help='Interactive chat mode')
    parser.add_argument('--prompt', type=str,
                        help='Single prompt to generate from')

    # Generation settings
    parser.add_argument('--max-length', type=int, default=100,
                        help='Max generation length (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling (default: 50)')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Max batches for evaluation (default: all)')

    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config for eval mode
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    model, tokenizer, checkpoint = load_model_from_checkpoint(
        args.checkpoint, args.config, args.model_type, device, args.tokenizer
    )

    # Print model info
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f"Model: {info['parameters']:,} params ({info.get('parameters_millions', info['parameters']/1e6):.1f}M)")
    elif hasattr(model, 'count_parameters'):
        params = model.count_parameters()
        print(f"Model: {params:,} params ({params/1e6:.1f}M)")

    # Print checkpoint metrics
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        val_loss = metrics.get('val_loss')
        if val_loss is not None:
            print(f"Checkpoint: epoch={checkpoint.get('epoch', '?')}, val_loss={val_loss:.4f}")

    # Run requested mode
    if args.eval:
        evaluate_model(model, tokenizer, config, device, args.model_type, args.max_batches)
    elif args.chat:
        chat_mode(model, tokenizer, device, args.model_type,
                  args.max_length, args.temperature, args.top_k)
    elif args.prompt:
        response = generate_response(
            model, tokenizer, args.prompt,
            args.max_length, args.temperature, args.top_k,
            str(device), args.model_type
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {response}")
    else:
        parser.print_help()
        print("\n[!] Specify --eval, --chat, or --prompt 'text'")


if __name__ == '__main__':
    main()
