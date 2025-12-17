"""Evaluation utilities for training validation."""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Iterator
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Container for evaluation results."""
    loss: float
    perplexity: float
    num_tokens: int
    num_batches: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'loss': self.loss,
            'perplexity': self.perplexity,
            'num_tokens': self.num_tokens,
            'num_batches': self.num_batches,
        }


class Evaluator:
    """
    Handles model evaluation on validation data.

    Supports both TorchTransformer and CustomTransformerWrapper models.

    Example:
        evaluator = Evaluator(model, device='cuda')

        # Evaluate on validation set
        result = evaluator.evaluate(val_dataloader)
        print(f"Val Loss: {result.loss:.4f}, Perplexity: {result.perplexity:.2f}")

        # Or evaluate a single batch
        batch_loss = evaluator.eval_batch(input_ids, labels)
    """

    def __init__(
        self,
        model: Any,
        device: str = 'cpu',
        vocab_size: Optional[int] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate (TorchTransformer or CustomTransformerWrapper)
            device: Device for evaluation
            vocab_size: Vocabulary size (auto-detected if not provided)
        """
        self.model = model
        self.device = device
        self.vocab_size = vocab_size or self._get_vocab_size()

    def _get_vocab_size(self) -> int:
        """Extract vocab_size from model."""
        if hasattr(self.model, 'vocab_size'):
            return self.model.vocab_size
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
            return self.model.model.config.get('vocab_size', 128)
        raise ValueError("Cannot determine vocab_size from model")

    def _get_model_for_eval(self):
        """Get the underlying model, unwrapping torch.compile() if needed."""
        model = self.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        return model

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: Iterator,
        max_batches: Optional[int] = None,
    ) -> EvalResult:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader yielding (input_ids, labels) batches
            max_batches: Maximum number of batches to evaluate (None = all)

        Returns:
            EvalResult with loss, perplexity, and statistics
        """
        model = self._get_model_for_eval()
        was_training = model.training if hasattr(model, 'training') else False

        if hasattr(model, 'eval'):
            model.eval()

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Unpack batch
            if isinstance(batch, (list, tuple)):
                input_ids, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                input_ids = batch['input_ids']
                labels = batch.get('labels', input_ids)
            else:
                input_ids = labels = batch

            # Move to device
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Compute loss
            loss, num_tokens = self._compute_batch_loss(input_ids, labels)

            total_loss += loss * num_tokens
            total_tokens += num_tokens
            num_batches += 1

        if was_training and hasattr(model, 'train'):
            model.train()

        if total_tokens == 0:
            return EvalResult(loss=float('inf'), perplexity=float('inf'),
                            num_tokens=0, num_batches=0)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow

        return EvalResult(
            loss=avg_loss,
            perplexity=perplexity,
            num_tokens=total_tokens,
            num_batches=num_batches,
        )

    def _compute_batch_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, int]:
        """
        Compute loss for a single batch.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len]

        Returns:
            Tuple of (loss_value, num_tokens)
        """
        model = self._get_model_for_eval()

        # Check if model has built-in loss computation
        if hasattr(model, 'forward'):
            # Try forward with labels (TorchTransformer style)
            try:
                outputs = model(input_ids, labels=labels)
                if isinstance(outputs, dict) and 'loss' in outputs:
                    # Count non-padding tokens (shifted for causal LM)
                    num_tokens = (labels[:, 1:] != -100).sum().item()
                    return outputs['loss'].item(), num_tokens
            except TypeError:
                pass

        # Fallback: compute loss manually
        if hasattr(model, 'model') and hasattr(model.model, 'forward'):
            # CustomTransformerWrapper
            logits = model.model.forward(input_ids.cpu().numpy())
            logits = torch.tensor(logits, device=self.device)
        else:
            # Direct forward
            outputs = model(input_ids)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean',
        )

        num_tokens = (shift_labels != -100).sum().item()
        return loss.item(), num_tokens

    @torch.no_grad()
    def eval_batch(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Evaluate a single batch and return loss.

        Args:
            input_ids: Input token IDs
            labels: Target token IDs (defaults to input_ids for LM)

        Returns:
            Loss value as float
        """
        if labels is None:
            labels = input_ids

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        loss, _ = self._compute_batch_loss(input_ids, labels)
        return loss


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss value."""
    return math.exp(min(loss, 100))  # Cap to avoid overflow
