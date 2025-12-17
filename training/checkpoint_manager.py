"""Checkpoint management for training resumption."""

import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and training resumption.

    Features:
    - Automatic checkpoint rotation (keeps N most recent)
    - latest.pt pointer for easy resumption
    - best.pt tracking for best model by metric
    - Full training state for resumption (epoch, step, lr, config)

    Example:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir='assets/models/experiment_01',
            model=model,
            experiment_name='tinystories_torch',
        )

        # Save at end of epoch
        checkpoint_manager.save(
            epoch=epoch + 1,
            global_step=global_step,
            train_config=config,
            learning_rate=lr,
            optimizer=optimizer,  # Optional, for TorchTransformer
            metrics={'val_loss': val_loss},
            is_best=(val_loss < best_val_loss),
        )

        # Resume from checkpoint
        resume_state = checkpoint_manager.load()
        if resume_state:
            start_epoch = resume_state['epoch']
            global_step = resume_state['global_step']
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model: Any,
        experiment_name: str,
        max_checkpoints: int = 5,
    ):
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            model: Model instance (TorchTransformer or CustomTransformerWrapper)
            experiment_name: Name for this experiment (used in filenames)
            max_checkpoints: Maximum number of epoch checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.experiment_name = experiment_name
        self.max_checkpoints = max_checkpoints

    def _get_model_to_save(self):
        """Get the underlying model, unwrapping torch.compile() if needed."""
        model = self.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        return model

    def save(
        self,
        epoch: int,
        global_step: int,
        train_config: dict,
        learning_rate: float,
        optimizer: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save a training checkpoint.

        Args:
            epoch: Current epoch number (1-indexed, after completing epoch)
            global_step: Total training steps completed
            train_config: Training configuration dict
            learning_rate: Current learning rate
            optimizer: Optional optimizer (for TorchTransformer with autograd)
            metrics: Optional dict of metrics (loss, perplexity, etc.)
            is_best: If True, also save as best.pt

        Returns:
            Path to saved checkpoint file
        """
        model = self._get_model_to_save()

        # Get model state dict
        if hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
            # CustomTransformerWrapper
            model_state = model.model.state_dict()
            model_config = model.model.get_config()
        elif hasattr(model, 'state_dict'):
            # TorchTransformer / BaseLanguageModel
            model_state = model.state_dict()
            model_config = getattr(model, 'model_config', {})
        else:
            raise ValueError("Model must have state_dict() method")

        checkpoint = {
            # Model state
            'model_state_dict': model_state,
            'model_config': model_config,

            # Training state
            'epoch': epoch,
            'global_step': global_step,
            'learning_rate': learning_rate,
            'train_config': train_config,

            # Optimizer state (for TorchTransformer)
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,

            # Metrics
            'metrics': metrics or {},

            # Metadata
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name,
        }

        # Save epoch checkpoint
        filename = f"checkpoint_epoch{epoch}_step{global_step}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

        # Always update latest pointer
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint updated: {best_path}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return filepath

    def load(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        optimizer: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint and restore model state.

        Args:
            checkpoint_path: Specific checkpoint to load.
                           If None, loads 'latest.pt'.
                           Can also be 'best' to load best.pt.
            optimizer: Optional optimizer to restore state into

        Returns:
            Dict with training state for resumption, or None if no checkpoint found.
            Contains: epoch, global_step, learning_rate, train_config, metrics
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest.pt"
        elif checkpoint_path == 'best':
            checkpoint_path = self.checkpoint_dir / "best.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return None

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Handle torch.compile() state dict keys
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        model = self._get_model_to_save()

        # Restore model state
        if hasattr(model, 'model') and hasattr(model.model, 'load_state_dict'):
            # CustomTransformerWrapper
            model.model.load_state_dict(state_dict)
        elif hasattr(model, 'load_state_dict'):
            # TorchTransformer / BaseLanguageModel
            model.load_state_dict(state_dict)
        else:
            raise ValueError("Model must have load_state_dict() method")

        # Restore optimizer state if provided
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Restored from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")

        return {
            'epoch': checkpoint['epoch'],
            'global_step': checkpoint['global_step'],
            'learning_rate': checkpoint['learning_rate'],
            'train_config': checkpoint.get('train_config', {}),
            'metrics': checkpoint.get('metrics', {}),
        }

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only max_checkpoints most recent."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old_ckpt in checkpoints[self.max_checkpoints:]:
            old_ckpt.unlink()
            print(f"Removed old checkpoint: {old_ckpt.name}")

    def get_latest(self) -> Optional[Path]:
        """Get path to latest checkpoint if it exists."""
        latest_path = self.checkpoint_dir / "latest.pt"
        return latest_path if latest_path.exists() else None

    def get_best(self) -> Optional[Path]:
        """Get path to best checkpoint if it exists."""
        best_path = self.checkpoint_dir / "best.pt"
        return best_path if best_path.exists() else None

    def list_checkpoints(self) -> Dict[str, Any]:
        """List all available checkpoints with metadata."""
        checkpoints = {}

        for ckpt_path in self.checkpoint_dir.glob("*.pt"):
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                checkpoints[ckpt_path.name] = {
                    'epoch': checkpoint.get('epoch'),
                    'global_step': checkpoint.get('global_step'),
                    'timestamp': checkpoint.get('timestamp'),
                    'metrics': checkpoint.get('metrics', {}),
                }
            except Exception as e:
                checkpoints[ckpt_path.name] = {'error': str(e)}

        return checkpoints
