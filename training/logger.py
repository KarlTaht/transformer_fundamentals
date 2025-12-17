"""Structured training logger with JSON export for visualization."""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict


@dataclass
class TrainingMetrics:
    """Single training step metrics."""
    step: int
    epoch: int
    loss: float
    learning_rate: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    grad_norm: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationMetrics:
    """Validation metrics at a given step."""
    step: int
    epoch: int
    loss: float
    perplexity: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TrainingRun:
    """Complete training run metadata and metrics."""
    experiment_name: str
    model_type: str
    model_config: Dict[str, Any]
    train_config: Dict[str, Any]
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: Optional[str] = None
    train_metrics: List[TrainingMetrics] = field(default_factory=list)
    val_metrics: List[ValidationMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_name': self.experiment_name,
            'model_type': self.model_type,
            'model_config': self.model_config,
            'train_config': self.train_config,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'train_metrics': [asdict(m) for m in self.train_metrics],
            'val_metrics': [asdict(m) for m in self.val_metrics],
        }


class TrainingLogger:
    """
    Structured logger for training runs.

    Logs to both console and JSON file for later visualization.

    Example:
        logger = TrainingLogger(
            log_dir='assets/logs',
            experiment_name='tinystories_torch',
            model_type='TorchTransformer',
            model_config=model.get_model_info(),
            train_config=config,
        )

        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            logger.log_step(step, epoch, loss, lr)

            if step % val_interval == 0:
                val_result = evaluate(val_loader)
                logger.log_validation(step, epoch, val_result.loss, val_result.perplexity)

        logger.finish()
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        model_type: str,
        model_config: Dict[str, Any],
        train_config: Dict[str, Any],
        console_interval: int = 100,
        save_interval: int = 500,
    ):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save log files
            experiment_name: Name for this experiment
            model_type: Type of model being trained
            model_config: Model configuration dict
            train_config: Training configuration dict
            console_interval: Steps between console prints
            save_interval: Steps between JSON saves
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.console_interval = console_interval
        self.save_interval = save_interval

        # Initialize training run
        self.run = TrainingRun(
            experiment_name=experiment_name,
            model_type=model_type,
            model_config=model_config,
            train_config=train_config,
        )

        # JSON log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.json"

        # Timing
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_tokens = 0

        print(f"Training logger initialized: {self.log_file}")

    def log_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        tokens_processed: Optional[int] = None,
        **extras,
    ):
        """
        Log a training step.

        Args:
            step: Global step number
            epoch: Current epoch
            loss: Training loss
            learning_rate: Current learning rate
            grad_norm: Optional gradient norm
            tokens_processed: Total tokens processed (for throughput)
            **extras: Additional metrics to log
        """
        # Compute throughput
        tokens_per_sec = None
        if tokens_processed is not None:
            elapsed = time.time() - self.last_log_time
            if elapsed > 0:
                tokens_per_sec = (tokens_processed - self.last_tokens) / elapsed
            self.last_tokens = tokens_processed
            self.last_log_time = time.time()

        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            tokens_per_sec=tokens_per_sec,
            extras=extras if extras else {},
        )
        self.run.train_metrics.append(metrics)

        # Console output
        if step % self.console_interval == 0:
            self._print_step(metrics)

        # Save to JSON periodically
        if step % self.save_interval == 0:
            self._save_json()

    def log_validation(
        self,
        step: int,
        epoch: int,
        loss: float,
        perplexity: float,
    ):
        """
        Log validation metrics.

        Args:
            step: Global step number
            epoch: Current epoch
            loss: Validation loss
            perplexity: Validation perplexity
        """
        metrics = ValidationMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            perplexity=perplexity,
        )
        self.run.val_metrics.append(metrics)

        print(f"  [Val] Step {step} | Loss: {loss:.4f} | Perplexity: {perplexity:.2f}")
        self._save_json()

    def _print_step(self, metrics: TrainingMetrics):
        """Print step info to console."""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)

        parts = [
            f"Step {metrics.step:6d}",
            f"Epoch {metrics.epoch}",
            f"Loss: {metrics.loss:.4f}",
            f"LR: {metrics.learning_rate:.2e}",
        ]

        if metrics.grad_norm is not None:
            parts.append(f"GradNorm: {metrics.grad_norm:.2f}")

        if metrics.tokens_per_sec is not None:
            parts.append(f"Tok/s: {metrics.tokens_per_sec:.0f}")

        parts.append(f"[{hours:02d}:{minutes:02d}]")

        print(" | ".join(parts))

    def _save_json(self):
        """Save current run to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.run.to_dict(), f, indent=2)

    def finish(self):
        """Mark training as finished and save final log."""
        self.run.finished_at = datetime.now().isoformat()
        self._save_json()

        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print(f"\nTraining finished in {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Log saved to: {self.log_file}")

    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get most recent training metrics."""
        if self.run.train_metrics:
            return self.run.train_metrics[-1]
        return None

    def get_best_validation(self) -> Optional[ValidationMetrics]:
        """Get best validation metrics by loss."""
        if not self.run.val_metrics:
            return None
        return min(self.run.val_metrics, key=lambda m: m.loss)


def load_training_log(log_path: str) -> TrainingRun:
    """Load a training log from JSON file."""
    with open(log_path, 'r') as f:
        data = json.load(f)

    run = TrainingRun(
        experiment_name=data['experiment_name'],
        model_type=data['model_type'],
        model_config=data['model_config'],
        train_config=data['train_config'],
        started_at=data['started_at'],
        finished_at=data.get('finished_at'),
    )

    for m in data.get('train_metrics', []):
        run.train_metrics.append(TrainingMetrics(**m))

    for m in data.get('val_metrics', []):
        run.val_metrics.append(ValidationMetrics(**m))

    return run
