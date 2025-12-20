"""Data loading utilities for training run comparison."""

from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training import TrainingRun, load_training_log


def get_default_log_dir() -> Path:
    """Get the default log directory."""
    return PROJECT_ROOT / "assets" / "logs"


def discover_training_logs(log_dir: Optional[str] = None) -> List[Path]:
    """
    Discover all JSON log files in the log directory.

    Args:
        log_dir: Directory to search. Defaults to assets/logs/

    Returns:
        List of Path objects for each .json log file, sorted by modification time (newest first)
    """
    log_path = Path(log_dir) if log_dir else get_default_log_dir()

    if not log_path.exists():
        return []

    logs = list(log_path.glob("*.json"))
    # Sort by modification time, newest first
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return logs


def get_run_display_name(run: TrainingRun, log_path: Path) -> str:
    """
    Create a human-readable display name for a training run.

    Format: "{experiment_name} ({model_type}) - {date}"

    Args:
        run: TrainingRun object
        log_path: Path to the log file

    Returns:
        Human-readable display name
    """
    # Parse date from filename: {name}_{YYYYMMDD_HHMMSS}.json
    try:
        parts = log_path.stem.rsplit("_", 2)
        if len(parts) >= 2:
            date_part = parts[-2]  # YYYYMMDD
            date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
        else:
            date_str = run.started_at[:10] if run.started_at else "unknown"
    except (IndexError, ValueError):
        date_str = run.started_at[:10] if run.started_at else "unknown"

    return f"{run.experiment_name} ({run.model_type}) - {date_str}"


def load_runs_from_paths(log_paths: List[str]) -> Dict[str, TrainingRun]:
    """
    Load multiple training runs from log file paths.

    Args:
        log_paths: List of paths to JSON log files

    Returns:
        Dict mapping display name -> TrainingRun
    """
    runs = {}
    for log_path_str in log_paths[:4]:  # Limit to 4 runs for comparison
        log_path = Path(log_path_str)
        try:
            run = load_training_log(str(log_path))
            name = get_run_display_name(run, log_path)
            runs[name] = run
        except Exception as e:
            print(f"Warning: Failed to load {log_path}: {e}")
    return runs


def get_log_choices(log_dir: Optional[str] = None) -> List[str]:
    """
    Get list of log file paths as strings for Gradio dropdown.

    Args:
        log_dir: Directory to search. Defaults to assets/logs/

    Returns:
        List of log file paths as strings
    """
    logs = discover_training_logs(log_dir)
    return [str(log) for log in logs]
