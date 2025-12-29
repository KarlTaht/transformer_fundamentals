#!/usr/bin/env python
"""Verify that the transformer_fundamentals package is installed correctly."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def verify_imports():
    """Test all major imports."""
    errors = []

    # Core ML
    try:
        import torch
        import numpy as np
    except ImportError as e:
        errors.append(f"Core ML: {e}")

    # Models
    try:
        from models import TorchTransformer, CustomTransformer, TransformerConfig
    except ImportError as e:
        errors.append(f"Models: {e}")

    # Training
    try:
        from training import CheckpointManager, Evaluator, TrainingLogger, TrainingRun
    except ImportError as e:
        errors.append(f"Training: {e}")

    # Tools
    try:
        from tools import DATASET_REGISTRY, analyze_dataset, train_tokenizer
    except ImportError as e:
        errors.append(f"Tools: {e}")

    # Visualizer
    try:
        from visualizer import launch, create_app, create_loss_curves
        import gradio as gr
        import plotly.graph_objects as go
    except ImportError as e:
        errors.append(f"Visualizer: {e}")

    return errors


def verify_directories():
    """Check expected directory structure exists."""
    required_dirs = [
        "models",
        "training",
        "tools",
        "visualizer",
        "scripts",
        "configs",
    ]
    # These are created on first use, so just warn if missing
    optional_dirs = [
        "assets/logs",
        "assets/models",
        "assets/datasets",
    ]

    missing_required = []
    missing_optional = []

    for d in required_dirs:
        if not (PROJECT_ROOT / d).exists():
            missing_required.append(d)

    for d in optional_dirs:
        if not (PROJECT_ROOT / d).exists():
            missing_optional.append(d)

    return missing_required, missing_optional


def main():
    print("Verifying transformer_fundamentals installation...\n")

    # Check imports
    print("Checking imports...")
    import_errors = verify_imports()
    if import_errors:
        print("  FAILED:")
        for e in import_errors:
            print(f"    - {e}")
    else:
        print("  OK: All imports successful")

    # Check directories
    print("\nChecking directories...")
    missing_required, missing_optional = verify_directories()
    if missing_required:
        print("  FAILED: Missing required directories:")
        for d in missing_required:
            print(f"    - {d}")
    else:
        print("  OK: All required directories present")

    if missing_optional:
        print("  INFO: Optional directories not yet created (will be created on first use):")
        for d in missing_optional:
            print(f"    - {d}")

    # Summary
    print("\n" + "=" * 50)
    if import_errors or missing_required:
        print("FAILED: Fix errors above before using the package")
        sys.exit(1)
    else:
        print("SUCCESS: Installation verified!")
        sys.exit(0)


if __name__ == "__main__":
    main()
