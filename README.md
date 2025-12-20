# Transformers from Scratch

This educational repository provides low-level implementations of transformers and supporting tools and scripts to get started with training them. 

Guiding Principles:
* Focus on the models & learning mechanism. This is the core of what makes transformers work and the key intention of this repo. 
* Simplicity over optimization. There are many finely tuned transformer implementations, the purpose of this repository is a fundamental understanding!
* Readability over conciseness. You'll find longer variable names and comments to walk through exactly what is happening

The repository includes:

### Models

* Transformer models at three different levels of abstraction:
- *Custom Transfomrer*: A tensor-level implementation with explicitly defined backpropagation
- *Torch Transfomer*: A PyTorch implementation almost exactly mirrors the custom transformer to show the abstractions by PyTorch
* *Baseline Transformer*: A PyTorch transformer with minor optimizations to improve the performance - both in compute efficiency and model capability. 

### Tools

Tools for dataset management and tokenization. Run with `python -m tools.<module>`.

#### Dataset Download

Download datasets from HuggingFace Hub:

```bash
python -m tools.download_dataset roneneldan/TinyStories
python -m tools.download_dataset HuggingFaceFW/fineweb --config sample-10BT
```

#### Tokenization

Unified tokenization toolkit with three subcommands:

```bash
# Analyze token distribution (helps choose vocab size)
python -m tools.tokenization analyze tinystories --subset 10000

# Train a custom BPE tokenizer from registry dataset
python -m tools.tokenization train registry tinystories 4096

# Train from JSONL file(s)
python -m tools.tokenization train file corpus.jsonl 8192

# Pre-tokenize dataset for faster training
python -m tools.tokenization pretokenize tinystories path/to/tokenizer --max-length 256
```

Python API:

```python
from tools.tokenization import (
    analyze_dataset,           # Get vocab recommendations
    train_tokenizer,           # Train from registry dataset
    train_tokenizer_from_files,# Train from JSONL
    pretokenize_dataset,       # Pre-tokenize for training
    load_pretokenized,         # Load pre-tokenized data
)

# Analyze and get recommendations
report, recommendations = analyze_dataset('tinystories', subset_size=10000)
print(report.summary())

# Train a small tokenizer
tokenizer_path = train_tokenizer('tinystories', vocab_size=4096)

# Pre-tokenize for efficient training
output = pretokenize_dataset('tinystories', tokenizer_path, max_length=256)
```

### Training

Unified training script supporting both model types:

```bash
# Train TorchTransformer (PyTorch autograd)
python scripts/train.py --config configs/torch_tinystories.yaml --model-type torch

# Train CustomTransformer (manual backprop)
python scripts/train.py --config configs/custom_tinystories.yaml --model-type custom

# Resume training from checkpoint
python scripts/train.py --config configs/torch_tinystories.yaml --model-type torch --resume

# Resume from specific checkpoint
python scripts/train.py --config configs/torch_tinystories.yaml --model-type torch --resume assets/models/experiment/best.pt
```

Training features:
- Learning rate scheduling (warmup + cosine/linear decay)
- Checkpoint management with best model tracking
- JSON logging for visualization
- Dynamic batch padding for efficiency
- bfloat16 mixed precision training

#### Configuration

Training is configured via YAML files in `configs/`:

```yaml
experiment_name: tinystories_torch

data:
  dataset: tinystories
  tokenizer: gpt2  # or path to custom tokenizer
  max_length: 256
  subset_size: 100000  # optional

model:
  d_model: 256
  n_heads: 4
  n_blocks: 6
  d_ffn: 1024
  max_seq_len: 256
  dtype: bfloat16

training:
  batch_size: 32
  num_epochs: 10
  learning_rate: 3e-4
  min_learning_rate: 3e-5
  lr_decay: cosine
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  eval_every: 500
```

#### Training Utilities (Python API)

```python
from training import (
    CheckpointManager,  # Save/load checkpoints with rotation
    Evaluator,          # Validation loss and perplexity
    TrainingLogger,     # JSON-compatible structured logging
)

# Checkpoint management
checkpoint_manager = CheckpointManager(
    checkpoint_dir='assets/models/my_experiment',
    model=model,
    experiment_name='my_experiment',
)
checkpoint_manager.save(epoch=1, global_step=1000, ...)
resume_state = checkpoint_manager.load()

# Evaluation
evaluator = Evaluator(model, device='cuda')
result = evaluator.evaluate(val_loader)
print(f"Val Loss: {result.loss:.4f}, Perplexity: {result.perplexity:.2f}")

# Logging
logger = TrainingLogger(
    log_dir='assets/logs',
    experiment_name='my_experiment',
    model_type='TorchTransformer',
    model_config={...},
    train_config={...},
)
logger.log_step(step=100, epoch=0, loss=2.5, learning_rate=1e-4)
logger.log_validation(step=100, epoch=0, loss=2.3, perplexity=10.0)
logger.finish()
```

### Validation & Evaluation

The `validate.py` script provides model evaluation and interactive text generation:

```bash
# Evaluate on validation set (loss, perplexity)
python scripts/validate.py \
    --checkpoint assets/models/test_torch/best.pt \
    --config configs/test_torch.yaml \
    --eval

# Single prompt generation
python scripts/validate.py \
    --checkpoint assets/models/test_torch/best.pt \
    --config configs/test_torch.yaml \
    --prompt "Once upon a time"

# Interactive chat mode
python scripts/validate.py \
    --checkpoint assets/models/test_torch/best.pt \
    --config configs/test_torch.yaml \
    --chat

# CustomTransformer evaluation
python scripts/validate.py \
    --checkpoint assets/models/test_custom/best.pt \
    --config configs/test_custom.yaml \
    --model-type custom \
    --eval
```

#### Generation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-length` | 100 | Maximum tokens to generate |
| `--temperature` | 0.8 | Sampling temperature (higher = more random) |
| `--top-k` | 50 | Top-k sampling (0 = disabled) |
| `--max-batches` | all | Limit eval batches for quick testing |

#### Chat Mode Commands

In interactive chat mode:
- Type prompts and press Enter to generate
- `temp 0.5` - Adjust temperature
- `topk 40` - Adjust top-k sampling
- `quit` / `exit` / `q` - Exit chat

### Visualizer

Web-based training comparison tool for analyzing and comparing training runs:

```bash
# Launch the visualizer
python -m visualizer

# With public sharing link
python -m visualizer --share

# Custom port
python -m visualizer --port 8080
```

#### Features

- **Loss Curves**: Overlay training and validation loss for up to 4 runs
- **Validation Metrics**: Dual y-axis plot showing loss and perplexity
- **FLOPs-Normalized**: Loss vs cumulative compute for fair model size comparison
- **Learning Rate**: Visualize LR schedules across runs
- **Throughput**: Compare tokens/second performance
- **Config Comparison**: Side-by-side model and training configuration tables
- **Summary Table**: Quick comparison of key metrics (params, best loss, TFLOPs)

#### Python API

```python
from visualizer import (
    create_app,              # Create Gradio app
    launch,                  # Launch the visualizer
    create_loss_curves,      # Generate loss plot
    create_flops_normalized_loss,  # Compute-normalized plot
    estimate_model_params,   # Estimate params from config
)
from training import load_training_log

# Load and visualize a training run
run = load_training_log('assets/logs/my_experiment.json')
fig = create_loss_curves({'my_run': run})
fig.show()
```

### Assets

This includes datasets, models, and outputs from experiments. More details in the assets folder. 


