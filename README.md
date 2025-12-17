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

### Scripts

Core scripts for training, validating, and evaluating transformer models. 

### Assets

This includes datasets, models, and outputs from experiments. More details in the assets folder. 


