## Assets

Assets include large files which are necessary to support the ML development but shouldn't be committed to the repository. We break this into three broad categories:

* Datasets - Core files used for training
* Models - Trained model files
* Outputs - Experimental results and log files

### Downloading Datasets

To download a dataset, we leverage Huggingface. Please note that some datasets require an account to access. This can be easily setup:

```bash
# To download a dataset: 
python tools/download_dataset.py --name <dataset>
```

For the scope of this project, I recommend TinyStories

### Tokenizers

In order to use a transformer, you need a need to map raw text to tokens. For this repo, you can create a custom tokenizer using byte-pair encoding (BPE). Optionally, you can pre-tokenize the dataset which will accelerate iterative development at the cost of more storage and a pre-computation step. Useful for learning! 

```bash
To determine what size vocabulary to use:
python tools/tokenization.py --dataset <path_to_dataset> --analyze

# To train a tokenizer with a specific vocabulary
python tools/tokenization.py --dataset <path_to_dataset> --vocab_size <size>

# To pre-tokenize a datasset:
python tools/tokenization.py --dataset <path_to_dataset> --tokenizer <path_to_tokenizer>
```

### Models

Model checkpoints are stored in assets as well. During training, we periodically store checkpoints. Any stored checkpoint can then be reloaded as needed. 

### Outputs

Specific experimental logs are stored here. This allows you to maintain a history of training logs, loss curves, and evaluation. We utilize DuckDB because it's easy and optimized for column datastore (plus it's easy!)


