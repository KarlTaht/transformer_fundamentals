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

Tools are specific scripts intended for setting up your environment and necessary assets. This includes validating necessary dependencies and scripts to help download and tokenize datasets.

### Scripts

Core scripts for training, validating, and evaluating transformer models. 

### Assets

This includes datasets, models, and outputs from experiments. More details in the assets folder. 


