"""Unified tokenization toolkit: analysis, training, and pre-tokenization.

This module provides three main capabilities:
1. **analyze** - Analyze token distributions to understand vocabulary needs
2. **train** - Train custom BPE tokenizers optimized for specific datasets
3. **pretokenize** - Pre-tokenize datasets for faster training

CLI Usage:
    # Analyze token distribution in a dataset
    python -m tools.tokenization analyze tinystories --subset 10000

    # Train a custom tokenizer from registry dataset
    python -m tools.tokenization train registry tinystories 4096

    # Train from JSONL file(s)
    python -m tools.tokenization train file corpus.jsonl 8192

    # Pre-tokenize a dataset
    python -m tools.tokenization pretokenize tinystories path/to/tokenizer --max-length 256

Python API:
    from tools.tokenization import TokenAnalyzer, train_tokenizer, pretokenize_dataset

    # Analysis
    analyzer = TokenAnalyzer(tokenizer)
    report = analyzer.analyze_dataset('tinystories')
    print(report.summary())

    # Training
    tokenizer_path = train_tokenizer('tinystories', vocab_size=4096)

    # Pre-tokenization
    output_path = pretokenize_dataset('tinystories', tokenizer_path, max_length=256)
"""

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator, Union

import numpy as np
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerFast

from .dataset_registry import (
    load_dataset_from_registry,
    get_dataset_config,
    get_datasets_dir,
    get_models_dir,
    DATASET_REGISTRY,
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EdgeCaseStats:
    """Statistics about edge case tokens in the dataset."""

    single_char_tokens: Dict[int, int] = field(default_factory=dict)
    punctuation_tokens: Dict[int, int] = field(default_factory=dict)
    number_tokens: Dict[int, int] = field(default_factory=dict)
    hyphenated_tokens: Dict[int, int] = field(default_factory=dict)
    whitespace_tokens: Dict[int, int] = field(default_factory=dict)
    rare_tokens: Dict[int, int] = field(default_factory=dict)

    total_single_char_occurrences: int = 0
    total_punctuation_occurrences: int = 0
    total_number_occurrences: int = 0
    total_hyphenated_occurrences: int = 0
    total_whitespace_occurrences: int = 0
    total_rare_occurrences: int = 0

    def summary(self) -> str:
        """Return a formatted summary of edge case statistics."""
        lines = [
            "Edge Case Statistics:",
            f"  Single char: {len(self.single_char_tokens):,} unique, {self.total_single_char_occurrences:,} occurrences",
            f"  Punctuation: {len(self.punctuation_tokens):,} unique, {self.total_punctuation_occurrences:,} occurrences",
            f"  Numbers: {len(self.number_tokens):,} unique, {self.total_number_occurrences:,} occurrences",
            f"  Hyphenated: {len(self.hyphenated_tokens):,} unique, {self.total_hyphenated_occurrences:,} occurrences",
            f"  Whitespace: {len(self.whitespace_tokens):,} unique, {self.total_whitespace_occurrences:,} occurrences",
            f"  Rare: {len(self.rare_tokens):,} unique, {self.total_rare_occurrences:,} occurrences",
        ]
        return "\n".join(lines)


@dataclass
class CoveragePoint:
    """A point on the coverage CDF curve."""

    num_tokens: int
    coverage: float
    cumulative_count: int


@dataclass
class TokenAnalysisReport:
    """Complete analysis report for a dataset."""

    dataset_name: str
    tokenizer_name: str
    total_tokens: int
    unique_tokens: int
    vocab_size: int
    vocab_utilization: float
    token_frequencies: Counter
    coverage_cdf: List[CoveragePoint]
    edge_cases: EdgeCaseStats
    tokens_for_90_coverage: int = 0
    tokens_for_95_coverage: int = 0
    tokens_for_99_coverage: int = 0
    tokens_for_999_coverage: int = 0

    def summary(self) -> str:
        """Return a formatted summary of the analysis."""
        lines = [
            "=" * 60,
            f"Token Analysis Report: {self.dataset_name}",
            "=" * 60,
            "",
            f"Tokenizer: {self.tokenizer_name}",
            f"Tokenizer vocab size: {self.vocab_size:,}",
            "",
            "Dataset Statistics:",
            f"  Total tokens: {self.total_tokens:,}",
            f"  Unique tokens used: {self.unique_tokens:,}",
            f"  Vocabulary utilization: {self.vocab_utilization:.2%}",
            "",
            "Coverage Milestones (tokens needed for X% coverage):",
            f"  90% coverage: {self.tokens_for_90_coverage:,} tokens",
            f"  95% coverage: {self.tokens_for_95_coverage:,} tokens",
            f"  99% coverage: {self.tokens_for_99_coverage:,} tokens",
            f"  99.9% coverage: {self.tokens_for_999_coverage:,} tokens",
            "",
            self.edge_cases.summary(),
            "",
            "Top 20 Most Frequent Tokens:",
        ]

        for token_id, count in self.token_frequencies.most_common(20):
            pct = count / self.total_tokens * 100
            lines.append(f"  {token_id:6d}: {count:8,} ({pct:5.2f}%)")

        return "\n".join(lines)

    def get_coverage_at_vocab_size(self, vocab_size: int) -> float:
        """Get coverage percentage for a given vocabulary size."""
        for point in self.coverage_cdf:
            if point.num_tokens >= vocab_size:
                return point.coverage
        return 1.0

    def get_vocab_size_for_coverage(self, target_coverage: float) -> int:
        """Get minimum vocabulary size needed for target coverage."""
        for point in self.coverage_cdf:
            if point.coverage >= target_coverage:
                return point.num_tokens
        return self.unique_tokens


@dataclass
class TokenizerRecommendation:
    """Recommendation for tokenizer configuration."""

    target_coverage: float
    recommended_vocab_size: int
    actual_coverage: float
    tokens_saved: int
    embedding_reduction: float

    def summary(self) -> str:
        """Return formatted recommendation."""
        return (
            f"Target: {self.target_coverage:.1%} coverage\n"
            f"  Recommended vocab size: {self.recommended_vocab_size:,}\n"
            f"  Actual coverage: {self.actual_coverage:.2%}\n"
            f"  Tokens saved vs GPT-2: {self.tokens_saved:,}\n"
            f"  Embedding param reduction: {self.embedding_reduction:.1%}"
        )


# =============================================================================
# Token Analyzer
# =============================================================================


class TokenAnalyzer:
    """Analyzes token distributions in datasets for tokenizer optimization."""

    def __init__(self, tokenizer, rare_threshold: int = 10):
        """
        Initialize the analyzer.

        Args:
            tokenizer: HuggingFace tokenizer (e.g., GPT2TokenizerFast)
            rare_threshold: Tokens with count below this are considered "rare"
        """
        self.tokenizer = tokenizer
        self.rare_threshold = rare_threshold
        self.vocab_size = len(tokenizer)
        self._decode_cache: Dict[int, str] = {}

    def _decode_token(self, token_id: int) -> str:
        """Decode a token ID to its string representation with caching."""
        if token_id not in self._decode_cache:
            try:
                self._decode_cache[token_id] = self.tokenizer.decode([token_id])
            except Exception:
                self._decode_cache[token_id] = ""
        return self._decode_cache[token_id]

    def _classify_token(self, token_id: int, token_str: str) -> List[str]:
        """Classify a token into edge case categories."""
        categories = []

        if len(token_str) == 1:
            categories.append("single_char")

        if token_str.isspace() or (token_str.startswith(" ") and len(token_str) == 1):
            categories.append("whitespace")

        stripped = token_str.strip()
        if stripped:
            punct_count = sum(1 for c in stripped if not c.isalnum() and not c.isspace())
            if punct_count > len(stripped) / 2:
                categories.append("punctuation")

        if "-" in token_str or "\u2013" in token_str or "\u2014" in token_str:
            categories.append("hyphenated")

        if any(c.isdigit() for c in token_str):
            categories.append("number")

        return categories

    def analyze_tokens(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> Tuple[Counter, int]:
        """Tokenize texts and count token frequencies."""
        token_counts = Counter()
        total_tokens = 0
        batch_size = 1000

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            encoded = self.tokenizer(
                batch,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )

            for input_ids in encoded["input_ids"]:
                token_counts.update(input_ids)
                total_tokens += len(input_ids)

            if show_progress and (i // batch_size) % 10 == 0:
                print(f"  Processed {min(i + batch_size, len(texts)):,}/{len(texts):,} texts...")

        return token_counts, total_tokens

    def compute_coverage_cdf(
        self,
        token_frequencies: Counter,
        total_tokens: int,
    ) -> List[CoveragePoint]:
        """Compute the cumulative distribution function of token coverage."""
        sorted_tokens = token_frequencies.most_common()
        cdf_points = []
        cumulative = 0

        for i, (token_id, count) in enumerate(sorted_tokens, 1):
            cumulative += count
            coverage = cumulative / total_tokens

            # Sample points to avoid huge lists
            if (
                i <= 100
                or (i % 10 == 0 and i <= 1000)
                or (i % 100 == 0 and i <= 10000)
                or i % 1000 == 0
                or i == len(sorted_tokens)
            ):
                cdf_points.append(
                    CoveragePoint(
                        num_tokens=i,
                        coverage=coverage,
                        cumulative_count=cumulative,
                    )
                )

        return cdf_points

    def detect_edge_cases(self, token_frequencies: Counter) -> EdgeCaseStats:
        """Detect and categorize edge case tokens."""
        stats = EdgeCaseStats()

        for token_id, count in token_frequencies.items():
            token_str = self._decode_token(token_id)
            categories = self._classify_token(token_id, token_str)

            if "single_char" in categories:
                stats.single_char_tokens[token_id] = count
                stats.total_single_char_occurrences += count

            if "punctuation" in categories:
                stats.punctuation_tokens[token_id] = count
                stats.total_punctuation_occurrences += count

            if "number" in categories:
                stats.number_tokens[token_id] = count
                stats.total_number_occurrences += count

            if "hyphenated" in categories:
                stats.hyphenated_tokens[token_id] = count
                stats.total_hyphenated_occurrences += count

            if "whitespace" in categories:
                stats.whitespace_tokens[token_id] = count
                stats.total_whitespace_occurrences += count

            if count < self.rare_threshold:
                stats.rare_tokens[token_id] = count
                stats.total_rare_occurrences += count

        return stats

    def _find_coverage_milestone(self, cdf_points: List[CoveragePoint], target: float) -> int:
        """Find number of tokens needed for target coverage."""
        for point in cdf_points:
            if point.coverage >= target:
                return point.num_tokens
        return cdf_points[-1].num_tokens if cdf_points else 0

    def analyze_dataset(
        self,
        dataset_name: str,
        subset_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> TokenAnalysisReport:
        """Analyze token distribution in a registered dataset."""
        if show_progress:
            print(f"Loading dataset: {dataset_name}")

        train_dataset, _, text_column, _ = load_dataset_from_registry(dataset_name)

        if subset_size is not None and subset_size < len(train_dataset):
            train_dataset = train_dataset.select(range(subset_size))

        if show_progress:
            print(f"Analyzing {len(train_dataset):,} examples...")

        texts = train_dataset[text_column]
        token_frequencies, total_tokens = self.analyze_tokens(texts, show_progress)

        if show_progress:
            print("Computing coverage CDF...")
        cdf_points = self.compute_coverage_cdf(token_frequencies, total_tokens)

        if show_progress:
            print("Detecting edge cases...")
        edge_cases = self.detect_edge_cases(token_frequencies)

        report = TokenAnalysisReport(
            dataset_name=dataset_name,
            tokenizer_name=getattr(self.tokenizer, "name_or_path", "unknown"),
            total_tokens=total_tokens,
            unique_tokens=len(token_frequencies),
            vocab_size=self.vocab_size,
            vocab_utilization=len(token_frequencies) / self.vocab_size,
            token_frequencies=token_frequencies,
            coverage_cdf=cdf_points,
            edge_cases=edge_cases,
            tokens_for_90_coverage=self._find_coverage_milestone(cdf_points, 0.90),
            tokens_for_95_coverage=self._find_coverage_milestone(cdf_points, 0.95),
            tokens_for_99_coverage=self._find_coverage_milestone(cdf_points, 0.99),
            tokens_for_999_coverage=self._find_coverage_milestone(cdf_points, 0.999),
        )

        if show_progress:
            print("Analysis complete!")

        return report

    def analyze_texts(
        self,
        texts: List[str],
        name: str = "custom",
        show_progress: bool = True,
    ) -> TokenAnalysisReport:
        """Analyze token distribution in arbitrary texts."""
        if show_progress:
            print(f"Analyzing {len(texts):,} texts...")

        token_frequencies, total_tokens = self.analyze_tokens(texts, show_progress)
        cdf_points = self.compute_coverage_cdf(token_frequencies, total_tokens)
        edge_cases = self.detect_edge_cases(token_frequencies)

        return TokenAnalysisReport(
            dataset_name=name,
            tokenizer_name=getattr(self.tokenizer, "name_or_path", "unknown"),
            total_tokens=total_tokens,
            unique_tokens=len(token_frequencies),
            vocab_size=self.vocab_size,
            vocab_utilization=len(token_frequencies) / self.vocab_size,
            token_frequencies=token_frequencies,
            coverage_cdf=cdf_points,
            edge_cases=edge_cases,
            tokens_for_90_coverage=self._find_coverage_milestone(cdf_points, 0.90),
            tokens_for_95_coverage=self._find_coverage_milestone(cdf_points, 0.95),
            tokens_for_99_coverage=self._find_coverage_milestone(cdf_points, 0.99),
            tokens_for_999_coverage=self._find_coverage_milestone(cdf_points, 0.999),
        )

    def recommend_vocab_size(
        self,
        report: TokenAnalysisReport,
        target_coverages: List[float] = [0.90, 0.95, 0.99, 0.999],
        base_vocab_size: int = 50257,
    ) -> List[TokenizerRecommendation]:
        """Generate tokenizer size recommendations for different coverage targets."""
        recommendations = []

        for target in target_coverages:
            vocab_needed = report.get_vocab_size_for_coverage(target)
            actual_coverage = report.get_coverage_at_vocab_size(vocab_needed)
            tokens_saved = base_vocab_size - vocab_needed
            embedding_reduction = tokens_saved / base_vocab_size

            recommendations.append(
                TokenizerRecommendation(
                    target_coverage=target,
                    recommended_vocab_size=vocab_needed,
                    actual_coverage=actual_coverage,
                    tokens_saved=tokens_saved,
                    embedding_reduction=embedding_reduction,
                )
            )

        return recommendations

    def compare_datasets(self, reports: List[TokenAnalysisReport]) -> str:
        """Compare token distributions across multiple datasets."""
        lines = [
            "=" * 80,
            "Dataset Comparison",
            "=" * 80,
            "",
        ]

        header = f"{'Dataset':<25} {'Total Tokens':>15} {'Unique':>10} {'Util%':>8} {'95%Cov':>8} {'99%Cov':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for report in reports:
            lines.append(
                f"{report.dataset_name:<25} "
                f"{report.total_tokens:>15,} "
                f"{report.unique_tokens:>10,} "
                f"{report.vocab_utilization:>7.1%} "
                f"{report.tokens_for_95_coverage:>8,} "
                f"{report.tokens_for_99_coverage:>8,}"
            )

        return "\n".join(lines)

    def export_cdf_data(
        self,
        report: TokenAnalysisReport,
        output_path: Optional[Path] = None,
    ) -> Optional[str]:
        """Export CDF data for plotting."""
        lines = ["num_tokens,coverage,cumulative_count"]

        for point in report.coverage_cdf:
            lines.append(f"{point.num_tokens},{point.coverage:.6f},{point.cumulative_count}")

        csv_content = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(csv_content)
            return None

        return csv_content

    def get_essential_tokens(
        self,
        report: TokenAnalysisReport,
        coverage: float = 0.99,
    ) -> List[Tuple[int, str, int]]:
        """Get the most essential tokens for a given coverage level."""
        vocab_needed = report.get_vocab_size_for_coverage(coverage)

        essential = []
        for token_id, count in report.token_frequencies.most_common(vocab_needed):
            token_str = self._decode_token(token_id)
            essential.append((token_id, token_str, count))

        return essential


def analyze_dataset(
    dataset_name: str,
    subset_size: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[TokenAnalysisReport, List[TokenizerRecommendation]]:
    """
    Convenience function to analyze a dataset and get tokenizer recommendations.

    Uses GPT-2 tokenizer as baseline for analysis.

    Args:
        dataset_name: Name of dataset in registry
        subset_size: If set, only analyze this many examples
        show_progress: Whether to show progress

    Returns:
        Tuple of (TokenAnalysisReport, List[TokenizerRecommendation])
    """
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    analyzer = TokenAnalyzer(tokenizer)
    report = analyzer.analyze_dataset(dataset_name, subset_size, show_progress)
    recommendations = analyzer.recommend_vocab_size(report)

    return report, recommendations


# =============================================================================
# Tokenizer Training
# =============================================================================


def _create_bpe_tokenizer(
    text_iterator: Iterator[str],
    num_texts: int,
    vocab_size: int,
    min_frequency: int = 2,
    show_progress: bool = True,
) -> "Tokenizer":
    """Internal helper to create and train a BPE tokenizer."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    if show_progress:
        num_threads = os.cpu_count() or 1
        print(f"Training with {num_threads} CPU threads available")

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        show_progress=show_progress,
    )

    tokenizer.train_from_iterator(text_iterator, trainer=trainer, length=num_texts)

    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    return tokenizer


def _save_tokenizer(tokenizer, output_dir: Path, show_progress: bool = True) -> Path:
    """Internal helper to save a tokenizer in HuggingFace format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(output_dir / "tokenizer.json"))

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    hf_tokenizer.save_pretrained(str(output_dir))

    if show_progress:
        print(f"Tokenizer saved to: {output_dir}")
        print(f"Actual vocab size: {hf_tokenizer.vocab_size:,}")

    return output_dir


def train_tokenizer(
    dataset_name: str,
    vocab_size: int,
    output_dir: Optional[Path] = None,
    subset_size: Optional[int] = None,
    min_frequency: int = 2,
    show_progress: bool = True,
) -> Path:
    """
    Train a custom BPE tokenizer from a registry dataset.

    Args:
        dataset_name: Name of dataset in registry (e.g., 'tinystories')
        vocab_size: Target vocabulary size
        output_dir: Where to save (default: assets/models/tokenizers/)
        subset_size: If set, only train on this many examples
        min_frequency: Minimum frequency for a token to be included
        show_progress: Whether to show progress

    Returns:
        Path to the saved tokenizer directory

    Example:
        tokenizer_path = train_tokenizer('tinystories', vocab_size=4096)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    """
    if show_progress:
        print(f"Loading dataset: {dataset_name}")

    train_dataset, _, text_column, _ = load_dataset_from_registry(dataset_name)

    if subset_size is not None and subset_size < len(train_dataset):
        train_dataset = train_dataset.select(range(subset_size))

    if show_progress:
        print(f"Training tokenizer on {len(train_dataset):,} examples...")
        print(f"Target vocab size: {vocab_size:,}")

    def text_iterator() -> Iterator[str]:
        for i in range(len(train_dataset)):
            yield train_dataset[i][text_column]

    tokenizer = _create_bpe_tokenizer(
        text_iterator(),
        len(train_dataset),
        vocab_size,
        min_frequency,
        show_progress,
    )

    if output_dir is None:
        output_dir = get_models_dir() / "tokenizers" / f"{dataset_name}_bpe_{vocab_size}"
    else:
        output_dir = Path(output_dir)

    return _save_tokenizer(tokenizer, output_dir, show_progress)


def train_tokenizer_from_files(
    paths: Union[str, Path, List[Union[str, Path]]],
    vocab_size: int,
    output_dir: Optional[Path] = None,
    subset_size: Optional[int] = None,
    min_frequency: int = 2,
    text_key: str = "text",
    show_progress: bool = True,
) -> Path:
    """
    Train a custom BPE tokenizer from JSONL file(s).

    Args:
        paths: Path(s) to JSONL file(s) with {"text": "..."} entries
        vocab_size: Target vocabulary size
        output_dir: Where to save (default: assets/models/tokenizers/)
        subset_size: If set, only train on this many examples
        min_frequency: Minimum frequency for a token to be included
        text_key: Key in JSON objects containing the text
        show_progress: Whether to show progress

    Returns:
        Path to the saved tokenizer directory
    """
    import json

    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    else:
        paths = [Path(p) for p in paths]

    texts = []
    for path in paths:
        if show_progress:
            print(f"Loading texts from: {path}")

        with open(path) as f:
            file_count = 0
            for line in f:
                data = json.loads(line)
                texts.append(data[text_key])
                file_count += 1

        if show_progress:
            print(f"  Loaded {file_count:,} texts from {path.name}")

    if show_progress:
        print(f"Total: {len(texts):,} texts from {len(paths)} file(s)")

    if subset_size is not None and subset_size < len(texts):
        texts = texts[:subset_size]

    if show_progress:
        print(f"Training tokenizer on {len(texts):,} examples...")
        print(f"Target vocab size: {vocab_size:,}")

    def text_iterator() -> Iterator[str]:
        for text in texts:
            yield text

    tokenizer = _create_bpe_tokenizer(
        text_iterator(),
        len(texts),
        vocab_size,
        min_frequency,
        show_progress,
    )

    if output_dir is None:
        if len(paths) > 1:
            name = "combined"
        else:
            name = paths[0].stem
            if name in ("train", "val", "test"):
                name = paths[0].parent.name
        output_dir = get_models_dir() / "tokenizers" / f"{name}_bpe_{vocab_size}"
    else:
        output_dir = Path(output_dir)

    return _save_tokenizer(tokenizer, output_dir, show_progress)


# =============================================================================
# Pre-tokenization
# =============================================================================


def pretokenize_dataset(
    dataset_name: str,
    tokenizer_path: Union[str, Path],
    max_length: int = 1024,
    output_dir: Optional[Path] = None,
    num_proc: Optional[int] = None,
    batch_size: int = 1000,
    show_progress: bool = True,
) -> Path:
    """
    Pre-tokenize a dataset and save to disk for fast training.

    Args:
        dataset_name: Name of dataset in registry
        tokenizer_path: Path to tokenizer directory or HuggingFace model name
        max_length: Maximum sequence length
        output_dir: Where to save (default: alongside dataset)
        num_proc: Number of processes (default: CPU count)
        batch_size: Batch size for tokenization
        show_progress: Whether to show progress

    Returns:
        Path to the saved pre-tokenized dataset directory
    """
    if num_proc is None:
        num_proc = os.cpu_count() or 1

    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.is_absolute():
        candidates = [
            tokenizer_path,
            get_datasets_dir() / "HuggingFaceFW" / "fineweb" / "tokenizers" / tokenizer_path,
            get_datasets_dir().parent / "models" / "tokenizers" / tokenizer_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                tokenizer_path = candidate
                break

    if show_progress:
        print(f"Loading tokenizer from: {tokenizer_path}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    if show_progress:
        print(f"  Vocab size: {len(tokenizer):,}")
        print(f"Loading dataset: {dataset_name}")

    train_dataset, val_dataset, text_column, dataset_path = load_dataset_from_registry(dataset_name)

    if show_progress:
        print(f"  Train examples: {len(train_dataset):,}")
        if val_dataset:
            print(f"  Val examples: {len(val_dataset):,}")

    if output_dir is None:
        tokenizer_name = tokenizer_path.name
        output_dir = (
            dataset_path.parent / f"{dataset_path.name}_pretokenized" / f"{tokenizer_name}_len{max_length}"
        )
    else:
        output_dir = Path(output_dir)

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    if show_progress:
        print(f"\nTokenizing train set with {num_proc} processes...")

    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train" if show_progress else None,
    )

    train_output = output_dir / "train"
    train_output.parent.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"Saving train to: {train_output}")

    train_tokenized.save_to_disk(str(train_output))

    if show_progress:
        print(f"  Saved {len(train_tokenized):,} tokenized examples")

    if val_dataset is not None:
        if show_progress:
            print(f"\nTokenizing validation set...")

        val_tokenized = val_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val" if show_progress else None,
        )

        val_output = output_dir / "validation"
        if show_progress:
            print(f"Saving validation to: {val_output}")

        val_tokenized.save_to_disk(str(val_output))

        if show_progress:
            print(f"  Saved {len(val_tokenized):,} tokenized examples")

    if show_progress:
        print(f"\nPre-tokenization complete!")
        print(f"Output directory: {output_dir}")

    return output_dir


def load_pretokenized(
    pretokenized_path: Union[str, Path],
    split: str = "train",
) -> Dataset:
    """
    Load a pre-tokenized dataset from disk.

    Args:
        pretokenized_path: Path to the pre-tokenized dataset directory
        split: Which split to load ('train' or 'validation')

    Returns:
        HuggingFace Dataset with tokenized data
    """
    pretokenized_path = Path(pretokenized_path)
    split_path = pretokenized_path / split

    if not split_path.exists():
        available = [d.name for d in pretokenized_path.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"Split '{split}' not found at {pretokenized_path}. " f"Available splits: {available}"
        )

    return load_from_disk(str(split_path))


def get_pretokenized_path(
    dataset_name: str,
    tokenizer_name: str,
    max_length: int,
    datasets_dir: Optional[Path] = None,
) -> Path:
    """Get the expected path for a pre-tokenized dataset."""
    config = get_dataset_config(dataset_name)
    datasets_dir = datasets_dir or get_datasets_dir()

    dataset_path = datasets_dir / config["path"]
    return dataset_path.parent / f"{dataset_path.name}_pretokenized" / f"{tokenizer_name}_len{max_length}"


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point with subcommands for analyze, train, and pretokenize."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tokenization toolkit: analyze, train, and pretokenize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze token distribution
  python -m tools.tokenization analyze tinystories --subset 10000

  # Train tokenizer from registry dataset
  python -m tools.tokenization train registry tinystories 4096

  # Train tokenizer from JSONL files
  python -m tools.tokenization train file corpus.jsonl 8192

  # Pre-tokenize a dataset
  python -m tools.tokenization pretokenize tinystories ./tokenizer --max-length 256
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- analyze subcommand ---
    analyze_parser = subparsers.add_parser("analyze", help="Analyze token distribution in a dataset")
    analyze_parser.add_argument("dataset", help="Dataset name in registry")
    analyze_parser.add_argument("--subset", type=int, help="Subset size for analysis")
    analyze_parser.add_argument("--export-cdf", help="Export CDF data to CSV file")

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train a custom BPE tokenizer")
    train_subparsers = train_parser.add_subparsers(dest="source", required=True)

    # train registry
    train_registry = train_subparsers.add_parser("registry", help="Train from dataset registry")
    train_registry.add_argument("dataset", help="Dataset name in registry")
    train_registry.add_argument("vocab_size", type=int, help="Target vocabulary size")
    train_registry.add_argument("--output", "-o", help="Output directory")
    train_registry.add_argument("--subset", type=int, help="Subset size for training")
    train_registry.add_argument("--min-freq", type=int, default=2, help="Min token frequency")

    # train file
    train_file = train_subparsers.add_parser("file", help="Train from JSONL file(s)")
    train_file.add_argument("files", nargs="+", help="JSONL file path(s)")
    train_file.add_argument("vocab_size", type=int, help="Target vocabulary size")
    train_file.add_argument("--output", "-o", help="Output directory")
    train_file.add_argument("--subset", type=int, help="Subset size for training")
    train_file.add_argument("--min-freq", type=int, default=2, help="Min token frequency")
    train_file.add_argument("--text-key", default="text", help="JSON key for text field")

    # --- pretokenize subcommand ---
    pretok_parser = subparsers.add_parser("pretokenize", help="Pre-tokenize a dataset")
    pretok_parser.add_argument("dataset", help="Dataset name in registry")
    pretok_parser.add_argument("tokenizer", help="Path to tokenizer or HF model name")
    pretok_parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    pretok_parser.add_argument("--output", "-o", help="Output directory")
    pretok_parser.add_argument("--num-proc", type=int, help="Number of processes")

    args = parser.parse_args()

    if args.command == "analyze":
        report, recommendations = analyze_dataset(args.dataset, args.subset)
        print(report.summary())
        print("\nRecommendations:")
        for rec in recommendations:
            print(rec.summary())
            print()
        if args.export_cdf:
            analyzer = TokenAnalyzer.__new__(TokenAnalyzer)
            analyzer.export_cdf_data(report, Path(args.export_cdf))
            print(f"CDF exported to: {args.export_cdf}")

    elif args.command == "train":
        if args.source == "registry":
            train_tokenizer(
                dataset_name=args.dataset,
                vocab_size=args.vocab_size,
                output_dir=args.output,
                subset_size=args.subset,
                min_frequency=args.min_freq,
            )
        elif args.source == "file":
            train_tokenizer_from_files(
                paths=args.files,
                vocab_size=args.vocab_size,
                output_dir=args.output,
                subset_size=args.subset,
                min_frequency=args.min_freq,
                text_key=args.text_key,
            )

    elif args.command == "pretokenize":
        pretokenize_dataset(
            dataset_name=args.dataset,
            tokenizer_path=args.tokenizer,
            max_length=args.max_length,
            output_dir=args.output,
            num_proc=args.num_proc,
        )


if __name__ == "__main__":
    main()
