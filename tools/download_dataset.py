"""HuggingFace utilities for downloading datasets."""

from pathlib import Path
from typing import Optional, Union

from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

from .dataset_registry import get_datasets_dir


def download_dataset(
    name: str,
    config: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    **kwargs,
):
    """
    Download a dataset from HuggingFace Hub.

    Args:
        name: Dataset name (e.g., 'roneneldan/TinyStories', 'HuggingFaceFW/fineweb')
        config: Dataset configuration/subset name (e.g., 'sample-10BT', 'de-en')
        output_dir: Where to save the dataset. If None, uses assets/datasets/{name}
        split: Specific split to download (e.g., 'train', 'test'). If None, downloads all.
        cache_dir: Custom cache directory. Defaults to ~/.cache/huggingface/datasets
        **kwargs: Additional arguments passed to load_dataset()

    Returns:
        Dataset object

    Example:
        >>> from tools.download_dataset import download_dataset
        >>> dataset = download_dataset('roneneldan/TinyStories', split='train')
        >>> dataset = download_dataset('HuggingFaceFW/fineweb', config='sample-100BT')
    """
    if cache_dir:
        cache_dir = Path(cache_dir).expanduser().resolve()

    if output_dir:
        output_dir = Path(output_dir).expanduser().resolve()
    else:
        # Default to assets/datasets/{name}
        safe_name = name.replace("/", "_")
        output_dir = get_datasets_dir() / safe_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {name}")
    if config:
        print(f"Config: {config}")
    if split:
        print(f"Split: {split}")

    dataset = load_dataset(
        name,
        name=config,  # 'name' parameter in load_dataset is actually the config
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
        **kwargs,
    )

    print(f"Saving to: {output_dir}")
    dataset.save_to_disk(str(output_dir))

    print(f"Dataset downloaded: {name}")
    return dataset


def main():
    """CLI interface for downloading datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets from HuggingFace Hub")
    parser.add_argument("name", help="Dataset name (e.g., 'roneneldan/TinyStories')")
    parser.add_argument("--config", "-c", help="Dataset configuration/subset")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--split", "-s", help="Specific split to download")
    parser.add_argument("--cache-dir", help="Cache directory")

    args = parser.parse_args()

    download_dataset(
        name=args.name,
        config=args.config,
        output_dir=args.output,
        split=args.split,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
