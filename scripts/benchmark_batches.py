"""Benchmark batch sizes and CPU tokenization overhead on GPUHub."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from src.data import ARTDataset, collate_batch
from src.model_embeddings import EmbeddingClassifier
from src.model_rnn import RNNClassifier
from src.train_embeddings import run_epoch as embedding_epoch
from src.train_rnn import run_epoch as rnn_epoch

ROOT = Path(__file__).resolve().parents[1]


def measure(dataset, batch_size, model_name, matrix, device):
    torch.manual_seed(42)
    if model_name == "embeddings":
        model = EmbeddingClassifier(matrix, hidden_dim=256).to(device)
        epoch_fn = embedding_epoch
    else:
        model = RNNClassifier(matrix, rnn_type="gru", rnn_hidden_dim=128).to(device)
        epoch_fn = rnn_epoch
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(42),
    )

    # Initialize CUDA/cuDNN before starting the timed epoch.
    epoch_fn(model, loader, device, optimizer, max_batches=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    epoch_fn(model, loader, device, optimizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_mb = float("nan")
    seconds = time.perf_counter() - start
    print(
        f"{batch_size:>5} | {seconds:>7.2f} s | "
        f"{len(dataset)/seconds:>8.0f} examples/s | "
        f"{peak_mb:>7.1f} MiB peak PyTorch allocation",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("embeddings", "gru"), default="embeddings")
    parser.add_argument("--batches", nargs="+", type=int,
                        default=[128, 256, 512, 1024])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; architecture: {args.model}")
    with (ROOT / "data/vocab.json").open() as f:
        vocab = json.load(f)
    split = load_from_disk(str(ROOT / "data/art"))["train"]
    dataset = ARTDataset(split, vocab)
    matrix = torch.from_numpy(np.load(ROOT / "data/embedding_matrix.npy"))

    print("\nUncached data, current default batch size:")
    measure(dataset, 128, args.model, matrix, device)

    print("\nCaching tokenized examples once (CPU RAM):")
    start = time.perf_counter()
    cached = [dataset[i] for i in range(len(dataset))]
    print(f"Cached {len(cached):,} examples in {time.perf_counter()-start:.2f} s")

    print("\nCached data, batch-size comparison:")
    for size in args.batches:
        measure(cached, size, args.model, matrix, device)


if __name__ == "__main__":
    main()
