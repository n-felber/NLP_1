
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_from_disk

from src.data import ARTDataset, TEXT_FIELDS, collate_batch

ROOT = Path(__file__).resolve().parents[1]


class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=256):
        super().__init__()

        # Pretrained embeddings remain frozen.
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0,
        )

        embedding_dim = embedding_matrix.shape[1]

        # Four separate representations, concatenated.
        self.classifier = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def mean_pool(self, token_ids):
        """Average embeddings, excluding padding."""
        embeddings = self.embedding(token_ids)

        mask = (token_ids != 0).unsqueeze(-1)

        summed = (embeddings * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)

        return summed / lengths

    def forward(self, batch):
        # Each field becomes one 300-dimensional vector.
        vectors = [
            self.mean_pool(batch[field])
            for field in TEXT_FIELDS
        ]

        # [batch_size, 4 * 300]
        features = torch.cat(vectors, dim=1)

        # [batch_size, 2] raw logits
        return self.classifier(features)


if __name__ == "__main__":
    import json

    torch.manual_seed(42)

    matrix = torch.from_numpy(
        np.load(ROOT / "data/embedding_matrix.npy")
    )

    model = EmbeddingClassifier(matrix)

    with (ROOT / "data/vocab.json").open() as file:
        vocab = json.load(file)

    train = load_from_disk(str(ROOT / "data/art"))["train"]

    loader = DataLoader(
        ARTDataset(train, vocab),
        batch_size=3,
        shuffle=False,
        collate_fn=collate_batch,
    )

    batch = next(iter(loader))

    logits = model(batch)
    probabilities = logits.softmax(dim=1)

    # Dataset labels 1/2 -> PyTorch labels 0/1.
    targets = batch["labels"] - 1
    loss = nn.CrossEntropyLoss()(logits, targets)

    loss.backward()

    print("Logits shape:", tuple(logits.shape))
    print("Probabilities:", probabilities.detach())
    print("Loss:", loss.item())
    print("Embeddings frozen:", not model.embedding.weight.requires_grad)
    print("Classifier receives gradients:", model.classifier[0].weight.grad is not None)
