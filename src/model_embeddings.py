
"""Frozen word2vec mean-pooling classifier for ART."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data import TEXT_FIELDS

ROOT = Path(__file__).resolve().parents[1]


class EmbeddingClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        hidden_dim=256,
        dropout=0.0,
        use_layer_norm=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True, padding_idx=0
        )
        embedding_dim = embedding_matrix.shape[1]

        # Preserve the original classifier structure for old checkpoints.
        self.classifier = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.normalization = (
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

    def mean_pool(self, token_ids):
        embeddings = self.embedding(token_ids)
        mask = (token_ids != 0).unsqueeze(-1)
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def forward(self, batch):
        vectors = [self.mean_pool(batch[field]) for field in TEXT_FIELDS]
        features = torch.cat(vectors, dim=1)

        hidden = self.classifier[0](features)
        hidden = self.normalization(hidden)
        hidden = self.classifier[1](hidden)
        hidden = self.dropout(hidden)
        return self.classifier[2](hidden)  # Raw logits


if __name__ == "__main__":
    from datasets import load_from_disk
    from torch.utils.data import DataLoader
    from src.data import ARTDataset, collate_batch

    torch.manual_seed(42)
    matrix = torch.from_numpy(np.load(ROOT / "data/embedding_matrix.npy"))
    model = EmbeddingClassifier(matrix, dropout=0.2, use_layer_norm=True)

    with (ROOT / "data/vocab.json").open(encoding="utf-8") as file:
        vocab = json.load(file)
    train = load_from_disk(str(ROOT / "data/art"))["train"]
    batch = next(iter(DataLoader(
        ARTDataset(train, vocab), batch_size=3, collate_fn=collate_batch
    )))

    logits = model(batch)
    loss = nn.CrossEntropyLoss()(logits, batch["labels"] - 1)
    loss.backward()
    print("Logits shape:", tuple(logits.shape))
    print("Loss:", loss.item())
    print("Embeddings frozen:", not model.embedding.weight.requires_grad)
    print("Classifier receives gradients:", model.classifier[0].weight.grad is not None)
