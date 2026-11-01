
"""Frozen word2vec + two-layer GRU/LSTM classifier for ART."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.data import TEXT_FIELDS

ROOT = Path(__file__).resolve().parents[1]


class RNNClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        rnn_type="gru",
        rnn_hidden_dim=128,
        classifier_hidden_dim=256,
        dropout=0.0,
        rnn_dropout=0.0,
        use_layer_norm=False,
    ):
        super().__init__()
        if rnn_type not in ("gru", "lstm"):
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True, padding_idx=0
        )
        rnn_class = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=embedding_matrix.shape[1],
            hidden_size=rnn_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=rnn_dropout,  # Between RNN layers only
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * rnn_hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(classifier_hidden_dim, 2),
        )
        self.normalization = (
            nn.LayerNorm(classifier_hidden_dim)
            if use_layer_norm else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.rnn_type = rnn_type

    def encode_field(self, token_ids, lengths):
        embeddings = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.rnn(packed)
        hidden = state[0] if self.rnn_type == "lstm" else state
        return hidden[-1]

    def forward(self, batch):
        vectors = [
            self.encode_field(batch[field], batch[f"{field}_lengths"])
            for field in TEXT_FIELDS
        ]
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn", choices=("gru", "lstm"), default="gru")
    args = parser.parse_args()

    torch.manual_seed(42)
    matrix = torch.from_numpy(np.load(ROOT / "data/embedding_matrix.npy"))
    model = RNNClassifier(
        matrix, rnn_type=args.rnn, dropout=0.2,
        rnn_dropout=0.2, use_layer_norm=True
    )

    with (ROOT / "data/vocab.json").open(encoding="utf-8") as file:
        vocab = json.load(file)
    train = load_from_disk(str(ROOT / "data/art"))["train"]
    batch = next(iter(DataLoader(
        ARTDataset(train, vocab), batch_size=3, collate_fn=collate_batch
    )))

    logits = model(batch)
    loss = nn.CrossEntropyLoss()(logits, batch["labels"] - 1)
    loss.backward()
    print("RNN type:", args.rnn.upper())
    print("Logits shape:", tuple(logits.shape))
    print("Loss:", loss.item())
    print("Embeddings frozen:", not model.embedding.weight.requires_grad)
    print("RNN receives gradients:", model.rnn.weight_ih_l0.grad is not None)
    print("Classifier receives gradients:", model.classifier[0].weight.grad is not None)
