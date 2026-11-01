
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from src.data import ARTDataset, TEXT_FIELDS, collate_batch

ROOT = Path(__file__).resolve().parents[1]


class RNNClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        rnn_type="gru",
        rnn_hidden_dim=128,
        classifier_hidden_dim=256,
    ):
        super().__init__()

        if rnn_type not in ("gru", "lstm"):
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        # Same frozen word2vec embeddings as Model A.
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0,
        )

        embedding_dim = embedding_matrix.shape[1]
        rnn_class = nn.GRU if rnn_type == "gru" else nn.LSTM

        # Two-layer recurrent network, shared across all fields.
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(4 * rnn_hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(classifier_hidden_dim, 2),
        )

        self.rnn_type = rnn_type

    def encode_field(self, token_ids, lengths):
        """Return the final non-padding hidden state."""
        embeddings = self.embedding(token_ids)

        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, state = self.rnn(packed)

        # LSTM returns (hidden_state, cell_state).
        # GRU returns only hidden_state.
        if self.rnn_type == "lstm":
            hidden, _ = state
        else:
            hidden = state

        # Last layer's final hidden state: [batch, hidden_dim]
        return hidden[-1]

    def forward(self, batch):
        vectors = [
            self.encode_field(
                batch[field],
                batch[f"{field}_lengths"],
            )
            for field in TEXT_FIELDS
        ]

        features = torch.cat(vectors, dim=1)
        return self.classifier(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rnn", choices=("gru", "lstm"), default="gru"
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    matrix = torch.from_numpy(
        np.load(ROOT / "data/embedding_matrix.npy")
    )

    model = RNNClassifier(matrix, rnn_type=args.rnn)

    with (ROOT / "data/vocab.json").open(
        encoding="utf-8"
    ) as file:
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

    targets = batch["labels"] - 1
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    print("RNN type:", args.rnn.upper())
    print("Logits shape:", tuple(logits.shape))
    print("Probabilities:", probabilities.detach())
    print("Loss:", loss.item())
    print("Embeddings frozen:", not model.embedding.weight.requires_grad)
    print("RNN receives gradients:", model.rnn.weight_ih_l0.grad is not None)
    print("Classifier receives gradients:", model.classifier[0].weight.grad is not None)
