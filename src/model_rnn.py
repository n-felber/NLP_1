"""Frozen word2vec + two-layer GRU/LSTM classifier for ART."""
import argparse
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.data import TEXT_FIELDS


class RNNClassifier(nn.Module):
    def __init__(
        self, embedding_matrix, rnn_type="gru", rnn_hidden_dim=128,
        classifier_hidden_dim=256, dropout=0.0, rnn_dropout=0.0,
        use_layer_norm=False, bidirectional=False,
    ):
        super().__init__()
        if rnn_type not in ("gru", "lstm"):
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True, padding_idx=0,
        )
        rnn_class = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_class(
            input_size=embedding_matrix.shape[1], hidden_size=rnn_hidden_dim,
            num_layers=2, batch_first=True, dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(4 * rnn_hidden_dim * directions, classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(classifier_hidden_dim, 2),
        )
        self.normalization = (
            nn.LayerNorm(classifier_hidden_dim) if use_layer_norm else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

    def encode_field(self, token_ids, lengths):
        embeddings = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, state = self.rnn(packed)
        hidden = state[0] if self.rnn_type == "lstm" else state
        # Final layer: two directions occupy the last two hidden-state slots.
        if self.bidirectional:
            return torch.cat([hidden[-2], hidden[-1]], dim=1)
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
        return self.classifier[2](hidden)


def main():
    parser = argparse.ArgumentParser(description="Test GRU/LSTM forward/backward")
    parser.add_argument("--rnn", choices=("gru", "lstm"), default="gru")
    parser.add_argument("--bidirectional", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    matrix = torch.randn(32, 300)
    matrix[0] = 0
    model = RNNClassifier(
        matrix, rnn_type=args.rnn, rnn_hidden_dim=16,
        classifier_hidden_dim=64, dropout=0.2, rnn_dropout=0.2,
        use_layer_norm=True, bidirectional=args.bidirectional,
    )
    batch = {}
    for field in TEXT_FIELDS:
        batch[field] = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
        batch[f"{field}_lengths"] = torch.tensor([3, 2, 1])
    logits = model(batch)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 0]))
    loss.backward()
    assert logits.shape == (3, 2)
    assert not model.embedding.weight.requires_grad
    assert any(p.grad is not None for p in model.rnn.parameters())
    assert model.classifier[0].in_features == (128 if args.bidirectional else 64)
    print(f"{args.rnn.upper()} bidirectional={args.bidirectional}: passed")
    print(f"Logits: {tuple(logits.shape)}; loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
