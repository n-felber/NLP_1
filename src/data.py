
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import encode

ROOT = Path(__file__).resolve().parents[1]

TEXT_FIELDS = (
    "observation_1",
    "observation_2",
    "hypothesis_1",
    "hypothesis_2",
)


class ARTDataset(Dataset):
    def __init__(self, split, vocab):
        self.split = split
        self.vocab = vocab

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        row = self.split[index]

        result = {
            field: torch.tensor(
                encode(row[field], self.vocab),
                dtype=torch.long,
            )
            for field in TEXT_FIELDS
        }

        result["label"] = row["label"]
        return result


def collate_batch(examples):
    batch = {
        "labels": torch.tensor(
            [example["label"] for example in examples],
            dtype=torch.long,
        )
    }

    for field in TEXT_FIELDS:
        sequences = [example[field] for example in examples]
        lengths = torch.tensor(
            [len(sequence) for sequence in sequences],
            dtype=torch.long,
        )

        if (lengths == 0).any():
            raise ValueError(f"Empty sequence in {field}")

        batch[field] = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0,
        )
        batch[f"{field}_lengths"] = lengths

    return batch


if __name__ == "__main__":
    with (ROOT / "data/vocab.json").open() as file:
        vocab = json.load(file)

    train = load_from_disk(str(ROOT / "data/art"))["train"]

    loader = DataLoader(
        ARTDataset(train, vocab),
        batch_size=3,  # Small batch for testing only
        shuffle=False,
        collate_fn=collate_batch,
    )

    batch = next(iter(loader))

    print("=== BATCH ===")
    for field in TEXT_FIELDS:
        print(
            f"{field}: "
            f"shape={tuple(batch[field].shape)}, "
            f"lengths={batch[field + '_lengths'].tolist()}"
        )

    print("Labels:", batch["labels"].tolist())

    # Demonstrate packing with the first field.
    packed = pack_padded_sequence(
        batch["observation_1"],
        batch["observation_1_lengths"],
        batch_first=True,
        enforce_sorted=False,
    )

    print("Active batch sizes:", packed.batch_sizes.tolist())
