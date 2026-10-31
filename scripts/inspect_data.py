
from pathlib import Path
from pprint import pprint
from collections import Counter

from datasets import load_from_disk

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "art"

dataset = load_from_disk(str(DATA_DIR))
train = dataset["train"]

print("=== DATASET ===")
print(dataset)

print("\n=== FEATURES ===")
print(train.features)

print("\n=== EXAMPLES ===")
for i in range(3):
    print(f"\nExample {i + 1}:")
    pprint(train[i])


print("\n=== LABEL DISTRIBUTION ===")
counts = Counter(train["label"])

for label, count in sorted(counts.items()):
    print(f"Label {label}: {count:,} ({count / len(train):.1%})")


print("\n=== TEXT LENGTHS ===")
for column in ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2"]:
    lengths = sorted(len(text.split()) for text in train[column])

    print(
        f"{column}: "
        f"median={lengths[len(lengths) // 2]}, "
        f"95th percentile={lengths[int(0.95 * (len(lengths) - 1))]}, "
        f"max={lengths[-1]}"
    )
