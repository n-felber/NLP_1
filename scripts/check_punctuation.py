
import re
from pathlib import Path

from datasets import load_from_disk

DATA_DIR = Path(__file__).resolve().parents[1] / "data/art"
train = load_from_disk(str(DATA_DIR))["train"]

COLUMNS = [
    "observation_1",
    "observation_2",
    "hypothesis_1",
    "hypothesis_2",
]

# Approximate sentence boundaries (e.g. "Hello. She left.")
SENTENCE_BOUNDARY = re.compile(r'[.!?]["”]?\s+(?=["“]?[A-Z])')

CHECKS = {
    "Comma": lambda text: "," in text,
    "Quotation marks": lambda text: any(
        char in text for char in '"“”'
    ),
    "Multiple sentences": lambda text: bool(
        SENTENCE_BOUNDARY.search(text)
    ),
}

for name, check in CHECKS.items():
    print(f"\n=== {name.upper()} ===")

    for column in COLUMNS:
        matches = [text for text in train[column] if check(text)]

        print(
            f"{column}: {len(matches):,}/{len(train):,} "
            f"({len(matches) / len(train):.2%})"
        )

        for example in matches[:2]:
            print(f"  Example: {example}")
