
import gzip
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

from src.preprocessing import tokenize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMBEDDINGS_PATH = (
    DATA_DIR / "embeddings/word2vec-google-news-300"
    / "word2vec-google-news-300.gz"
)

TEXT_COLUMNS = (
    "observation_1",
    "observation_2",
    "hypothesis_1",
    "hypothesis_2",
)

SPECIAL_OOV = ("to", "a", "and", "of", "'s", ",", "?", "!")


def find_known_words(candidates):
    """Scan pretrained vocabulary without loading vectors."""
    found = set()

    with gzip.open(EMBEDDINGS_PATH, "rb") as file:
        vocab_size, dimensions = map(
            int, file.readline().split()
        )

        for _ in range(vocab_size):
            word = bytearray()

            while True:
                char = file.read(1)
                if not char:
                    raise EOFError("Unexpected end of file")
                if char == b" ":
                    break
                if char != b"\n":
                    word.extend(char)

            token = word.decode("utf-8", errors="replace")

            if token in candidates:
                found.add(token)

            file.read(dimensions * 4)

    return found


def main():
    train = load_from_disk(str(DATA_DIR / "art"))["train"]

    counts = Counter(
        token
        for row in train
        for column in TEXT_COLUMNS
        for token in tokenize(row[column])
    )

    print(f"Training vocabulary: {len(counts):,} unique tokens")

    candidates = set(counts) | {
        word.lower() for word in counts
    }

    print("Checking pretrained word2vec vocabulary...")
    found = find_known_words(candidates)

    # Reserve IDs for special tokens.
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word in SPECIAL_OOV:
        vocab[word] = len(vocab)

    # Add words with pretrained embeddings.
    # Sorting ensures reproducible IDs.
    for word in sorted(counts):
        if word in vocab:
            continue

        if word in found or word.lower() in found:
            vocab[word] = len(vocab)

    output = DATA_DIR / "vocab.json"

    with output.open("w", encoding="utf-8") as file:
        json.dump(vocab, file, indent=2, ensure_ascii=False)

    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Saved to: {output}")

    print("\nSpecial token IDs:")
    for word in ("<PAD>", "<UNK>", *SPECIAL_OOV):
        print(f"  {word}: {vocab[word]}")


if __name__ == "__main__":
    main()
