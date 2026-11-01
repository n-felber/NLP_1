
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = (
    DATA_DIR / "embeddings/word2vec-google-news-300"
    / "word2vec-google-news-300.gz"
)

DIM = 300
SEED = 42
RANDOM_WORDS = ("<UNK>", "to", "a", "and", "of", "'s", ",", "?", "!")
SPECIAL_WORDS = {"<PAD>", *RANDOM_WORDS}


def main():
    with (DATA_DIR / "vocab.json").open(encoding="utf-8") as file:
        vocab = json.load(file)

    # One row per vocabulary ID, with 300 dimensions.
    matrix = np.zeros((len(vocab), DIM), dtype=np.float32)

    # Fixed random vectors for the agreed OOV tokens.
    rng = np.random.default_rng(SEED)
    for word in RANDOM_WORDS:
        matrix[vocab[word]] = rng.normal(0, 0.1, DIM)

    # Match original case first, lowercase as fallback.
    candidates = defaultdict(list)
    for word in vocab:
        if word in SPECIAL_WORDS:
            continue

        candidates[word].append((word, 2))
        if word.lower() != word:
            candidates[word.lower()].append((word, 1))

    priority = np.zeros(len(vocab), dtype=np.uint8)

    print("Reading pretrained word2vec vectors...")

    with gzip.open(MODEL_PATH, "rb") as file:
        total, dimensions = map(int, file.readline().split())
        assert dimensions == DIM

        for _ in range(total):
            word_bytes = bytearray()

            while True:
                char = file.read(1)
                if not char:
                    raise EOFError("Unexpected end of file")
                if char == b" ":
                    break
                if char != b"\n":
                    word_bytes.extend(char)

            word = word_bytes.decode("utf-8", errors="replace")
            raw_vector = file.read(DIM * 4)

            if word in candidates:
                vector = np.frombuffer(raw_vector, dtype="<f4")

                for token, score in candidates[word]:
                    idx = vocab[token]
                    if score > priority[idx]:
                        matrix[idx] = vector
                        priority[idx] = score

    missing = [
        word for word in vocab
        if word not in SPECIAL_WORDS
        and priority[vocab[word]] == 0
    ]

    if missing:
        raise RuntimeError(
            f"Missing pretrained vectors: {missing[:10]}"
        )

    output = DATA_DIR / "embedding_matrix.npy"
    np.save(output, matrix)

    print(f"Embedding matrix shape: {matrix.shape}")
    print(f"Pretrained vectors: {(priority > 0).sum():,}")
    print(f"Random vectors: {len(RANDOM_WORDS)}")
    print(f"Padding vector: {matrix[vocab['<PAD>']].sum()}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
