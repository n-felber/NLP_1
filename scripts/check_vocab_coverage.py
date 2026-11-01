
import gzip
import re
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/art"
EMBEDDINGS_PATH = (
    ROOT / "data/embeddings/word2vec-google-news-300"
    / "word2vec-google-news-300.gz"
)

TEXT_COLUMNS = [
    "observation_1",
    "observation_2",
    "hypothesis_1",
    "hypothesis_2",
]



from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()


def tokenize(text):
    tokens = tokenizer.tokenize(text)
    return [
        token for token in tokens
        if any(char.isalnum() for char in token)
    ]



def find_words_in_embeddings(words):
    """Scan the full vocabulary without loading vectors into RAM."""
    found = set()

    with gzip.open(EMBEDDINGS_PATH, "rb") as file:
        vocab_size, dimensions = map(int, file.readline().split())

        for _ in range(vocab_size):
            word = bytearray()

            while True:
                char = file.read(1)
                if not char:
                    raise EOFError("Unexpected end of embeddings file")
                if char == b" ":
                    break
                if char == b"\n" and not word:
                    continue
                word.extend(char)

            token = word.decode("utf-8", errors="replace")
            if token in words:
                found.add(token)

            # Skip the vector: we only need the vocabulary.
            file.read(dimensions * 4)

    return found


def main():
    train = load_from_disk(str(DATA_PATH))["train"]

    counts = Counter(
        token
        for row in train
        for column in TEXT_COLUMNS
        for token in tokenize(row[column])
    )

    print(f"Total tokens: {sum(counts.values()):,}")
    print(f"Unique tokens: {len(counts):,}")

    candidates = set(counts) | {word.lower() for word in counts}

    print("\nScanning Google News vocabulary...")
    found = find_words_in_embeddings(candidates)
    
    
    print("\n=== TOP 30 UNKNOWN WORDS ===")

    unknown = Counter({
        word: count
        for word, count in counts.items()
        if word not in found and word.lower() not in found
    })

    for word, count in unknown.most_common(30):
        print(f"{word:<25} {count:>5}")

    print(f"\nUnique unknown words: {len(unknown):,}")
    print(f"Total unknown occurrences: {sum(unknown.values()):,}")

    
    print("\n=== NEGATION TOKEN COVERAGE ===")
    for word in ["did", "n't", "not", "never"]:
        if word in candidates:
            print(f"{word!r}: {word in found}")


    strategies = {
        "Original case": lambda w: w in found,
        "Lowercase": lambda w: w.lower() in found,
        "Original + fallback": lambda w: (
            w in found or w.lower() in found
        ),
    }

    print("\n=== VOCABULARY COVERAGE ===")
    for name, matches in strategies.items():
        known_tokens = sum(
            count for word, count in counts.items() if matches(word)
        )
        known_unique = sum(matches(word) for word in counts)

        print(f"\n{name}")
        print(f"Token coverage: {known_tokens / sum(counts.values()):.2%}")
        print(f"Unique word coverage: {known_unique / len(counts):.2%}")


if __name__ == "__main__":
    main()
