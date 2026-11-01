
import json
from pathlib import Path

from nltk.tokenize import TreebankWordTokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = TreebankWordTokenizer()

KEEP_PUNCTUATION = {",", "?", "!"}


def tokenize(text):
    """Tokenize text according to our preprocessing decisions."""
    tokens = TOKENIZER.tokenize(text)

    return [
        token
        for token in tokens
        if any(char.isalnum() for char in token)
        or token in KEEP_PUNCTUATION
    ]


def encode(text, vocab):
    """Convert text into vocabulary IDs."""
    return [
        vocab.get(token, vocab["<UNK>"])
        for token in tokenize(text)
    ]


if __name__ == "__main__":
    from datasets import load_from_disk

    with (ROOT / "data/vocab.json").open(encoding="utf-8") as file:
        vocab = json.load(file)

    train = load_from_disk(str(ROOT / "data/art"))["train"]
    text = train[0]["observation_1"]

    print("Original:", text)
    print("Tokens:  ", tokenize(text))
    print("IDs:     ", encode(text, vocab))
