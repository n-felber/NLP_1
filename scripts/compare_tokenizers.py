
import re
from pathlib import Path

from datasets import load_from_disk
from nltk.tokenize import TreebankWordTokenizer

DATA_DIR = Path(__file__).resolve().parents[1] / "data/art"

train = load_from_disk(str(DATA_DIR))["train"]

treebank = TreebankWordTokenizer()


def regex_tokenize(text):
    return re.findall(r"\b\w+(?:['’]\w+)*\b", text)


examples = [
    train[0]["observation_1"],
    train[0]["hypothesis_2"],
    train[2]["observation_2"],
    train[2]["hypothesis_1"],
]

for text in examples:
    print(f"\nOriginal: {text}")
    print(f"Regex:    {regex_tokenize(text)}")
    print(f"Treebank: {treebank.tokenize(text)}")
