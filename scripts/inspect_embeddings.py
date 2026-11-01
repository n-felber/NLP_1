
from pathlib import Path

from gensim.models import KeyedVectors

EMBEDDINGS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/embeddings/word2vec-google-news-300"
    / "word2vec-google-news-300.gz"
)

print("Loading sample of word2vec embeddings...")

# Load only the first 100,000 vectors to save memory.
model = KeyedVectors.load_word2vec_format(
    str(EMBEDDINGS_PATH),
    binary=True,
    limit=100_000,
)

print(f"Loaded words: {len(model):,}")
print(f"Vector dimensions: {model.vector_size}")

print("\n=== VOCABULARY CHECK ===")
for word in ["panda", "Panda", "zoo", "Zoo", "lost", "didn't"]:
    print(f"{word}: {word in model.key_to_index}")
