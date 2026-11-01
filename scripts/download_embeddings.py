
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

os.environ["GENSIM_DATA_DIR"] = str(EMBEDDINGS_DIR)

import gensim.downloader as api

MODEL_NAME = "word2vec-google-news-300"


def main():
    print(f"Downloading {MODEL_NAME}...")

    path = api.load(MODEL_NAME, return_path=True)

    print(f"Embeddings saved to: {path}")


if __name__ == "__main__":
    main()
