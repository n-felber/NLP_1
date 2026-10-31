
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_NAME = "amarfurt/art-nlp-hs26"


def main():
    print(f"Downloading {DATASET_NAME}...")

    dataset = load_dataset(DATASET_NAME, cache_dir=str(DATA_DIR / "cache"))

    dataset.save_to_disk(str(DATA_DIR / "art"))

    for name, split in dataset.items():
        print(f"{name}: {len(split):,} examples")

    print("Dataset saved to data/art/")


if __name__ == "__main__":
    main()
