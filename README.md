# NLP HS26 — Project 1

ART abductive commonsense reasoning with pretrained word2vec embeddings and PyTorch.

## Setup and Training

### 1. Clone and install dependencies

```bash
git clone https://github.com/n-felber/NLP_1.git
cd NLP_1

# Skip installation if uv is already available.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --locked
```

For a private repository, authenticate with GitHub before cloning.

### 2. Prepare data and embeddings (first setup only)

```bash
uv run python scripts/download_data.py
uv run python scripts/download_embeddings.py
uv run python -m scripts.build_vocab
uv run python scripts/build_embeddings.py
```

These commands download the ART dataset and pretrained Google News word2vec embeddings, build the training vocabulary, and create the embedding matrix.

All generated artifacts are saved in the gitignored `data/` directory. These steps only need to be repeated if the files are missing or the preprocessing changes.

### 3. Authenticate experiment tracking

```bash
uv run wandb login
```

Training experiments are logged to Weights & Biases.

### 4. Train Model A — Word Embeddings + Classifier

```bash
uv run python -m src.train_embeddings
```

Default training configuration:

- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.01
- Batch size: 128 (initial value, adjustable for GPU memory)
- Maximum epochs: 30
- Early stopping patience: 8
- Early stopping and checkpoint selection: highest validation accuracy

Training and validation loss/accuracy are logged to W&B. The best model checkpoint is saved locally in `checkpoints/`.

### 5. Train Model B — RNN + Classifier

*To be added after implementing Model B.*
