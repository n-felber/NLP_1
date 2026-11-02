# NLP HS26 — Project 1: Abductive Commonsense Reasoning

**ART hypothesis selection with frozen word2vec embeddings and PyTorch.** This project compares a mean-pooled embedding classifier (Model A) with a two-layer GRU classifier (Model B) on the HSLU course split of ART. The implementation, training runs, and final evaluation were developed with assistance from **OpenAI ChatGPT**; the student reviewed the code, made project decisions, and ran the experiments.

## Data and task

For each example, the model receives observations `O1` and `O2` and two candidate intermediate explanations `H1` and `H2`. It predicts which hypothesis better explains the observations (label **1 or 2**). The course dataset [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26) contains **16,801 training**, **1,000 validation**, and **1,532 test** examples.

- **Preprocessing:** NLTK Treebank tokenization; retain casing, stopwords, negation and original word forms; selectively retain commas, `?`, and `!`; no truncation. Build the vocabulary from *training data only*. Keep tokens with a pretrained vector (original-case match, then lowercase fallback), plus reserved tokens. Other tokens map to `<UNK>`. `<PAD>` is a zero vector; `<UNK>` and reserved common/OOV tokens have fixed random vectors.
- **Batching:** Token IDs, dynamic padding per field and batch, and sequence lengths; ignore padding during mean pooling and pack sequences for the GRU.
- **Output:** Two logits, with labels converted from `1/2` to `0/1` for cross-entropy loss.

## Models

| | Model A — embeddings | Model B — GRU |
|---|---|---|
| Shared input | Frozen Google News word2vec (300 dimensions) | Same frozen word2vec |
| Field encoder | Mean of non-padding token vectors | Shared, two-layer **unidirectional GRU**; final hidden state |
| Combined features | Four 300-dimensional vectors concatenated (1,200 dimensions) | Four GRU representations concatenated |
| Classifier | Linear → ReLU → Linear (2 logits) | Linear → ReLU → Linear (2 logits) |
| Selected configuration | 256 classifier units; no dropout or LayerNorm | 64 GRU units; 256 classifier units; dropout and LayerNorm |

The four fields (`O1`, `O2`, `H1`, `H2`) are encoded **separately** before concatenation. The GRU shares parameters across fields; it does not concatenate all tokens into one sequence. Both models train only the non-embedding parameters.

## Results

The test set was evaluated **once**, after selecting checkpoints by validation performance. These are **accuracy percentages**, not scores optimized on test data.

| Model | Selected checkpoint validation | Final test | Correct on test |
|---|---:|---:|---:|
| Model A — baseline, seed 46 | 66.20% | **51.04%** | 782 / 1,532 |
| Model B — optimized GRU, trial 108 | 70.90% | **53.85%** | 825 / 1,532 |

The test split's majority-class baseline is **50.98%** (781/1,532). Both architectures suffer a substantial validation-to-test drop. Checkpoint loading and evaluation were checked by reproducing **66.20%** and **70.90%** on validation with the final evaluation pipeline. A difference between validation and test distributions is plausible but not established as the cause. **No model changes were made using test results.**

During development, Optuna TPE evaluated **200 trials per architecture**. To check whether the best individual scores were representative, baseline and tuned configurations were each retrained across five matched seeds (42–46):

| Configuration | Mean best validation accuracy | Sample SD (percentage points) |
|---|---:|---:|
| Model A, baseline | 65.74% | 0.52 |
| Model A, tuned | 65.44% | 0.93 |
| GRU, baseline | 66.50% | 0.61 |
| GRU, tuned | 68.22% | 1.57 |

Thus tuning did **not** improve Model A on average (−0.30 pp), while the GRU showed a **+1.72 pp** mean improvement (with uncertainty from five runs). See [`docs/final_experiments.md`](docs/final_experiments.md) for full experimental settings, interpretation, and checkpoint identifiers; [`docs/decisions.md`](docs/decisions.md) preserves the earlier design rationale and historical search-space decisions. Machine-readable final metrics and checkpoint hashes are in [`reports/final_test/summary.json`](reports/final_test/summary.json).

## Setup

Requirements: Python **3.12+**, [`uv`](https://docs.astral.sh/uv/), sufficient disk space for the Google News vectors, and an optional CUDA GPU for faster training. Dependencies are pinned through `uv.lock`.

```bash
git clone https://github.com/n-felber/NLP_1.git
cd NLP_1
uv sync --locked

# Prepare all local data; train/validation/test splits are downloaded together.
uv run python scripts/download_data.py
uv run python scripts/download_embeddings.py
uv run python -m scripts.build_vocab
uv run python scripts/build_embeddings.py
```

The downloads and generated data are **not versioned in Git**. The embedding download uses Gensim's `word2vec-google-news-300` model and may be large. All data preparation scripts are in `scripts/`.

## Running experiments

For the original training-script baselines (these default to batch size 128 and at most 30 epochs):

```bash
uv run wandb login
uv run python -m src.train_embeddings
uv run python -m src.train_rnn --rnn gru
```

The final **matched-seed** baseline comparison instead used **batch size 1,024**, at most **50 epochs**, and **patience 8** for both configurations; it is distinct from those original baseline runs. Its commands are:

```bash
uv run python -m src.stability --configuration baseline --model both --seeds 42 43 44 45 46 --epochs 50 --patience 8 --device cuda --wandb-entity YOUR_WANDB_ENTITY
uv run python -m src.stability --configuration tuned --model both --seeds 42 43 44 45 46 --epochs 50 --patience 8 --device cuda --wandb-entity YOUR_WANDB_ENTITY
```

The expanded HPO studies used `src.hpo` with `--model embeddings` or `--model gru`, `--trials 200`, a fixed batch size of 1,024, and up to 50 epochs. Parallel workers use Optuna JournalStorage. Both HPO and stability scripts operate on **train and validation**, not test. Metrics (`train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`), hyperparameters, and run identifiers are recorded in [Weights & Biases](https://forge.coreweave.com/wandb/noe-felber-hochschule-luzern/nlp-hs26-project1).

## Final evaluation and artifacts

The selected checkpoints were evaluated using:

```bash
uv run python -m scripts.evaluate_final --model both --device cuda
```

This script requires the **exact selected checkpoints** in `checkpoints/`, in addition to the prepared `data/` files. Checkpoints are **not included in this repository**; their filenames and SHA-256 digests are recorded in `reports/final_test/summary.json`. The command refuses to overwrite an existing test summary. Consequently, the archived metrics can be inspected directly, but exact checkpoint-level reproduction requires obtaining the original checkpoint files; retraining may produce different outcomes. Do not use the test split for further tuning.

## References and tools

- Bhagavatula et al. (2020), [*Abductive Commonsense Reasoning*](https://arxiv.org/abs/1908.05739); [HSLU ART split](https://huggingface.co/datasets/amarfurt/art-nlp-hs26).
- Mikolov et al. (2013), [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781); pretrained Google News word2vec vectors distributed via [Gensim Data](https://github.com/RaRe-Technologies/gensim-data).
- **Libraries/services:** PyTorch, Hugging Face Datasets, NLTK, Gensim, Optuna (TPE), Weights & Biases, `uv`; **AI assistance:** OpenAI ChatGPT for planning, coding, analysis, and documentation.
