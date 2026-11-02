# Final experiments, selections, and evaluation (HS26 Project 1)

This is a **retrospective record of the experiments actually run**. `docs/decisions.md` records the original decisions and earlier proposed HPO ranges; later experiments expanded those ranges. `EXPECTATIONS.md` preserves the initial test-accuracy predictions without post hoc changes.

## Dataset, metrics, and evaluation protocol

- HSLU ART course split: **16,801 train / 1,000 validation / 1,532 test**.
- Each example contains two observations and two candidate hypotheses; predict the correct hypothesis (label 1 or 2). Optimize **validation accuracy** and save each run's best-validation checkpoint.
- Use training data for parameter fitting and validation data for hyperparameter search, early stopping, and final model selection. Hold test out until the chosen checkpoints are frozen.
- Frozen pretrained Google News word2vec embeddings (300d) in both models; train the classifier only for Model A and the GRU plus classifier for Model B. Two logits with cross-entropy; AdamW.

## Preprocessing and architecture

Treebank tokenizer; casing retained; no stemming, lemmatization, stopword filtering or truncation. Keep `,`, `?`, `!` among standalone punctuation tokens. The **implemented** train-derived vocabulary (`scripts/build_vocab.py`) contains tokens matched to pretrained word2vec plus `<PAD>`, `<UNK>`, and special-OOV tokens (`to`, `a`, `and`, `of`, `'s`, `,`, `?`, `!`). All other tokens map to `<UNK>`, including unseen validation/test words. The special/OOV vectors are random but fixed; `<PAD>` has a zero vector. This is narrower than the earlier vocabulary plan in `docs/decisions.md`.

Each of O1/O2/H1/H2 is encoded individually. Model A mean-pools each field's non-padding word embeddings, concatenates four 300d vectors, and applies a two-layer ReLU classifier. Model B runs each field through the **same two-layer GRU**, uses the final non-padding hidden state, concatenates four vectors, and applies a two-layer ReLU classifier. Padding is dynamic; the GRU uses packed sequences. The search allowed bidirectionality, but the selected GRU is **unidirectional**.

## Development experiments

**Original baseline training:** learning rate 0.001, weight decay 0.01, batch size 128, up to 30 epochs, early stopping patience 8. Observed baseline validation accuracy: **65.3%** (Model A) and **68.4%** (GRU). These historical single-run results are distinct from the later matched-seed comparison.

**Expanded HPO:** two separate Optuna TPE/JournalStorage studies; **200 completed trials per architecture**; fixed batch size **1,024**; up to **50 epochs**; patience **8**; common training seed **42**. Search space:

| Parameter | Model A | GRU |
|---|---|---|
| Learning rate | 1e-6 to 0.1, log-uniform | same |
| AdamW weight decay | 0, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4 | same |
| Classifier dropout | 0–0.8 | same |
| Classifier hidden size | 64, 128, 256, 512 | same |
| LayerNorm | on/off | same |
| GRU hidden size | — | 16, 32, 64, 128, 256 |
| GRU inter-layer dropout | — | 0–0.6 |
| Bidirectional GRU | — | on/off |

Best individual validation scores: **66.7%** (Model A, trial 52, best epoch 38) and **70.9%** (GRU, trial 108, best epoch 26). TPE's adaptive sampling means trial-wise correlations and fANOVA importances are not causal effects. Only **11** of the 200 GRU trials were bidirectional; their lower observed scores cannot establish a controlled architectural disadvantage.

## Matched-seed stability and baseline comparison

Each configuration was trained on **seeds 42–46**, with identical batch size **1,024**, epoch cap **50**, patience **8**, data and selection criterion. The baseline configurations retain the original learning rate/regularization/hidden dimensions, but are **not exact repetitions of the historical batch-128 baseline training**.

| Model and configuration | Mean best val. accuracy | Sample SD (pp) | Individual best val. accuracies (seeds 42–46) |
|---|---:|---:|---|
| Model A — baseline | 65.74% | 0.52 | 65.6, 65.9, 64.9, 66.1, 66.2 |
| Model A — tuned | 65.44% | 0.93 | 66.7, 65.5, 65.0, 64.2, 65.8 |
| GRU — baseline | 66.50% | 0.61 | 66.6, 66.1, 65.8, 66.6, 67.4 |
| GRU — tuned | 68.22% | 1.57 | 70.9, 68.3, 67.1, 67.6, 67.2 |

The tuned-minus-baseline difference averaged **−0.30 percentage points for Model A** and **+1.72 pp for GRU**. The GRU improved in four of five seeds; the optimized GRU also beat the optimized Model A in all five. Only five seeds were tested, so these comparisons are descriptive rather than conclusive statistical evidence. Model A's baseline was selected for its marginally higher mean and lower variability; the tuned GRU was selected for its higher mean performance.

## Final checkpoint selection — made before test evaluation

| Model | Validation-selected checkpoint | Selection val. accuracy | Best epoch |
|---|---|---:|---:|
| Model A | `stability_baseline_embeddings_s46_tv4ie5ev_best.pt` | 66.20% | 28 |
| GRU | `hpo_gru_expanded_v2_t108_72n7o6dt_best.pt` | 70.90% | 26 |

Model A: original baseline hyperparameters, **seed 46**, classifier hidden size 256, AdamW LR 0.001 and weight decay 0.01, no classifier dropout or LayerNorm. GRU: **seed 42**, GRU hidden size 64, classifier hidden size 256, LR 0.0012430355, weight decay 0.1, classifier dropout 0.6990971, GRU inter-layer dropout 0.5905069, LayerNorm enabled, unidirectional.

## Final one-time test results

The selected models were evaluated on all **1,532 test examples** using `scripts/evaluate_final.py`. The pipeline was checked on validation and reproduced **66.20% and 70.90% exactly** for those same checkpoints.

| Measure | Model A | GRU |
|---|---:|---:|
| Selected checkpoint validation accuracy | 66.20% | 70.90% |
| **Final test accuracy** | **51.04%** | **53.85%** |
| Test correct | 782 / 1,532 | 825 / 1,532 |
| Test cross-entropy loss | 0.8827 | 0.9535 |
| Validation-to-test accuracy drop | 15.16 pp | 17.05 pp |

Test-set class counts: 781 label-1 and 751 label-2 examples; predicting the majority class every time would achieve **50.98%**. Confusion matrices (true rows, predicted columns, label order 1/2):

- Model A: `[[378, 403], [347, 404]]`.
- GRU: `[[414, 367], [340, 411]]`.

The GRU outperformed the embedding baseline on test by **2.81 percentage points** (43 additional correct predictions), but **both models were close to the test majority baseline** despite markedly higher validation accuracy. This indicates poor measured generalization from validation to test. A change in how the course's test hypotheses were constructed has been suggested as a possible explanation, but **that explanation has not been independently verified**. The reported test results were not used to retune or reselect a model.

Machine-readable results and SHA-256 hashes of the selected checkpoint files are retained in [`../reports/final_test/summary.json`](../reports/final_test/summary.json). The weight files themselves are stored outside Git.

## Experiment records and attribution

`src/hpo.py` implements expanded parallel HPO, `scripts/analyze_hpo.py` analyzes the studies and W&B histories, `src/stability.py` provides the five-seed controlled comparison, and `scripts/evaluate_final.py` evaluates the preselected test checkpoints. W&B logs train/validation loss and accuracy for each training run.

Original ART benchmark: Bhagavatula et al. (2020), *Abductive Commonsense Reasoning*. Course data: [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26). Embeddings: pretrained Google News word2vec via Gensim (Mikolov et al., 2013). Libraries/tools: PyTorch, Hugging Face Datasets, NLTK, Gensim, Optuna, W&B, `uv`. OpenAI ChatGPT assisted with implementation, planning, analysis and writing; experimental runs were executed by the student.
