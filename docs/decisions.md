# Decisions

## Embeddings

- **Available options:** word2vec, GloVe, fastText.
- **My choice:** Pretrained Google News word2vec (300 dimensions).
- **Reasoning / implications:** I chose word2vec because its pretrained vectors capture semantic relationships without needing to train embeddings from scratch. Google News provides a large English vocabulary and well-established 300-dimensional vectors. I will use the same frozen embeddings for both models; the checkpoint is relatively large, and out-of-vocabulary words need handling.

## Casing

- **Available options:** Preserve original casing, lowercase all text, or preserve casing with lowercase fallback.
- **My choice:** Preserve original casing; use the lowercase embedding only if the original-cased token is not found.
- **Reasoning / implications:** I want to retain potentially useful capitalization information. This strategy also gave the highest measured word2vec token coverage (88.32%, versus 87.49% with full lowercasing). I may compare validation performance with full lowercasing later.

## Tokenization

- **Available options:** Simple regex tokenizer or NLTK TreebankWordTokenizer.
- **My choice:** NLTK TreebankWordTokenizer.
- **Reasoning / implications:** I chose Treebank because it splits contractions such as `didn't` into `did` and `n't`, preserving negation as recognizable word2vec tokens. Its measured token coverage (88.21%) is very close to regex (88.32%).

## Punctuation

- **Available options:** Keep all punctuation, remove all punctuation, or selectively keep useful marks.
- **My choice:** Keep commas, question marks, and exclamation marks; remove periods and quotation marks. Preserve apostrophes within contractions (e.g., `n't`).
- **Reasoning / implications:** I keep punctuation that can signal clause structure, questions, or emphasis, while removing mostly redundant sentence endings and rare quotation marks. The ART fields already separate observations and hypotheses.

## Stopwords

- **Available options:** Remove stopwords or keep them.
- **My choice:** Keep all stopwords.
- **Reasoning / implications:** I want to preserve negations such as `not`, `never`, and `n't`, which can reverse the meaning of a hypothesis and change which explanation is correct. Other function words may also carry important relationships, so I will not apply stopword filtering.

## Stemming and Lemmatization

- **Available options:** Stemming, lemmatization, or neither.
- **My choice:** Neither; retain original word forms.
- **Reasoning / implications:** I want to preserve distinctions such as tense and number that could affect which hypothesis is plausible. The pretrained word2vec vectors also represent original word forms, so changing them could alter the intended embedding lookup.

## Truncation

- **Available options:** Truncate to a fixed maximum length or keep complete sequences.
- **My choice:** Do not truncate; keep every token.
- **Reasoning / implications:** I do not want to remove potentially decisive information. ART texts are short, so I will retain complete sequences and use padding for batching.

## Unknown and Rare Words

- **Available options:** Remove unknown words, map all unknown words to `<UNK>`, or give frequent unknown words separate embeddings.
- **My choice:** Keep all words with pretrained vectors, even if rare in ART. Give `to`, `a`, `and`, `of`, and `'s` distinct, fixed, randomly initialized 300-dimensional vectors; map other unknown words to one shared, fixed `<UNK>` vector. Use a separate zero vector for `<PAD>`.
- **Reasoning / implications:** These five words account for about 96.7% of unknown-token occurrences in the training data. Separate vectors preserve their identities, whereas a single `<UNK>` would make them indistinguishable. All embeddings remain frozen and are shared by both models; the random vectors carry no pretrained semantic meaning.

## Vocabulary

- **Available options:** Use the full pretrained word2vec vocabulary or build a vocabulary from ART training data only.
- **My choice:** Build the vocabulary from the training split only, retaining all its tokens after the agreed preprocessing.
- **Reasoning / implications:** I want a compact embedding matrix and a consistent way to handle words not seen during training. Tokens absent from the training vocabulary will map to `<UNK>` in validation and test, even if word2vec has a vector for them. This simplifies implementation, although it means I cannot benefit from pretrained vectors for those unseen tokens.

## Input Representation

- **Available options:** Represent O1, O2, H1, and H2 separately and concatenate them, or form and compare two candidate stories (`O1–H1–O2` and `O1–H2–O2`).
- **My choice:** Encode O1, O2, H1, and H2 separately, then concatenate their representations before the classifier.
- **Reasoning / implications:** I want the classifier to compare both hypotheses directly while keeping each field identifiable by its position in the input vector. With 300-dimensional field representations, the embedding-only classifier receives 1,200 features. Interactions between fields are learned by the classifier rather than during independent encoding.

## Sequence Representations

- **Available options:** Average or sum word embeddings for the embedding-only model; use an RNN’s final hidden state or pool its outputs for the RNN model.
- **My choice:** Use the mean of the word embeddings for each input field in the embedding-only model, and the final hidden state for each field in the RNN model.
- **Reasoning / implications:** Averaging keeps embedding values on a comparable scale across different sequence lengths. The RNN's final hidden state summarizes the sequence in order. I will exclude padding tokens from the average and use the actual final non-padding hidden state for the RNN.

## RNN Type

- **Available options:** GRU or LSTM.
- **My choice:** Use GRU for Model B, while keeping LSTM available for a possible later comparison.
- **Reasoning / implications:** I chose GRU because ART sequences are short and GRUs are simpler, with fewer parameters than comparable LSTMs. I use the same recurrent encoder for all four fields, processing each independently before concatenating their final hidden states. This keeps the model relatively efficient while retaining word-order information.

## Batching

- **Available options:** Small fixed batches, or the largest batch size that fits in GPU memory.
- **My choice:** Use the largest practical batch size that fits in memory, but keep multiple batches per epoch even if the entire dataset would fit. I will determine the final size after testing the model on the training hardware.
- **Reasoning / implications:** Larger batches can improve hardware utilization, but I still want multiple parameter updates per epoch. The feasible size depends on the model and GPU memory.

## Dynamic Padding and Packed Sequences

- **Available options:** Pad to a fixed global length or dynamically per batch; process padded RNN sequences or pack them.
- **My choice:** Dynamically pad each input field to the longest sequence in its batch using `pad_sequence`. For the RNN, use `pack_padded_sequence`; for mean pooling, exclude padding tokens.
- **Reasoning / implications:** I want to preserve complete sequences without wasting computation on padding. Packing lets the RNN process only active sequences at each timestep; the DataLoader batch size itself stays fixed.

## Classifier Output and Loss

- **Available options:** One output logit with `BCEWithLogitsLoss`, or two output logits with `CrossEntropyLoss`.
- **My choice:** Use two output logits (one for each hypothesis) and `CrossEntropyLoss`. Apply softmax only when probabilities are needed for interpretation or inference.
- **Reasoning / implications:** I want separate scores for H1 and H2 and to choose the higher-scoring hypothesis. `CrossEntropyLoss` applies log-softmax internally to the raw logits, so I must **not apply softmax before the loss** (which would apply normalization twice). I use softmax only when I need probabilities for interpretation or inference. I will map ART labels `1 → 0` and `2 → 1` for training.

## Optimizer

- **Available options:** Adam or AdamW.
- **My choice:** AdamW; start with weight decay `0.01` for the baseline.
- **Reasoning / implications:** I want adaptive optimization with decoupled weight decay. The initial weight decay is a baseline setting, not a tuned optimum.

## Initial Learning Rate

- **Available options:** Use a fixed initial learning rate or tune it immediately.
- **My choice:** Start at `1e-3` and compare learning rates in later experiments.
- **Reasoning / implications:** I want a reasonable baseline before spending time on hyperparameter optimization.

## Training Duration and Early Stopping

- **Available options:** Train for a fixed number of epochs or stop based on validation performance; monitor validation accuracy or validation loss.
- **My choice:** Set a maximum of 30 epochs and use early stopping with patience 8 based on **validation accuracy**. Save the checkpoint with the **highest validation accuracy**.
- **Reasoning / implications:** I want stopping and checkpoint selection to optimize the metric I care about: accuracy. I will maximize validation accuracy, retain the best-performing checkpoint, and continue logging validation loss for diagnostic purposes. A patience of 8 gives the first baseline run room to improve.

## Hyperparameter Optimization

- **Available options:** Manual tuning, grid/random search, W&B Sweeps, or Optuna's TPE-based Bayesian optimization.
- **My choice:** Use Optuna (TPE) to optimize hyperparameters jointly, with separate studies for Model A and GRU Model B. I will continue tracking each run in W&B and save the studies in SQLite to allow resuming.
- **Reasoning / implications:** Joint optimization can find interactions between settings that sequential tuning could miss. I will maximize the best validation accuracy, keep the test set untouched, and use an initial budget of roughly 20–30 trials per model if time allows.

## Hyperparameter Search Space

- **Available options:** Tune learning rate, weight decay, dropout, normalization, hidden dimensions, and batch size; tune them independently or jointly.
- **My choice:** Jointly search learning rate `1e-6`–`1e-3` (log scale), weight decay `{0, 1e-4, 1e-3, 1e-2, 1e-1}`, classifier dropout `0`–`0.5`, LayerNorm on/off, classifier hidden dimension `{128, 256, 512}`, and batch size `{64, 128, 256}`. For GRU, also search hidden dimension `{64, 128, 256}` and dropout between recurrent layers `0`–`0.4`.
- **Reasoning / implications:** Both baseline models overfit, so I want to test dropout and weight decay alongside model capacity and learning rate. LayerNorm is an optional normalization step rather than a substitute for regularization. Very small learning rates might require more epochs, so I will interpret undertrained trials carefully.
