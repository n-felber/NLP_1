"""One-time test evaluation of the two validation-selected ART checkpoints.

No training, hyperparameter search, validation peeking, or model selection here.
Run only AFTER the checkpoint choices have been frozen.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader

from src.data import ARTDataset, TEXT_FIELDS, collate_batch
from src.model_embeddings import EmbeddingClassifier
from src.model_rnn import RNNClassifier

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "embeddings": "stability_baseline_embeddings_s46_tv4ie5ev_best.pt",
    "gru": "hpo_gru_expanded_v2_t108_72n7o6dt_best.pt",
}


def make_model(name, matrix, config):
    if name == "embeddings":
        return EmbeddingClassifier(
            matrix, hidden_dim=config["classifier_hidden_dim"],
            dropout=config.get("dropout", 0.0),
            use_layer_norm=config.get("layer_norm", False),
        )
    return RNNClassifier(
        matrix, rnn_type="gru", rnn_hidden_dim=config["rnn_hidden_dim"],
        classifier_hidden_dim=config["classifier_hidden_dim"],
        dropout=config.get("dropout", 0.0),
        rnn_dropout=config.get("rnn_dropout", 0.0),
        use_layer_norm=config.get("layer_norm", False),
        bidirectional=config.get("bidirectional", False),
    )


def evaluate(name, checkpoint_path, dataset, matrix, device, batch_size):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = make_model(name, matrix, config)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_batch)
    count = correct = loss_sum = 0
    confusion = [[0, 0], [0, 0]]  # rows: true label 1/2; columns: prediction 1/2
    errors = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {f: batch[f].to(device) for f in TEXT_FIELDS}
            if name == "gru":
                inputs.update({f + "_lengths": batch[f + "_lengths"]
                               for f in TEXT_FIELDS})
            targets = (batch["labels"] - 1).to(device)
            logits = model(inputs)
            loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
            predictions = logits.argmax(dim=1).cpu()
            gold = targets.cpu()
            for local_index, (g, p) in enumerate(zip(gold.tolist(), predictions.tolist())):
                confusion[g][p] += 1
                if g != p:
                    original = dataset.split[count + local_index]
                    errors.append({
                        "index": count + local_index, "gold": g + 1,
                        "predicted": p + 1,
                        **{f: original[f] for f in TEXT_FIELDS},
                    })
            correct += (predictions == gold).sum().item()
            count += len(gold)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "model": name, "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": digest, "selection_val_accuracy": saved.get("val_accuracy"),
        "selection_epoch": saved.get("epoch"), "config": config,
        "test_examples": count, "test_correct": correct,
        "test_accuracy": correct / count, "test_loss": loss_sum / count,
        "confusion_matrix_true_rows_pred_cols": confusion,
        "test_recall_label_1": confusion[0][0] / sum(confusion[0]),
        "test_recall_label_2": confusion[1][1] / sum(confusion[1]),
        "errors": errors,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["both", *CHECKPOINTS], default="both")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--out-dir", type=Path, default=ROOT / "reports/final_test")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    models = list(CHECKPOINTS) if args.model == "both" else [args.model]
    paths = {name: args.checkpoint_dir / CHECKPOINTS[name] for name in models}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        p.error("Missing selected checkpoints: " + ", ".join(missing))
    output = args.out_dir / "summary.json"
    if output.exists():
        p.error(f"Test already evaluated at {output}. Do not rerun or retune on test results.")
    for required in (ROOT / "data/art", ROOT / "data/vocab.json",
                     ROOT / "data/embedding_matrix.npy"):
        if not required.exists():
            p.error(f"Missing prepared data: {required}")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    with (ROOT / "data/vocab.json").open(encoding="utf-8") as file:
        vocab = json.load(file)
    dataset = ARTDataset(load_from_disk(str(ROOT / "data/art"))["test"], vocab)
    matrix = torch.from_numpy(np.load(ROOT / "data/embedding_matrix.npy"))
    results = {}
    for name in models:
        result = evaluate(name, paths[name], dataset, matrix,
                          torch.device(device), args.batch_size)
        errors = result.pop("errors")
        results[name] = result
        print(f"{name}: test accuracy={result['test_accuracy']:.2%} "
              f"({result['test_correct']}/{result['test_examples']})")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / f"errors_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "gold", "predicted", *TEXT_FIELDS])
            writer.writeheader()
            writer.writerows(errors)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Final results written once to: {output}")


if __name__ == "__main__":
    main()
