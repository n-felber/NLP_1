"""Compare original baseline and tuned configurations across matched random seeds.

Runs only on TRAIN and VALIDATION. The test split is never loaded.
"""
import argparse
import csv
import math
import random
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from src.data import collate_batch
from src.hpo import BATCH_SIZE, CACHE_PATH, ROOT, ensure_cache
from src.model_embeddings import EmbeddingClassifier
from src.model_rnn import RNNClassifier
from src.train_embeddings import run_epoch as embedding_epoch
from src.train_rnn import run_epoch as rnn_epoch

# Best configurations from the 200-trial expanded_v2 studies. No new tuning.
WINNERS = {
    "embeddings": {
        "learning_rate": 0.003996004654022659,
        "weight_decay": 0.0,
        "dropout": 0.0861815451720809,
        "layer_norm": False,
        "classifier_hidden_dim": 256,
    },
    "gru": {
        "learning_rate": 0.0012430354675090896,
        "weight_decay": 0.1,
        "dropout": 0.6990971478917894,
        "layer_norm": True,
        "classifier_hidden_dim": 256,
        "rnn_hidden_dim": 64,
        "rnn_dropout": 0.5905068882070185,
        "bidirectional": False,
    },
}


# Original training-script defaults, except batch size/epoch cap (controlled below).
# Baseline: original hyperparameters, but same data pipeline, batch size 1024,
# maximum 50 epochs, patience 8, and seeds as the tuned stability experiment.
BASELINES = {
    "embeddings": {
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "dropout": 0.0,
        "layer_norm": False,
        "classifier_hidden_dim": 256,
    },
    "gru": {
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "dropout": 0.0,
        "layer_norm": False,
        "classifier_hidden_dim": 256,
        "rnn_hidden_dim": 128,
        "rnn_dropout": 0.0,
        "bidirectional": False,
    },
}


def build_model(name, embedding_matrix, config):
    if name == "embeddings":
        return EmbeddingClassifier(
            embedding_matrix, hidden_dim=config["classifier_hidden_dim"],
            dropout=config["dropout"], use_layer_norm=config["layer_norm"],
        )
    return RNNClassifier(
        embedding_matrix, rnn_type="gru",
        rnn_hidden_dim=config["rnn_hidden_dim"],
        classifier_hidden_dim=config["classifier_hidden_dim"],
        dropout=config["dropout"], rnn_dropout=config["rnn_dropout"],
        use_layer_norm=config["layer_norm"],
        bidirectional=config["bidirectional"],
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(name, seed, args, train_data, val_data, matrix, device):
    seed_everything(seed)
    config = (BASELINES if args.configuration == "baseline" else WINNERS)[name]
    model = build_model(name, matrix, config).to(device)
    train = DataLoader(
        train_data, batch_size=8 if args.smoke_test else BATCH_SIZE,
        shuffle=True, collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(seed),
    )
    val = DataLoader(
        val_data, batch_size=8 if args.smoke_test else BATCH_SIZE,
        shuffle=False, collate_fn=collate_batch,
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    epoch_fn = embedding_epoch if name == "embeddings" else rnn_epoch
    name_for_run = f"stability-{args.configuration}-{name}-seed{seed}"
    with wandb.init(
        project="nlp-hs26-project1", entity=args.wandb_entity,
        group=f"stability-{args.configuration}-{name}", name=name_for_run,
        config={**config, "seed": seed, "architecture": name,
                "epochs": args.epochs, "patience": args.patience,
                "batch_size": BATCH_SIZE, "configuration": args.configuration,
                "source_study": ("original_baseline_hparams" if
                                  args.configuration == "baseline" else "expanded_v2")},
        mode="disabled" if args.smoke_test else "online",
    ) as run:
        checkpoint = ROOT / "checkpoints" / f"stability_{args.configuration}_{name}_s{seed}_{run.id}_best.pt"
        if not args.smoke_test:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
        best_acc, best_epoch, stale = -1.0, 0, 0
        last_epoch = 0
        for epoch in range(1, 2 if args.smoke_test else args.epochs + 1):
            max_batches = 2 if args.smoke_test else None
            train_loss, train_acc = epoch_fn(
                model, train, device, optimizer, max_batches=max_batches
            )
            val_loss, val_acc = epoch_fn(
                model, val, device, max_batches=1 if args.smoke_test else None
            )
            if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
                raise RuntimeError(f"Non-finite loss: {name}, seed={seed}, epoch={epoch}")
            run.log({"epoch": epoch, "train/loss": train_loss,
                     "train/accuracy": train_acc, "val/loss": val_loss,
                     "val/accuracy": val_acc})
            print(f"{name}, seed {seed}, epoch {epoch:02d}: "
                  f"train={train_acc:.3f} val={val_acc:.3f}", flush=True)
            last_epoch = epoch
            if val_acc > best_acc:
                best_acc, best_epoch, stale = val_acc, epoch, 0
                if not args.smoke_test:
                    torch.save({"model_state_dict": model.state_dict(),
                                "epoch": epoch, "val_accuracy": val_acc,
                                "config": {**config, "seed": seed,
                                           "batch_size": BATCH_SIZE}}, checkpoint)
            else:
                stale += 1
                if stale >= args.patience:
                    break
        run.summary["best_val_accuracy"] = best_acc
        run.summary["best_epoch"] = best_epoch
        run.summary["last_epoch"] = last_epoch
        return {"model": name, "configuration": args.configuration, "seed": seed,
                "best_val_accuracy": best_acc,
                "best_epoch": best_epoch, "last_epoch": last_epoch,
                "checkpoint": str(checkpoint) if not args.smoke_test else "",
                "wandb_url": run.url if not args.smoke_test else ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("both", "embeddings", "gru"), default="both")
    parser.add_argument("--configuration", choices=("tuned", "baseline"), default="tuned")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--wandb-entity", default="noe-felber-hochschule-luzern")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Epochs, patience and seeds must be positive; seeds must be unique")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    torch.set_num_threads(1)
    ensure_cache()
    cached = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    train_data = cached["train"][:16] if args.smoke_test else cached["train"]
    val_data = cached["validation"][:16] if args.smoke_test else cached["validation"]
    matrix = torch.from_numpy(np.load(ROOT / "data" / "embedding_matrix.npy"))
    models = ("embeddings", "gru") if args.model == "both" else (args.model,)
    rows = []
    report_dir = ROOT / "reports" / "stability"
    output_csv = report_dir / ("baseline_results.csv" if args.configuration == "baseline"
                               else "results.csv")
    if not args.smoke_test:
        report_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        for seed in args.seeds:
            rows.append(run_one(model, seed, args, train_data, val_data, matrix, device))
            if not args.smoke_test:
                with output_csv.open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        values = [r["best_val_accuracy"] for r in rows if r["model"] == model]
        std = stdev(values) if len(values) > 1 else 0.0
        print(f"{model}: mean={mean(values):.3%} ± {std:.3%} SD, "
              f"min={min(values):.3%}, max={max(values):.3%}")
    if args.smoke_test:
        print("Stability smoke test passed (not a meaningful accuracy estimate).")
    else:
        print(f"Saved all runs to {output_csv}")


if __name__ == "__main__":
    main()
