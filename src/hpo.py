"""Optuna TPE search for ART Model A (embeddings) and Model B (GRU)."""
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import torch
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader

from src.data import ARTDataset, collate_batch
from src.model_embeddings import EmbeddingClassifier
from src.model_rnn import RNNClassifier
from src.train_embeddings import run_epoch as embedding_epoch
from src.train_rnn import run_epoch as rnn_epoch

ROOT = Path(__file__).resolve().parents[1]


def suggest_params(trial, model_name):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3, 1e-2, 1e-1]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "layer_norm": trial.suggest_categorical("layer_norm", [False, True]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "classifier_hidden_dim": trial.suggest_categorical("classifier_hidden_dim", [128, 256, 512]),
    }
    if model_name == "gru":
        params["rnn_hidden_dim"] = trial.suggest_categorical("rnn_hidden_dim", [64, 128, 256])
        params["rnn_dropout"] = trial.suggest_float("rnn_dropout", 0.0, 0.4)
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("embeddings", "gru"), required=True)
    parser.add_argument("--trials", type=int, default=20, help="Target total trials in the saved study")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--smoke-test", action="store_true", help="Run one tiny trial without saving or contacting W&B")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Device: {device}; model: {args.model}")

    with (ROOT / "data/vocab.json").open(encoding="utf-8") as file:
        vocab = json.load(file)
    splits = load_from_disk(str(ROOT / "data/art"))
    train_data = ARTDataset(splits["train"], vocab)
    val_data = ARTDataset(splits["validation"], vocab)
    matrix = torch.from_numpy(np.load(ROOT / "data/embedding_matrix.npy"))
    epoch_fn = embedding_epoch if args.model == "embeddings" else rnn_epoch
    study_name = f"art_{args.model}_tpe"

    def objective(trial):
        params = suggest_params(trial, args.model)
        torch.manual_seed(args.seed)  # Reinitialize each trial reproducibly.
        if args.model == "embeddings":
            model = EmbeddingClassifier(
                matrix, hidden_dim=params["classifier_hidden_dim"],
                dropout=params["dropout"],
                use_layer_norm=params["layer_norm"],
            )
        else:
            model = RNNClassifier(
                matrix, rnn_type="gru",
                rnn_hidden_dim=params["rnn_hidden_dim"],
                classifier_hidden_dim=params["classifier_hidden_dim"],
                dropout=params["dropout"],
                rnn_dropout=params["rnn_dropout"],
                use_layer_norm=params["layer_norm"],
            )
        model = model.to(device)
        actual_batch_size = 8 if args.smoke_test else params["batch_size"]
        train_loader = DataLoader(
            train_data, batch_size=actual_batch_size, shuffle=True,
            collate_fn=collate_batch,
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_loader = DataLoader(
            val_data, batch_size=actual_batch_size, shuffle=False,
            collate_fn=collate_batch,
        )
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=params["learning_rate"], weight_decay=params["weight_decay"],
        )

        run_config = {
            **params, "architecture": args.model, "seed": args.seed,
            "epochs": args.epochs, "patience": args.patience,
            "optuna_study": study_name, "optuna_trial": trial.number,
        }
        with wandb.init(
            project="nlp-hs26-project1", group=f"hpo-{args.model}",
            name=f"hpo-{args.model}-t{trial.number:03d}",
            config=run_config,
            mode="disabled" if args.smoke_test else "online",
        ) as run:
            checkpoint = (ROOT / "checkpoints" /
                          f"hpo_{args.model}_t{trial.number:03d}_{run.id}_best.pt")
            if not args.smoke_test:
                checkpoint.parent.mkdir(exist_ok=True)
            best_accuracy, best_epoch, stale_epochs = -1.0, 0, 0
            for epoch in range(1, (1 if args.smoke_test else args.epochs) + 1):
                limit = 2 if args.smoke_test else None
                train_loss, train_acc = epoch_fn(
                    model, train_loader, device, optimizer, max_batches=limit
                )
                val_loss, val_acc = epoch_fn(
                    model, val_loader, device,
                    max_batches=1 if args.smoke_test else None,
                )
                run.log({
                    "epoch": epoch,
                    "train/loss": train_loss, "train/accuracy": train_acc,
                    "val/loss": val_loss, "val/accuracy": val_acc,
                })
                print(f"Trial {trial.number:03d}, epoch {epoch:02d}: "
                      f"train acc={train_acc:.3f}, val acc={val_acc:.3f}", flush=True)
                if val_acc > best_accuracy:
                    best_accuracy, best_epoch, stale_epochs = val_acc, epoch, 0
                    if not args.smoke_test:
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "epoch": epoch, "val_accuracy": val_acc,
                            "config": run_config,
                        }, checkpoint)
                else:
                    stale_epochs += 1
                    if stale_epochs >= args.patience:
                        break
            run.summary["best_val_accuracy"] = best_accuracy
            run.summary["best_epoch"] = best_epoch
            if not args.smoke_test:
                trial.set_user_attr("checkpoint", str(checkpoint))
                trial.set_user_attr("wandb_url", run.url)
                trial.set_user_attr("best_epoch", best_epoch)
            return best_accuracy

    if args.smoke_test:
        study = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective, n_trials=1)
        print("HPO smoke test passed.")
        return

    study_dir = ROOT / "studies"
    study_dir.mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=study_name, direction="maximize",
        storage=f"sqlite:///{study_dir / 'optuna.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.NopPruner(),  # Early stopping is already active.
    )
    remaining = max(0, args.trials - len(study.trials))
    print(f"Existing trials: {len(study.trials)}; new trials: {remaining}")
    if remaining:
        study.optimize(objective, n_trials=remaining)
    if study.best_trials:
        best = study.best_trial
        print(f"Best trial: {best.number}; accuracy: {best.value:.3f}")
        print(f"Parameters: {best.params}")
        print(f"Checkpoint: {best.user_attrs.get('checkpoint')}")
        print(f"Study saved: {study_dir / 'optuna.db'}")


if __name__ == "__main__":
    main()
