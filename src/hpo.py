"""Expanded, parallel Optuna TPE search for ART (fixed 1,024 batches).

New JournalStorage studies preserve all previous studies/checkpoints.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import optuna
import torch
import wandb
from datasets import load_from_disk
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from torch.utils.data import DataLoader

from src.data import ARTDataset, collate_batch
from src.model_embeddings import EmbeddingClassifier
from src.model_rnn import RNNClassifier
from src.train_embeddings import run_epoch as embedding_epoch
from src.train_rnn import run_epoch as rnn_epoch

ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 1024
CACHE_PATH = ROOT / "data/hpo_tokenized.pt"
STUDY_VERSION = "expanded_v2"


def suggest_params(trial, model_name):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.8),
        "layer_norm": trial.suggest_categorical("layer_norm", [False, True]),
        "classifier_hidden_dim": trial.suggest_categorical("classifier_hidden_dim", [64, 128, 256, 512]),
    }
    if model_name == "gru":
        params["rnn_hidden_dim"] = trial.suggest_categorical("rnn_hidden_dim", [16, 32, 64, 128, 256])
        params["rnn_dropout"] = trial.suggest_float("rnn_dropout", 0.0, 0.6)
        params["bidirectional"] = trial.suggest_categorical("bidirectional", [False, True])
    return params


def known_good_configs(model_name):
    """Two reproducible anchors; all values are inside the expanded search space."""
    if model_name == "embeddings":
        return [
            dict(learning_rate=0.0007764938896254405, weight_decay=0.1,
                 dropout=0.012205715178020535, layer_norm=True,
                 classifier_hidden_dim=128),
            dict(learning_rate=1e-3, weight_decay=0.01, dropout=0.0,
                 layer_norm=False, classifier_hidden_dim=256),
        ]
    return [
        dict(learning_rate=0.0009922445325797901, weight_decay=0.1,
             dropout=0.42536281377153073, layer_norm=True,
             classifier_hidden_dim=256, rnn_hidden_dim=64,
             rnn_dropout=0.37118651886594817, bidirectional=False),
        dict(learning_rate=1e-3, weight_decay=0.01, dropout=0.0,
             layer_norm=False, classifier_hidden_dim=256,
             rnn_hidden_dim=128, rnn_dropout=0.0, bidirectional=False),
    ]


def cache_signature():
    digest = hashlib.sha256()
    for name in ("data/vocab.json", "src/preprocessing.py", "data/art/dataset_dict.json"):
        path = ROOT / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def ensure_cache(rebuild=False):
    signature = cache_signature()
    if CACHE_PATH.exists() and not rebuild:
        cached = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
        if cached["signature"] == signature:
            print(f"Using token cache: {CACHE_PATH}", flush=True)
            return
    print("Tokenizing training and validation data once...", flush=True)
    with (ROOT / "data/vocab.json").open(encoding="utf-8") as f:
        vocab = json.load(f)
    splits = load_from_disk(str(ROOT / "data/art"))
    cached = {"signature": signature}
    for split in ("train", "validation"):
        dataset = ARTDataset(splits[split], vocab)
        cached[split] = [dataset[i] for i in range(len(dataset))]
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(".tmp")
    torch.save(cached, temporary)
    os.replace(temporary, CACHE_PATH)
    print(f"Cached {len(cached['train']):,} train examples and "
          f"{len(cached['validation']):,} validation examples.", flush=True)


def journal_path(model_name):
    return ROOT / "studies" / f"optuna_{model_name}_{STUDY_VERSION}.journal"


def create_study(args, worker_id=0, existing=0):
    path = journal_path(args.model)
    sampler = optuna.samplers.TPESampler(
        seed=args.seed + existing * 31 + worker_id,
        constant_liar=True,  # Penalize pending trials to avoid near-duplicates.
        multivariate=True,   # Learn interactions among hyperparameters.
    )
    return optuna.create_study(
        study_name=f"art_{args.model}_{STUDY_VERSION}_tpe",
        direction="maximize",
        storage=JournalStorage(JournalFileBackend(str(path))),
        sampler=sampler,
        load_if_exists=True,
    )


def run_worker(args, worker_id, n_trials, existing=0):
    # Each process initializes CUDA independently (spawn, never fork after CUDA).
    torch.set_num_threads(1)
    device_name = args.device
    if device_name == "auto":
        device_name = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)

    if args.smoke_test:
        with (ROOT / "data/vocab.json").open(encoding="utf-8") as f:
            vocab = json.load(f)
        splits = load_from_disk(str(ROOT / "data/art"))
        train_data = ARTDataset(splits["train"], vocab)
        val_data = ARTDataset(splits["validation"], vocab)
    else:
        cached = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
        train_data, val_data = cached["train"], cached["validation"]

    matrix = torch.from_numpy(np.load(ROOT / "data/embedding_matrix.npy"))
    epoch_fn = embedding_epoch if args.model == "embeddings" else rnn_epoch
    study_name = f"art_{args.model}_{STUDY_VERSION}_tpe"

    def objective(trial):
        params = suggest_params(trial, args.model)
        torch.manual_seed(args.seed)  # Same initialization seed across configurations.
        if args.model == "embeddings":
            model = EmbeddingClassifier(
                matrix, hidden_dim=params["classifier_hidden_dim"],
                dropout=params["dropout"], use_layer_norm=params["layer_norm"],
            )
        else:
            model = RNNClassifier(
                matrix, rnn_type="gru", rnn_hidden_dim=params["rnn_hidden_dim"],
                classifier_hidden_dim=params["classifier_hidden_dim"],
                dropout=params["dropout"], rnn_dropout=params["rnn_dropout"],
                use_layer_norm=params["layer_norm"],
                bidirectional=params["bidirectional"],
            )
        model = model.to(device)
        batch_size = 8 if args.smoke_test else BATCH_SIZE
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            collate_fn=collate_batch,
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch,
        )
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=params["learning_rate"], weight_decay=params["weight_decay"],
        )
        config = {
            **params, "batch_size": BATCH_SIZE, "architecture": args.model,
            "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
            "optuna_study": study_name, "optuna_trial": trial.number,
            "search_version": STUDY_VERSION,
        }
        with wandb.init(
            project="nlp-hs26-project1", entity=args.wandb_entity,
            group=f"hpo-{args.model}-{STUDY_VERSION}",
            name=f"hpo-{args.model}-{STUDY_VERSION}-t{trial.number:03d}",
            config=config, mode="disabled" if args.smoke_test else "online",
        ) as run:
            checkpoint = (ROOT / "checkpoints" /
                          f"hpo_{args.model}_{STUDY_VERSION}_t{trial.number:03d}_{run.id}_best.pt")
            if not args.smoke_test:
                checkpoint.parent.mkdir(exist_ok=True)
            best_accuracy, best_epoch, stale_epochs = -1.0, 0, 0
            for epoch in range(1, (1 if args.smoke_test else args.epochs) + 1):
                train_loss, train_acc = epoch_fn(
                    model, train_loader, device, optimizer,
                    max_batches=2 if args.smoke_test else None,
                )
                val_loss, val_acc = epoch_fn(
                    model, val_loader, device,
                    max_batches=1 if args.smoke_test else None,
                )
                # Very high suggested learning rates may diverge.
                if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                    run.summary["status"] = "pruned_nonfinite_loss"
                    raise optuna.TrialPruned("Non-finite train/validation loss")
                run.log({
                    "epoch": epoch, "train/loss": train_loss,
                    "train/accuracy": train_acc, "val/loss": val_loss,
                    "val/accuracy": val_acc,
                })
                print(f"[worker {worker_id}, trial {trial.number:03d}] "
                      f"epoch {epoch:02d}: train={train_acc:.3f} "
                      f"val={val_acc:.3f}", flush=True)
                if val_acc > best_accuracy:
                    best_accuracy, best_epoch, stale_epochs = val_acc, epoch, 0
                    if not args.smoke_test:
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "epoch": epoch, "val_accuracy": val_acc,
                            "config": config,
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
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
    else:
        study = create_study(args, worker_id, existing)
    study.optimize(objective, n_trials=n_trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("embeddings", "gru"), required=True)
    parser.add_argument("--trials", type=int, default=60, help="Target number of finished trials in the expanded study")
    parser.add_argument("--workers", type=int, default=1, help="Parallel OS processes (e.g. 2, 4, or 10)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--wandb-entity", default=None, help="Choose the SAME W&B team/entity for every experiment")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.workers < 1:
        parser.error("--trials and --workers must be positive")

    if args.smoke_test:
        run_worker(args, worker_id=0, n_trials=1)
        print("HPO smoke test passed.")
        return

    journal_path(args.model).parent.mkdir(parents=True, exist_ok=True)
    ensure_cache(args.rebuild_cache)
    study = create_study(args)
    if not study.trials:
        # Queue the best previous configurations (and baselines) before new suggestions.
        for params in known_good_configs(args.model):
            study.enqueue_trial(params, skip_if_exists=True)
        print("Queued previous winner and baseline for comparison.", flush=True)
    finished = sum(t.state.is_finished() for t in study.trials)
    remaining = max(0, args.trials - finished)
    workers = min(args.workers, remaining) if remaining else 0
    print(f"Study: {study.study_name}; finished: {finished}; "
          f"new: {remaining}; workers: {workers}; batch: {BATCH_SIZE}", flush=True)
    if workers:
        counts = [remaining // workers + (i < remaining % workers)
                  for i in range(workers)]
        # spawn avoids the CUDA-fork initialization problem.
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(run_worker, args, i, count, finished)
                       for i, count in enumerate(counts)]
            for future in futures:
                future.result()  # Surface worker exceptions.
    study = create_study(args, existing=finished + remaining)
    if study.best_trials:
        best = study.best_trial
        print(f"Best trial: {best.number}; accuracy: {best.value:.3f}")
        print(f"Parameters: {best.params}; fixed batch_size: {BATCH_SIZE}")
        print(f"Checkpoint: {best.user_attrs.get('checkpoint')}")
        print(f"Study saved: {journal_path(args.model)}")


if __name__ == "__main__":
    main()
