
"""Train the frozen-word2vec baseline classifier on ART."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader

from src.data import ARTDataset, TEXT_FIELDS, collate_batch
from src.model_embeddings import EmbeddingClassifier

ROOT = Path(__file__).resolve().parents[1]


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_examples = 0

    with torch.set_grad_enabled(training):
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break

            inputs = {field: batch[field].to(device) for field in TEXT_FIELDS}
            targets = (batch["labels"] - 1).to(device)  # ART: 1/2 -> 0/1

            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)  # Raw logits, no softmax

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            n = targets.size(0)
            total_loss += loss.item() * n
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += n

    return total_loss / total_examples, total_correct / total_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    with (ROOT / "data/vocab.json").open(encoding="utf-8") as file:
        vocab = json.load(file)
    data = load_from_disk(str(ROOT / "data/art"))
    matrix = torch.from_numpy(np.load(ROOT / "data/embedding_matrix.npy"))
    model = EmbeddingClassifier(matrix, hidden_dim=args.hidden_dim).to(device)

    train_loader = DataLoader(
        ARTDataset(data["train"], vocab), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        ARTDataset(data["validation"], vocab), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_batch,
    )
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_name = (
        f"emb-lr{args.learning_rate:g}-bs{args.batch_size}"
        f"-wd{args.weight_decay:g}"
    )
    with wandb.init(
        project="nlp-hs26-project1", name=run_name,
        config={**vars(args), "architecture": "embeddings_mean_pool"},
        mode="disabled" if args.smoke_test else "online",
    ) as run:
        if args.smoke_test:
            train_loss, train_acc = run_epoch(
                model, train_loader, device, optimizer, max_batches=2,
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, device, max_batches=1,
            )
            print(f"Smoke test passed: train loss={train_loss:.4f}, "
                  f"val loss={val_loss:.4f}, val acc={val_acc:.3f}")
            return  # No checkpoint or W&B upload during smoke test

        checkpoint_dir = ROOT / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / f"model_a_{run.id}_best.pt"
        best_accuracy, stale_epochs = -1.0, 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device)

            run.log({
                "epoch": epoch,
                "train/loss": train_loss, "train/accuracy": train_acc,
                "val/loss": val_loss, "val/accuracy": val_acc,
            })
            print(f"Epoch {epoch:02d}: train acc={train_acc:.3f}, "
                  f"val acc={val_acc:.3f}, val loss={val_loss:.4f}")

            if val_acc > best_accuracy:
                best_accuracy, stale_epochs = val_acc, 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "config": vars(args),
                }, checkpoint_path)
                run.summary["best_val_accuracy"] = best_accuracy
                run.summary["best_epoch"] = epoch
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"Early stopping after {epoch} epochs")
                    break

        print(f"Best validation accuracy: {best_accuracy:.3f}")
        print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()
