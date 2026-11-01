"""Analyze completed expanded_v2 Optuna HPO studies; optional W&B stop epochs.

Run from project root: uv run python scripts/analyze_hpo.py --with-wandb
"""
import argparse
import csv
from pathlib import Path
from statistics import mean, median

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

ROOT = Path(__file__).resolve().parents[1]
VERSION = "expanded_v2"


def pct(value):
    return f"{100 * value:.2f}%" if value is not None else "—"


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_wandb_epochs(model, entity):
    import wandb

    group = f"hpo-{model}-{VERSION}"
    runs = wandb.Api(timeout=60).runs(
        f"{entity}/nlp-hs26-project1", filters={"group": group}, per_page=50
    )
    result = {}
    for run in runs:
        config = run.config
        if config.get("optuna_study") not in (None, f"art_{model}_{VERSION}_tpe"):
            continue
        number = config.get("optuna_trial")
        if number is None and run.name and "-t" in run.name:
            suffix = run.name.rsplit("-t", 1)[-1]
            number = int(suffix) if suffix.isdigit() else None
        if number is None:
            continue
        summary = run.summary
        result[int(number)] = {
            "last_epoch": summary.get("epoch"),
            "wandb_best_epoch": summary.get("best_epoch"),
            "wandb_best_accuracy": summary.get("best_val_accuracy"),
        }
    return result


def summarize_distribution(rows, column="value"):
    values = np.array([r[column] for r in rows if r[column] is not None], dtype=float)
    if not len(values):
        return "No completed trials"
    return (f"n={len(values)}, median={pct(float(np.median(values)))}, "
            f"IQR={pct(float(np.percentile(values, 25)))}–"
            f"{pct(float(np.percentile(values, 75)))}, "
            f"best={pct(float(np.max(values)))}")


def plot_figures(rows, model, out, importance):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping PNG plots (CSV and report still saved).")
        return

    complete = sorted((r for r in rows if r["value"] is not None),
                      key=lambda r: r["trial"])
    if complete:
        xs = [r["trial"] for r in complete]
        ys = [100 * r["value"] for r in complete]
        best_so_far = np.maximum.accumulate(ys)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.scatter(xs, ys, s=12, alpha=0.45, label="Completed trial")
        ax.plot(xs, best_so_far, linewidth=2, label="Best so far")
        ax.set(xlabel="Trial number", ylabel="Best validation accuracy (%)",
               title=f"{model}: accuracy across HPO trials")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{model}_accuracy.png", dpi=150)
        plt.close(fig)

    if importance:
        labels, values = list(importance), list(importance.values())
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.48 * len(labels))))
        ax.barh(labels[::-1], values[::-1])
        ax.set(xlabel="Relative importance (sums to 1)",
               title=f"{model}: Optuna importance of sampled parameters")
        fig.tight_layout()
        fig.savefig(out / f"{model}_importance.png", dpi=150)
        plt.close(fig)

    if model == "gru":
        groups = [
            [100 * r["value"] for r in complete if r.get("bidirectional") is value]
            for value in (False, True)
        ]
        if all(groups):
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.boxplot(groups, showfliers=True)
            ax.set_xticks([1, 2], ["Unidirectional", "Bidirectional"])
            ax.set(ylabel="Best validation accuracy (%)",
                   title="GRU: observed outcomes by direction")
            fig.tight_layout()
            fig.savefig(out / "gru_direction.png", dpi=150)
            plt.close(fig)


def analyze(model, output, fetch_wandb, entity, max_epochs, patience):
    path = ROOT / "studies" / f"optuna_{model}_{VERSION}.journal"
    if not path.exists():
        print(f"Missing journal: {path}. Skipping {model}.")
        return []
    study = optuna.load_study(
        study_name=f"art_{model}_{VERSION}_tpe",
        storage=JournalStorage(JournalFileBackend(str(path))),
    )
    extra = load_wandb_epochs(model, entity) if fetch_wandb else {}
    rows = []
    for trial in study.trials:
        e = extra.get(trial.number, {})
        last_epoch = e.get("last_epoch")
        best_epoch = trial.user_attrs.get("best_epoch")
        state = trial.state.name
        rows.append({
            "trial": trial.number, "state": state,
            "value": trial.value if state == "COMPLETE" else None,
            "best_epoch": best_epoch,
            "last_epoch": int(last_epoch) if last_epoch is not None else None,
            "hit_epoch_cap": (int(last_epoch) >= max_epochs
                              if last_epoch is not None else None),
            "seconds": ((trial.datetime_complete - trial.datetime_start).total_seconds()
                        if trial.datetime_complete and trial.datetime_start else None),
            **trial.params,
            "checkpoint": trial.user_attrs.get("checkpoint", ""),
            "wandb_url": trial.user_attrs.get("wandb_url", ""),
        })
    save_csv(output / f"{model}_trials.csv", rows)
    completed = [r for r in rows if r["value"] is not None]
    lines = [f"## {model.upper()} — {len(completed)} completed trials",
             "States: " + ", ".join(
                 f"{s}: {sum(r['state'] == s for r in rows)}"
                 for s in sorted(set(r["state"] for r in rows))),
             f"Validation distribution: {summarize_distribution(completed)}"]
    if completed:
        winner = max(completed, key=lambda r: r["value"])
        lines.append(f"Winner: trial {winner['trial']}, {pct(winner['value'])}, "
                     f"best epoch {winner['best_epoch']}; checkpoint: `{winner['checkpoint']}`")
        near_cap = [r for r in completed if r["best_epoch"] is not None
                    and r["best_epoch"] >= max_epochs - 4]
        lines.append(f"Best epoch in last 5 allowed epochs: {len(near_cap)}/{len(completed)} "
                     "(suggestive of ongoing improvement, not proof)")
    if fetch_wandb:
        observed = [r for r in completed if r["last_epoch"] is not None]
        capped = [r for r in observed if r["hit_epoch_cap"]]
        early = [r for r in observed if r["last_epoch"] < max_epochs]
        late_peak = [r for r in observed if r["best_epoch"] is not None and
                     r["best_epoch"] >= r["last_epoch"] - 4]
        lines.extend([
            f"W&B last-epoch summaries matched: {len(observed)}/{len(completed)}",
            f"Reached epoch limit ({max_epochs}): {len(capped)}/{len(observed)}",
            f"Stopped before limit: {len(early)}/{len(observed)}",
            f"Best epoch within last 5 trained epochs: {len(late_peak)}/{len(observed)}",
            "Early stopping suggests a plateau under patience " + str(patience) +
            "; it is not proof of full convergence.",
        ])
    else:
        lines.append("Stopping-limit counts unavailable from Optuna alone. "
                     "Rerun with --with-wandb to retrieve final epoch summaries.")
    if model == "gru":
        for direction in (False, True):
            group = [r for r in completed if r.get("bidirectional") is direction]
            lines.append(f"Bidirectional={direction}: {summarize_distribution(group)}")
        lines.append("Group comparisons are observational: Optuna sampled other parameters "
                     "differently in each group; this is not a controlled causal effect.")
    importance = {}
    if len(completed) >= 10:
        try:
            evaluator = optuna.importance.FanovaImportanceEvaluator(seed=42)
            importance = optuna.importance.get_param_importances(study, evaluator=evaluator)
            method = "fANOVA"
        except (ImportError, RuntimeError, ValueError) as exc:
            lines.append(f"fANOVA unavailable ({type(exc).__name__}); trying PED-ANOVA.")
            try:
                evaluator = optuna.importance.PedAnovaImportanceEvaluator(
                    baseline_quantile=0.1, evaluate_on_local=True
                )
                importance = optuna.importance.get_param_importances(study, evaluator=evaluator)
                method = "PED-ANOVA, top 10%"
            except (ImportError, RuntimeError, ValueError) as inner:
                lines.append(f"Importance unavailable: {inner}")
        if importance:
            save_csv(output / f"{model}_importance.csv", [
                {"parameter": k, "importance": v} for k, v in importance.items()
            ])
            lines.append(f"Hyperparameter importance ({method}; relative, noncausal):")
            lines += [f"- {k}: {v:.3f}" for k, v in importance.items()]
    plot_figures(rows, model, output, importance)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("both", "embeddings", "gru"), default="both")
    parser.add_argument("--with-wandb", action="store_true",
                        help="Fetch run summaries to distinguish epoch cap vs early stopping")
    parser.add_argument("--wandb-entity", default="user-university")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "hpo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lines = ["# Expanded Optuna HPO analysis", "", "Validation metrics are selected on a shared "
             "1,000-example split; tuning across many trials creates selection bias.", ""]
    models = ("embeddings", "gru") if args.model == "both" else (args.model,)
    for model in models:
        lines.extend(analyze(model, args.output, args.with_wandb,
                             args.wandb_entity, args.epochs, args.patience))
        lines.append("")
    report = args.output / "summary.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved report and CSV/PNG outputs in {args.output}")


if __name__ == "__main__":
    main()
