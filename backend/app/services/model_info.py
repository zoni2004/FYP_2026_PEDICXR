"""Reads the saved test results of the current model (cxr_cap/results/*.csv)."""
import csv

from .. import config

METRICS = ["auc", "recall", "specificity", "precision", "f1", "acc"]


def load_results() -> tuple[dict[str, dict[str, float]], list[dict[str, float]]]:
    summary, folds = {}, []
    s_path = config.RESULTS_DIR / "summary_5fold.csv"
    f_path = config.RESULTS_DIR / "fold_results_all5.csv"
    if s_path.exists():
        with s_path.open(newline="") as f:
            for row in csv.reader(f):
                if len(row) == 3 and row[0].startswith("test_"):
                    name = row[0][len("test_"):]
                    if name in METRICS:
                        summary[name] = {"mean": float(row[1]), "std": float(row[2])}
    if f_path.exists():
        with f_path.open(newline="") as f:
            for row in csv.DictReader(f):
                folds.append({"fold": float(row["fold"]), "best_epoch": float(row["best_epoch"]),
                              **{m: float(row[f"test_{m}"]) for m in METRICS if f"test_{m}" in row}})
    return summary, folds
