import numpy as np
from typing import Dict, Optional, Sequence

from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence[int]] = None,
) -> Dict:
    """
    Compute multilabel classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, shape (n_samples, 17), multi-hot encoded.
    y_pred : np.ndarray
        Predicted labels, shape (n_samples, 17), multi-hot encoded.
    labels : optional sequence of ints
        Label indices to consider; defaults to range(17).

    Returns
    -------
    Dict with:
      - per_label_f1: {1..17 -> float}
      - per_label_precision: {1..17 -> float}
      - per_label_recall: {1..17 -> float}
      - micro_f1: float
      - macro_f1: float
    """
    if labels is None:
        labels = list(range(17))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    per_label_f1: Dict[int, float] = {}
    per_label_precision: Dict[int, float] = {}
    per_label_recall: Dict[int, float] = {}

    for idx in labels:
        # SDG numbers are 1-based
        sdg_id = idx + 1
        y_t = y_true[:, idx]
        y_p = y_pred[:, idx]

        per_label_f1[sdg_id] = float(
            f1_score(y_t, y_p, zero_division=0)
        )
        per_label_precision[sdg_id] = float(
            precision_score(y_t, y_p, zero_division=0)
        )
        per_label_recall[sdg_id] = float(
            recall_score(y_t, y_p, zero_division=0)
        )

    micro_f1 = float(
        f1_score(y_true, y_pred, average="micro", zero_division=0)
    )
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    return {
        "per_label_f1": per_label_f1,
        "per_label_precision": per_label_precision,
        "per_label_recall": per_label_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def print_metrics(metrics: Dict, model_name: str) -> None:
    """
    Pretty-print per-label precision/recall/F1 and overall micro/macro F1.
    """
    per_f1 = metrics["per_label_f1"]
    per_p = metrics["per_label_precision"]
    per_r = metrics["per_label_recall"]

    print(f"\n=== Metrics for {model_name} ===")
    print(f"{'SDG':>4} | {'Prec':>7} | {'Recall':>7} | {'F1':>7}")
    print("-" * 32)

    for sdg in range(1, 18):
        p = per_p.get(sdg, 0.0)
        r = per_r.get(sdg, 0.0)
        f = per_f1.get(sdg, 0.0)
        print(f"{sdg:>4} | {p:7.3f} | {r:7.3f} | {f:7.3f}")

    print("-" * 32)
    print(f"Micro F1: {metrics['micro_f1']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}\n")

