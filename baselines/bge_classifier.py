import argparse
import json
from pathlib import Path
from typing import List, Tuple
import sys

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on sys.path for `evaluation` imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_metrics, print_metrics


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMB_TRAIN_BASE = Path("data/bge_embeddings_train.npz")
EMB_TRAIN_AUG = Path("data/bge_embeddings_train_augmented.npz")
EMB_TEST = Path("data/bge_embeddings_test.npz")


def sdgi_to_xy() -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """Load SDGi train/test and convert labels to 17-dim multi-hot."""
    ds = load_dataset("UNDP/sdgi-corpus")
    train = ds["train"]
    test = ds["test"]

    def process_split(split):
        texts: List[str] = []
        labels: List[np.ndarray] = []
        for ex in split:
            texts.append(ex["text"])
            ys = ex.get("labels") or []
            y_vec = np.zeros(17, dtype=int)
            for sdg in ys:
                if 1 <= sdg <= 17:
                    y_vec[sdg - 1] = 1
            labels.append(y_vec)
        return texts, np.stack(labels, axis=0)

    X_train, y_train = process_split(train)
    X_test, y_test = process_split(test)
    return X_train, y_train, X_test, y_test


def load_sdgx(path: Path) -> Tuple[List[str], np.ndarray]:
    """Load SDGX from JSONL and return texts and 17-dim multi-hot labels."""
    texts: List[str] = []
    labels: List[np.ndarray] = []

    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                ex = json.loads(ln)
            except json.JSONDecodeError:
                continue

            text = ex.get("text")
            if not text:
                continue

            y_vec = np.zeros(17, dtype=int)
            ex_type = ex.get("type")
            if ex_type == "easy":
                primary = ex.get("primary_sdg")
                if primary is not None:
                    try:
                        p = int(primary)
                        if 1 <= p <= 17:
                            y_vec[p - 1] = 1
                    except Exception:
                        pass
            elif ex_type == "hard":
                for sdg in ex.get("sdgs") or []:
                    try:
                        s = int(sdg)
                    except Exception:
                        continue
                    if 1 <= s <= 17:
                        y_vec[s - 1] = 1

            if y_vec.sum() == 0:
                continue

            texts.append(text)
            labels.append(y_vec)

    if not texts:
        raise RuntimeError(f"No valid SDGX examples found in {path}")

    return texts, np.stack(labels, axis=0)


def get_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_or_load_embeddings(
    augment: bool = False,
    sdgx_path: Path = Path("data/sdgx_clean.jsonl"),
    model_name: str = "BAAI/bge-m3",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute or load BGE-M3 embeddings for SDGi (and optional SDGX augmentation).
    """
    train_emb_path = EMB_TRAIN_AUG if augment else EMB_TRAIN_BASE

    if train_emb_path.exists() and EMB_TEST.exists():
        train_data = np.load(train_emb_path)
        test_data = np.load(EMB_TEST)
        return (
            train_data["X_train"],
            train_data["y_train"],
            test_data["X_test"],
            test_data["y_test"],
        )

    print("Loading SDGi and computing BGE-M3 embeddings...")
    X_train_texts, y_train, X_test_texts, y_test = sdgi_to_xy()

    if augment:
        print(f"Augmenting train set with SDGX from {sdgx_path}...")
        sdgx_texts, sdgx_labels = load_sdgx(sdgx_path)
        X_train_texts = X_train_texts + sdgx_texts
        y_train = np.concatenate([y_train, sdgx_labels], axis=0)

    model = SentenceTransformer(model_name)
    # Cap sequence length and use small batch size to avoid OOM on GPUs
    model.max_seq_length = 512
    X_train_emb = model.encode(X_train_texts, batch_size=8, show_progress_bar=True)
    X_test_emb = model.encode(X_test_texts, batch_size=8, show_progress_bar=True)

    np.savez_compressed(
        train_emb_path,
        X_train=X_train_emb.astype("float32"),
        y_train=y_train,
    )
    np.savez_compressed(
        EMB_TEST,
        X_test=X_test_emb.astype("float32"),
        y_test=y_test,
    )

    return X_train_emb, y_train, X_test_emb, y_test


class BGEClassifier(nn.Module):
    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 17),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute multi-label F1 (only examples with >1 true label) and rare-label F1 for SDG 6,7,14.
    """
    metrics = compute_metrics(y_true, y_pred)

    # Multi-label subset: y_true.sum(axis=1) > 1
    mask_multi = y_true.sum(axis=1) > 1
    if mask_multi.any():
        multi_metrics = compute_metrics(y_true[mask_multi], y_pred[mask_multi])
        metrics["multi_label_macro_f1"] = multi_metrics["macro_f1"]
        metrics["multi_label_micro_f1"] = multi_metrics["micro_f1"]
    else:
        metrics["multi_label_macro_f1"] = 0.0
        metrics["multi_label_micro_f1"] = 0.0

    # Rare-label F1: SDG 6, 7, 14
    rare_indices = [5, 6, 13]  # 0-based indices for SDG6,7,14
    rare_metrics = compute_metrics(y_true, y_pred, labels=rare_indices)
    metrics["rare_macro_f1"] = rare_metrics["macro_f1"]
    metrics["rare_micro_f1"] = rare_metrics["micro_f1"]

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BGE-M3 + FFN baseline on SDGi, with optional SDGX augmentation."
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment SDGi train set with SDGX from data/sdgx_clean.jsonl.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--sdgx-path",
        type=Path,
        default=Path("data/sdgx_clean.jsonl"),
        help="Path to SDGX JSONL file.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Augment with SDGX: {args.augment}")

    X_train_emb, y_train, X_test_emb, y_test = compute_or_load_embeddings(
        augment=args.augment,
        sdgx_path=args.sdgx_path,
    )

    X_train_t = torch.from_numpy(X_train_emb).float()
    X_test_t = torch.from_numpy(X_test_emb).float()
    y_train_t = torch.from_numpy(y_train).float()
    y_test_t = torch.from_numpy(y_test).float()

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = BGEClassifier(input_dim=X_train_emb.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = args.epochs
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)
        print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred_scores = model(X_test_t.to(device)).cpu().numpy()
    y_pred = (y_pred_scores >= 0.5).astype(int)

    metrics = compute_additional_metrics(y_test, y_pred)

    model_name = "BGE-M3 + FFN (baseline)" if not args.augment else "BGE-M3 + FFN (SDGX-augmented)"
    print_metrics(metrics, model_name=model_name)
    print(
        f"Multi-label F1 (macro): {metrics['multi_label_macro_f1']:.3f}, "
        f"multi-label F1 (micro): {metrics['multi_label_micro_f1']:.3f}"
    )
    print(
        f"Rare-label F1 (SDG6/7/14) macro: {metrics['rare_macro_f1']:.3f}, "
        f"micro: {metrics['rare_micro_f1']:.3f}"
    )

    if args.augment:
        preds_path = RESULTS_DIR / "bge_augmented_preds.npz"
        metrics_path = RESULTS_DIR / "bge_augmented_metrics.json"
    else:
        preds_path = RESULTS_DIR / "bge_frozen_preds.npz"
        metrics_path = RESULTS_DIR / "bge_frozen_metrics.json"

    np.savez_compressed(preds_path, y_true=y_test, y_pred=y_pred)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to {preds_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()


