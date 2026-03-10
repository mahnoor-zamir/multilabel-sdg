import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
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

EMB_TRAIN_PATH = Path("data/bge_embeddings_train.npz")
EMB_TEST_PATH = Path("data/bge_embeddings_test.npz")
MODEL_PATH = RESULTS_DIR / "bge_learned_threshold_model.pt"
THRESHOLDS_PATH = RESULTS_DIR / "learned_thresholds.json"


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


def load_sdgx_with_meta(
    path: Path,
) -> Tuple[List[str], np.ndarray, List[str], List[str]]:
    """
    Load SDGX from JSONL and return:
      - texts
      - 17-dim multi-hot labels
      - types ("easy"/"hard")
      - pairs (e.g. "10_16" for hard examples, "" for easy)
    """
    texts: List[str] = []
    labels: List[np.ndarray] = []
    types: List[str] = []
    pairs: List[str] = []

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
            pair = ""
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
                pair = str(ex.get("pair") or "")

            if y_vec.sum() == 0:
                continue

            texts.append(text)
            labels.append(y_vec)
            types.append(str(ex_type))
            pairs.append(pair)

    if not texts:
        raise RuntimeError(f"No valid SDGX examples found in {path}")

    return texts, np.stack(labels, axis=0), types, pairs


def get_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_or_load_embeddings(
    sdgx_path: Path = Path("data/sdgx_clean.jsonl"),
    model_name: str = "BAAI/bge-m3",
    augment: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute or load BGE-M3 embeddings for SDGi train/test.
    If augment=True, append SDGX examples to the train set only before encoding.
    """
    if EMB_TRAIN_PATH.exists() and EMB_TEST_PATH.exists() and not augment:
        train_data = np.load(EMB_TRAIN_PATH)
        test_data = np.load(EMB_TEST_PATH)
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
        texts_sdgx, labels_sdgx, _, _ = load_sdgx_with_meta(sdgx_path)
        X_train_texts = X_train_texts + texts_sdgx
        y_train = np.concatenate([y_train, labels_sdgx], axis=0)

    model = SentenceTransformer(model_name)
    model.max_seq_length = 512
    X_train_emb = model.encode(X_train_texts, batch_size=8, show_progress_bar=True)
    X_test_emb = model.encode(X_test_texts, batch_size=8, show_progress_bar=True)

    if not augment:
        np.savez_compressed(
            EMB_TRAIN_PATH,
            X_train=X_train_emb.astype("float32"),
            y_train=y_train,
        )
        np.savez_compressed(
            EMB_TEST_PATH,
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
        )
        # Per-label thresholds (logit space); sigmoid(thresholds) ∈ (0,1)
        self.thresholds = nn.Parameter(torch.zeros(17))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        thresh = torch.sigmoid(self.thresholds).unsqueeze(0)
        return (probs > thresh).float()


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


def train_and_evaluate(
    augment: bool,
    sdgx_path: Path,
    epochs: int,
    device: torch.device,
) -> None:
    X_train_emb, y_train, X_test_emb, y_test = compute_or_load_embeddings(
        sdgx_path=sdgx_path, augment=augment
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

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            probs = torch.sigmoid(logits)
            loss = criterion(probs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)
        print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred_scores = model.predict_proba(X_test_t.to(device)).cpu().numpy()
        y_pred = model.predict(X_test_t.to(device)).cpu().numpy().astype(int)

    metrics = compute_additional_metrics(y_test, y_pred)

    label = "BGE-M3 + FFN (learned thresholds)"
    if augment:
        label += " + SDGX augment"
    print_metrics(metrics, model_name=label)
    print(
        f"Multi-label F1 (macro): {metrics['multi_label_macro_f1']:.3f}, "
        f"multi-label F1 (micro): {metrics['multi_label_micro_f1']:.3f}"
    )
    print(
        f"Rare-label F1 (SDG6/7/14) macro: {metrics['rare_macro_f1']:.3f}, "
        f"micro: {metrics['rare_micro_f1']:.3f}"
    )

    # Save predictions and metrics
    preds_path = RESULTS_DIR / "bge_learned_threshold_preds.npz"
    metrics_path = RESULTS_DIR / "bge_learned_threshold_metrics.json"
    np.savez_compressed(preds_path, y_true=y_test, y_pred=y_pred)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save model weights
    torch.save(model.state_dict(), MODEL_PATH)

    # Save learned thresholds
    with torch.no_grad():
        thr_vals = torch.sigmoid(model.thresholds).cpu().numpy().tolist()
    thresholds_dict = {f"SDG{i+1}": float(v) for i, v in enumerate(thr_vals)}
    with THRESHOLDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, ensure_ascii=False, indent=2)

    print("\nLearned per-label thresholds (sigmoid):")
    for i, v in enumerate(thr_vals, start=1):
        print(f"  SDG{i}: {v:.3f}")

    print(f"\nSaved predictions to {preds_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved learned thresholds to {THRESHOLDS_PATH}")
    print(f"Saved model weights to {MODEL_PATH}")


def evaluate_on_sdgx(sdgx_path: Path, device: torch.device) -> None:
    """
    Evaluate the trained BGE classifier head with learned thresholds on SDGX:
      - overall F1
      - easy-only F1
      - hard-only F1
      - per-pair F1 for each hard pair
    """
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model weights not found at {MODEL_PATH}. "
            f"Run baseline training with learned thresholds first."
        )

    print(f"Evaluating on SDGX at {sdgx_path} using model {MODEL_PATH}")

    texts, y_true, types, pairs = load_sdgx_with_meta(sdgx_path)

    encoder = SentenceTransformer("BAAI/bge-m3")
    encoder.max_seq_length = 512
    X_sdgx_emb = encoder.encode(texts, batch_size=8, show_progress_bar=True)
    X_sdgx_t = torch.from_numpy(X_sdgx_emb).float().to(device)

    classifier = BGEClassifier(input_dim=X_sdgx_emb.shape[1]).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    with torch.no_grad():
        y_pred = classifier.predict(X_sdgx_t).cpu().numpy().astype(int)

    # Overall metrics
    metrics_all = compute_metrics(y_true, y_pred)

    types_arr = np.array(types)
    pairs_arr = np.array(pairs)

    easy_mask = types_arr == "easy"
    hard_mask = types_arr == "hard"

    def safe_metrics(mask: np.ndarray) -> Dict[str, float]:
        if mask.any():
            m = compute_metrics(y_true[mask], y_pred[mask])
            return {"micro_f1": m["micro_f1"], "macro_f1": m["macro_f1"]}
        return {"micro_f1": 0.0, "macro_f1": 0.0}

    easy_metrics = safe_metrics(easy_mask)
    hard_metrics = safe_metrics(hard_mask)

    per_pair: Dict[str, Dict[str, float]] = {}
    unique_pairs = sorted(
        {p for p, t in zip(pairs_arr, types_arr) if t == "hard" and p}
    )
    for pair in unique_pairs:
        mask = hard_mask & (pairs_arr == pair)
        per_pair[pair] = safe_metrics(mask)

    result = {
        "overall": {
            "micro_f1": metrics_all["micro_f1"],
            "macro_f1": metrics_all["macro_f1"],
        },
        "easy": easy_metrics,
        "hard": hard_metrics,
        "per_pair": per_pair,
    }

    out_path = RESULTS_DIR / "bge_learned_threshold_sdgx_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== SDGX evaluation with BGE-M3 + FFN (learned thresholds) ===")
    print(f"Overall SDGX  - micro F1: {result['overall']['micro_f1']:.3f}, "
          f"macro F1: {result['overall']['macro_f1']:.3f}")
    print(f"Easy-only     - micro F1: {result['easy']['micro_f1']:.3f}, "
          f"macro F1: {result['easy']['macro_f1']:.3f}")
    print(f"Hard-only     - micro F1: {result['hard']['micro_f1']:.3f}, "
          f"macro F1: {result['hard']['macro_f1']:.3f}")
    print("\nPer-pair F1 (hard examples):")
    for pair, m in per_pair.items():
        print(
            f"  Pair {pair}: micro F1 = {m['micro_f1']:.3f}, "
            f"macro F1 = {m['macro_f1']:.3f}"
        )
    print(f"\nSaved SDGX learned-threshold evaluation to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BGE-M3 + FFN baseline on SDGi with learned per-label thresholds."
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment SDGi train set with SDGX from --sdgx-path.",
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
    parser.add_argument(
        "--evaluate-sdgx",
        action="store_true",
        help="Evaluate trained classifier head with learned thresholds on SDGX only.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if args.evaluate_sdgx:
        evaluate_on_sdgx(args.sdgx_path, device)
        return

    print(f"Augment with SDGX: {args.augment}")
    train_and_evaluate(
        augment=args.augment,
        sdgx_path=args.sdgx_path,
        epochs=args.epochs,
        device=device,
    )


if __name__ == "__main__":
    main()

