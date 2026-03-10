import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score


SDG_DEFINITIONS: Dict[int, str] = {
    1: "End poverty in all its forms everywhere.",
    2: "End hunger, achieve food security and improved nutrition, and promote sustainable agriculture.",
    3: "Ensure healthy lives and promote well-being for all at all ages.",
    4: "Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all.",
    5: "Achieve gender equality and empower all women and girls.",
    6: "Ensure availability and sustainable management of water and sanitation for all.",
    7: "Ensure access to affordable, reliable, sustainable and modern energy for all.",
    8: "Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all.",
    9: "Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation.",
    10: "Reduce inequality within and among countries.",
    11: "Make cities and human settlements inclusive, safe, resilient and sustainable.",
    12: "Ensure sustainable consumption and production patterns.",
    13: "Take urgent action to combat climate change and its impacts.",
    14: "Conserve and sustainably use the oceans, seas and marine resources for sustainable development.",
    15: "Protect, restore and promote sustainable use of terrestrial ecosystems and halt biodiversity loss.",
    16: "Promote peaceful and inclusive societies, provide access to justice for all and build effective, accountable institutions.",
    17: "Strengthen the means of implementation and revitalize the global partnership for sustainable development.",
}

SDG_WEAK_PAIRS = ["10_16", "10_11", "4_10"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sdgi() -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    ds = load_dataset("UNDP/sdgi-corpus")
    train = ds["train"]
    test = ds["test"]

    def process_split(split):
        texts = []
        labels = []
        for ex in split:
            texts.append(ex["text"])
            ys = ex.get("labels") or []
            y = np.zeros(17, dtype=int)
            for sdg in ys:
                if 1 <= sdg <= 17:
                    y[sdg - 1] = 1
            labels.append(y)
        return texts, np.stack(labels, axis=0)

    X_train, y_train = process_split(train)
    X_test, y_test = process_split(test)
    return X_train, y_train, X_test, y_test


def load_sdgx(path: Path):
    texts, labels, types, pairs = [], [], [], []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            ex = json.loads(ln)
            text = ex.get("text")
            if not text:
                continue
            y = np.zeros(17, dtype=int)
            t = ex.get("type")
            pair = str(ex.get("pair") or "")
            if t == "easy":
                p = ex.get("primary_sdg")
                if p is not None and 1 <= int(p) <= 17:
                    y[int(p) - 1] = 1
            elif t == "hard":
                for s in ex.get("sdgs") or []:
                    try:
                        s_int = int(s)
                    except Exception:
                        continue
                    if 1 <= s_int <= 17:
                        y[s_int - 1] = 1
            if y.sum() == 0:
                continue
            texts.append(text)
            labels.append(y)
            types.append(str(t))
            pairs.append(pair)
    return texts, np.stack(labels, axis=0), types, pairs


def build_contrastive_examples(
    sdgx_texts, sdgx_types, sdgx_pairs, train_sdgi
) -> List[InputExample]:
    examples: List[InputExample] = []

    # SDGX weak hard pairs
    for text, t, pair in zip(sdgx_texts, sdgx_types, sdgx_pairs):
        if t != "hard" or pair not in SDG_WEAK_PAIRS:
            continue
        try:
            a_str, b_str = pair.split("_")
            a, b = int(a_str), int(b_str)
        except Exception:
            continue
        pos_a = SDG_DEFINITIONS[a]
        pos_b = SDG_DEFINITIONS[b]
        neg_candidates = [k for k in SDG_DEFINITIONS.keys() if k not in (a, b)]
        if not neg_candidates:
            continue
        neg_def = SDG_DEFINITIONS[neg_candidates[0]]
        examples.append(InputExample(texts=[text, pos_a, neg_def]))
        examples.append(InputExample(texts=[text, pos_b, neg_def]))

    # In-batch SDGi pairs (up to 50 examples per SDG)
    per_sdg_texts: Dict[int, List[str]] = {i: [] for i in range(1, 18)}
    for ex in train_sdgi:
        ys = ex.get("labels") or []
        for sdg in ys:
            if 1 <= sdg <= 17 and len(per_sdg_texts[sdg]) < 50:
                per_sdg_texts[sdg].append(ex["text"])
    for sdg, texts in per_sdg_texts.items():
        definition = SDG_DEFINITIONS[sdg]
        for text in texts:
            examples.append(InputExample(texts=[text, definition]))

    return examples


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> Dict:
    if labels is None:
        labels = list(range(17))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_label_f1: Dict[int, float] = {}
    per_label_p: Dict[int, float] = {}
    per_label_r: Dict[int, float] = {}
    for idx in labels:
        sdg_id = idx + 1
        y_t = y_true[:, idx]
        y_p = y_pred[:, idx]
        per_label_f1[sdg_id] = float(f1_score(y_t, y_p, zero_division=0))
        per_label_p[sdg_id] = float(precision_score(y_t, y_p, zero_division=0))
        per_label_r[sdg_id] = float(recall_score(y_t, y_p, zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "per_label_f1": per_label_f1,
        "per_label_precision": per_label_p,
        "per_label_recall": per_label_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Contrastive fine-tuning of BGE-M3 on weak SDG pairs."
    )
    parser.add_argument(
        "--sdgx-path",
        type=Path,
        default=Path("data/sdgx_clean.jsonl"),
        help="Path to sdgx_clean.jsonl (on Kaggle or local).",
    )
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)

    # Paths
    sdgx_path = args.sdgx_path
    if not sdgx_path.exists():
        print(
            f"Warning: {sdgx_path} not found; adjust --sdgx-path to your Kaggle dataset path."
        )

    # Load data
    X_train_texts, y_train, X_test_texts, y_test = load_sdgi()
    ds = load_dataset("UNDP/sdgi-corpus")
    train_sdgi = ds["train"]

    sdgx_texts, sdgx_labels, sdgx_types, sdgx_pairs = load_sdgx(sdgx_path)
    print("SDGi train/test sizes:", len(X_train_texts), len(X_test_texts))
    print("SDGX size:", len(sdgx_texts))

    # Build contrastive examples
    train_examples = build_contrastive_examples(
        sdgx_texts, sdgx_types, sdgx_pairs, train_sdgi
    )
    print("Total contrastive examples:", len(train_examples))

    # Contrastive fine-tuning
    model_name = "BAAI/bge-m3"
    contrastive_model = SentenceTransformer(model_name, device=device)
    contrastive_model.max_seq_length = 512
    train_dataloader = DataLoader(train_examples, batch_size=32, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(contrastive_model)

    num_epochs = 3
    warmup_steps = 100
    output_dir = Path("bge_m3_contrastive")
    output_dir.mkdir(parents=True, exist_ok=True)

    contrastive_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )
    contrastive_model.save(str(output_dir))
    print("Saved fine-tuned encoder to", output_dir)

    # Embed SDGi
    print("Embedding SDGi with contrastive encoder...")
    X_train_emb = contrastive_model.encode(
        X_train_texts, batch_size=32, show_progress_bar=True
    )
    X_test_emb = contrastive_model.encode(
        X_test_texts, batch_size=32, show_progress_bar=True
    )

    X_train = torch.from_numpy(X_train_emb).float().to(device)
    X_test = torch.from_numpy(X_test_emb).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    train_ds = TensorDataset(X_train, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    class FFN(nn.Module):
        def __init__(self, input_dim: int = 1024):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 17),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    ffn = FFN(input_dim=X_train_emb.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs_ffn = 30
    for epoch in range(1, num_epochs_ffn + 1):
        epoch_loss = 0.0
        ffn.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = ffn(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)
        print(f"Epoch {epoch}/{num_epochs_ffn} - loss: {epoch_loss:.4f}")

    # SDGi metrics
    ffn.eval()
    with torch.no_grad():
        y_pred_scores = ffn(X_test).cpu().numpy()
    y_pred = (y_pred_scores >= 0.5).astype(int)

    metrics_sdgi = compute_metrics(y_test, y_pred)
    print("Overall SDGi micro F1:", metrics_sdgi["micro_f1"])
    print("Overall SDGi macro F1:", metrics_sdgi["macro_f1"])

    # Weak pair metrics on SDGi
    def pair_metrics(pair: str) -> Dict[str, float]:
        a_str, b_str = pair.split("_")
        a, b = int(a_str), int(b_str)
        ia, ib = a - 1, b - 1
        mask = (y_test[:, ia] == 1) & (y_test[:, ib] == 1)
        if not mask.any():
            return {"micro_f1": 0.0, "macro_f1": 0.0}
        m = compute_metrics(y_test[mask][:, [ia, ib]], y_pred[mask][:, [ia, ib]])
        return {"micro_f1": m["micro_f1"], "macro_f1": m["macro_f1"]}

    pair_results_sdgi = {p: pair_metrics(p) for p in SDG_WEAK_PAIRS}
    print("SDGi weak pair metrics:", pair_results_sdgi)

    # SDGX hard evaluation
    types_arr = np.array(sdgx_types)
    hard_mask = types_arr == "hard"
    texts_hard = [t for t, m in zip(sdgx_texts, hard_mask) if m]
    labels_hard = sdgx_labels[hard_mask]
    pairs_hard = np.array(sdgx_pairs)[hard_mask]

    print("Embedding SDGX hard examples with contrastive encoder...")
    X_hard_emb = contrastive_model.encode(
        texts_hard, batch_size=32, show_progress_bar=True
    )
    X_hard = torch.from_numpy(X_hard_emb).float().to(device)
    ffn.eval()
    with torch.no_grad():
        y_pred_hard_scores = ffn(X_hard).cpu().numpy()
    y_pred_hard = (y_pred_hard_scores >= 0.5).astype(int)

    metrics_hard_all = compute_metrics(labels_hard, y_pred_hard)
    print("Overall SDGX hard-only micro F1:", metrics_hard_all["micro_f1"])
    print("Overall SDGX hard-only macro F1:", metrics_hard_all["macro_f1"])

    def per_pair_metrics_hard(pair: str) -> Dict[str, float]:
        mask = pairs_hard == pair
        if not mask.any():
            return {"micro_f1": 0.0, "macro_f1": 0.0}
        m = compute_metrics(labels_hard[mask], y_pred_hard[mask])
        return {"micro_f1": m["micro_f1"], "macro_f1": m["macro_f1"]}

    all_pairs = sorted(set(p for p in pairs_hard if p))
    pair_results_sdgx = {p: per_pair_metrics_hard(p) for p in all_pairs}
    print("SDGX hard per-pair metrics:", pair_results_sdgx)

    # Save everything
    metrics_all = {
        "sdgi": metrics_sdgi,
        "sdgi_weak_pairs": pair_results_sdgi,
        "sdgx_hard_overall": {
            "micro_f1": metrics_hard_all["micro_f1"],
            "macro_f1": metrics_hard_all["macro_f1"],
        },
        "sdgx_hard_pairs": pair_results_sdgx,
    }
    out_path = Path("contrastive_metrics.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)
    torch.save(ffn.state_dict(), "ffn_contrastive_sdgi.pt")
    print("Saved metrics to", out_path)
    print("Saved FFN head to ffn_contrastive_sdgi.pt")


if __name__ == "__main__":
    main()

