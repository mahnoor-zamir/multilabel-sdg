import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset


SDG_DEFINITIONS: Dict[int, Dict[str, str]] = {
    1: {
        "title": "No Poverty",
        "description": "End poverty in all its forms everywhere. Economic growth must be inclusive to provide sustainable jobs and promote equality. Targets include eradicating extreme poverty, reducing poverty by half, implementing social protection systems, and ensuring equal rights to economic resources.",
    },
    2: {
        "title": "Zero Hunger",
        "description": "End hunger, achieve food security and improved nutrition and promote sustainable agriculture. The food and agriculture sector offers key solutions for development, and is central for hunger and poverty eradication. Targets include ending hunger, achieving food security, improving nutrition, and promoting sustainable agriculture.",
    },
    3: {
        "title": "Good Health and Well-Being",
        "description": "Ensure healthy lives and promote well-being for all at all ages. Targets include reducing maternal mortality, ending preventable deaths of newborns and children, ending epidemics of AIDS, tuberculosis, malaria, achieving universal health coverage, and reducing deaths from hazardous chemicals.",
    },
    4: {
        "title": "Quality Education",
        "description": "Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all. Obtaining a quality education is the foundation to improving people's lives and sustainable development. Targets include free primary and secondary education, early childhood development, equal access to technical and vocational training.",
    },
    5: {
        "title": "Gender Equality",
        "description": "Achieve gender equality and empower all women and girls. Gender equality is not only a fundamental human right, but a necessary foundation for a peaceful, prosperous and sustainable world. Targets include ending discrimination and violence against women, eliminating child marriage, ensuring participation in leadership, universal reproductive rights.",
    },
    6: {
        "title": "Clean Water and Sanitation",
        "description": "Ensure availability and sustainable management of water and sanitation for all. Clean, accessible water for all is an essential part of the world we want to live in. Targets include safe drinking water, adequate sanitation, improving water quality, increasing water-use efficiency, protecting water-related ecosystems.",
    },
    7: {
        "title": "Affordable and Clean Energy",
        "description": "Ensure access to affordable, reliable, sustainable and modern energy for all. Energy is central to nearly every major challenge and opportunity. Targets include universal access to energy services, increasing share of renewable energy, doubling energy efficiency improvements, and expanding clean energy infrastructure.",
    },
    8: {
        "title": "Decent Work and Economic Growth",
        "description": "Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all. Targets include sustaining per capita economic growth, higher economic productivity, promoting development-oriented policies, full employment, eradicating forced labour and child labour, protecting labour rights.",
    },
    9: {
        "title": "Industry, Innovation and Infrastructure",
        "description": "Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation. Investments in infrastructure are crucial to achieving sustainable development. Targets include developing reliable infrastructure, promoting inclusive industrialization, increasing access to financial services, upgrading technology, research and innovation.",
    },
    10: {
        "title": "Reduced Inequalities",
        "description": "Reduce inequality within and among countries. Policies should be universal in principle, paying attention to the needs of disadvantaged and marginalized populations. Targets include income growth of bottom 40%, promoting social and economic inclusion, eliminating discriminatory laws, improving regulation of financial markets, development assistance.",
    },
    11: {
        "title": "Sustainable Cities and Communities",
        "description": "Make cities and human settlements inclusive, safe, resilient and sustainable. Cities must provide opportunities for all, with access to basic services, energy, housing, transportation. Targets include access to safe housing, sustainable transport, inclusive urbanization, protecting cultural heritage, disaster risk reduction, air quality.",
    },
    12: {
        "title": "Responsible Consumption and Production",
        "description": "Ensure sustainable consumption and production patterns. Targets include sustainable management of natural resources, halving per capita food waste, sustainable management of chemicals and waste, reducing waste generation, encouraging sustainable practices in companies, sustainable public procurement, sustainable tourism.",
    },
    13: {
        "title": "Climate Action",
        "description": "Take urgent action to combat climate change and its impacts. Climate change is a global challenge that affects everyone, everywhere. Targets include strengthening resilience to climate hazards, integrating climate change measures into policy, improving education on climate change, implementing commitments under the UN Framework Convention on Climate Change.",
    },
    14: {
        "title": "Life Below Water",
        "description": "Conserve and sustainably use the oceans, seas and marine resources for sustainable development. Careful management of this essential global resource is a key feature of a sustainable future. Targets include reducing marine pollution, protecting marine ecosystems, ending overfishing, conserving coastal areas, ending harmful fisheries subsidies.",
    },
    15: {
        "title": "Life on Land",
        "description": "Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt and reverse land degradation and halt biodiversity loss. Targets include conserving forests, restoring degraded land, combating desertification, protecting biodiversity and natural habitats, preventing invasive species.",
    },
    16: {
        "title": "Peace, Justice and Strong Institutions",
        "description": "Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels. Targets include reducing violence, ending abuse and trafficking, promoting rule of law, reducing corruption, developing transparent institutions, ensuring public access to information.",
    },
    17: {
        "title": "Partnerships for the Goals",
        "description": "Strengthen the means of implementation and revitalize the global partnership for sustainable development. Targets include mobilizing domestic resources, developed countries implementing development assistance commitments, technology transfer, capacity building, promoting fair multilateral trading systems, policy coherence for sustainable development.",
    },
}


def _extract_language(example_metadata) -> str:
    if isinstance(example_metadata, dict):
        return example_metadata.get("language", "unknown")
    if hasattr(example_metadata, "language"):
        return getattr(example_metadata, "language", "unknown")
    return "unknown"


def _build_standard_record(example: Dict, split: str) -> Dict:
    text = example.get("text", "")
    labels = example.get("labels", [])
    metadata = example.get("metadata", {})

    if isinstance(labels, list):
        label_list = [int(l) for l in labels if 1 <= int(l) <= 17]
    else:
        label_list = [int(labels)] if labels is not None else []

    language = _extract_language(metadata)
    size = metadata.get("size", "unknown") if isinstance(metadata, dict) else "unknown"

    return {
        "text": text,
        "labels": sorted(set(label_list)),
        "language": language,
        "size": size,
        "split": split,
    }


def load_sdgi_corpus() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=== Loading UNDP SDGi Corpus ===")
    dataset = load_dataset("UNDP/sdgi-corpus")

    records_train: List[Dict] = []
    records_test: List[Dict] = []

    for split_name, split_data in dataset.items():
        print(f"Processing split: {split_name} ({len(split_data)} examples)")
        for ex in split_data:
            record = _build_standard_record(ex, split=split_name)
            if split_name == "train":
                records_train.append(record)
            elif split_name == "test":
                records_test.append(record)

    train_df = pd.DataFrame(records_train)
    test_df = pd.DataFrame(records_test)

    return train_df, test_df


def compute_multihot(labels_series: pd.Series, num_labels: int = 17) -> np.ndarray:
    n = len(labels_series)
    multihot = np.zeros((n, num_labels), dtype=np.int64)
    for i, label_list in enumerate(labels_series):
        for l in label_list:
            if 1 <= l <= num_labels:
                multihot[i, l - 1] = 1
    return multihot


def _print_header(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("=== SDGi Corpus Statistics ===")
    total = len(train_df) + len(test_df)
    print(f"Total examples: {total} (train: {len(train_df)}, test: {len(test_df)})\n")


def _print_label_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("Label distribution:")

    def expand_counts(df: pd.DataFrame) -> Counter:
        all_labels: List[int] = []
        for labels in df["labels"]:
            all_labels.extend(labels)
        return Counter(all_labels)

    train_counts = expand_counts(train_df)
    test_counts = expand_counts(test_df)

    for sdg in range(1, 18):
        train_c = sum(1 for labels in train_df["labels"] if sdg in labels)
        test_c = sum(1 for labels in test_df["labels"] if sdg in labels)
        print(f"SDG{sdg}:  train={train_c} test={test_c}")
    print()


def _print_multilabel_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    def label_counts(df: pd.DataFrame) -> List[int]:
        return [len(labels) for labels in df["labels"]]

    train_counts = label_counts(train_df)
    test_counts = label_counts(test_df)
    all_counts = train_counts + test_counts
    total = len(all_counts)

    def pct(count: int) -> float:
        return 100.0 * count / total if total > 0 else 0.0

    single = sum(1 for c in all_counts if c == 1)
    two = sum(1 for c in all_counts if c == 2)
    three_plus = sum(1 for c in all_counts if c >= 3)
    avg_labels = float(np.mean(all_counts)) if total > 0 else 0.0

    print("Multi-label statistics:")
    print(f"Single label:  {pct(single):5.2f}% of examples")
    print(f"2 labels:      {pct(two):5.2f}%")
    print(f"3+ labels:     {pct(three_plus):5.2f}%")
    print(f"Avg labels per example: {avg_labels:.2f}\n")


def _print_language_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    combined = pd.concat([train_df, test_df], ignore_index=True)
    total = len(combined)
    lang_counts = combined["language"].value_counts().to_dict()

    def pct(count: int) -> float:
        return 100.0 * count / total if total > 0 else 0.0

    en = lang_counts.get("en", 0)
    fr = lang_counts.get("fr", 0)
    es = lang_counts.get("es", 0)

    print("Language distribution:")
    print(f"EN: {pct(en):5.2f}%  FR: {pct(fr):5.2f}%  ES: {pct(es):5.2f}%\n")


def _approx_token_length(text: str) -> int:
    return max(1, len(text.split()))


def _print_length_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    combined = pd.concat([train_df, test_df], ignore_index=True)
    lengths = combined["text"].astype(str).apply(_approx_token_length).tolist()
    total = len(lengths)

    short = sum(1 for l in lengths if l < 512)
    medium = sum(1 for l in lengths if 512 <= l < 2048)
    long = sum(1 for l in lengths if l >= 2048)

    def pct(count: int) -> float:
        return 100.0 * count / total if total > 0 else 0.0

    print("Text length (tokens approx):")
    print(f"Short (<512):  {pct(short):5.2f}%")
    print(f"Medium (512-2048): {pct(medium):5.2f}%")
    print(f"Long (>2048):  {pct(long):5.2f}%\n")


def _print_label_cooccurrence(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    combined = pd.concat([train_df, test_df], ignore_index=True)
    pair_counts: Counter = Counter()

    for labels in combined["labels"]:
        unique = sorted(set(labels))
        if len(unique) < 2:
            continue
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair = (unique[i], unique[j])
                pair_counts[pair] += 1

    top_pairs = pair_counts.most_common(10)
    print("Label co-occurrence (top 10 pairs):")
    for (a, b), count in top_pairs:
        print(f"SDG_{a} + SDG_{b}: {count} examples")
    print()


def save_parquet(train_df: pd.DataFrame, test_df: pd.DataFrame, base_dir: str) -> None:
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)

    train_path = os.path.join(base_dir, "data", "sdgi_train.parquet")
    test_path = os.path.join(base_dir, "data", "sdgi_test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Saved processed dataset:")
    print(f"- train -> {train_path}")
    print(f"- test  -> {test_path}\n")


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_df, test_df = load_sdgi_corpus()

    # Convert labels to multi-hot arrays (not printed, but ready for models)
    y_train = compute_multihot(train_df["labels"])
    y_test = compute_multihot(test_df["labels"])
    print(f"Multi-hot label matrix shapes: train={y_train.shape}, test={y_test.shape}\n")

    _print_header(train_df, test_df)
    _print_label_distribution(train_df, test_df)
    _print_multilabel_stats(train_df, test_df)
    _print_language_distribution(train_df, test_df)
    _print_length_stats(train_df, test_df)
    _print_label_cooccurrence(train_df, test_df)

    save_parquet(train_df, test_df, project_root)


if __name__ == "__main__":
    main()

