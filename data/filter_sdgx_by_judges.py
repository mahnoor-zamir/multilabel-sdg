import argparse
import json
from pathlib import Path
from typing import Any, Dict


def is_easy_high_confidence(obj: Dict[str, Any], mode: str = "strict") -> bool:
    """
    Decide whether to keep an EASY example based on two-judge agreement.

    strict:  both judges' primary_sdg == ground truth primary_sdg
    medium:  at least one judge's primary_sdg == ground truth primary_sdg
    """
    if obj.get("type") != "easy":
        return False

    ex_primary = obj.get("primary_sdg")
    if ex_primary is None:
        return False

    groq = obj.get("judge_groq") or {}
    gem = obj.get("judge_gemini") or {}

    # Skip anything with explicit errors
    if groq.get("error") or gem.get("error"):
        return False

    g_primary = groq.get("primary_sdg")
    m_primary = gem.get("primary_sdg")
    if g_primary is None or m_primary is None:
        return False

    try:
        ex_p = int(ex_primary)
        g_p = int(g_primary)
        m_p = int(m_primary)
    except Exception:
        return False

    if mode == "strict":
        return g_p == ex_p and m_p == ex_p
    # medium or anything else: at least one judge correct
    return g_p == ex_p or m_p == ex_p


def is_hard_high_confidence(obj: Dict[str, Any], mode: str = "strict") -> bool:
    """
    Decide whether to keep a HARD example based on two-judge agreement.

    strict:  both judges' relevant_sdgs contain all target sdgs
    medium:  either judge's relevant_sdgs contains at least one target sdg
    """
    if obj.get("type") != "hard":
        return False

    target_sdgs = obj.get("sdgs") or []
    try:
        target_sdgs = [int(s) for s in target_sdgs]
    except Exception:
        return False
    if not target_sdgs:
        return False

    groq = obj.get("judge_groq") or {}
    gem = obj.get("judge_gemini") or {}

    # Skip anything with explicit errors
    if groq.get("error") or gem.get("error"):
        return False

    g_rel = groq.get("relevant_sdgs") or []
    m_rel = gem.get("relevant_sdgs") or []
    try:
        g_rel = [int(s) for s in g_rel]
        m_rel = [int(s) for s in m_rel]
    except Exception:
        return False

    tset = set(target_sdgs)
    gset = set(g_rel)
    mset = set(m_rel)

    if mode == "strict":
        return tset.issubset(gset) and tset.issubset(mset)

    # medium or anything else: at least one judge finds at least one target SDG
    return bool(tset & gset) or bool(tset & mset)


def filter_sdgx(
    valid_path: Path,
    output_path: Path,
    mode: str = "strict",
) -> None:
    """
    Read sdgx_valid_sample.jsonl and write a filtered subset where both judges
    agree with the ground truth labels according to the chosen mode.
    """
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_path}")

    kept_easy = kept_hard = 0
    total_easy = total_hard = 0

    with valid_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            ex_type = obj.get("type")
            if ex_type == "easy":
                total_easy += 1
                if is_easy_high_confidence(obj, mode=mode):
                    kept_easy += 1
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            elif ex_type == "hard":
                total_hard += 1
                if is_hard_high_confidence(obj, mode=mode):
                    kept_hard += 1
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Filtering mode: {mode}")
    print(f"EASY examples: kept {kept_easy} / {total_easy}")
    print(f"HARD examples: kept {kept_hard} / {total_hard}")
    print(f"Total kept:    {kept_easy + kept_hard}")
    print(f"Filtered subset written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter SDGX examples by two-judge agreement."
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=Path("data") / "sdgx_valid_sample.jsonl",
        help="Path to validated SDGX JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "sdgx_filtered.jsonl",
        help="Path to write filtered subset.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["strict", "medium"],
        default="strict",
        help="Filtering mode: 'strict' (both judges agree) or 'medium' (at least one).",
    )
    args = parser.parse_args()

    filter_sdgx(args.valid, args.output, mode=args.mode)


if __name__ == "__main__":
    main()

