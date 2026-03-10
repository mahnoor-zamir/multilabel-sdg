import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml
from groq import Groq
from tqdm import tqdm

from generate_sdgx import SDG_DEFINITIONS

@dataclass
class ValidationConfig:
    judge_model: str
    rate_limit_rpm: int
    max_retries: int
    sample_size: int
    input_path: Path
    output_path: Path


def load_config(config_path: Path, sample_size: int = 100) -> ValidationConfig:
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    groq_cfg = cfg["groq"]

    root = config_path.parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    input_path = data_dir / "sdgx_raw.jsonl"
    output_path = data_dir / "sdgx_valid_sample.jsonl"

    return ValidationConfig(
      judge_model=groq_cfg.get("secondary_model", groq_cfg["primary_model"]),
        rate_limit_rpm=int(groq_cfg.get("rate_limit_rpm", 30)),
        max_retries=int(groq_cfg.get("max_retries", 3)),
        sample_size=sample_size,
        input_path=input_path,
        output_path=output_path,
    )


def format_sdg_definitions() -> str:
    lines: List[str] = []
    for sdg in range(1, 18):
        info = SDG_DEFINITIONS[sdg]
        lines.append(f"SDG {sdg}: {info['title']}\n{info['description']}\n")
    return "\n".join(lines)


def build_validation_prompt(text: str) -> str:
    sdg_defs = format_sdg_definitions()
    return f"""You are an expert in UN Sustainable Development Goals 
with deep knowledge of policy documents.

Read this text carefully and identify which of the 17 SDGs 
it genuinely relates to.

Text:
{text}

SDG Reference:
{sdg_defs}

For each SDG ask: does this text meaningfully discuss 
topics, targets, or initiatives related to this goal?

Return ONLY valid JSON:
{{
  "relevant_sdgs": [list of relevant SDG numbers],
  "primary_sdg": most relevant SDG number,
  "confidence": "high/medium/low",
  "reasoning": "one sentence max"
}}"""


def parse_model_json(content: str) -> Dict[str, Any]:
    """
    Robust JSON extraction from Groq response.
    """
    content = content.strip()
    # Fast path: whole content is JSON
    if content.startswith("{") and content.endswith("}"):
        return json.loads(content)

    decoder = json.JSONDecoder()
    start = content.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    obj, _ = decoder.raw_decode(content[start:])
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON is not an object")
    return obj


def sample_examples(path: Path, k: int) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"SDGX file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []
    k = min(k, len(lines))
    sampled = random.sample(lines, k)
    out: List[Dict[str, Any]] = []
    for ln in sampled:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def validate_sample(config: ValidationConfig) -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable must be set.")

    client = Groq(api_key=api_key)

    examples = sample_examples(config.input_path, config.sample_size)
    if not examples:
        print("No examples found to validate.")
        return

    print(f"Validating random sample of {len(examples)} examples from {config.input_path}")

    stats = {
        "total": 0,                 # all processed examples
        "low_confidence": 0,
        "easy_primary_match": 0,    # easy examples where primary_sdg matches
        "easy_primary_mismatch": 0, # easy examples where primary_sdg differs
        "hard_both_found": 0,       # hard examples where both target SDGs appear in relevant_sdgs
        "hard_partial": 0,          # hard examples where at least one target SDG appears
        "hard_none": 0,             # hard examples where neither target SDG appears
        "unscored_rate_limit": 0,   # judge has RateLimitError
        "unscored_other_error": 0,  # other judge errors
    }

    with config.output_path.open("w", encoding="utf-8") as out_f:
        for ex in tqdm(examples, desc="Validating SDGX sample", unit="example"):
            text = ex.get("text", "")
            if not text:
                continue

            prompt = build_validation_prompt(text)

            result: Dict[str, Any] = {}
            for attempt in range(config.max_retries + 1):
                try:
                    resp = client.chat.completions.create(
                        model=config.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=512,
                        response_format={"type": "json_object"},
                    )
                    content = resp.choices[0].message.content or ""
                    result = parse_model_json(content)
                    break
                except Exception as e:
                    if attempt == config.max_retries:
                        result = {"error": repr(e)}
                    else:
                        time.sleep(2**attempt)

            combined = {**ex, "judge": result}
            out_f.write(json.dumps(combined, ensure_ascii=False) + "\n")

            stats["total"] += 1
            if not isinstance(result, dict):
                continue

            # Handle judge errors explicitly
            err = result.get("error")
            if err:
                err_str = str(err)
                if "RateLimitError" in err_str or "rate_limit_exceeded" in err_str:
                    stats["unscored_rate_limit"] += 1
                else:
                    stats["unscored_other_error"] += 1
                continue

            conf = str(result.get("confidence", "")).lower()
            if conf == "low":
                stats["low_confidence"] += 1

            ex_type = ex.get("type")

            # EASY: compare primary SDG
            if ex_type == "easy":
                ex_primary = ex.get("primary_sdg")
                judge_primary = result.get("primary_sdg")
                if ex_primary is not None and judge_primary is not None:
                    if int(ex_primary) == int(judge_primary):
                        stats["easy_primary_match"] += 1
                    else:
                        stats["easy_primary_mismatch"] += 1

            # HARD: check how many target SDGs were found
            if ex_type == "hard":
                target_sdgs = ex.get("sdgs") or []
                try:
                    target_sdgs = [int(s) for s in target_sdgs]
                except Exception:
                    target_sdgs = []
                relevant = result.get("relevant_sdgs") or []
                try:
                    relevant = [int(s) for s in relevant]
                except Exception:
                    relevant = []

                if target_sdgs:
                    found = [t for t in target_sdgs if t in relevant]
                    if len(found) == len(target_sdgs):
                        stats["hard_both_found"] += 1
                    elif len(found) > 0:
                        stats["hard_partial"] += 1
                    else:
                        stats["hard_none"] += 1

    print("\nValidation summary (sample):")
    print(f"  Total validated:              {stats['total']}")
    print(f"  Low confidence:               {stats['low_confidence']}")
    print(f"  Unscored (rate limit):        {stats['unscored_rate_limit']}")
    print(f"  Unscored (other error):       {stats['unscored_other_error']}")
    print(f"  EASY primary SDG matches:     {stats['easy_primary_match']}")
    print(f"  EASY primary SDG mismatches:  {stats['easy_primary_mismatch']}")
    print(f"  HARD both targets found:      {stats['hard_both_found']}")
    print(f"  HARD partial targets found:   {stats['hard_partial']}")
    print(f"  HARD no targets found:        {stats['hard_none']}")
    print(f"\nDetailed results written to: {config.output_path}")


def run_validation(sample_size: int = 100) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "config.yaml"
    cfg = load_config(config_path, sample_size=sample_size)
    validate_sample(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=100, help="Number of examples to validate")
    args = parser.parse_args()
    run_validation(sample_size=args.sample)

