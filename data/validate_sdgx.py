import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai
import yaml
from groq import Groq
from tqdm import tqdm

from .generate_sdgx import SDG_DEFINITIONS

@dataclass
class ValidationConfig:
    judge_model: str
    gemini_model: str
    rate_limit_rpm: int
    max_retries: int
    sample_size: int
    input_path: Path
    output_path: Path


def load_config(config_path: Path, sample_size: int = 100) -> ValidationConfig:
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    groq_cfg = cfg["groq"]
    gem_cfg = cfg.get("gemini", {})

    root = config_path.parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    input_path = data_dir / "sdgx_raw.jsonl"
    output_path = data_dir / "sdgx_valid_sample.jsonl"

    return ValidationConfig(
        judge_model=groq_cfg.get("primary_model", "llama-3.1-8b-instant"),
        gemini_model=gem_cfg.get("model", "gemini-2.0-flash"),
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
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY environment variable must be set.")
    gem_key = os.environ.get("GEMINI_API_KEY")
    if not gem_key:
        raise RuntimeError("GEMINI_API_KEY environment variable must be set.")

    groq_client = Groq(api_key=groq_key)
    genai.configure(api_key=gem_key)
    gem_model = genai.GenerativeModel(config.gemini_model)

    examples = sample_examples(config.input_path, config.sample_size)
    if not examples:
        print("No examples found to validate.")
        return

    print(f"Validating random sample of {len(examples)} examples from {config.input_path}")

    stats = {
        "total": 0,
        "low_confidence": 0,
        # Easy examples: how many judges match primary_sdg
        "easy_both_correct": 0,
        "easy_one_correct": 0,
        "easy_none_correct": 0,
        # Hard examples: whether both/one/none judges find both target SDGs
        "hard_both_both": 0,
        "hard_partial": 0,
        "hard_none": 0,
        # Errors
        "unscored_rate_limit": 0,
        "unscored_other_error": 0,
    }

    with config.output_path.open("w", encoding="utf-8") as out_f:
        for ex in tqdm(examples, desc="Validating SDGX sample", unit="example"):
            text = ex.get("text", "")
            if not text:
                continue

            prompt = build_validation_prompt(text)

            groq_result: Dict[str, Any] = {}
            gem_result: Dict[str, Any] = {}

            # Groq judge
            for attempt in range(config.max_retries + 1):
                try:
                    resp = groq_client.chat.completions.create(
                        model=config.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=512,
                        response_format={"type": "json_object"},
                    )
                    content = resp.choices[0].message.content or ""
                    groq_result = parse_model_json(content)
                    break
                except Exception as e:
                    if attempt == config.max_retries:
                        groq_result = {"error": repr(e)}
                    else:
                        time.sleep(2**attempt)

            # Gemini judge
            for attempt in range(config.max_retries + 1):
                try:
                    resp = gem_model.generate_content(prompt)
                    content = resp.text or ""
                    gem_result = parse_model_json(content)
                    break
                except Exception as e:
                    if attempt == config.max_retries:
                        gem_result = {"error": repr(e)}
                    else:
                        time.sleep(2**attempt)

            combined = {**ex, "judge_groq": groq_result, "judge_gemini": gem_result}
            out_f.write(json.dumps(combined, ensure_ascii=False) + "\n")

            stats["total"] += 1
            if not isinstance(groq_result, dict) or not isinstance(gem_result, dict):
                continue

            # Handle judge errors explicitly
            errors = [groq_result.get("error"), gem_result.get("error")]
            if any(errors):
                joined = " ".join(str(e) for e in errors if e)
                if "RateLimitError" in joined or "rate_limit_exceeded" in joined:
                    stats["unscored_rate_limit"] += 1
                else:
                    stats["unscored_other_error"] += 1
                continue

            # Use Groq confidence for now
            conf = str(groq_result.get("confidence", "")).lower()
            if conf == "low":
                stats["low_confidence"] += 1

            ex_type = ex.get("type")

            # EASY: compare primary SDG with two judges
            if ex_type == "easy":
                ex_primary = ex.get("primary_sdg")
                g_primary = groq_result.get("primary_sdg")
                m_primary = gem_result.get("primary_sdg")
                if ex_primary is not None and g_primary is not None and m_primary is not None:
                    ex_p = int(ex_primary)
                    g_p = int(g_primary)
                    m_p = int(m_primary)
                    if g_p == ex_p and m_p == ex_p:
                        stats["easy_both_correct"] += 1
                    elif g_p == ex_p or m_p == ex_p:
                        stats["easy_one_correct"] += 1
                    else:
                        stats["easy_none_correct"] += 1

            # HARD: check how many target SDGs were found by each judge
            if ex_type == "hard":
                target_sdgs = ex.get("sdgs") or []
                try:
                    target_sdgs = [int(s) for s in target_sdgs]
                except Exception:
                    target_sdgs = []
                g_rel = groq_result.get("relevant_sdgs") or []
                m_rel = gem_result.get("relevant_sdgs") or []
                try:
                    g_rel = [int(s) for s in g_rel]
                    m_rel = [int(s) for s in m_rel]
                except Exception:
                    g_rel, m_rel = [], []

                if target_sdgs:
                    tset = set(target_sdgs)
                    gset = set(g_rel)
                    mset = set(m_rel)
                    g_ok = tset.issubset(gset)
                    m_ok = tset.issubset(mset)
                    if g_ok and m_ok:
                        stats["hard_both_both"] += 1
                    elif (tset & gset) or (tset & mset):
                        stats["hard_partial"] += 1
                    else:
                        stats["hard_none"] += 1

    print("\nValidation summary (sample):")
    print(f"  Total validated:              {stats['total']}")
    print(f"  Low confidence:               {stats['low_confidence']}")
    print(f"  Unscored (rate limit):        {stats['unscored_rate_limit']}")
    print(f"  Unscored (other error):       {stats['unscored_other_error']}")
    print(f"  EASY both judges correct:     {stats['easy_both_correct']}")
    print(f"  EASY one judge correct:       {stats['easy_one_correct']}")
    print(f"  EASY no judge correct:        {stats['easy_none_correct']}")
    print(f"  HARD both judges found both:  {stats['hard_both_both']}")
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

