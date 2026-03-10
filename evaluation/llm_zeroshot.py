import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import urllib.error
import urllib.request

import numpy as np
from datasets import load_dataset
from groq import Groq

from evaluation.metrics import compute_metrics, print_metrics


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_sdgi_test() -> Tuple[List[str], np.ndarray]:
    """
    Load the SDGi test split and return texts and multi-hot labels.
    """
    ds = load_dataset("UNDP/sdgi-corpus")
    test = ds["test"]

    texts: List[str] = []
    labels: List[np.ndarray] = []
    for ex in test:
        texts.append(ex["text"])
        # SDGi stores label indices under the "labels" field
        ys = ex.get("labels") or []
        y_vec = np.zeros(17, dtype=int)
        for sdg in ys:
            if 1 <= sdg <= 17:
                y_vec[sdg - 1] = 1
        labels.append(y_vec)

    return texts, np.stack(labels, axis=0)


def build_zeroshot_prompt(text: str, max_chars: int = 2000) -> str:
    """
    Prompt that asks the LLM to assign relevant SDGs to the document.

    The SDGi documents can be very long; we truncate to max_chars to
    avoid exceeding Groq per-request token limits.
    """
    if len(text) > max_chars:
        text = text[:max_chars]

    return f"""You are an expert in the UN Sustainable Development Goals (SDGs).

You will be given a document. Your task is to decide, for each of the 17 SDGs,
whether the document is meaningfully about that goal.

Document:
\"\"\"{text}\"\"\"

Return ONLY valid JSON with this exact schema:
{{
  "relevant_sdgs": [list of SDG numbers between 1 and 17],
  "primary_sdg": most relevant SDG number, or null if none,
  "confidence": "high" | "medium" | "low",
  "reasoning": "one short sentence"
}}
"""


def parse_model_json(content: str) -> Dict[str, Any]:
    """
    Robust JSON extraction from LLM response.
    """
    content = content.strip()
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


def call_ollama(model: str, prompt: str) -> str:
    """
    Call a local Ollama model via the /api/chat endpoint.
    Expects Ollama to be running on http://localhost:11434.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    # Ollama returns {"message": {"role": "assistant", "content": "..."}}
    message = obj.get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError(f"Ollama response missing content: {obj}")
    return content


def run_zeroshot(
    model: str = "llama-3.3-70b-versatile",
    max_examples: int = 200,
    min_interval: float = 4.0,
    backend: str = "groq",
    dump_path: Path | None = None,
    max_chars: int = 2000,
    max_tokens: int = 256,
) -> None:
    """
    Run zero-shot SDG classification on the SDGi test set.

    backend:
      - "groq": use Groq API (requires GROQ_API_KEY)
      - "ollama": use local Ollama at http://localhost:11434
    """
    client: Groq | None = None
    if backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable must be set.")
        client = Groq(api_key=api_key)

    texts, y_true = load_sdgi_test()
    n_total = len(texts)
    if max_examples and max_examples < n_total:
        n_total = max_examples

    print(f"Running zero-shot on {n_total} SDGi test examples with {model} via {backend}")

    # Initialize predictions and load any existing results if resuming
    y_pred = np.zeros_like(y_true)
    completed_indices = set()

    if dump_path is not None and dump_path.exists():
        print(f"Resuming from existing dump file: {dump_path}")
        with dump_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = obj.get("index")
                parsed = obj.get("parsed") or {}
                if idx is None or not isinstance(idx, int):
                    continue
                if idx >= len(y_true):
                    continue
                completed_indices.add(idx)
                rel = parsed.get("relevant_sdgs") or []
                try:
                    rel_ints = [int(s) for s in rel if 1 <= int(s) <= 17]
                except Exception:
                    rel_ints = []
                if rel_ints:
                    for sdg in rel_ints:
                        y_pred[idx, sdg - 1] = 1
    last_request_time = 0.0

    dump_f = None
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        # Append to existing file to support resume
        dump_f = dump_path.open("a", encoding="utf-8")

    for i, text in enumerate(texts[:n_total]):
        if i in completed_indices:
            continue
        prompt = build_zeroshot_prompt(text, max_chars=max_chars)

        # Simple rate limiting
        now = time.time()
        elapsed = now - last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        last_request_time = time.time()

        parsed: Dict[str, Any] = {}
        raw_content: str | None = None
        for attempt in range(3):
            try:
                if backend == "groq":
                    assert client is not None
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=max_tokens,
                        # Relax strict JSON validation; rely on parse_model_json instead
                        # to reduce json_validate_failed errors on long texts.
                        # response_format={"type": "json_object"},
                    )
                    content = resp.choices[0].message.content or ""
                elif backend == "ollama":
                    content = call_ollama(model=model, prompt=prompt)
                else:
                    raise ValueError(f"Unknown backend '{backend}', expected 'groq' or 'ollama'.")

                raw_content = content
                parsed = parse_model_json(content)
                break
            except Exception as e:
                if attempt == 2:
                    parsed = {"error": repr(e)}
                else:
                    backoff = 2 ** attempt
                    print(f"Error on example {i} (attempt {attempt+1}): {e}. Sleeping {backoff}s")
                    time.sleep(backoff)

        # Optional raw dump for debugging
        if dump_f is not None:
            dump_obj = {
                "index": i,
                "text": text,
                "backend": backend,
                "model": model,
                "raw": raw_content,
                "parsed": parsed,
            }
            dump_f.write(json.dumps(dump_obj, ensure_ascii=False) + "\n")

        rel = parsed.get("relevant_sdgs") or []
        try:
            rel_ints = [int(s) for s in rel if 1 <= int(s) <= 17]
        except Exception:
            rel_ints = []

        if rel_ints:
            for sdg in rel_ints:
                y_pred[i, sdg - 1] = 1

        # Show progress more frequently for long runs
        if (i + 1) % 10 == 0 or i + 1 == n_total:
            print(f"Processed {i+1}/{n_total} examples", flush=True)

    if dump_f is not None:
        dump_f.close()

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, model_name=f"LLM zero-shot ({model})")

    preds_path = RESULTS_DIR / "llm_zeroshot_preds_sdgi.npz"
    np.savez_compressed(preds_path, y_true=y_true, y_pred=y_pred)

    metrics_path = RESULTS_DIR / "llm_zeroshot_metrics_sdgi.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved zero-shot predictions to {preds_path}")
    print(f"Saved zero-shot metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run zero-shot SDG classification on SDGi test set using Groq or Ollama."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Model name to use (Groq or Ollama).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=200,
        help="Maximum number of test examples to evaluate (sample-based run).",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=4.0,
        help="Minimum seconds between requests for simple rate limiting.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["groq", "ollama"],
        default="groq",
        help="Backend to use: 'groq' (API) or 'ollama' (local).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum number of characters from each document to send to the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum completion tokens for the model.",
    )
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=None,
        help="Optional path to write raw model outputs as JSONL for debugging.",
    )
    args = parser.parse_args()

    run_zeroshot(
        model=args.model,
        max_examples=args.max_examples,
        min_interval=args.min_interval,
        backend=args.backend,
        dump_path=args.dump_path,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()

