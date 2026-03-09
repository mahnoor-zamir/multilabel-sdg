import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from groq import Groq
from tqdm import tqdm


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


@dataclass
class GenerationConfig:
    primary_model: str
    rate_limit_rpm: int
    max_retries: int
    easy_per_sdg: int
    hard_per_pair: int
    sdg14_easy_boost: int
    sdg6_easy_boost: int
    sdg7_easy_boost: int
    confused_pairs: List[Tuple[int, int]]
    min_words: int
    max_words: int
    output_path: Path
    failed_path: Path


def load_config(config_path: Path) -> GenerationConfig:
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    groq_cfg = cfg["groq"]
    gen_cfg = cfg["generation"]
    pairs = cfg.get("confused_pairs", [])
    confused_pairs = [tuple(p) for p in pairs]

    data_dir = config_path.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return GenerationConfig(
        primary_model=groq_cfg["primary_model"],
        rate_limit_rpm=int(groq_cfg.get("rate_limit_rpm", 30)),
        max_retries=int(groq_cfg.get("max_retries", 3)),
        easy_per_sdg=int(gen_cfg["easy_per_sdg"]),
        hard_per_pair=int(gen_cfg["hard_per_pair"]),
        sdg14_easy_boost=int(gen_cfg.get("sdg14_easy_boost", gen_cfg["easy_per_sdg"])),
        sdg6_easy_boost=int(gen_cfg.get("sdg6_easy_boost", gen_cfg["easy_per_sdg"])),
        sdg7_easy_boost=int(gen_cfg.get("sdg7_easy_boost", gen_cfg["easy_per_sdg"])),
        confused_pairs=confused_pairs,
        min_words=int(gen_cfg["min_words"]),
        max_words=int(gen_cfg["max_words"]),
        output_path=data_dir / "sdgx_raw.jsonl",
        failed_path=data_dir / "sdgx_failed.jsonl",
    )


def build_easy_prompt(sdg: int) -> str:
    info = SDG_DEFINITIONS[sdg]
    return f"""You are generating training data for an SDG text 
classification system.

Generate a realistic policy document excerpt that 
clearly and unambiguously discusses:
SDG {sdg}: {info['title']} — {info['description']}

Requirements:
- {SDG_DEFINITIONS[sdg]['title']} must be the only substantive SDG theme
- {SDG_DEFINITIONS[sdg]['title']} content should dominate the text
- 200-400 words
- Written like a real UN or government policy document
- Unambiguously about THIS specific SDG, not others
- Include concrete targets, initiatives, or indicators
- Vary writing style across examples:
  styles = ["formal UN report", "government policy brief", 
            "NGO program description", "national review excerpt",
            "municipal action plan"]
- Do NOT mention "SDG", "Goal", or any SDG numbers 
  in the generated text itself

Return ONLY valid JSON with no preamble or explanation:
{{"text": "...", "primary_sdg": {sdg}, "type": "easy",
  "style": "..."}}"""


def build_hard_prompt(sdg_n: int, sdg_m: int) -> str:
    info_n = SDG_DEFINITIONS[sdg_n]
    info_m = SDG_DEFINITIONS[sdg_m]
    pair = f"{sdg_n}_{sdg_m}"
    return f"""You are generating training data for an SDG text 
classification system.

Generate a realistic policy document excerpt that 
genuinely and equally relates to BOTH:
- SDG {sdg_n}: {info_n['title']} — {info_n['description']}
- SDG {sdg_m}: {info_m['title']} — {info_m['description']}

Requirements:
- 200-400 words
- Both SDGs must be naturally and equally present
- Written like a real policy document
- Subtle multi-label — the connection to each SDG 
  should emerge from context, not be stated explicitly
- This should be genuinely ambiguous, challenging 
  to classify even for experts
- Do NOT mention "SDG", "Goal", or SDG numbers 
  in the generated text

Return ONLY valid JSON with no preamble:
{{"text": "...", "sdgs": [{sdg_n}, {sdg_m}], "type": "hard",
  "pair": "{pair}"}}"""


class RateLimiter:
    """
    Simple rate limiter that spaces requests without
    blocking other coroutines more than necessary.
    """

    def __init__(self, rpm: int, max_concurrent: int = 10) -> None:
        self.min_interval = 60.0 / float(rpm)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._last_request_time: float = 0.0
        self._interval_lock = asyncio.Lock()

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        async with self._interval_lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_time
            sleep_for = self.min_interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            self._last_request_time = asyncio.get_event_loop().time()

    def release(self) -> None:
        self._semaphore.release()


async def call_groq_json(
    client: Groq,
    model: str,
    prompt: str,
    rate_limiter: RateLimiter,
    max_retries: int = 3,
) -> Dict[str, Any]:
    for attempt in range(max_retries + 1):
        await rate_limiter.acquire()
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=800,
                ),
            )
            content = resp.choices[0].message.content or ""

            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError(f"No JSON in response: {content[:200]}")

            parsed = json.loads(content[start:end])

            if not parsed.get("text", "").strip():
                raise ValueError("Empty text field in response")

            return parsed

        except json.JSONDecodeError:
            if attempt == max_retries:
                raise
            await asyncio.sleep(2**attempt)
        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str:
                wait = 60 * (attempt + 1)
                print(
                    f"\nRate limited (attempt {attempt + 1}), waiting {wait}s..."
                )
                await asyncio.sleep(wait)
            elif attempt == max_retries:
                raise
            else:
                await asyncio.sleep(2**attempt)
        finally:
            rate_limiter.release()


def load_existing_indices(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    counts: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = f"{obj.get('type','?')}::{obj.get('primary_sdg', obj.get('pair','?'))}"
            counts[key] = counts.get(key, 0) + 1
    return counts


async def generate_all(config: GenerationConfig, test_mode: bool = False) -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable must be set.")

    client = Groq(api_key=api_key)
    rate_limiter = RateLimiter(rpm=config.rate_limit_rpm, max_concurrent=10)

    existing = load_existing_indices(config.output_path)
    total_target_easy = 0
    total_target_hard = 0

    plan: List[Tuple[str, Dict[str, Any], str]] = []

    # Easy examples per SDG
    for sdg in range(1, 18):
        base = config.easy_per_sdg
        if sdg == 14:
            target = config.sdg14_easy_boost
        elif sdg in (6, 7):
            target = config.sdg6_easy_boost if sdg == 6 else config.sdg7_easy_boost
        else:
            target = base
        total_target_easy += target

        key = f"easy::{sdg}"
        done = existing.get(key, 0)
        remaining = max(0, target - done)

        for _ in range(remaining):
            plan.append(("easy", {"primary_sdg": sdg, "type": "easy"}, key))

    # Hard examples for confused pairs
    for n, m in config.confused_pairs:
        pair_key = f"{n}_{m}"
        target = config.hard_per_pair
        total_target_hard += target

        key = f"hard::{pair_key}"
        done = existing.get(key, 0)
        remaining = max(0, target - done)

        for _ in range(remaining):
            plan.append(
                ("hard", {"sdgs": [n, m], "type": "hard", "pair": pair_key}, key)
            )

    if test_mode:
        plan = []
        for _ in range(3):
            plan.append(("easy", {"primary_sdg": 1, "type": "easy"}, "easy::1"))
        if config.confused_pairs:
            n, m = config.confused_pairs[0]
            pair_key = f"{n}_{m}"
            for _ in range(2):
                plan.append(
                    (
                        "hard",
                        {"sdgs": [n, m], "type": "hard", "pair": pair_key},
                        f"hard::{pair_key}",
                    )
                )

    total_target = total_target_easy + total_target_hard
    already_done = sum(existing.values())
    remaining_total = len(plan)

    print(
        f"Planned generation: easy={total_target_easy}, hard={total_target_hard}, "
        f"total={total_target} (already in file: {already_done}, remaining: {remaining_total})"
    )

    if remaining_total == 0:
        print("No remaining examples to generate.")
        return

    est_minutes = remaining_total / float(config.rate_limit_rpm)
    print(
        f"Generating {remaining_total} examples...\n"
        f"Est. time at {config.rate_limit_rpm} RPM with 10 concurrent: "
        f"~{est_minutes:.1f} minutes (ideal, ignoring retries)"
    )

    file_lock = asyncio.Lock()
    stats = defaultdict(int)
    token_count = 0
    processed = 0
    start_time = time.time()

    async def process_one(kind: str, meta: Dict[str, Any], key: str) -> None:
        nonlocal token_count, processed
        try:
            prompt = (
                build_easy_prompt(meta["primary_sdg"])
                if kind == "easy"
                else build_hard_prompt(meta["sdgs"][0], meta["sdgs"][1])
            )
            result = await call_groq_json(
                client,
                config.primary_model,
                prompt,
                rate_limiter,
                max_retries=config.max_retries,
            )
        except Exception as e:
            stats["failed"] += 1
            async with file_lock:
                with config.failed_path.open("a", encoding="utf-8") as ff:
                    ff.write(
                        json.dumps(
                            {
                                "kind": kind,
                                "key": key,
                                "error": repr(e),
                                "meta": meta,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            return

        meta_out = {**meta, **result}
        text = meta_out.get("text", "")
        if not text:
            stats["empty"] += 1
            return

        async with file_lock:
            with config.output_path.open("a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(meta_out, ensure_ascii=False) + "\n")

        token_count += len(text.split())
        stats["success"] += 1
        processed += 1

        if stats["success"] % 100 == 0:
            elapsed = time.time() - start_time
            rate = stats["success"] / elapsed * 60.0 if elapsed > 0 else 0.0
            remaining = remaining_total - processed
            print(
                f"\nCheckpoint: {stats['success']} done, "
                f"{remaining} remaining, "
                f"{rate:.1f}/min actual rate"
            )

    tasks = [asyncio.create_task(process_one(kind, meta, key)) for (kind, meta, key) in plan]

    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Generating SDGX",
        unit="example",
    ):
        await coro

    elapsed_total = time.time() - start_time
    print(
        f"Done: {stats['success']} success, "
        f"{stats['failed']} failed, "
        f"{stats['empty']} empty."
    )
    print(
        f"Total time: {elapsed_total/60.0:.1f} minutes, "
        f"approx tokens: {token_count}"
    )


def run_generation(test: bool = False) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "config.yaml"
    cfg = load_config(config_path)
    asyncio.run(generate_all(cfg, test_mode=test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_generation(test=args.test)


