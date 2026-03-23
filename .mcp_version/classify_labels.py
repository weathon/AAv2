"""
Classify each unique label in sampled_dataset.json into a top-level
anti_aesthetics class from classes_new.json using Kimi via OpenRouter.

Usage:
    python classify_labels.py [--workers 10] [--output label_classes.json]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dotenv
dotenv.load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = os.getenv("CLASSIFY_MODEL", "anthropic/claude-sonnet-4")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY")

MAX_RETRIES = 5
DATASET_PATH = Path(__file__).parent / "sampled_dataset.json"
CLASSES_PATH = "classes_big.json"
OUTPUT_PATH = Path(__file__).parent / "label_classes.json"

# Load classes once at module level so workers inherit it via fork
with open(CLASSES_PATH) as _f:
    CLASSES_JSON = json.load(_f)

AA_CLASSES = CLASSES_JSON["anti_aesthetics"]
VALID_BOTTOM_CLASSES = sum(list([list(i.values()) for i in list(AA_CLASSES.values())]), [])


def classify_single(label: str) -> dict:
    """Ask Kimi which bottom-level anti_aesthetics class this label belongs to."""
    from openai import OpenAI

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    classes_text = json.dumps(AA_CLASSES, indent=2, ensure_ascii=False)
    prompt = (
        f"Here is the full anti_aesthetics taxonomy:\n```json\n{classes_text}\n```\n\n"
        f"Label: \"{label}\"\n\n"
        f"Which bottom-level class does this label belong to? "
        f"Valid classes: {VALID_BOTTOM_CLASSES}\n\n"
        f"Reply with ONLY the class name, nothing else."
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
            )
            msg = response.choices[0].message
            raw = msg.content or ""
            answer = raw.strip().strip('"').strip("'").strip("`")
            # Validate
            if answer in VALID_BOTTOM_CLASSES:
                return {"label": label, "class": answer}
            # Try fuzzy match (e.g. "clarity and focus" -> "clarity_and_focus")
            normalized = answer.lower().replace(" ", "_").replace("-", "_")
            for cls in VALID_BOTTOM_CLASSES:
                if normalized == cls or normalized in cls or cls in normalized:
                    return {"label": label, "class": cls}
            raise ValueError(
                f"LLM returned '{answer}' which is not in {VALID_BOTTOM_CLASSES}"
            )
        except Exception as exc:
            last_error = exc
            wait = min(2 ** attempt, 30)
            print(f"[RETRY {attempt}/{MAX_RETRIES}] {label[:60]}: {exc}", flush=True)
            time.sleep(wait)

    return {"label": label, "class": None, "error": str(last_error)}


def main():
    parser = argparse.ArgumentParser(description="Classify labels into bottom-level classes")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    args = parser.parse_args()

    with open(args.dataset) as f:
        data = json.load(f)
    unique_labels = sorted(set(data["labels"]))
    print(f"[INFO] {len(unique_labels)} unique labels to classify")
    print(f"[INFO] Valid bottom classes: {VALID_BOTTOM_CLASSES}")

    results: dict[str, str] = {}

    # Load existing output if resuming
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"[INFO] Loaded {len(results)} existing classifications")

    todo = [l for l in unique_labels if l not in results]
    print(f"[INFO] {len(todo)} remaining")

    if not todo:
        print("[INFO] All labels already classified.")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(classify_single, label): label for label in todo}
            done_count = 0
            for future in as_completed(futures):
                label = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"label": label, "class": None, "error": str(exc)}

                done_count += 1
                cls = result.get("class")
                if cls:
                    results[label] = cls
                    print(f"[{done_count}/{len(todo)}] {label[:60]} -> {cls}", flush=True)
                else:
                    print(f"[{done_count}/{len(todo)}] FAIL: {label[:60]} -> {result.get('error')}", flush=True)

                # Checkpoint every 20
                if done_count % 20 == 0:
                    with open(output_path, "w") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    classified = sum(1 for v in results.values() if v)
    print(f"\n[DONE] {classified}/{len(unique_labels)} classified -> {output_path}")


if __name__ == "__main__":
    main()
