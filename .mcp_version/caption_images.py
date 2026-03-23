"""
Multi-process image captioning using Kimi via OpenRouter.

For each unique image in sampled_dataset.json, merges all its labels into one
prompt, then asks Kimi to produce a detailed caption that explicitly encodes
the style / aesthetic elements visible in the image.

Usage:
    python caption_images.py [--workers 8] [--output captions.json]

Env vars:
    OPENROUTER_API_KEY   required
    LLM_MODEL            default moonshotai/kimi-k2.5
    LLM_BASE_URL         default https://openrouter.ai/api/v1
"""

import argparse
import base64
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dotenv
dotenv.load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.4-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY")

MAX_RETRIES = 5
DATASET_PATH = Path(__file__).parent / "sampled_dataset.json"
CHECKPOINT_PATH = Path(__file__).parent / "caption_checkpoint.json"
OUTPUT_PATH = Path(__file__).parent / "captions.json"

SYSTEM_PROMPT = """\
You are an expert image analyst. Given an image and a set of style labels,
produce a single cohesive caption describing what the image depicts and how
it looks — weaving content and style into one unified description.

The provided labels are a good reference point but may not be perfect. Use your
own judgement: drop labels that don't match, add style elements you notice that
the labels missed. Trust the image over the labels.

Rules:
- Output ONLY the caption text — no preamble, no labels, no markdown.
- Keep it to 2-3 sentences. Be concise.
- NEVER use the words "anti-aesthetic", "anti-aesthetics", or similar meta
  terms. Just describe the actual visual effects, style, and content directly.
- Weave subject and style naturally. Example: "A dog stands in a sun-scorched
  field, its fur blown out by harsh overhead light that crushes the grass
  into flat yellow."
- Ground style claims in visual evidence (tones, textures, lighting,
  composition, color, focus, grain, etc.).
- No bullet points, no bold, no label names.\
"""


def encode_image(path: str) -> str:
    """Read an image file and return a base64-encoded data URL."""
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def build_merged_dataset(dataset_path: str) -> dict[str, list[str]]:
    """Merge labels by image path. Returns {image_path: [label1, label2, ...]}."""
    with open(dataset_path) as f:
        data = json.load(f)
    merged: dict[str, list[str]] = defaultdict(list)
    for img, label in zip(data["images"], data["labels"]):
        merged[img].append(label)
    return dict(merged)


def caption_single(image_path: str, labels: list[str]) -> dict:
    """Call Kimi to caption a single image. Runs in a worker process."""
    from openai import OpenAI

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    label_text = "\n".join(f"- {l}" for l in labels)
    user_content = [
        {"type": "text", "text": (
            f"Here are the aesthetic/style labels for this image:\n{label_text}\n\n"
            "Produce a detailed caption that explicitly encodes each style/aesthetic "
            "element you observe in the image, grounded in visual evidence."
        )},
        {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
    ]

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=4096,
                extra_body={
                    "reasoning": {
                        "effort": "low"
                    }
                },
            )
            caption = response.choices[0].message.content.strip()
            return {"image": image_path, "labels": labels, "caption": caption}
        except Exception as exc:
            last_error = exc
            wait = min(2 ** attempt, 30)
            print(f"[RETRY {attempt}/{MAX_RETRIES}] {image_path}: {exc}", flush=True)
            time.sleep(wait)

    return {"image": image_path, "labels": labels, "caption": None,
            "error": str(last_error)}


def load_checkpoint() -> dict[str, dict]:
    """Load already-completed captions from checkpoint."""
    try:
        with open(CHECKPOINT_PATH) as f:
            items = json.load(f)
        return {item["image"]: item for item in items}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_checkpoint(results: dict[str, dict]):
    """Persist completed captions to checkpoint."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Caption images with Kimi")
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    args = parser.parse_args()

    merged = build_merged_dataset(args.dataset)
    print(f"[INFO] {len(merged)} unique images, merging from dataset")

    completed = load_checkpoint()
    todo = {k: v for k, v in merged.items() if k not in completed}
    print(f"[INFO] {len(completed)} already done, {len(todo)} remaining")

    if not todo:
        print("[INFO] All images already captioned.")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(caption_single, img, labels): img
                for img, labels in todo.items()
            }
            done_count = 0
            for future in as_completed(futures):
                img = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"image": img, "labels": merged[img], "caption": None,
                              "error": str(exc)}
                completed[img] = result
                done_count += 1
                status = "OK" if result.get("caption") else f"FAIL: {result.get('error', '?')}"
                print(f"[{done_count}/{len(todo)}] {Path(img).name}: {status}", flush=True)

                # Checkpoint every 50 images
                if done_count % 50 == 0:
                    save_checkpoint(completed)
                    print(f"[CHECKPOINT] Saved {len(completed)} results", flush=True)

        save_checkpoint(completed)

    # Write final output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(list(completed.values()), f, ensure_ascii=False, indent=2)

    success = sum(1 for r in completed.values() if r.get("caption"))
    failed = len(completed) - success
    print(f"\n[DONE] {success} captioned, {failed} failed -> {output_path}")


if __name__ == "__main__":
    main()
