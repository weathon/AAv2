"""
MCP server for wide-spectrum aesthetics image generation.

Exposes three image-generation tools:
- generate_flux: FLUX.1-Krea-dev model with NAG (Negative-prompt Aligned Guidance)
  for fine-grained control over aesthetic direction. Runs on separate microservice (flux_server.py).
- generate_z_image: Cloud-based Z-image model via Replicate API.
- generate_using_nano_banana: Cloud-based Nano Banana model via Replicate API.

All tools return MCP Image objects that can be viewed directly in the client.
Optionally returns HPSv3 aesthetic scores (good images: 8-15; low = anti-aesthetic success).

Run with:
    # Start Flux server (separate terminal):
    python image_generation/flux_server.py

    # Then start MCP server:
    uv run image_generation/server.py
"""

import io
import json
import os
import sys
import uuid
import base64
import time

import dotenv
import replicate
import torch
import requests
from fastmcp import FastMCP
from fastmcp.utilities.types import Image as MCPImage
from openai import OpenAI
from PIL import Image

dotenv.load_dotenv()

mcp = FastMCP("Image Generation")

DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")

COMMITS_JSON = os.path.join(os.path.dirname(__file__), "commits.json")

# ---------------------------------------------------------------------------
# Session cost tracking
# ---------------------------------------------------------------------------

cost: float = 0.0

COST_PER_REPLICATE_IMAGE = 0.03       # z_image and nano_banana
COST_PER_CAPTION = 0.000156           # Qwen3-VL captioning via OpenRouter

FLUX_SERVER_URL = os.getenv("FLUX_SERVER_URL", "http://127.0.0.1:5001")

# ---------------------------------------------------------------------------
# Aesthetic scoring (HPSv3) — loaded at startup
# ---------------------------------------------------------------------------

hps_dir = os.path.join(os.path.dirname(__file__), "..", "HPSv3")
if hps_dir not in sys.path:
    sys.path.insert(0, hps_dir)
from hpsv3 import HPSv3RewardInferencer

_inferencer = HPSv3RewardInferencer(device=os.getenv("HPS_DEVICE", "cuda:1"))

_captioning_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def _caption_pil_image(image: Image.Image, max_retries: int = 4) -> str:
    """Caption a PIL image using Qwen3-VL via OpenRouter (physical facts only)."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    for attempt in range(1, max_retries + 1):
        try:
            completion = _captioning_client.chat.completions.create(
                extra_body={},
                model="qwen/qwen3-vl-30b-a3b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Caption this image based on physical facts in the image, "
                                    "ignore aesthetics and styles. Only describe what you see in "
                                    "the image, do not add any interpretation, imagination, or styles. Be "
                                    "concise and objective. The caption should be a single short "
                                    "sentence describe the main content of the image. Do not "
                                    "mention the style or aesthetics (or bad aesthetics) of the image. Focus on "
                                    "physical facts like objects, colors, and their relationships. "
                                    "Do not add any information that cannot be directly observed "
                                    "from the image."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_str}"},
                            },
                        ],
                    }
                ],
            )
            caption = completion.choices[0].message.content
            if caption is None or (isinstance(caption, str) and not caption.strip()):
                raise RuntimeError("Empty caption content from LLM")
            global cost
            api_cost = getattr(completion.usage, "cost", None)
            if api_cost is None and hasattr(completion.usage, "model_extra"):
                api_cost = completion.usage.model_extra.get("cost")
            cost += float(api_cost) if api_cost is not None else COST_PER_CAPTION
            return caption
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return "An image."


def _score_pil_images(images: list) -> list:
    """Score PIL images with HPSv3. Returns list of float aesthetic scores.

    Captions each image via Qwen3-VL (physical description, no aesthetics),
    then feeds caption + image to HPSv3.
    Score range has no hard bounds; good images typically score 10-15.
    For anti-aesthetic goals, lower scores indicate success.
    """
    captions = []
    for i, img in enumerate(images):
        caption = _caption_pil_image(img)
        captions.append(caption)
        print(f"[CAPTION {i+1}] {caption}", flush=True)

    scores = []
    for i in range(0, len(images), 5):
        batch_images = images[i : i + 5]
        batch_captions = captions[i : i + 5]
        with torch.no_grad():
            rewards = _inferencer.reward(prompts=batch_captions, image_paths=batch_images)
        scores.extend([reward[0].item() for reward in rewards])

    return scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEBUG_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", ".mcp_version", "123.jpg")

def _debug_image() -> MCPImage:
    with open(_DEBUG_IMAGE_PATH, "rb") as f:
        return MCPImage(data=f.read(), format="jpeg")


def _pil_to_mcp_image(image: Image.Image) -> MCPImage:
    """Convert a PIL Image to a FastMCP Image (WebP for smaller payload)."""
    buffered = io.BytesIO()
    try:
        image.save(buffered, format="WEBP", quality=85, optimize=True)
        return MCPImage(data=buffered.getvalue(), format="webp")
    except Exception:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return MCPImage(data=buffered.getvalue(), format="png")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def generate_flux(
    prompt: str,
    negative_prompt: str,
    nag_scale: float,
    nag_alpha: float,
    nag_tau: float,
    num_of_images: int,
    return_aesthetic_score: bool = True,
) -> list[MCPImage | str]:
    """Generate images using FLUX.1-Krea-dev with NAG (Negative-prompt Aligned Guidance).

    NAG allows explicit negative prompts to steer the model away from unwanted aesthetics,
    making it suitable for both pro (high-aesthetics) and anti (low-aesthetics) generation.

    Flux runs on a separate microservice (flux_server.py).

    Args:
        prompt: Text description of the desired image.
        negative_prompt: Text describing what to avoid in the generated image.
        nag_scale: Strength of NAG effect (1-12). Higher = stronger negative guidance.
            Recommended starting value: 5. 
        nag_alpha: Blending coefficient for NAG (0-1). Higher = stronger effect.
            Recommended starting value: 0.3.
        nag_tau: Threshold controlling which tokens NAG applies to (0-10).
            Higher = weaker/more selective effect. Recommended starting value: 5.
        num_of_images: Number of images to generate. Do not exceed 5 — each image
            requires a full inference pass and generation time grows linearly.
        return_aesthetic_score: If True, score each image with HPSv3 and append scores
            to the result. Good images score 10-15; low scores = anti-aesthetic success.

    Returns:
        List of generated images as MCPImage objects, optionally followed by a
        text entry with HPSv3 aesthetic scores.

    Hint: When the scale is > 8, set tau to around 5 and alpha < 0.5 to avoid overly harsh guidance that can lead to failed generations.
    """
    if DEBUG:
        return [_debug_image() for _ in range(num_of_images)] + ["[DEBUG] Fake scores: ['9.0000'] * n"]

    # Call Flux microservice
    response = requests.post(
        f"{FLUX_SERVER_URL}/generate",
        json={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "nag_scale": nag_scale,
            "nag_alpha": nag_alpha,
            "nag_tau": nag_tau,
            "num_of_images": num_of_images,
        },
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Flux server error: {response.status_code} - {response.text}"
        )

    data = response.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Flux generation failed: {data.get('error', 'Unknown error')}")

    # Convert base64 images to MCPImage objects
    images_b64 = data.get("images", [])
    pil_images = []
    results = []

    for img_b64 in images_b64:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(img_b64)
        # Open as PIL Image (for aesthetic scoring if needed)
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_images.append(pil_img)
        # Convert to MCPImage
        results.append(MCPImage(data=image_bytes, format="webp"))

    if return_aesthetic_score:
        scores = _score_pil_images(pil_images)
        score_strs = [f"{s:.4f}" for s in scores]
        results.append(
            f"Aesthetic scores (HPSv3): {score_strs}\n"
            "(Good images typically score 8-15; low scores indicate anti-aesthetic success.)"
        )

    results.append(f"Cost this call: $0.0000 (local GPU) | Session total: ${cost:.4f}")
    return results


@mcp.tool()
def generate_z_image(
    prompt: str,
    negative_prompt: str,
    scale: float,
    num_of_images: int,
    return_aesthetic_score: bool = True,
) -> list[MCPImage | str]:
    """Generate images using Z-image via the Replicate API.

    Args:
        prompt: Text description of the desired image.
        negative_prompt: Text describing what to avoid in the generated image.
        scale: Guidance scale controlling prompt adherence (1-15).
            Higher values follow the prompt more strictly.
            Recommended starting value: 5.
        num_of_images: Number of images to generate. Do not exceed 5 — each image
            incurs a separate Replicate API call with associated time and cost.
        return_aesthetic_score: If True, score each image with HPSv3 and append scores
            to the result. Good images score 8-15; low scores = anti-aesthetic success.

    Returns:
        List of generated images as MCPImage objects, optionally followed by a
        text entry with HPSv3 aesthetic scores.
    """
    if DEBUG:
        return [_debug_image() for _ in range(num_of_images)] + ["[DEBUG] Fake scores: ['9.0000'] * n"]

    global cost
    pil_images = []
    for _ in range(num_of_images):
        output = replicate.run(
            "prunaai/z-image:eb865cc448032613678cd0e4e99548671cdff1286bc04f0f605b3fc10fffe3aa",
            input={
                "width": 1024,
                "height": 1024,
                "prompt": prompt,
                "output_format": "webp",
                "guidance_scale": scale,
                "output_quality": 90,
                "negative_prompt": negative_prompt,
                "num_inference_steps": 28,
            },
        )
        image_data = output.read()
        pil_images.append(Image.open(io.BytesIO(image_data)))

    cost += COST_PER_REPLICATE_IMAGE * num_of_images
    results = [_pil_to_mcp_image(img) for img in pil_images]

    if return_aesthetic_score:
        scores = _score_pil_images(pil_images)
        score_strs = [f"{s:.4f}" for s in scores]
        results.append(
            f"Aesthetic scores (HPSv3): {score_strs}\n"
            "(Good images typically score 8-15; low scores indicate anti-aesthetic success.)"
        )

    results.append(f"Cost this call: ${COST_PER_REPLICATE_IMAGE * num_of_images:.4f} | Session total: ${cost:.4f}")
    return results


@mcp.tool()
def generate_using_nano_banana(
    prompt: str,
    num_of_images: int,
    return_aesthetic_score: bool = True,
) -> list[MCPImage | str]:
    """Generate images using Nano Banana via the Replicate API.

    Args:
        prompt: Text description of the desired image.
        num_of_images: Number of images to generate. Do not exceed 5 — each image
            incurs a separate Replicate API call with associated time and cost.
        return_aesthetic_score: If True, score each image with HPSv3 and append scores
            to the result. Good images score 8-15; low scores = anti-aesthetic success.

    Returns:
        List of generated images as MCPImage objects, optionally followed by a
        text entry with HPSv3 aesthetic scores.
    """
    if DEBUG:
        return [_debug_image() for _ in range(num_of_images)] + ["[DEBUG] Fake scores: ['9.0000'] * n"]

    global cost
    pil_images = []
    for _ in range(num_of_images):
        output = replicate.run(
            "google/nano-banana",
            input={
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "output_format": "jpg",
            },
        )
        image_data = output.read()
        pil_images.append(Image.open(io.BytesIO(image_data)))

    cost += COST_PER_REPLICATE_IMAGE * num_of_images
    results = [_pil_to_mcp_image(img) for img in pil_images]

    if return_aesthetic_score:
        scores = _score_pil_images(pil_images)
        score_strs = [f"{s:.4f}" for s in scores]
        results.append(
            f"Aesthetic scores (HPSv3): {score_strs}\n"
            "(Good images typically score 8-15; low scores indicate anti-aesthetic success.)"
        )

    results.append(f"Cost this call: ${COST_PER_REPLICATE_IMAGE * num_of_images:.4f} | Session total: ${cost:.4f}")
    return results


@mcp.tool()
def init() -> str:
    """Initialize a new session. MUST be called at the start of every session.

    Resets the session cost tracker to $0.00.

    Returns:
        Confirmation that the session has been initialized.
    """
    global cost
    cost = 0.0
    return "Session initialized. Cost tracker reset to $0.00."


@mcp.tool()
def commit(entries: list) -> str:
    """Commit a batch of image generation configurations for later bulk generation.

    Each entry in the list is a dict with the following keys:
        - model (str): One of "flux", "z_image", or "nano_banana".
        - prompt (str): Positive text prompt.
        - negative_prompt (str): Negative text prompt (use empty string for nano_banana).
        - other_parameters (dict): Model-specific parameters, e.g.:
            flux:      {"nag_scale": 7, "nag_alpha": 0.5, "nag_tau": 5}
            z_image:   {"scale": 5}
            nano_banana: {}

    Args:
        entries: List of generation configuration dicts (100-200 entries recommended).

    Returns:
        Confirmation string with commit ID and entry count.
    """
    commit_id = str(uuid.uuid4())[:8]

    try:
        with open(COMMITS_JSON, "r") as f:
            commits = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        commits = {}

    commits[commit_id] = {
        "entries": entries,
        "size": len(entries),
    }

    with open(COMMITS_JSON, "w") as f:
        json.dump(commits, f, indent=2)

    return f"Committed {len(entries)} entries with ID: {commit_id}"


@mcp.tool()
def add_agent_cost(amount: float) -> str:
    """Add external agent (LLM) cost to the session cost tracker.

    The MCP client should call this after each LLM inference to keep the
    server-side total in sync with actual spend.

    Args:
        amount: Cost in USD to add.

    Returns:
        Updated session total cost string.
    """
    global cost
    cost += amount
    return f"Added ${amount:.6f} | Session total: ${cost:.4f}"


@mcp.tool()
def log_action(msg: str = "") -> str:
    """Log a message to console."""
    print(msg, flush=True)
    return "log successfully"


if __name__ == "__main__":
    print("Starting MCP server...")
    mcp.run(transport="http")
