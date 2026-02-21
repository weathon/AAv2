"""
MCP server for wide-spectrum aesthetics image generation.

Exposes three image-generation tools:
- generate_flux: FLUX.1-Krea-dev model with NAG (Negative-prompt Aligned Guidance)
  for fine-grained control over aesthetic direction. Runs on separate microservice (flux_server.py).
- generate_z_image: Cloud-based Z-image model via Replicate API.
- generate_using_nano_banana: Cloud-based Nano Banana model via Replicate API.

All tools return MCP Image objects that can be viewed directly in the client.
Optionally returns HPSv3 aesthetic scores (good images: 10-15; low = anti-aesthetic success).

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

import dotenv
import replicate
import torch
import requests
from fastmcp import FastMCP
from fastmcp.utilities.types import Image as MCPImage
from PIL import Image

dotenv.load_dotenv()

mcp = FastMCP("Image Generation")

_MAX_RETRIES = 4

import time
import concurrent.futures as _cf

_GENERATION_TIMEOUT = 240  # seconds per attempt

def _with_retry(fn, *args, **kwargs):
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            with _cf.ThreadPoolExecutor(max_workers=1) as _exe:
                future = _exe.submit(fn, *args, **kwargs)
                return future.result(timeout=_GENERATION_TIMEOUT)
        except _cf.TimeoutError:
            last_exc = TimeoutError(f"generation timed out after {_GENERATION_TIMEOUT}s")
            print(f"[retry] attempt {attempt + 1}/{_MAX_RETRIES} timed out", flush=True)
        except Exception as exc:
            last_exc = exc
            print(f"[retry] attempt {attempt + 1}/{_MAX_RETRIES} failed: {exc}", flush=True)
        if attempt < _MAX_RETRIES - 1:
            time.sleep([5, 10, 20, 40][attempt])  # Exponential backoff
    raise last_exc

DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")

COMMITS_JSON = os.path.join(os.path.dirname(__file__), "commits.json")

# ---------------------------------------------------------------------------
# Session cost tracking
# ---------------------------------------------------------------------------

cost: float = 0.0

COST_PER_REPLICATE_IMAGE = 0.03       # z_image and nano_banana

FLUX_SERVER_URL = os.getenv("FLUX_SERVER_URL", "http://127.0.0.1:5001")

# ---------------------------------------------------------------------------
# Aesthetic scoring (HPSv3) — loaded at startup
# ---------------------------------------------------------------------------

hps_dir = os.path.join(os.path.dirname(__file__), "..", "HPSv3")
if hps_dir not in sys.path:
    sys.path.insert(0, hps_dir)
from hpsv3 import HPSv3RewardInferencer

_inferencer = HPSv3RewardInferencer(device=os.getenv("HPS_DEVICE", "cuda:1"))

def _score_pil_images(images: list, eval_prompt: str) -> list:
    """Score PIL images with HPSv3. Returns list of float aesthetic scores.

    Uses the provided eval_prompt as the caption for all images.
    Score range has no hard bounds; good images typically score 10-15.
    For anti-aesthetic goals, lower scores indicate success.
    """
    captions = [eval_prompt] * len(images)

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


def _generate_flux(
    prompt: str,
    negative_prompt: str,
    nag_scale: float,
    nag_alpha: float,
    nag_tau: float,
    num_of_images: int,
    eval_prompt: str,
) -> list[MCPImage | str]:
    """Generate images using FLUX.1-Krea-dev with NAG (Negative-prompt Aligned Guidance).

    NAG allows explicit negative prompts to steer the model away from unwanted aesthetics,
    making it suitable for both pro (high-aesthetics) and anti (low-aesthetics) generation.

    Flux runs on a separate microservice (flux_server.py).

    Args:
        prompt: Text description of the desired image.
        negative_prompt: Text describing what to avoid in the generated image.
        nag_scale: Strength of NAG effect (1-6). Higher = stronger negative guidance.
            Recommended starting value: 3.
        nag_alpha: Blending coefficient for NAG (0-0.5). Higher = stronger effect.
            Recommended starting value: 0.25.
        nag_tau: Threshold controlling which tokens NAG applies to (1-5).
            Higher = stronger effect. Recommended starting value: 2.5.
        num_of_images: Number of images to generate. Do not exceed 5 — each image
            requires a full inference pass and generation time grows linearly.
        eval_prompt: Neutral physical description of the image content used for HPSv3
            scoring. Describe only observable objects.
            **Do NOT include any pro- or anti-aesthetics elements, make a descriptive caption as-if it is a normal image only.**
            **You CANNOT mention elements that is required, such as 'sad emotion' or 'noise', it should be under 10 words.**
            Example: "an apple on a wooden table".

    Returns:
        List of generated images as MCPImage objects followed by a text entry with
        HPSv3 aesthetic scores. Good images score 10-15; low scores = anti-aesthetic success.

    Hint: When the scale is > 8, set tau to around 5 and alpha < 0.5 to avoid overly harsh guidance that can lead to failed generations.
    **First start with recommended values and then adjust based on whether you want a stronger or more subtle effect.**
    """
    if DEBUG:
        return [_debug_image() for _ in range(num_of_images)]# #+ #["[DEBUG] Fake scores: ['9.0000'] * n"]

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

    scores = _score_pil_images(pil_images, eval_prompt)
    score_strs = [f"{s:.4f}" for s in scores]
    results.append(
        f"Aesthetic scores (HPSv3): {score_strs}\n"
        "(Good images typically score 10-15; low scores indicate anti-aesthetic success.)"
    )

    results.append(f"Cost this call: $0.0000 (local GPU) | Session total: ${cost:.4f}")
    return results

@mcp.tool()
def generate_flux(
    prompt: str,
    negative_prompt: str,
    nag_scale: float,
    nag_alpha: float,
    nag_tau: float,
    num_of_images: int,
    eval_prompt: str,
) -> list[MCPImage | str]:
    """Generate images using FLUX.1-Krea-dev with NAG (Negative-prompt Aligned Guidance).

    NAG allows explicit negative prompts to steer the model away from unwanted aesthetics,
    making it suitable for both pro (high-aesthetics) and anti (low-aesthetics) generation.

    Flux runs on a separate microservice (flux_server.py).

    Args:
        prompt: Text description of the desired image.
        negative_prompt: Text describing what to avoid in the generated image.
        nag_scale: Strength of NAG effect (1-6). Higher = stronger negative guidance.
            Recommended starting value: 3.
        nag_alpha: Blending coefficient for NAG (0-0.5). Higher = stronger effect.
            Recommended starting value: 0.25.
        nag_tau: Threshold controlling which tokens NAG applies to (1-5).
            Higher = stronger effect. Recommended starting value: 2.5.
        num_of_images: Number of images to generate. Do not exceed 5 — each image
            requires a full inference pass and generation time grows linearly.
        eval_prompt: Neutral physical description of the image content used for HPSv3
            scoring. Describe only observable objects.
            **Do NOT include any pro- or anti-aesthetics elements, make a descriptive caption as-if it is a normal image only.**
            **You CANNOT mention elements that is required, such as 'sad emotion' or 'noise', it should be under 10 words.**
            Example: "an apple on a wooden table".

    Returns:
        List of generated images as MCPImage objects followed by a text entry with
        HPSv3 aesthetic scores. Good images score 10-15; low scores = anti-aesthetic success.

    Hint: When the scale is > 8, set tau to around 5 and alpha < 0.5 to avoid overly harsh guidance that can lead to failed generations.
    **First start with recommended values and then adjust based on whether you want a stronger or more subtle effect.**
    """
    return _with_retry(
        _generate_flux,
        prompt=prompt,
        negative_prompt=negative_prompt,
        nag_scale=nag_scale,
        nag_alpha=nag_alpha,
        nag_tau=nag_tau,
        num_of_images=num_of_images,
        eval_prompt=eval_prompt,
    )

def _generate_z_image(
    prompt: str,
    negative_prompt: str,
    scale: float,
    num_of_images: int,
    eval_prompt: str,
) -> list[MCPImage | str]:
    if DEBUG:
        return [_debug_image() for _ in range(num_of_images)] #+ #["[DEBUG] Fake scores: ['9.0000'] * n"]

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

    scores = _score_pil_images(pil_images, eval_prompt)
    score_strs = [f"{s:.4f}" for s in scores]
    results.append(
        f"Aesthetic scores (HPSv3): {score_strs}\n"
        "(Good images typically score 10-15; low scores indicate anti-aesthetic success.)"
    )

    results.append(f"Cost this call: ${COST_PER_REPLICATE_IMAGE * num_of_images:.4f} | Session total: ${cost:.4f}")
    return results




@mcp.tool()
def generate_using_z_image(
    prompt: str,
    negative_prompt: str,
    scale: float,
    num_of_images: int,
    eval_prompt: str,
) -> list[MCPImage | str]:
    """Generate images using Z-image via the Replicate API.

    Args:
        prompt: Text description of the desired image.
        negative_prompt: Text describing what to avoid in the generated image.
        scale: Guidance scale controlling prompt adherence (1-15).
            Higher values follow the prompt more strictly.
            Recommended starting value: 7.
        num_of_images: Number of images to generate. Do not exceed 5 — each image
            incurs a separate Replicate API call with associated time and cost.
        eval_prompt: Neutral physical description of the image content used for HPSv3
            scoring. Describe only observable objects.
            Do NOT include any pro- or anti-aesthetics elements, make a descriptive caption as-if it is a normal image only.
        **Do NOT include any pro- or anti-aesthetics elements, make a descriptive caption as-if it is a normal image only.**    
            Example: "an apple on a wooden table".

    Returns:
        List of generated images as MCPImage objects followed by a text entry with
        HPSv3 aesthetic scores. Good images score 10-15; low scores = anti-aesthetic success.
    """
    return _with_retry(
        _generate_z_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        scale=scale,
        num_of_images=num_of_images,
        eval_prompt=eval_prompt,
    )


def _generate_using_nano_banana(
    prompt: str,
    num_of_images: int,
    eval_prompt: str,
) -> list[MCPImage | str]:
    if DEBUG:
        return [_debug_image() for _ in range(num_of_images)] #+ #["[DEBUG] Fake scores: ['9.0000'] * n"]

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

    scores = _score_pil_images(pil_images, eval_prompt)
    score_strs = [f"{s:.4f}" for s in scores]
    results.append(
        f"Aesthetic scores (HPSv3): {score_strs}\n"
        "(Good images typically score 10-15; low scores indicate anti-aesthetic success.)"
    )

    results.append(f"Cost this call: ${COST_PER_REPLICATE_IMAGE * num_of_images:.4f} | Session total: ${cost:.4f}")
    return results

@mcp.tool()
def generate_using_nano_banana(
    prompt: str,
    num_of_images: int,
    eval_prompt: str,
) -> list[MCPImage | str]:
    """Generate images using Nano Banana via the Replicate API.

    Args:
        prompt: Text description of the desired image.
        num_of_images: Number of images to generate. Do not exceed 5 — each image
            incurs a separate Replicate API call with associated time and cost.
        eval_prompt: Neutral physical description of the image content used for HPSv3
            scoring. Describe only observable objects.
            Do NOT include any pro- or anti-aesthetics elements, make a descriptive caption as-if it is a normal image only.
          **You CANNOT mention elements that is required, such as 'sad emotion' or 'noise', it should be under 10 words.**    
            Example: "an apple on a wooden table".

    Returns:
        List of generated images as MCPImage objects followed by a text entry with
        HPSv3 aesthetic scores. Good images score 10-15; low scores = anti-aesthetic success.
    """
    return _with_retry(
        _generate_using_nano_banana,
        prompt=prompt,
        num_of_images=num_of_images,
        eval_prompt=eval_prompt,
    )

def _generate_using_seedream(
    prompt: str,
    num_of_images: int,
    eval_prompt: str,
) -> list[MCPImage | str]:
    if DEBUG:
        return [_debug_image() for _ in range(num_of_images)]

    global cost
    pil_images = []
    for _ in range(num_of_images):
        output = replicate.run(
            "bytedance/seedream-4.5",
            input={
                "size": "custom",
                "width": 2048,
                "height": 2048,
                "prompt": prompt,
                "max_images": 1,
                "aspect_ratio": "1:1",
                "sequential_image_generation": "disabled",
            },
        )
        image_data = output[0].read()
        pil_images.append(Image.open(io.BytesIO(image_data)))

    cost += COST_PER_REPLICATE_IMAGE * num_of_images
    results = [_pil_to_mcp_image(img) for img in pil_images]

    scores = _score_pil_images(pil_images, eval_prompt)
    score_strs = [f"{s:.4f}" for s in scores]
    results.append(
        f"Aesthetic scores (HPSv3): {score_strs}\n"
        "(Good images typically score 10-15; low scores indicate anti-aesthetic success.)"
    )

    results.append(f"Cost this call: ${COST_PER_REPLICATE_IMAGE * num_of_images:.4f} | Session total: ${cost:.4f}")
    return results


@mcp.tool()
def generate_using_seedream(
    prompt: str,
    num_of_images: int,
    eval_prompt: str,
) -> list[MCPImage | str]:
    """Generate images using ByteDance Seedream-4.5 via the Replicate API.

    A text-to-image model. Does not support negative prompts.

    Args:
        prompt: Text description of the desired image.
        num_of_images: Number of images to generate. Do not exceed 5 — each image
            incurs a separate Replicate API call with associated time and cost.
        eval_prompt: Neutral physical description of the image content used for HPSv3
            scoring. Describe only observable objects, no aesthetic language.
            **Must be under 10 words, simple subject-verb(-object) structure.**
            Example: "an apple on a wooden table".

    Returns:
        List of generated images as MCPImage objects followed by a text entry with
        HPSv3 aesthetic scores. Good images score 10-15; low scores = anti-aesthetic success.
    """
    return _with_retry(
        _generate_using_seedream,
        prompt=prompt,
        num_of_images=num_of_images,
        eval_prompt=eval_prompt,
    )


def _generate_using_sdxl(
    prompt: str,
    negative_prompt: str,
    num_of_images: int,
    eval_prompt: str,
    guidance_scale: float = 5.0,
    prompt_strength: float = 0.8,
) -> list[MCPImage | str]:
    if DEBUG:
        return [_debug_image() for _ in range(num_of_images)] #+ #["[DEBUG] Fake scores: ['9.0000'] * n"]

    global cost
    pil_images = []
    for _ in range(num_of_images):
        output = replicate.run(
            "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
            input={
                    "width": 768,
                    "height": 768,
                    "prompt": prompt,
                    "refine": "expert_ensemble_refiner",
                    "scheduler": "K_EULER",
                    "lora_scale": 0.6,
                    "num_outputs": 1,
                    "guidance_scale": guidance_scale,
                    "apply_watermark": False,
                    "high_noise_frac": 0.8,
                    "negative_prompt": negative_prompt,
                    "prompt_strength": prompt_strength,
                    "num_inference_steps": 25,
                    "disable_safety_checker": True,
            },
        )[0]
        image_data = output.read()
        pil_images.append(Image.open(io.BytesIO(image_data)))

    cost += COST_PER_REPLICATE_IMAGE * num_of_images
    results = [_pil_to_mcp_image(img) for img in pil_images]

    scores = _score_pil_images(pil_images, eval_prompt)
    score_strs = [f"{s:.4f}" for s in scores]
    results.append(
        f"Aesthetic scores (HPSv3): {score_strs}\n"
        "(Good images typically score 10-15; low scores indicate anti-aesthetic success.)"
    )

    results.append(f"Cost this call: ${COST_PER_REPLICATE_IMAGE * num_of_images:.4f} | Session total: ${cost:.4f}")
    return results




@mcp.tool()
def generate_using_sdxl(
    prompt: str,
    negative_prompt: str,
    num_of_images: int,
    eval_prompt: str,
    guidance_scale: float = 5.0,
    prompt_strength: float = 0.8,
) -> list[MCPImage | str]:
    """Generate images using Stable Diffusion via the Replicate API.

    Args:
        prompt: Text description of the desired image.
        negative_prompt: Text describing what to avoid in the generated image.
        num_of_images: Number of images to generate. Do not exceed 5 — each image
            incurs a separate Replicate API call with associated time and cost.
        eval_prompt: Neutral physical description of the image content used for HPSv3
            scoring. Describe only observable objects.
            Do NOT include any pro- or anti-aesthetics elements, make a descriptive caption as-if it is a normal image only.
          **You CANNOT mention elements that is required, such as 'sad emotion' or 'noise', it should be under 10 words.**    
            Example: "an apple on a wooden table".
        guidance_scale: Controls how strongly the model follows the prompt (1-15).
            Higher values = more adherence to the prompt. Recommended starting value: 5.
        prompt_strength: Controls how much the initial noise is influenced by the prompt (0-1).
            Higher values = stronger influence. Recommended starting value: 0.8.

    Returns:
        List of generated images as MCPImage objects followed by a text entry with
        HPSv3 aesthetic scores. Good images score 10-15; low scores = anti-aesthetic success.
    """
    return _with_retry(
        _generate_using_sdxl,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_of_images=num_of_images,
        eval_prompt=eval_prompt,
        guidance_scale=guidance_scale,
        prompt_strength=prompt_strength,
    )

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
            flux:      {"nag_scale": 4, "nag_alpha": 0.3, "nag_tau": 3}
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



def execute_model(model: str, params: dict) -> list[MCPImage | str]:
    if model == "flux":
        return _generate_flux(
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt", ""),
            nag_scale=params.get("nag_scale", 3),
            nag_alpha=params.get("nag_alpha", 0.25),
            nag_tau=params.get("nag_tau", 2.5),
            num_of_images=params.get("num_of_images", 1),
            eval_prompt=params["eval_prompt"],
        )
    elif model == "z_image":
        return _generate_z_image(
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt", ""),
            scale=params.get("scale", 7),
            num_of_images=params.get("num_of_images", 1),
            eval_prompt=params["eval_prompt"],
        )
    elif model == "nano_banana":
        return _generate_using_nano_banana(
            prompt=params["prompt"],
            num_of_images=params.get("num_of_images", 1),
            eval_prompt=params["eval_prompt"],
        )
    elif model == "sdxl":
        return _generate_using_sdxl(
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt", ""),
            num_of_images=params.get("num_of_images", 1),
            eval_prompt=params["eval_prompt"],
            guidance_scale=params.get("guidance_scale", 5.0),
            prompt_strength=params.get("prompt_strength", 0.8),
        )
    elif model == "seedream":
        return _generate_using_seedream(
            prompt=params["prompt"],
            num_of_images=params.get("num_of_images", 1),
            eval_prompt=params["eval_prompt"],
        )
    else:
        return [f"Error: Unknown model '{model}' in job entry."]


@mcp.tool()
def batch_generate(jobs: dict) -> list[MCPImage | str]:
    """Generate images concurrently across multiple models in a single call.

    `jobs` is a dict mapping model name to its parameters:
        {
            "flux":        {"prompt": ..., "negative_prompt": ..., "num_of_images": ...,
                            "eval_prompt": ..., "nag_scale": ..., "nag_alpha": ..., "nag_tau": ...},
            "z_image":     {"prompt": ..., "negative_prompt": ..., "num_of_images": ...,
                            "eval_prompt": ..., "scale": ...},
            "nano_banana": {"prompt": ..., "num_of_images": ..., "eval_prompt": ...},
            "sdxl":        {"prompt": ..., "negative_prompt": ..., "num_of_images": ...,
                            "eval_prompt": ..., "guidance_scale": ..., "prompt_strength": ...},
            "seedream":    {"prompt": ..., "num_of_images": ..., "eval_prompt": ...},
        }
    Only include the models you want to run. All included models run in parallel.

    Returns:
        Flat list of images and score strings. Each model's output is preceded
        by a separator line identifying the model and prompt.
    """
    with _cf.ThreadPoolExecutor(max_workers=int(1e100)) as executor:
        futures = {model: executor.submit(execute_model, model, params)
                   for model, params in jobs.items()}

    flat: list[MCPImage | str] = []
    for i, (model, future) in enumerate(futures.items()):
        prompt_preview = jobs[model].get("prompt", "")[:80]
        flat.append(f"--- Job {i+1} | model={model} | prompt={prompt_preview!r} ---")
        flat.extend(future.result())
    return flat
    


  



if __name__ == "__main__":
    print("Starting MCP server...")
    mcp.run(transport="http")
