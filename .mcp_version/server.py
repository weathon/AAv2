"""
MCP Server exposing dataset curation tools.

Converted from OpenAI Agents SDK to Model Context Protocol (MCP).
All 9 tools (`init` + search, sample, commit, undo_commit, status,
sample_from_committed, aesthetics_rate, log_actions) are exposed
as MCP tools that any MCP-compatible client can call.
"""

import os
import sys
import json
import uuid
import time
import random
import datetime
import base64
import contextlib
from io import BytesIO

import dotenv
dotenv.load_dotenv()

import torch
import numpy as np
from PIL import Image as PILImage
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image as MCPImage
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import weave

# ---------------------------------------------------------------------------
# Lightweight imports at startup; heavy resources are lazy-loaded via `init`
# ---------------------------------------------------------------------------

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../openai_sdk_tools"))

from image_utils import grid_stack  # reuse original, no ToolOutputImage dep

captioning_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

dataset_commits: dict = {}
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "agent_log.txt")
DATASET_JSON = os.path.join(os.path.dirname(__file__), "..", "dataset.json")
DATASET_ROOT = os.getenv("DATASET_ROOT", "/home/wg25r/Downloads/ds/train")
WEAVE_PROJECT = os.getenv("WEAVE_PROJECT", "aas2-mcp-server")
_IS_INITIALIZED = False
_INIT_REQUIRED_MSG = "Server resources are not initialized. You need to call `init` first."

# Lazy-loaded resources (populated by `init`)
model = None
inferencer = None
ava_embeddings_tensor = None
ls_embeddings_tensor = None
lapis_embeddings_tensor = None
ava_names_list = None
ls_names_list = None
lapis_names_list = None
dataset_map = {
    "photos": "ava",
    "dreamcore": "liminal_space",
    "artwork": "lapis",
}
_loader_summary: dict = {}

# ---------------------------------------------------------------------------
# Helpers (not exposed as tools)
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _require_init(tool_name: str):
    if _IS_INITIALIZED:
        return None
    return _INIT_REQUIRED_MSG


def _load_heavy_resources() -> tuple[float, dict]:
    global _IS_INITIALIZED
    global model, inferencer
    global ava_embeddings_tensor, ls_embeddings_tensor, lapis_embeddings_tensor
    global ava_names_list, ls_names_list, lapis_names_list
    global dataset_map, _loader_summary

    start = time.time()

    with contextlib.redirect_stdout(sys.stderr):
        from dataset_loader import (
            model as loaded_model,
            ava_embeddings_tensor as loaded_ava_embeddings_tensor,
            ls_embeddings_tensor as loaded_ls_embeddings_tensor,
            lapis_embeddings_tensor as loaded_lapis_embeddings_tensor,
            ava_names_list as loaded_ava_names_list,
            ls_names_list as loaded_ls_names_list,
            lapis_names_list as loaded_lapis_names_list,
            dataset_map as loaded_dataset_map,
            dataset_loader_summary,
        )
        from hpsv3 import HPSv3RewardInferencer

    model = loaded_model
    ava_embeddings_tensor = loaded_ava_embeddings_tensor
    ls_embeddings_tensor = loaded_ls_embeddings_tensor
    lapis_embeddings_tensor = loaded_lapis_embeddings_tensor
    ava_names_list = loaded_ava_names_list
    ls_names_list = loaded_ls_names_list
    lapis_names_list = loaded_lapis_names_list
    dataset_map = loaded_dataset_map

    inferencer_device = os.getenv("HPS_DEVICE", "cuda:1")
    with contextlib.redirect_stdout(sys.stderr):
        inferencer = HPSv3RewardInferencer(device=inferencer_device)
    _loader_summary = dataset_loader_summary()
    _loader_summary["inferencer_device"] = inferencer_device

    _IS_INITIALIZED = True
    elapsed = round(time.time() - start, 2)
    return elapsed, _loader_summary


def _pil_to_mcp_image(image: PILImage.Image) -> MCPImage:
    """Convert a PIL Image to a FastMCP Image (WebP for smaller payload)."""
    buffered = BytesIO()
    try:
        image.save(buffered, format="WEBP", quality=85, optimize=True)
        return MCPImage(data=buffered.getvalue(), format="webp")
    except Exception:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return MCPImage(data=buffered.getvalue(), format="png")


def _caption_single_image(path: str, max_retries: int = 4) -> str:
    image = PILImage.open(path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    for attempt in range(1, max_retries + 1):
        try:
            completion = captioning_client.chat.completions.create(
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
                                    "the image, do not add any interpretation or imagination. Be "
                                    "concise and objective. The caption should be a single short "
                                    "sentence describe the main content of the image. Do not "
                                    "mention the style or aesthetics of the image. Focus on "
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
            _log(f"[LOG] Generated caption for {path}: {caption}")
            return caption
        except Exception as e:
            _log(f"[WARN] Captioning attempt {attempt}/{max_retries} failed for {path}: {e}")
            if attempt < max_retries:
                time.sleep(2**attempt)
            else:
                _log(f"[ERROR] All {max_retries} captioning attempts failed for {path}, using fallback.")
                return "An image."


def _rate_images(image_paths: list[str]) -> str:
    captions = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {
            executor.submit(_caption_single_image, path, max_retries=4): idx
            for idx, path in enumerate(image_paths)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            captions[idx] = future.result()

    scores = []
    for i in range(0, len(image_paths), 4):
        batch_prompts = captions[i : i + 4]
        batch_paths = image_paths[i : i + 4]
        with torch.no_grad():
            rewards = inferencer.reward(prompts=batch_prompts, image_paths=batch_paths)
        scores.extend([reward[0].item() for reward in rewards])

    hist = np.histogram(scores, bins=10)
    hist_str = f"Score histogram: {hist[0].tolist()}, bins: {hist[1].tolist()}"
    raw_scores = [f"{score:.4f}" for score in scores]
    return hist_str + "\nRaw Scores: " + str(raw_scores)


def _get_embeddings_and_names(dataset: str):
    if dataset == "photos":
        return ava_embeddings_tensor, ava_names_list
    elif dataset == "dreamcore":
        return ls_embeddings_tensor, ls_names_list
    else:
        return lapis_embeddings_tensor, lapis_names_list


def _apply_negative_filter(embeddings, names, negative_prompts: list[str] | None, negative_threshold: float):
    if not negative_prompts:
        return set()
    combined_mask = torch.zeros(len(embeddings), dtype=torch.bool)
    for neg in negative_prompts:
        q_emb = model.process([{"text": neg}]).cpu().float()
        sim = torch.nn.functional.cosine_similarity(embeddings, q_emb)
        combined_mask |= sim > negative_threshold
    target_indices = torch.where(combined_mask)[0].tolist()
    return {names[i].item() for i in target_indices}


def _search_impl(
    query: str,
    dataset: str,
    negative_prompts: list[str],
    negative_threshold: float,
    t: int,
    return_paths: bool = False,
):
    _log(f"[LOG] Searching for '{query}' in dataset '{dataset}' ...")
    embeddings, names = _get_embeddings_and_names(dataset)
    excluded = _apply_negative_filter(embeddings, names, negative_prompts, negative_threshold)

    query_embedding = model.process([{"text": query}]).cpu()
    res = torch.nn.functional.cosine_similarity(embeddings, query_embedding.float())

    # Compute similarity distribution histogram (excluding negatively-filtered images)
    excluded_indices = {i for i, n in enumerate(names) if n.item() in excluded}
    valid_mask = torch.ones(len(res), dtype=torch.bool)
    for idx in excluded_indices:
        valid_mask[idx] = False
    valid_scores = res[valid_mask].numpy()
    hist = np.histogram(valid_scores, bins=10)
    sim_distribution = f"Similarity distribution: counts={hist[0].tolist()}, bins=[{', '.join(f'{b:.3f}' for b in hist[1].tolist())}]"

    selected_images = []
    top_scores = []
    for idx in torch.argsort(res, descending=True):
        if names[idx].item() not in excluded:
            selected_images.append(names[idx].item())
            top_scores.append(f"{res[idx].item():.4f}")
        if len(selected_images) >= t:
            break

    paths = []
    for name in selected_images:
        path = f"{DATASET_ROOT}/{dataset_map[dataset]}/{name}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file missing: {path}")
        paths.append(path)

    score_info = f"Top-{len(top_scores)} scores: [{', '.join(top_scores)}]\n{sim_distribution}"

    if return_paths:
        return paths, score_info
    return grid_stack(paths, row_size=5), score_info


def _sample_impl(
    query: str,
    dataset: str,
    min_threshold: float,
    max_threshold: float,
    negative_prompts: list[str],
    negative_threshold: float,
) -> list[str]:
    embeddings, names = _get_embeddings_and_names(dataset)
    excluded = _apply_negative_filter(embeddings, names, negative_prompts, negative_threshold)

    query_embedding = model.process([{"text": query}]).cpu()
    res = torch.nn.functional.cosine_similarity(embeddings, query_embedding.float())

    mask = torch.logical_and(res >= min_threshold, res <= max_threshold)
    candidate_indices = torch.where(mask)[0].tolist()
    selected_images = [names[i].item() for i in candidate_indices if names[i].item() not in excluded]

    paths = []
    for name in selected_images:
        path = f"{DATASET_ROOT}/{dataset_map[dataset]}/{name}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file missing: {path}")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("Dataset Curation Server", host="0.0.0.0", port=8765)


@mcp.tool()
def init():
    """Initialize embeddings and models. Call this once before using any other tool."""
    if _IS_INITIALIZED:
        return "Already initialized."

    try:
        elapsed, summary = _load_heavy_resources()
    except Exception as e:
        return f"Initialization failed: {e}"

    return (
        f"Initialization complete in {elapsed}s. "
        f"rows={summary.get('total_rows', 'n/a')}, "
        f"embedding_dim={summary.get('embedding_dim', 'n/a')}."
    )


@mcp.tool()
def search(
    query: str,
    dataset: str,
    negative_prompts: list[str] = None,
    negative_threshold: float = 0.3,
    t: int = 10,
) -> list:
    """Search for top-k images matching the query.

    Args:
        query: Text query for semantic image search.
        dataset: One of "photos", "dreamcore", or "artwork".
        negative_prompts: List of negative text prompts to filter out (3-5 items max).
        negative_threshold: Cosine similarity threshold for negative filtering.
        t: Number of top results to return.

    Returns a grid preview image of the top-k matches.
    """
    init_error = _require_init("search")
    if init_error:
        return init_error

    if negative_prompts is None:
        negative_prompts = []

    try:
        result, score_info = _search_impl(query, dataset, negative_prompts, negative_threshold, t)
        if result is None:
            return [f"No Image Found\n{score_info}"]
        return [
            _pil_to_mcp_image(result),
            f"Showing top {t} results for '{query}' in {dataset}.\n{score_info}",
        ]
    except Exception as e:
        _log(f"[ERROR] Search failed: {e}")
        return [f"Error: {e}"]


@mcp.tool()
def sample(
    query: str,
    dataset: str,
    min_threshold: float,
    max_threshold: float,
    count: int = 5,
    negative_prompts: list[str] = None,
    negative_threshold: float = 0.2,
) -> list:
    """Sample random images within a similarity score range.

    Args:
        query: Text query for semantic image search.
        dataset: One of "photos", "dreamcore", or "artwork".
        min_threshold: Minimum cosine similarity.
        max_threshold: Maximum cosine similarity (usually 1.0 unless excluding with negative_prompts).
        count: Number of images to sample.
        negative_prompts: Negative text prompts to exclude.
        negative_threshold: Threshold for negative filtering.

    Returns a grid of sampled images for threshold calibration.
    """
    init_error = _require_init("sample")
    if init_error:
        return init_error

    if negative_prompts is None:
        negative_prompts = []

    _log(f"[LOG] Sampling for '{query}' in dataset '{dataset}' between {min_threshold} and {max_threshold} ...")

    paths = _sample_impl(query, dataset, min_threshold, max_threshold, negative_prompts, negative_threshold)
    if len(paths) == 0:
        return ["No Image Found"]

    sampled_paths = random.sample(paths, min(count, len(paths)))
    _log(f"[LOG] Sampled {len(sampled_paths)} images from {len(paths)} candidates.")

    whole_image = grid_stack(sampled_paths, row_size=5)

    return [
        _pil_to_mcp_image(whole_image),
        f"Sampled {len(sampled_paths)} from {len(paths)} candidates.",
    ]


@mcp.tool()
def aesthetics_rate(
    query: str,
    dataset: str,
    min_threshold: float,
    max_threshold: float,
    negative_prompts: list[str] = None,
    negative_threshold: float = 0.2,
    sample_size: int = 100,
) -> str:
    """Rate the aesthetics scores of images matching the query.

    Args:
        query: Text query for semantic image search.
        dataset: One of "photos", "dreamcore", or "artwork".
        min_threshold: Minimum cosine similarity.
        max_threshold: Maximum cosine similarity (usually 1.0 unless excluding with negative_prompts).
        negative_prompts: Negative text prompts to exclude.
        negative_threshold: Threshold for negative filtering.
        sample_size: Max number of images to rate (25-50 recommended).

    Returns a string describing the distribution of aesthetics scores.
    """
    init_error = _require_init("aesthetics_rate")
    if init_error:
        return init_error

    if negative_prompts is None:
        negative_prompts = []

    _log(f"[LOG] Rating aesthetics for '{query}' in dataset '{dataset}' between {min_threshold} and {max_threshold} ...")

    paths = _sample_impl(query, dataset, min_threshold, max_threshold, negative_prompts, negative_threshold)
    if len(paths) == 0:
        return "No images found matching the criteria."

    if len(paths) > sample_size:
        paths_to_rate = random.sample(paths, sample_size)
        _log(f"[LOG] Sampled {sample_size} images from {len(paths)} total candidates for rating.")
    else:
        paths_to_rate = paths
        _log(f"[LOG] Rating all {len(paths)} matching images.")

    scores = _rate_images(paths_to_rate)
    _log(f"[LOG] Aesthetics scores: {scores}")
    return f"Aesthetics scores for {len(paths_to_rate)} images: {scores}"


@mcp.tool()
def commit(
    query: str,
    dataset: str,
    threshold: float,
    negative_prompts: list[str] = None,
    negative_threshold: float = 0.2,
    message: str = "",
) -> str:
    """Commit all images with similarity >= threshold to the dataset.

    Args:
        query: Text query used for the search.
        dataset: One of "photos", "dreamcore", or "artwork".
        threshold: Minimum cosine similarity threshold (0.0-1.0). Usually 1.0 unless excluding with negative_prompts.
        negative_prompts: Negative text prompts to exclude.
        negative_threshold: Threshold for negative filtering.
        message: Descriptive tags for this commit (sub-element, aesthetic direction, etc.).

    Returns confirmation with commit ID and image count.
    """
    init_error = _require_init("commit")
    if init_error:
        return init_error

    if negative_prompts is None:
        negative_prompts = []

    _log(f"[LOG] Committing with message: {message}")

    embeddings, names = _get_embeddings_and_names(dataset)
    excluded = _apply_negative_filter(embeddings, names, negative_prompts, negative_threshold)

    query_embedding = model.process([{"text": query}]).cpu()
    res = torch.nn.functional.cosine_similarity(embeddings, query_embedding.float())

    mask = res >= threshold
    candidate_indices = torch.where(mask)[0].tolist()
    selected_images = [names[i].item() for i in candidate_indices if names[i].item() not in excluded]

    images = []
    for name in selected_images:
        path = f"{DATASET_ROOT}/{dataset_map[dataset]}/{name}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file missing: {path}")
        images.append(path)

    commit_id = str(uuid.uuid4())[:8]
    dataset_commits[commit_id] = {
        "query": query,
        "dataset": dataset,
        "threshold": threshold,
        "negative_prompts": negative_prompts,
        "negative_threshold": negative_threshold,
        "message": message,
        "images": images,
        "size": len(images),
    }

    with open(DATASET_JSON, "w") as f:
        json.dump(dataset_commits, f, indent=2)

    return f"Committed with ID: {commit_id}, message: {message} with {len(images)} images."


@mcp.tool()
def undo_commit(commit_id: str) -> str:
    """Remove a commit from the dataset by its commit ID.

    Args:
        commit_id: The 8-character commit ID to remove.
    """
    init_error = _require_init("undo_commit")
    if init_error:
        return init_error

    if commit_id not in dataset_commits:
        return f"Commit ID {commit_id} not found."

    removed_commit = dataset_commits.pop(commit_id)
    with open(DATASET_JSON, "w") as f:
        json.dump(dataset_commits, f, indent=2)

    _log(f"[LOG] Removed commit {commit_id}: {removed_commit['message']}")
    return f"Removed commit {commit_id}: {removed_commit['message']} with {removed_commit['size']} images."


@mcp.tool()
def status() -> str:
    """Show all commit history including commit IDs and image counts."""
    init_error = _require_init("status")
    if init_error:
        return init_error

    if len(dataset_commits) == 0:
        return "No commits yet."

    total_images = sum(c["size"] for c in dataset_commits.values())
    result = f"Total commits: {len(dataset_commits)}, Total images: {total_images}\n\nCommit History:\n"
    for cid, info in dataset_commits.items():
        result += f"- [{cid}] {info['message']} ({info['size']} images)\n"
    return result


@mcp.tool()
def sample_from_committed(commit_id: str, n: int = 20) -> list:
    """Sample n random images from a committed batch to review.

    Args:
        commit_id: The 8-character commit ID to sample from.
        n: Number of images to sample.
    """
    init_error = _require_init("sample_from_committed")
    if init_error:
        return init_error

    if commit_id not in dataset_commits:
        return [f"Commit ID {commit_id} not found."]

    commit_info = dataset_commits[commit_id]
    images = commit_info["images"]
    if len(images) == 0:
        return ["No images in this commit."]

    sample_size = min(n, len(images))
    sampled_paths = random.sample(images, sample_size)
    _log(f"[LOG] Sampled {sample_size} images from commit {commit_id}")

    whole_image = grid_stack(sampled_paths, row_size=5)
    return [
        _pil_to_mcp_image(whole_image),
        f"Sampled {sample_size} images from commit {commit_id}.",
    ]


@mcp.tool()
def test_image() -> list:
    """Return the test image (123.jpg) for verifying image transport works.

    Returns the image so the model can describe what it sees.
    """
    init_error = _require_init("test_image")
    if init_error:
        return init_error

    image_path = os.path.join(os.path.dirname(__file__), "123.jpg")
    img = PILImage.open(image_path)
    return [
        _pil_to_mcp_image(img),
        "This is the test image. Please describe what you see.",
    ]


@mcp.tool()
def log_actions(msg: str = "") -> str:
    """Log the agent's thoughts, reasoning for the next step, and brief summary after each function call.

    Args:
        msg: The message to log.
    """
    init_error = _require_init("log_actions")
    if init_error:
        return init_error

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {msg}\n"
    with open(LOG_FILE, "a") as f:
        f.write(entry)
    _log(f"[LOG] {msg}")
    return "Logged."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with contextlib.redirect_stdout(sys.stderr):
        weave.init(WEAVE_PROJECT)
    _log(f"[INIT] weave enabled for project '{WEAVE_PROJECT}'.")

    # Load existing dataset commits
    if os.path.exists(DATASET_JSON):
        try:
            with open(DATASET_JSON, "r") as f:
                dataset_commits.update(json.load(f))
        except json.JSONDecodeError:
            dataset_commits.clear()

    mcp.run(transport="streamable-http")
