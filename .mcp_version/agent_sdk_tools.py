"""MCP tools for dataset curation, used by the Agent SDK agent."""

import os
import sys
import json
import uuid
import time
import random
import datetime
import contextlib
import tempfile

import torch
import numpy as np
from PIL import Image as PILImage

from claude_agent_sdk import tool

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../openai_sdk_tools"))

from image_utils import grid_stack

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)
dataset_commits: dict = {}
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "agent_log.txt")
DATASET_JSON = os.path.join(os.path.dirname(__file__), "..", "dataset.json")
DATASET_ROOT = os.getenv("DATASET_ROOT", "/home/wg25r/Downloads/ds/train")
_IS_INITIALIZED = False
_INIT_REQUIRED_MSG = "Server resources are not initialized. You need to call `init` first."

# Lazy-loaded resources
model = None
ava_embeddings_tensor = None
ls_embeddings_tensor = None
lapis_embeddings_tensor = None
ava_names_list = None
ls_names_list = None
lapis_names_list = None
dataset_map = {"photos": "ava", "dreamcore": "ls", "artwork": "lapis"}
_loader_summary: dict = {}
_img_counter = 0


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _require_init() -> str | None:
    if _IS_INITIALIZED:
        return None
    return _INIT_REQUIRED_MSG


def _save_grid_to_tmp(pil_image: PILImage.Image) -> str:
    """Save a PIL image to a temp file and return the path."""
    global _img_counter
    _img_counter += 1
    path = os.path.join(TMP_DIR, f"grid_{_img_counter}.jpg")
    pil_image.save(path, format="JPEG", quality=90)
    return path


def _get_embeddings_and_names(dataset: str):
    if dataset == "photos":
        return ava_embeddings_tensor, ava_names_list
    elif dataset == "dreamcore":
        return ls_embeddings_tensor, ls_names_list
    else:
        return lapis_embeddings_tensor, lapis_names_list


def _apply_negative_filter(embeddings, names, negative_prompts, negative_threshold):
    if not negative_prompts:
        return set()
    combined_mask = torch.zeros(len(embeddings), dtype=torch.bool)
    for neg in negative_prompts:
        q_emb = model.process([{"text": neg}]).cpu().float()
        sim = torch.nn.functional.cosine_similarity(embeddings, q_emb)
        combined_mask |= sim > negative_threshold
    target_indices = torch.where(combined_mask)[0].tolist()
    return {names[i].item() for i in target_indices}


def _search_impl(query, dataset, negative_prompts, negative_threshold, t, return_paths=False):
    _log(f"[LOG] Searching for '{query}' in dataset '{dataset}' ...")
    embeddings, names = _get_embeddings_and_names(dataset)
    excluded = _apply_negative_filter(embeddings, names, negative_prompts, negative_threshold)

    query_embedding = model.process([{"text": query}]).cpu()
    res = torch.nn.functional.cosine_similarity(embeddings, query_embedding.float())

    excluded_indices = {i for i, n in enumerate(names) if n.item() in excluded}
    valid_mask = torch.ones(len(res), dtype=torch.bool)
    for idx in excluded_indices:
        valid_mask[idx] = False
    valid_scores = res[valid_mask].numpy()
    hist = np.histogram(valid_scores, bins=10)
    sim_distribution = f"Similarity distribution: counts={hist[0].tolist()}, bins=[{', '.join(f'{b:.3f}' for b in hist[1].tolist())}]"

    selected_images, top_scores = [], []
    for idx in torch.argsort(res, descending=True):
        if names[idx].item() not in excluded:
            selected_images.append(names[idx].item())
            top_scores.append(f"{res[idx].item():.4f}")
        if len(selected_images) >= t:
            break

    paths = []
    for name in selected_images:
        path = f"{DATASET_ROOT}/{dataset_map[dataset]}/{name}"
        if os.path.exists(path):
            paths.append(path)

    score_info = f"Top-{len(top_scores)} scores: [{', '.join(top_scores)}]\n{sim_distribution}"

    if return_paths:
        return paths, score_info
    if not paths:
        return None, score_info
    return grid_stack(paths, row_size=5), score_info


def _sample_impl(query, dataset, min_threshold, max_threshold, negative_prompts, negative_threshold):
    embeddings, names = _get_embeddings_and_names(dataset)
    excluded = _apply_negative_filter(embeddings, names, negative_prompts, negative_threshold)

    query_embedding = model.process([{"text": query}]).cpu()
    res = torch.nn.functional.cosine_similarity(embeddings, query_embedding.float())

    mask = torch.logical_and(res >= min_threshold, res <= max_threshold)
    candidate_indices = torch.where(mask)[0].tolist()
    selected = [names[i].item() for i in candidate_indices if names[i].item() not in excluded]

    paths = []
    for name in selected:
        path = f"{DATASET_ROOT}/{dataset_map[dataset]}/{name}"
        if os.path.exists(path):
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool("init", "Initialize embeddings and models. Call once before using any other tool.", {})
async def tool_init(args):
    global _IS_INITIALIZED, model
    global ava_embeddings_tensor, ls_embeddings_tensor, lapis_embeddings_tensor
    global ava_names_list, ls_names_list, lapis_names_list
    global dataset_map, _loader_summary

    if _IS_INITIALIZED:
        return {"content": [{"type": "text", "text": "Already initialized."}]}

    start = time.time()
    with contextlib.redirect_stdout(sys.stderr):
        from dataset_loader import (
            model as m, ava_embeddings_tensor as a_e, ls_embeddings_tensor as l_e,
            lapis_embeddings_tensor as la_e, ava_names_list as a_n, ls_names_list as l_n,
            lapis_names_list as la_n, dataset_map as dm, dataset_loader_summary,
        )
    model = m
    ava_embeddings_tensor, ls_embeddings_tensor, lapis_embeddings_tensor = a_e, l_e, la_e
    ava_names_list, ls_names_list, lapis_names_list = a_n, l_n, la_n
    dataset_map = dm
    _loader_summary = dataset_loader_summary()
    _IS_INITIALIZED = True
    elapsed = round(time.time() - start, 2)
    msg = f"Initialization complete in {elapsed}s. rows={_loader_summary.get('total_rows', 'n/a')}, embedding_dim={_loader_summary.get('embedding_dim', 'n/a')}."
    return {"content": [{"type": "text", "text": msg}]}


@tool("search", "Search for top-k images matching a query.", {
    "query": {"type": "string", "description": "Text query for semantic image search."},
    "dataset": {"type": "string", "description": 'One of "photos", "dreamcore", or "artwork".'},
    "negative_prompts": {"type": "array", "items": {"type": "string"}, "description": "Negative text prompts to filter out (3-5 max)."},
    "negative_threshold": {"type": "number", "description": "Cosine similarity threshold for negative filtering. Default 0.3."},
    "t": {"type": "integer", "description": "Number of top results. Default 10."},
})
async def tool_search(args):
    err = _require_init()
    if err:
        return {"content": [{"type": "text", "text": err}]}

    query = args["query"]
    dataset = args["dataset"]
    negative_prompts = args.get("negative_prompts", [])
    negative_threshold = args.get("negative_threshold", 0.3)
    t = args.get("t", 10)

    result, score_info = _search_impl(query, dataset, negative_prompts, negative_threshold, t)
    if result is None:
        return {"content": [{"type": "text", "text": f"No Image Found\n{score_info}"}]}

    img_path = _save_grid_to_tmp(result)
    return {"content": [
        {"type": "text", "text": f"Grid image saved at: {img_path}\nUse the Read tool to view it.\nShowing top {t} results for '{query}' in {dataset}.\n{score_info}"},
    ]}


@tool("sample", "Sample random images within a similarity score range.", {
    "query": {"type": "string", "description": "Text query."},
    "dataset": {"type": "string", "description": 'One of "photos", "dreamcore", or "artwork".'},
    "min_threshold": {"type": "number", "description": "Minimum cosine similarity."},
    "max_threshold": {"type": "number", "description": "Maximum cosine similarity."},
    "count": {"type": "integer", "description": "Number of images to sample. Default 5."},
    "negative_prompts": {"type": "array", "items": {"type": "string"}, "description": "Negative text prompts."},
    "negative_threshold": {"type": "number", "description": "Threshold for negative filtering. Default 0.2."},
})
async def tool_sample(args):
    err = _require_init()
    if err:
        return {"content": [{"type": "text", "text": err}]}

    query = args["query"]
    dataset = args["dataset"]
    min_t = args["min_threshold"]
    max_t = args["max_threshold"]
    count = args.get("count", 5)
    negative_prompts = args.get("negative_prompts", [])
    negative_threshold = args.get("negative_threshold", 0.2)

    paths = _sample_impl(query, dataset, min_t, max_t, negative_prompts, negative_threshold)
    if not paths:
        return {"content": [{"type": "text", "text": "No Image Found"}]}

    sampled = random.sample(paths, min(count, len(paths)))
    grid = grid_stack(sampled, row_size=5)
    img_path = _save_grid_to_tmp(grid)
    return {"content": [
        {"type": "text", "text": f"Grid image saved at: {img_path}\nUse the Read tool to view it.\nSampled {len(sampled)} from {len(paths)} candidates."},
    ]}


@tool("commit", "Commit images with similarity >= threshold to the dataset.", {
    "query": {"type": "string", "description": "Text query used for the search."},
    "dataset": {"type": "string", "description": 'One of "photos", "dreamcore", or "artwork".'},
    "threshold": {"type": "number", "description": "Minimum cosine similarity threshold (0.0-1.0)."},
    "negative_prompts": {"type": "array", "items": {"type": "string"}, "description": "Negative text prompts."},
    "negative_threshold": {"type": "number", "description": "Threshold for negative filtering. Default 0.2."},
    "message": {"type": "string", "description": "Descriptive tags for this commit."},
})
async def tool_commit(args):
    err = _require_init()
    if err:
        return {"content": [{"type": "text", "text": err}]}

    query = args["query"]
    dataset = args["dataset"]
    threshold = args["threshold"]
    negative_prompts = args.get("negative_prompts", [])
    negative_threshold = args.get("negative_threshold", 0.2)
    message = args.get("message", "")

    embeddings, names = _get_embeddings_and_names(dataset)
    excluded = _apply_negative_filter(embeddings, names, negative_prompts, negative_threshold)

    query_embedding = model.process([{"text": query}]).cpu()
    res = torch.nn.functional.cosine_similarity(embeddings, query_embedding.float())

    mask = res >= threshold
    candidate_indices = torch.where(mask)[0].tolist()
    selected = [names[i].item() for i in candidate_indices if names[i].item() not in excluded]

    images = []
    for name in selected:
        path = f"{DATASET_ROOT}/{dataset_map[dataset]}/{name}"
        if os.path.exists(path):
            images.append(path)

    commit_id = str(uuid.uuid4())[:8]
    dataset_commits[commit_id] = {
        "query": query, "dataset": dataset, "threshold": threshold,
        "negative_prompts": negative_prompts, "negative_threshold": negative_threshold,
        "message": message, "images": images, "size": len(images),
    }
    with open(DATASET_JSON, "w") as f:
        json.dump(dataset_commits, f, indent=2)

    return {"content": [{"type": "text", "text": f"Committed with ID: {commit_id}, message: {message} with {len(images)} images."}]}


@tool("undo_commit", "Remove a commit from the dataset.", {
    "commit_id": {"type": "string", "description": "The 8-character commit ID to remove."},
})
async def tool_undo_commit(args):
    err = _require_init()
    if err:
        return {"content": [{"type": "text", "text": err}]}

    commit_id = args["commit_id"]
    if commit_id not in dataset_commits:
        return {"content": [{"type": "text", "text": f"Commit ID {commit_id} not found."}]}

    removed = dataset_commits.pop(commit_id)
    with open(DATASET_JSON, "w") as f:
        json.dump(dataset_commits, f, indent=2)
    return {"content": [{"type": "text", "text": f"Removed commit {commit_id}: {removed['message']} with {removed['size']} images."}]}


@tool("status", "Show all commit history.", {})
async def tool_status(args):
    err = _require_init()
    if err:
        return {"content": [{"type": "text", "text": err}]}

    if not dataset_commits:
        return {"content": [{"type": "text", "text": "No commits yet."}]}

    total = sum(c["size"] for c in dataset_commits.values())
    lines = [f"Total commits: {len(dataset_commits)}, Total images: {total}\n\nCommit History:"]
    for cid, info in dataset_commits.items():
        lines.append(f"- [{cid}] {info['message']} ({info['size']} images)")
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool("sample_from_committed", "Sample images from a committed batch.", {
    "commit_id": {"type": "string", "description": "The 8-character commit ID."},
    "n": {"type": "integer", "description": "Number of images to sample. Default 20."},
})
async def tool_sample_from_committed(args):
    err = _require_init()
    if err:
        return {"content": [{"type": "text", "text": err}]}

    commit_id = args["commit_id"]
    n = args.get("n", 20)
    if commit_id not in dataset_commits:
        return {"content": [{"type": "text", "text": f"Commit ID {commit_id} not found."}]}

    images = [p for p in dataset_commits[commit_id]["images"] if os.path.exists(p)]
    if not images:
        return {"content": [{"type": "text", "text": "No images found on disk for this commit."}]}

    sampled = random.sample(images, min(n, len(images)))
    grid = grid_stack(sampled, row_size=5)
    img_path = _save_grid_to_tmp(grid)
    return {"content": [
        {"type": "text", "text": f"Grid image saved at: {img_path}\nUse the Read tool to view it.\nSampled {len(sampled)} images from commit {commit_id}."},
    ]}


@tool("log_actions", "Log the agent's thoughts and reasoning.", {
    "msg": {"type": "string", "description": "The message to log."},
})
async def tool_log_actions(args):
    msg = args.get("msg", "")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    _log(f"[LOG] {msg}")
    return {"content": [{"type": "text", "text": "Logged."}]}


# All tools list for the server
ALL_TOOLS = [
    tool_init, tool_search, tool_sample, tool_commit,
    tool_undo_commit, tool_status, tool_sample_from_committed, tool_log_actions,
]


def load_existing_commits():
    """Load existing dataset commits from disk."""
    if os.path.exists(DATASET_JSON):
        try:
            with open(DATASET_JSON, "r") as f:
                dataset_commits.update(json.load(f))
        except json.JSONDecodeError:
            dataset_commits.clear()
