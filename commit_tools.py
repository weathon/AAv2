import json
import uuid
import random
import os
import datetime
import torch
from agents import function_tool

from image_utils import grid_stack, encode
from dataset_loader import (
    model,
    ava_embeddings_tensor, ls_embeddings_tensor, lapis_embeddings_tensor,
    ava_names_list, ls_names_list, lapis_names_list,
    dataset_map
)

# Changed to dict structure with commit_id as keys
dataset_commits = {}

LOG_FILE = "agent_log.txt"

@function_tool(failure_error_function=None)
def log_actions(msg: str = ""):
    """Log the agent's thoughts, reasoning for the next step, and brief summary after each function call."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {msg}\n"
    with open(LOG_FILE, "a") as f:
        f.write(entry)
    print(f"[LOG] {msg}")
    return "Logged."

@function_tool(failure_error_function=None)
def commit(query: str, dataset: str, threshold: float, negative_prompts: list[str] = [], negative_threshold: float = 0.2, message: str = None):
    print(f"[LOG] Committing with message: {message}")
    # images = _search(query, dataset, negative_prompts, threshold=threshold, return_paths=True)
    print(f"[LOG] Sampling for '{query}' in dataset '{dataset}' with negative prompt '{negative_prompts}' and threshold {negative_threshold}")

    query_texts = [
        {"text": text} for text in negative_prompts
    ]

    embeddings = ava_embeddings_tensor if dataset == "photos" else ls_embeddings_tensor if dataset == "dreamcore" else lapis_embeddings_tensor
    names = ava_names_list if dataset == "photos" else ls_names_list if dataset == "dreamcore" else lapis_names_list

    combined_mask = torch.zeros(len(embeddings), dtype=torch.bool)

    for i, q in enumerate(query_texts):
        q_emb = model.process([q]).cpu().float()

        sim = torch.nn.functional.cosine_similarity(embeddings, q_emb)

        combined_mask |= (sim > negative_threshold)

        # print(f"[PROFILE] negative prompt {i}/{len(query_texts)}: process={t4p:.3f}s, sim={t4sim:.3f}s, mask={t4mask:.3f}s")

    target_indices = torch.where(combined_mask)[0].tolist()
    empty_images = {names[i].item() for i in target_indices}

    queries = [{"text": query}]

    query_embedding = model.process(queries).cpu()

    res = torch.nn.functional.cosine_similarity(embeddings, query_embedding.float())

    selected_images = []
    mask = res >= threshold
    candidate_indices = torch.where(mask)[0].tolist()
    selected_images = [names[i].item() for i in candidate_indices]

    print(f"[LOG] Sample results: {selected_images}.")

    images = [f"/home/wg25r/Downloads/ds/train/{dataset_map[dataset]}/{name}" for name in selected_images if os.path.exists(f"/home/wg25r/Downloads/ds/train/{dataset_map[dataset]}/{name}")]

    # Generate unique commit ID
    commit_id = str(uuid.uuid4())[:8]

    # Store commit with ID
    dataset_commits[commit_id] = {
        "query": query,
        "dataset": dataset,
        "threshold": threshold,
        "negative_prompts": negative_prompts,
        "negative_threshold": negative_threshold,
        "message": message,
        "images": images,
        "size": len(images)
    }

    # Save to dataset.json
    with open("dataset.json", "w") as f:
        json.dump(dataset_commits, f, indent=2)

    return f"Committed with ID: {commit_id}, message: {message} with {len(images)} images."

@function_tool(failure_error_function=None)
def undo_commit(commit_id: str):
    """Remove a commit from the dataset by its commit ID."""
    if commit_id not in dataset_commits:
        return f"Commit ID {commit_id} not found."

    removed_commit = dataset_commits.pop(commit_id)

    # Save updated dataset.json
    with open("dataset.json", "w") as f:
        json.dump(dataset_commits, f, indent=2)

    print(f"[LOG] Removed commit {commit_id}: {removed_commit['message']}")
    return f"Removed commit {commit_id}: {removed_commit['message']} with {removed_commit['size']} images."


@function_tool(failure_error_function=None)
def status():
    """Show all commit history including commit IDs and image counts."""
    if len(dataset_commits) == 0:
        return "No commits yet."

    total_images = sum(commit['size'] for commit in dataset_commits.values())
    result = f"Total commits: {len(dataset_commits)}, Total images: {total_images}\n\nCommit History:\n"

    for commit_id, commit_info in dataset_commits.items():
        result += f"- [{commit_id}] {commit_info['message']} ({commit_info['size']} images)\n"

    return result


@function_tool(failure_error_function=None)
def sample_from_committed(commit_id: str, n: int = 20):
    """Sample n random images from a committed batch to review."""
    if commit_id not in dataset_commits:
        return f"Commit ID {commit_id} not found."

    commit_info = dataset_commits[commit_id]
    images = commit_info['images']

    if len(images) == 0:
        return "No images in this commit."

    # Sample random images (up to n)
    sample_size = min(n, len(images))
    sampled_paths = random.sample(images, sample_size)

    print(f"[LOG] Sampled {sample_size} images from commit {commit_id}")

    # Create grid of sampled images
    whole_image = grid_stack(sampled_paths, row_size=5)
    result = encode(whole_image)

    return result
