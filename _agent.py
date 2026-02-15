# %%
from PIL import Image
import numpy as np

def vstack(images):
    if len(images) == 0:
        raise ValueError("Need 0 or more images")

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    width = max([img.size[0] for img in images])
    height = sum([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    y_pos = 0
    for img in images:
        stacked.paste(img, (0, y_pos))
        y_pos += img.size[1]
    return stacked


def hstack(images):
    if len(images) == 0:
        raise ValueError("Need 0 or more images")

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    width = sum([img.size[0] for img in images])
    height = max([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    x_pos = 0
    for img in images:
        stacked.paste(img, (x_pos, 0))
        x_pos += img.size[0]
    return stacked

def grid_stack(image_paths, row_size):
    target_width = 2048
    rows = []
    for i in range(0, len(image_paths), row_size):
        imgs = [Image.open(p) for p in image_paths[i:i + row_size]]
        aspect_ratios = [img.size[0] / img.size[1] for img in imgs]
        target_height = int(round(target_width / sum(aspect_ratios)))
        
        resized_imgs = []
        current_width = 0
        for j, img in enumerate(imgs):
            if j == len(imgs) - 1:
                new_w = target_width - current_width
            else:
                new_w = int(round(target_height * aspect_ratios[j]))
                current_width += new_w
            resized_imgs.append(img.resize((new_w, target_height), Image.BILINEAR))
            
        rows.append(hstack(resized_imgs))

    return vstack(rows)

# %%
import dotenv   
dotenv.load_dotenv()

# %%
import datasets
ds = datasets.load_dataset("weathon/ava_embeddings", split="train")

# %%
ds = ds.with_format("numpy")

# %%
import numpy as np

names = ds["name"]
sources = ds["source"]
arr = ds.data.column("embeddings").to_numpy()
arr = np.stack(arr, axis=0)
ava_dataset = ds.filter(lambda example: example["source"] == "ava")
ls_dataset = ds.filter(lambda example: example["source"] == "liminal_space")
lapis_dataset = ds.filter(lambda example: example["source"] == "lapis")
ava_embeddings = np.stack(ava_dataset["embeddings"], axis=0)
ls_embeddings = np.stack(ls_dataset["embeddings"], axis=0)
lapis_embeddings = np.stack(lapis_dataset["embeddings"], axis=0) 

# %%
ava_names = ava_dataset["name"]
ls_names = ls_dataset["name"]
lapis_names = lapis_dataset["name"]

# %%
import sys
sys.path.append("../")
from qwen3_vl_embedding import Qwen3VLEmbedder

model_name_or_path = "Qwen/Qwen3-VL-Embedding-8B"

model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path, device="cpu", attn_implementation="sdpa")

# %%
import torch
torch.tensor(arr).cpu().float()
torch.zeros(len(torch.tensor(arr).cpu().float()), dtype=torch.bool)

# %%
import base64
from PIL import Image
from agents.tool import ToolOutputImage

def encode(image: Image) -> ToolOutputImage:
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return ToolOutputImage(image_url=f"data:image/png;base64,{b64_str}", detail="high")

# %%
ava_embeddings_tensor = torch.tensor(ava_embeddings).float()
ls_embeddings_tensor = torch.tensor(ls_embeddings).float()
lapis_embeddings_tensor = torch.tensor(lapis_embeddings).float()

ava_names_list = list(ava_names)
ls_names_list = list(ls_names)
lapis_names_list = list(lapis_names)

# %%
from PIL import Image
import os

from agents import Agent, Runner, function_tool, set_default_openai_client, set_tracing_export_api_key
from openai import AsyncOpenAI
import torch
import time
from search_tools import rate_images

dataset_map = {
    "photos": "ava",
    "dreamcore": "liminal_space",
    "artwork": "lapis"
}

def _search(query: str, dataset: str, negative_prompts: list[str] = [], negative_threshold: float = 0.3, t: int = 10, return_paths: bool = False) -> ToolOutputImage:
    t0 = time.time()
    print(f"[LOG] Searching for '{query}' in dataset '{dataset}' with negative prompt '{negative_prompts}' and threshold {negative_threshold} for {t} items...")

    t1 = time.time()
    query_texts = [
        {"text": text} for text in negative_prompts
    ]
    # print(f"[PROFILE] build query_texts: {time.time()-t1:.3f}s")

    t2 = time.time()
    embeddings = ava_embeddings_tensor if dataset == "photos" else ls_embeddings_tensor if dataset == "dreamcore" else lapis_embeddings_tensor
    names = ava_names_list if dataset == "photos" else ls_names_list if dataset == "dreamcore" else lapis_names_list
    # print(f"[PROFILE] load embeddings & names: {time.time()-t2:.3f}s, shapes: embeddings={embeddings.shape}, names={len(names)}")

    t3 = time.time()
    combined_mask = torch.zeros(len(embeddings), dtype=torch.bool)
    # print(f"[PROFILE] init combined_mask: {time.time()-t3:.3f}s")

    t4 = time.time()
    for i, q in enumerate(query_texts):
        t4i = time.time()
        q_emb = model.process([q]).cpu().float()
        t4p = time.time() - t4i 
        
        t4s = time.time()
        sim = torch.nn.functional.cosine_similarity(embeddings, q_emb)
        t4sim = time.time() - t4s
        
        t4m = time.time()
        combined_mask |= (sim > negative_threshold)
        t4mask = time.time() - t4m
        
        # print(f"[PROFILE] negative prompt {i}/{len(query_texts)}: process={t4p:.3f}s, sim={t4sim:.3f}s, mask={t4mask:.3f}s")
    # print(f"[PROFILE] total negative loop: {time.time()-t4:.3f}s")

    t5 = time.time()
    target_indices = torch.where(combined_mask)[0].tolist()
    empty_images = {names[i].item() for i in target_indices}
    # print(f"[PROFILE] build empty_images: {time.time()-t5:.3f}s, count={len(empty_images)}")

    t6 = time.time()
    queries = [{"text": query}] 
    # print(f"[PROFILE] prep query: {time.time()-t6:.3f}s")

    t7 = time.time()
    query_embedding = model.process(queries).cpu()
    # print(f"[PROFILE] model.process query: {time.time()-t7:.3f}s, shape={query_embedding.shape}")

    t8 = time.time()
    res = torch.nn.functional.cosine_similarity(embeddings, query_embedding.float())
    # print(f"[PROFILE] cosine_similarity: {time.time()-t8:.3f}s")

    t9 = time.time()
    selected_images = []
    for idx in torch.argsort(res, descending=True):
        if names[idx].item() not in empty_images:
            selected_images.append(names[idx].item())
        if len(selected_images) >= t:
            break
    # print(f"[PROFILE] top-k loop: {time.time()-t9:.3f}s, iterations={len(res)}")

    print(f"[LOG] Search results: {selected_images}.")

    t10 = time.time()
    paths = [f"/home/wg25r/Downloads/ds/train/{dataset_map[dataset]}/{name}" for name in selected_images if os.path.exists(f"/home/wg25r/Downloads/ds/train/{dataset_map[dataset]}/{name}")]
    # print(f"[PROFILE] filter existing paths: {time.time()-t10:.3f}s, found={len(paths)}")
    if len(paths) == 0:
        return "No Image Found"
    t11 = time.time()
    whole_image = grid_stack(paths, row_size=5)
    # print(f"[PROFILE] grid_stack: {time.time()-t11:.3f}s")
    if return_paths:
        return paths
    t12 = time.time()
    result = encode(whole_image)
    # print(f"[PROFILE] encode: {time.time()-t12:.3f}s")

    # print(f"[PROFILE] TOTAL: {time.time()-t0:.3f}s")
    return result

@function_tool(failure_error_function=None)
def search(query: str, dataset: str, negative_prompts: list[str] = [], negative_threshold: float = 0.3, t: int = 10) -> ToolOutputImage:
    try:
        return _search(query, dataset, negative_prompts, negative_threshold, t)
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        raise e

# %%
# (query, dataset, min_threshold, max_threshold, count=5, negative_prompt=None, negative_threshold=0.2)
import random

def _sample(query: str, dataset: str, min_threshold: float, max_threshold: float, negative_prompts: list[str] = [], negative_threshold: float = 0.2):
    """Generic internal function to find images matching query and thresholds.

    Returns:
        list[str]: List of file paths to matching images
    """
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

    target_indices = torch.where(combined_mask)[0].tolist()
    empty_images = {names[i].item() for i in target_indices}

    queries = [{"text": query}]

    query_embedding = model.process(queries).cpu()

    res = torch.nn.functional.cosine_similarity(embeddings, query_embedding.float())

    # Find images within the threshold range
    mask = torch.logical_and(res >= min_threshold, res <= max_threshold)
    candidate_indices = torch.where(mask)[0].tolist()

    # Filter out negative prompt matches
    selected_images = [names[i].item() for i in candidate_indices if names[i].item() not in empty_images]

    # Convert to full paths and filter existing files
    paths = [f"/home/wg25r/Downloads/ds/train/{dataset_map[dataset]}/{name}" for name in selected_images if os.path.exists(f"/home/wg25r/Downloads/ds/train/{dataset_map[dataset]}/{name}")]

    return paths

@function_tool(failure_error_function=None)
def sample(query: str, dataset: str, min_threshold: float, max_threshold: float, count: int = 5, negative_prompts: list[str] = [], negative_threshold: float = 0.2):
    """Sample random images matching the query and display them as a grid."""
    print(f"[LOG] Sampling for '{query}' in dataset '{dataset}' with negative prompt '{negative_prompts}' and negative threshold {negative_threshold} between {min_threshold} and {max_threshold} for {count} items...")

    paths = _sample(query, dataset, min_threshold, max_threshold, negative_prompts, negative_threshold)

    if len(paths) == 0:
        return "No Image Found"

    # Sample random images from the results
    sampled_paths = random.sample(paths, min(count, len(paths)))

    print(f"[LOG] Sampled {len(sampled_paths)} images from {len(paths)} candidates.")

    whole_image = grid_stack(sampled_paths, row_size=5)
    result = encode(whole_image)

    return result

@function_tool(failure_error_function=None)
def aesthetics_rate(query: str, dataset: str, min_threshold: float, max_threshold: float, negative_prompts: list[str] = [], negative_threshold: float = 0.2, sample_size: int = 100):
    """Rate the aesthetics scores of images matching the query.

    Returns a string describing the distribution of aesthetics scores.
    """
    print(f"[LOG] Rating aesthetics for '{query}' in dataset '{dataset}' between {min_threshold} and {max_threshold}...")

    paths = _sample(query, dataset, min_threshold, max_threshold, negative_prompts, negative_threshold)

    if len(paths) == 0:
        return "No images found matching the criteria."

    # Sample if there are too many images
    if len(paths) > sample_size:
        paths_to_rate = random.sample(paths, sample_size)
        print(f"[LOG] Sampled {sample_size} images from {len(paths)} total candidates for rating.")
    else:
        paths_to_rate = paths
        print(f"[LOG] Rating all {len(paths)} matching images.")

    scores = rate_images(paths_to_rate)

    return f"Aesthetics scores for {len(paths_to_rate)} images: {scores}"


# %%
    

# commit(query, dataset, threshold, negative_prompt=None, negative_threshold=0.2, message=None)
import json
import uuid

# Changed to dict structure with commit_id as keys
dataset_commits = {}

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

# %%
out = _search("黑色纹理 表面 裂纹 焦黑 荒废感", "photos", negative_prompts=[
    "watermark",
    "empty image"
  ], negative_threshold=0.3, t=12) 

# %%
# convert base64 to image and display
import base64
from PIL import Image
from io import BytesIO
b64_str = out.image_url.split(",")[1]
img_data = b64_str.encode("utf-8")
img = Image.open(BytesIO(base64.b64decode(img_data)))
print(img.size)
img

# %%
# import weave
# weave.init("openai-agents")

# %%
import nest_asyncio
from agents import ModelSettings
from openai.types.shared import Reasoning
from openai import AsyncOpenAI
from agents import set_default_openai_client
from agents import Agent, Runner, function_tool, set_default_openai_client, set_tracing_export_api_key

import os
import dotenv
dotenv.load_dotenv()
# custom_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
# custom_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# set_default_openai_client(custom_client)
# set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))


with open("system_prompt.md", "r") as f:
    system_prompt = f.read()        


agent = Agent(name="Assistant",
              tools=[search, commit, sample, aesthetics_rate, undo_commit, status, sample_from_committed],
              instructions=system_prompt,
              model_settings=ModelSettings(
                reasoning=Reasoning(effort="medium"),
              ),
              model="gpt-5.2")
            #   model="google/gemini-3-flash-preview")

# Initialize dataset.json if it doesn't exist, or load existing commits
if os.path.exists("dataset.json"):
    with open("dataset.json", "r") as f:
        try:
            dataset_commits.update(json.load(f))
        except json.JSONDecodeError:
            dataset_commits.clear()
else:
    with open("dataset.json", "w") as f:
        json.dump({}, f)
 
result = Runner.run_sync(agent, "Psychedelic art", max_turns=200) 

