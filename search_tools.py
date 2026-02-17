import os
import time
import random
import torch
import wandb
from agents import function_tool
from agents.tool import ToolOutputImage

from image_utils import grid_stack, encode
from dataset_loader import (
    model,
    ava_embeddings_tensor, ls_embeddings_tensor, lapis_embeddings_tensor,
    ava_names_list, ls_names_list, lapis_names_list,
    dataset_map
)

DATASET_ROOT = os.getenv("DATASET_ROOT", "/home/wg25r/Downloads/ds/train")


from openai import OpenAI
import dotenv
dotenv.load_dotenv()

captioning_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)
from PIL import Image
import base64
from io import BytesIO

from hpsv3 import HPSv3RewardInferencer
inferencer = HPSv3RewardInferencer(device='cuda:1')
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def _caption_single_image(path, max_retries=4):
    """Caption a single image via OpenRouter API with retry logic."""
    image = Image.open(path)
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
                                "text": "Caption this image based on physical facts in the image, ignore aesthetics and styles. Only describe what you see in the image, do not add any interpretation or imagination. Be concise and objective. The caption should be a single short sentence describe the main content of the image. Do not mention the style or aesthetics of the image. Focus on physical facts like objects, colors, and their relationships. Do not add any information that cannot be directly observed from the image."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_str}"
                                }
                            }
                        ]
                    }
                ]
            )
            caption = completion.choices[0].message.content
            if caption is None or (isinstance(caption, str) and not caption.strip()):
                raise RuntimeError("Empty caption content from LLM")
            print(f"[LOG] Generated caption for {path}: {caption}")
            return caption
        except Exception as e:
            print(f"[WARN] Captioning attempt {attempt}/{max_retries} failed for {path}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"[ERROR] All {max_retries} captioning attempts failed for {path}, using fallback.")
                return "An image."

def rate_images(image_paths):
    # Caption images in parallel with 10 workers and 4 retries each
    captions = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {
            executor.submit(_caption_single_image, path, max_retries=4): idx
            for idx, path in enumerate(image_paths)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            captions[idx] = future.result()

    # Feed to HPSv3 in batches of 4
    scores = []
    for i in range(0, len(image_paths), 4):
        batch_prompts = captions[i:i+4]
        batch_paths = image_paths[i:i+4]
        with torch.no_grad():
            rewards = inferencer.reward(prompts=batch_prompts, image_paths=batch_paths)
        scores.extend([reward[0].item() for reward in rewards])
    hist = np.histogram(scores, bins=10)
    hist_str = f"Score histogram: {hist[0].tolist()}, bins: {hist[1].tolist()}"
    raw_scores = [f"{score:.4f}" for score in scores]
    return hist_str + "\nRaw Scores: " + str(raw_scores)


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

    # Compute similarity distribution histogram (excluding negatively-filtered images)
    valid_mask = torch.ones(len(res), dtype=torch.bool)
    for i in range(len(names)):
        if names[i].item() in empty_images:
            valid_mask[i] = False
    valid_scores = res[valid_mask].numpy()
    hist = np.histogram(valid_scores, bins=10)
    sim_distribution = f"Similarity distribution: counts={hist[0].tolist()}, bins=[{', '.join(f'{b:.3f}' for b in hist[1].tolist())}]"

    t9 = time.time()
    selected_images = []
    top_scores = []
    for idx in torch.argsort(res, descending=True):
        if names[idx].item() not in empty_images:
            selected_images.append(names[idx].item())
            top_scores.append(f"{res[idx].item():.4f}")
        if len(selected_images) >= t:
            break
    # print(f"[PROFILE] top-k loop: {time.time()-t9:.3f}s, iterations={len(res)}")

    score_info = f"Top-{len(top_scores)} scores: [{', '.join(top_scores)}]\n{sim_distribution}"
    print(f"[LOG] Search results: {selected_images}.")
    print(f"[LOG] {score_info}")

    t10 = time.time()
    paths = [f"{DATASET_ROOT}/{dataset_map[dataset]}/{name}" for name in selected_images]
    # print(f"[PROFILE] filter existing paths: {time.time()-t10:.3f}s, found={len(paths)}")
    if len(paths) == 0:
        return f"No Image Found\n{score_info}"
    t11 = time.time()
    whole_image = grid_stack(paths, row_size=5)
    # print(f"[PROFILE] grid_stack: {time.time()-t11:.3f}s")
    if return_paths:
        return paths
    t12 = time.time()
    result = encode(whole_image)
    # print(f"[PROFILE] encode: {time.time()-t12:.3f}s")

    # print(f"[PROFILE] TOTAL: {time.time()-t0:.3f}s")
    return [result, score_info]

@function_tool(failure_error_function=None)
def search(query: str, dataset: str, negative_prompts: list[str] = [], negative_threshold: float = 0.3, t: int = 10) -> ToolOutputImage:
    try:
        return _search(query, dataset, negative_prompts, negative_threshold, t)
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        raise e


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
    paths = [f"{DATASET_ROOT}/{dataset_map[dataset]}/{name}" for name in selected_images if os.path.exists(f"{DATASET_ROOT}/{dataset_map[dataset]}/{name}")]

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

    # Log collage image to wandb with query as caption
    if wandb.run is not None:
        wandb.log({"sample_result": wandb.Image(whole_image, caption=query)})

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
    print(f"[LOG] Aesthetics scores: {scores}")
    return f"Aesthetics scores for {len(paths_to_rate)} images: {scores}"
