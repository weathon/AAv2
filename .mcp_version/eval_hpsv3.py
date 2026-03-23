import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
with open("captions.json", "r") as f:
    captions = json.load(f)

import torch
import random
random.shuffle(captions)
image_paths = [i["image"] for i in captions] 
prompts = [i["caption"] for i in captions]

from hpsv3 import HPSv3RewardInferencer

# Initialize the model
inferencer = HPSv3RewardInferencer(device='cuda')

import hpsv2
import tqdm
from math import ceil
scores = []
bs = 10
with torch.no_grad():
    pbar = tqdm.tqdm(range(len(image_paths)), total=ceil(len(image_paths)/bs))
    for i in pbar:
        prompt_batch = prompts[i*bs:(i+1)*bs]
        image_batch = image_paths[i*bs:(i+1)*bs]
        rewards = inferencer.reward(prompts=prompt_batch, image_paths=image_batch)
        score = [reward[0].item() for reward in rewards]
        scores.extend(score)
        avg = sum(scores) / len(scores) 
        pbar.set_postfix(avg=f"{avg:.4f}", n=len(scores))
        with open("hpsv3.jsonl", "a") as f:
            for img, prompt, sc in zip(image_batch, prompt_batch, score):
                json.dump({"image": img, "caption": prompt, "score": sc}, f)
                f.write("\n")