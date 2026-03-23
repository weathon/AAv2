import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
with open("captions.json", "r") as f:
    captions = json.load(f)

import torch
image_paths = [i["image"] for i in captions] 
prompts = [i["caption"] for i in captions]
import hpsv2
import tqdm
scores = []
with torch.no_grad():
    pbar = tqdm.tqdm(zip(image_paths, prompts), total=len(image_paths))
    for image_path, prompt in pbar:
        result = hpsv2.score(image_path, prompt, hps_version="v2.1")
        score = result[0].item()
        scores.append(score)
        avg = sum(scores) / len(scores) 
        pbar.set_postfix(score=f"{score:.4f}", avg=f"{avg:.4f}", n=len(scores))
        with open("hpsv2.1.jsonl", "a") as f:
            json.dump({"image": image_path, "caption": prompt, "score": score}, f)
            f.write("\n")