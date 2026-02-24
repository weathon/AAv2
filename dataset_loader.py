import dotenv
dotenv.load_dotenv()

import datasets
import numpy as np
import torch
import sys

sys.path.append("../")
from qwen3_vl_embedding import Qwen3VLEmbedder

# Load dataset
ds = datasets.load_dataset("weathon/ava_embeddings", split="train")
ds = ds.with_format("numpy")

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

ava_names = ava_dataset["name"]
ls_names = ls_dataset["name"]
lapis_names = lapis_dataset["name"]

# Initialize model
model_name_or_path = "Qwen/Qwen3-VL-Embedding-8B"
model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path, device="cpu", attn_implementation="sdpa")

# Convert to tensors
ava_embeddings_tensor = torch.tensor(ava_embeddings).float()
ls_embeddings_tensor = torch.tensor(ls_embeddings).float()
lapis_embeddings_tensor = torch.tensor(lapis_embeddings).float()

ava_names_list = list(ava_names)
ls_names_list = list(ls_names)
lapis_names_list = list(lapis_names)

# Dataset mapping
dataset_map = {
    "photos": "ava",
    "dreamcore": "ls",
    "artwork": "lapis"
}
