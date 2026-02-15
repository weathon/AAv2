# Proposal: Agent-Curated Wide-Spectrum Aesthetics Dataset for Combating Toxic Positivity in Image Generation Models

## 1. Problem Statement

Current text-to-image generation models (e.g., Stable Diffusion, FLUX, Kolors) are trained with reward models and RLHF pipelines that systematically over-align outputs toward conventional, mainstream aesthetic preferences. Even when users explicitly request "anti-aesthetic" outputs -- such as grainy textures, chaotic compositions, clashing colors, or deliberately unsettling imagery -- these models default to producing conventionally beautiful images. This phenomenon, termed **"Toxic Positivity"** in the prior work (arXiv: 2512.11883v2), represents a fundamental failure of aesthetic pluralism: the models suppress legitimate, non-mainstream artistic expressions in favor of a single dominant aesthetic standard.

The root cause lies in the training data and reward signals used during alignment. Existing human preference datasets (e.g., HPDv2) and their derived reward models (e.g., HPSv2, PickScore, ImageReward) are overwhelmingly biased toward high-aesthetic, polished imagery. They lack representation of intentional anti-aesthetic content -- technically degraded, emotionally dark, compositionally chaotic, or stylistically raw images that are *deliberately* created as valid artistic expressions. As a result, models fine-tuned on these signals learn to equate "preferred" with "conventionally beautiful," erasing the full spectrum of human aesthetic intent.

## 2. Prior Work: Identifying the Problem

The paper **"Toxic Positivity in AI-Generated Imagery"** (arXiv: 2512.11883v2) formally identifies and characterizes this problem. Key contributions of the prior work include:

- **Defining Toxic Positivity**: Coining the term to describe the systematic suppression of non-mainstream aesthetic expressions by image generation models that are over-aligned to conventional beauty standards.
- **Empirical Evidence**: Demonstrating through experiments that state-of-the-art models consistently fail to produce requested anti-aesthetic outputs, instead "beautifying" prompts that explicitly call for ugliness, decay, distortion, or discomfort.
- **Taxonomy of Aesthetics**: Providing a structured taxonomy that decomposes aesthetics into fine-grained attributes across both pro-aesthetic and anti-aesthetic dimensions, covering categories such as color quality, clarity and focus, emotion, distortion, execution quality, lighting/exposure, structure/perspective, and context/setting.
- **Call for Wide-Spectrum Data**: Arguing that the solution requires training data and reward models that span the full aesthetic spectrum -- not just the conventionally beautiful end.

However, the prior work primarily *identifies* the problem and *proposes* the direction. **It does not provide a concrete solution** -- no wide-spectrum dataset, no anti-aesthetic-aware reward model, and no de-biased generation pipeline.

## 3. Our Solution: An Agentic Dataset Curation System

This project provides the **solution** to the problem identified above. We build an **autonomous AI agent system** that curates a wide-spectrum aesthetics dataset from existing image collections, guided by the taxonomy from the prior work. The agent intelligently searches, evaluates, and commits image batches that span the full aesthetic range -- from polished professional photography to deliberately raw, chaotic, and anti-aesthetic content.

### 3.1 System Architecture

The system consists of the following components:

#### 3.1.1 Multimodal Embedding Search Engine
- **Embedding Model**: Qwen3-VL-Embedding-8B, a vision-language embedding model that encodes both text queries and images into a shared semantic space.
- **Pre-computed Embeddings**: All images from three source datasets (AVA photographs, Lapis artwork, Liminal Space dreamcore) are pre-embedded and stored on HuggingFace (`weathon/ava_embeddings`), enabling real-time cosine similarity search.
- **Negative Prompt Filtering**: A novel filtering mechanism that excludes images matching orthogonal quality issues (watermarks, blank frames, text overlays) by computing cosine similarity against negative prompts and masking out above-threshold matches.

#### 3.1.2 Aesthetics Evaluation via HPSv3
- **HPSv3 Reward Model**: A state-of-the-art VLM-based human preference scorer (ICCV 2025) trained on HPDv3, a wide-spectrum preference dataset with 1.08M text-image pairs and 1.17M annotated comparisons.
- **Auto-Captioning Pipeline**: Each image is automatically captioned using GPT-5-chat via the OpenRouter API, focusing on objective physical facts rather than aesthetic judgments. These captions are then paired with the images for HPSv3 scoring.
- **Score Distribution Analysis**: Rather than applying a single quality threshold, the system computes and reports the full histogram of aesthetics scores for each batch, allowing the agent to reason about whether a low-scoring batch successfully achieves anti-aesthetic goals.

#### 3.1.3 Autonomous Curation Agent
- **Agent Framework**: Built on OpenAI's Agents SDK, the agent is powered by GPT-5.2 with medium reasoning effort, capable of multi-turn autonomous operation (up to 200 turns).
- **Tool Suite**: The agent has access to seven specialized tools:
  - `search`: Top-k semantic search with negative prompt filtering
  - `sample`: Random sampling within a cosine similarity range for threshold calibration
  - `aesthetics_rate`: HPSv3-based aesthetics score distribution analysis
  - `commit`: Batch commit of images above a similarity threshold to the dataset
  - `undo_commit`: Rollback mechanism for incorrect commits
  - `status`: Dataset composition monitoring
  - `sample_from_committed`: Visual review of committed batches
- **Curation Strategy**: The agent follows a structured workflow:
  1. Receive a target theme (e.g., "Psychedelic art")
  2. Perform broad exploratory searches to identify sub-concepts
  3. Decompose complex themes into specific visual sub-elements
  4. For each sub-element: search, sample, evaluate aesthetics, calibrate thresholds, and commit
  5. Monitor overall dataset balance and diversity

#### 3.1.4 Source Datasets
The system draws from three complementary image collections:
- **`photos` (AVA)**: Real-world photographs and edited photo art, spanning professional to amateur quality
- **`artwork` (Lapis)**: Traditional art dataset covering paintings, prints, and mixed media across centuries
- **`dreamcore` (Liminal Space)**: Surreal, unsettling liminal space imagery representing internet-born anti-aesthetic movements

### 3.2 Anti-Aesthetic Taxonomy

Based on the prior work's taxonomy, the system's `classes.json` defines a comprehensive classification schema with two top-level categories:

**Anti-Aesthetics** (9 subcategories, 60+ attributes):
- Realism & Style: surrealism, uncanny valley, dreamcore, weirdcore, outsider/naive style, psychedelic art
- Color Quality: wrong object color, clashing disharmony, toxic neon palette, sickly color cast
- Clarity & Focus: motion blur, datamosh, VHS decay, scanline texture, over-sharpened haloing
- Emotion: atmospheric distress, nostalgic unease, depersonalization/detachment
- Distortion: melted objects, non-Euclidean geometry, facial feature displacement
- Execution Quality: unfinished, analog decay, amateur snapshot energy, kitsch excess
- Lighting & Exposure: harsh flash, light leak, oppressive low contrast
- Structure & Perspective: scale inconsistency, endless corridor depth, tilted snapshot angle
- Context & Setting: liminal public space, backrooms infinite interior, dream symbol fragments

**Pro-Aesthetics** (7 subcategories, 40+ attributes):
- Photorealism, hyperrealism, cinematic quality, masterpiece execution
- Color harmony, HDR, cinematic grading
- Sharp focus, bokeh, 8K resolution
- Volumetric lighting, golden hour, chiaroscuro

### 3.3 Git-Like Version Control for Dataset Curation

A key design innovation is the **commit-based dataset management** system, inspired by Git:
- Each `commit` operation assigns a unique UUID-based ID to a batch of images
- Commits record full provenance: query text, dataset source, similarity threshold, negative prompts, and a human-readable message describing the curation intent
- `undo_commit` enables rollback of erroneous batches
- `status` provides a high-level view of all committed batches with image counts
- The full commit history is persisted in `dataset.json`, creating an auditable trail of curation decisions

## 4. Key Innovations

1. **Agent-as-Curator Paradigm**: Rather than relying on human annotators or simple rule-based filters, we delegate dataset curation to an autonomous AI agent that can reason about aesthetic intent, decompose complex themes, and iteratively refine search strategies.

2. **Inverted Aesthetics Scoring**: Traditional pipelines use high aesthetics scores as inclusion criteria. Our system *inverts* this logic for anti-aesthetic content: low HPSv3 scores can *validate* rather than disqualify images, confirming that they successfully deviate from mainstream preferences.

3. **Semantic Embedding Search with Negative Filtering**: The combination of Qwen3-VL vision-language embeddings with negative prompt masking enables precise retrieval of targeted visual sub-elements while excluding orthogonal quality artifacts.

4. **Wide-Spectrum by Design**: The system explicitly targets 200-300 images per theme across both high-aesthetic and anti-aesthetic content, ensuring the resulting dataset resists the monoculture bias present in existing preference datasets.

## 5. Expected Outcomes

- A **wide-spectrum aesthetics dataset** that includes deliberately curated anti-aesthetic content alongside conventional high-quality images, with full provenance metadata.
- A **reusable agentic curation framework** that can be applied to new aesthetic themes and source datasets with minimal human intervention.
- **Empirical validation** that the curated dataset, when used to train or fine-tune reward models, produces scores that better reflect the full range of human aesthetic intent -- including the deliberate pursuit of anti-aesthetic qualities.
- A foundation for **de-biasing image generation models** by providing training signal that respects aesthetic pluralism rather than enforcing toxic positivity.

## 6. Technical Stack

| Component | Technology |
|-----------|-----------|
| Embedding Model | Qwen3-VL-Embedding-8B |
| Reward Model | HPSv3 (Qwen2-VL-7B-Instruct backbone) |
| Captioning | GPT-5-chat via OpenRouter API |
| Agent LLM | GPT-5.2 via OpenAI Agents SDK |
| Embedding Dataset | HuggingFace (`weathon/ava_embeddings`) |
| Image Sources | AVA, Lapis (WikiArt), Liminal Space |
| Framework | OpenAI Agents SDK, PyTorch, HuggingFace Datasets |

## 7. Conclusion

The toxic positivity problem in image generation models stems from a data-level bias: existing preference datasets and reward models overwhelmingly favor conventional aesthetics. This project provides a concrete, scalable solution by building an autonomous agent that curates wide-spectrum aesthetic datasets spanning the full range from polished beauty to intentional anti-aesthetic expression. By combining vision-language embedding search, HPSv3-based aesthetics evaluation, and an agentic curation loop with Git-like version control, we create datasets that can de-bias reward models and, ultimately, restore aesthetic pluralism to image generation systems.
