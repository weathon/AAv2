"""
Direct test of HPSv3 aesthetic scoring without MCP.
Uses a placeholder caption to isolate HPSv3 version issues.
"""

import os
import sys
import torch
from PIL import Image

print("=" * 80)
print("HPSv3 Direct Test")
print("=" * 80)

# Test image path
test_image_path = "/home/wg25r/aas2/AAv2/.mcp_version/123.jpg"
if not os.path.exists(test_image_path):
    print(f"\n✗ Error: {test_image_path} not found")
    sys.exit(1)

print(f"\n[1/2] Loading test image: {test_image_path}")
try:
    image = Image.open(test_image_path)
    print(f"✓ Image loaded: {image.size} {image.mode}")
except Exception as e:
    print(f"✗ Error loading image: {e}")
    sys.exit(1)

# Load HPSv3 and score
print("\n[2/2] Loading HPSv3 and scoring...")
try:
    hps_dir = os.path.join(os.path.dirname(__file__), "..", "HPSv3")
    if hps_dir not in sys.path:
        sys.path.insert(0, hps_dir)
    from hpsv3 import HPSv3RewardInferencer

    device = os.getenv("HPS_DEVICE", "cuda:1")
    print(f"  Device: {device}")
    _inferencer = HPSv3RewardInferencer(device=device)
    print("  ✓ Inferencer loaded")

    # Score using placeholder caption
    caption = "an image"
    print(f"  Scoring with caption: '{caption}'")

    with torch.no_grad():
        rewards = _inferencer.reward(prompts=[caption], image_paths=[image])

    score = rewards[0][0].item()
    print(f"  ✓ Score: {score:.4f}")

    print("\n" + "=" * 80)
    print("SUCCESS: HPSv3 is working correctly")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}")
    print(f"  {str(e)}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 80)
    print("FAILURE: HPSv3 has version compatibility issues")
    print("=" * 80)
    sys.exit(1)
