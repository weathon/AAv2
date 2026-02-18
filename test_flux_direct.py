"""
Direct test of NAG Flux model without HTTP server.
Useful for debugging model issues in isolation.
"""

import os
import sys
import torch
from nag import NAGFluxPipeline, NAGFluxTransformer2DModel
from PIL import Image

print("=" * 80)
print("FLUX NAG Direct Test")
print("=" * 80)

try:
    print("\n[1/4] Loading Flux transformer...")
    transformer = NAGFluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    print("✓ Transformer loaded successfully")

    print("\n[2/4] Loading Flux pipeline...")
    pipe = NAGFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Krea-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    print("✓ Pipeline loaded successfully")

    device = os.getenv("FLUX_DEVICE", "cuda:1")
    print(f"\n[3/4] Moving to device: {device}")
    pipe.to(device)
    print(f"✓ Moved to {device}")

    print("\n[4/4] Generating image...")
    print("  Prompt: 'a serene mountain landscape with a lake at sunset'")
    print("  Negative: 'blurry, low quality'")
    print("  NAG scale: 7.0")

    image = pipe(
        "a serene mountain landscape with a lake at sunset",
        nag_negative_prompt="blurry, low quality",
        guidance_scale=0.0,
        nag_scale=7.0,
        nag_alpha=0.5,
        nag_tau=5.0,
        num_inference_steps=28,
        max_sequence_length=256,
    ).images[0]

    print("✓ Image generated successfully!")

    # Save the image
    output_path = "test_flux_output.png"
    image.save(output_path)
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Size: {image.size}")
    print(f"  Mode: {image.mode}")

    print("\n" + "=" * 80)
    print("SUCCESS: Flux NAG is working correctly")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}")
    print(f"\nMessage: {str(e)}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 80)
    print("FAILURE: Check error above")
    print("=" * 80)
    sys.exit(1)
