"""
Standalone HTTP server for FLUX.1-Krea-dev image generation with NAG.

Run with:
    python flux_server.py

Exposes POST /generate endpoint that accepts:
    {
        "prompt": str,
        "negative_prompt": str,
        "nag_scale": float,
        "nag_alpha": float,
        "nag_tau": float,
        "num_of_images": int
    }

Returns:
    {
        "images": [base64_webp_1, base64_webp_2, ...],
        "status": "success"
    }
"""

import io
import os
import base64
import torch
from flask import Flask, request, jsonify
from nag import NAGFluxPipeline, NAGFluxTransformer2DModel
from PIL import Image

app = Flask(__name__)

# Load Flux model at startup
print("Loading Flux transformer...")
transformer = NAGFluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-Krea-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

print("Loading Flux pipeline...")
pipe = NAGFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Krea-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.to(os.getenv("FLUX_DEVICE", "cuda:2"))
print("Flux model ready.")


def _pil_to_base64_webp(image: Image.Image) -> str:
    """Convert a PIL Image to base64 WebP string."""
    buffered = io.BytesIO()
    try:
        image.save(buffered, format="WEBP", quality=85, optimize=True)
    except Exception:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
    buffered.seek(0)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate images using Flux with NAG."""
    try:
        data = request.json
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        nag_scale = data.get("nag_scale", 7.0)
        nag_alpha = data.get("nag_alpha", 0.5)
        nag_tau = data.get("nag_tau", 5.0)
        num_of_images = data.get("num_of_images", 1)

        pil_images = []
        for _ in range(num_of_images):
            image = pipe(
                prompt,
                nag_negative_prompt=negative_prompt,
                guidance_scale=0.0,
                nag_scale=nag_scale,
                nag_alpha=nag_alpha,
                nag_tau=nag_tau,
                num_inference_steps=28,
                max_sequence_length=256,
            ).images[0]
            pil_images.append(image)

        images_b64 = [_pil_to_base64_webp(img) for img in pil_images]

        return jsonify({
            "images": images_b64,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.getenv("FLUX_SERVER_PORT", 5001))
    print(f"Starting Flux server on port {port}...")
    app.run(host="127.0.0.1", port=port, debug=False)
