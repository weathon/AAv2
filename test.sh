  curl -X POST http://127.0.0.1:5001/generate \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "a serene mountain landscape with a lake at sunset",
      "negative_prompt": "blurry, low quality",
      "nag_scale": 7,
      "nag_alpha": 0.5,
      "nag_tau": 5,
      "num_of_images": 1
    }' #| jq -r '.images[0]' | base64 -d > output.webp