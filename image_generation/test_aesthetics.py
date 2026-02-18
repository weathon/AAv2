"""
Test all image generation MCP tools with aesthetic scoring enabled.
"""

import requests
import json
import sys

MCP_URL = "http://127.0.0.1:8000"

def test_init():
    """Test init tool."""
    print("\n" + "="*80)
    print("TEST 1: init()")
    print("="*80)
    try:
        response = requests.post(f"{MCP_URL}/init", json={}, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Exception: {type(e).__name__}: {str(e)}")
        return False

def test_flux_with_aesthetics():
    """Test Flux with aesthetic scoring."""
    print("\n" + "="*80)
    print("TEST 2: generate_flux() with aesthetic score")
    print("="*80)
    payload = {
        "prompt": "a beautiful sunset over mountains with golden light",
        "negative_prompt": "blurry, low quality, dark",
        "nag_scale": 7,
        "nag_alpha": 0.5,
        "nag_tau": 5,
        "num_of_images": 1,
        "return_aesthetic_score": True
    }
    print(f"Sending request...")

    try:
        response = requests.post(f"{MCP_URL}/generate_flux", json=payload, timeout=300)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Generated successfully")
            print(f"  Response keys: {list(data.keys())[:5]}...")
            # Look for aesthetic score in the response
            for key in data:
                if isinstance(data[key], str) and "Aesthetic" in data[key]:
                    print(f"  {data[key][:100]}...")
            return True
        else:
            print(f"✗ Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ Exception: {type(e).__name__}: {str(e)}")
        return False

def test_z_image_with_aesthetics():
    """Test Z-image with aesthetic scoring."""
    print("\n" + "="*80)
    print("TEST 3: generate_z_image() with aesthetic score")
    print("="*80)
    payload = {
        "prompt": "vibrant abstract geometric art with neon colors",
        "negative_prompt": "realistic, dull colors, boring",
        "scale": 5,
        "num_of_images": 1,
        "return_aesthetic_score": True
    }
    print(f"Sending request...")

    try:
        response = requests.post(f"{MCP_URL}/generate_z_image", json=payload, timeout=300)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Generated successfully")
            print(f"  Response keys: {list(data.keys())[:5]}...")
            for key in data:
                if isinstance(data[key], str) and "Aesthetic" in data[key]:
                    print(f"  {data[key][:100]}...")
                if isinstance(data[key], str) and "Cost" in data[key]:
                    print(f"  {data[key]}")
            return True
        else:
            print(f"✗ Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ Exception: {type(e).__name__}: {str(e)}")
        return False

def test_nano_banana_with_aesthetics():
    """Test Nano Banana with aesthetic scoring."""
    print("\n" + "="*80)
    print("TEST 4: generate_using_nano_banana() with aesthetic score")
    print("="*80)
    payload = {
        "prompt": "serene forest with misty morning atmosphere and soft sunlight",
        "num_of_images": 1,
        "return_aesthetic_score": True
    }
    print(f"Sending request...")

    try:
        response = requests.post(f"{MCP_URL}/generate_using_nano_banana", json=payload, timeout=300)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Generated successfully")
            print(f"  Response keys: {list(data.keys())[:5]}...")
            for key in data:
                if isinstance(data[key], str) and "Aesthetic" in data[key]:
                    print(f"  {data[key][:100]}...")
                if isinstance(data[key], str) and "Cost" in data[key]:
                    print(f"  {data[key]}")
            return True
        else:
            print(f"✗ Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ Exception: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Image Generation MCP Server with Aesthetic Scoring")
    print(f"MCP URL: {MCP_URL}")

    results = {
        "init": test_init(),
        "flux": test_flux_with_aesthetics(),
        "z_image": test_z_image_with_aesthetics(),
        "nano_banana": test_nano_banana_with_aesthetics(),
    }

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)
