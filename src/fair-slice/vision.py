"""
Module: Vision (Computer Vision Pipeline)
Owner: Pablo

Responsible for:
  - Receiving a photo of a dish
  - Calling Vertex AI / Gemini for ingredient identification
  - Producing a per-pixel ingredient density map

See README.md Section 2 for the full contract specification.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from google import genai
from dotenv import load_dotenv
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE = 500           # Resize longest edge to this before any processing
MAX_INGREDIENTS = 8         # Hard cap — group beyond this
MIN_INGREDIENTS = 2         # Gemini must return at least this many
MIN_CHANNEL_MEAN_COVERAGE = 0.02  # canales con media < esto sobre dish_mask se eliminan


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def _init_model() -> genai.Client:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_AI_API_KEY not set in .env")
    return genai.Client(api_key=api_key)


def _call_gemini_combined(client: genai.Client, image_path: str, H: int, W: int) -> tuple[list[dict], np.ndarray]:
    """
    Single Gemini call that returns BOTH the ingredient list and the dish polygon.
    Saves one API call (faster) and ensures consistency between detection and labeling.

    Returns:
        ingredients: list of dicts with name + color_description
        dish_mask: bool ndarray (H, W) — True inside the dish
    """
    image = Image.open(image_path)

    prompt = """You are analyzing a food photo. Your task is to identify the dish boundary with PRECISION.

CRITICAL RULES for the polygon:
- Trace ONLY the outer edge of the food itself (pizza, cake, plate of food)
- Do NOT include the table, background, cutting board, napkins, or anything that is not food
- The polygon must fit INSIDE the food boundary, not outside it
- If unsure, trace slightly smaller rather than larger
- Use 30 points evenly spaced around the food edge
- Coordinates are fractions: x = column/image_width, y = row/image_height, both in [0,1]
- If the image has a clean/white background with the food on top, use that strong contrast to trace the true food edge precisely (including the crust/border)

Also identify the ingredients visible in the dish.

Return ONLY this JSON (no markdown, no explanation):
{
  "ingredients": [
    {"name": "dough", "color_description": "pale yellow-beige base covering entire pizza"},
    {"name": "tomato_sauce", "color_description": "bright red glossy patches"}
  ],
  "polygon": [[0.45, 0.08], [0.55, 0.09], ...]
}

The BASE ingredient (dough/bread/rice) MUST be first in the ingredients list.
Return 2-8 ingredients total."""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[image, prompt],
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)
    ingredients = data["ingredients"][:MAX_INGREDIENTS]
    if len(ingredients) < MIN_INGREDIENTS:
        raise ValueError(f"Gemini returned fewer than {MIN_INGREDIENTS} ingredients")

    # Build dish mask from polygon
    points = data["polygon"]
    pts = np.array(
        [[int(np.clip(p[0], 0, 1) * W), int(np.clip(p[1], 0, 1) * H)] for p in points],
        dtype=np.int32,
    )
    dish_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(dish_mask, [pts], 1)

    coverage = float(dish_mask.sum()) / float(H * W)

    # Validation: polygon must cover a reasonable area
    if coverage < 0.10:
        raise ValueError(f"Polygon too small (coverage={coverage:.2%})")
    if coverage > 0.75:
        raise ValueError(f"Polygon covers almost full image (coverage={coverage:.2%}) — likely wrong")

    # Smooth the mask edges with a small morphological close to eliminate jaggedness
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dish_mask = cv2.morphologyEx(dish_mask, cv2.MORPH_CLOSE, kernel)

    print(f"[vision] Gemini combined: {len(ingredients)} ingredients, "
          f"polygon {len(pts)} pts, coverage={coverage:.3f}")

    return ingredients, dish_mask.astype(bool)


def _fallback_dish_mask(H: int, W: int) -> np.ndarray:
    """Center ellipse fallback if Gemini fails."""
    print("[vision] Using fallback center ellipse mask")
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, (W // 2, H // 2), (int(W * 0.42), int(H * 0.42)), 0, 0, 360, 1, -1)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Per-ingredient color segmentation
# ---------------------------------------------------------------------------

_HSV_HINTS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "tomato":       (np.array([0, 100, 80]),   np.array([15, 255, 255])),
    "sauce":        (np.array([0,  80, 60]),   np.array([20, 255, 220])),
    "pepperoni":    (np.array([0, 120, 60]),   np.array([12, 255, 200])),
    "mozzarella":   (np.array([10,  20, 140]), np.array([45, 180, 255])),
    "cheese":       (np.array([10,  20, 130]), np.array([45, 180, 255])),
    "olive":        (np.array([35, 40,  20]),  np.array([90, 180, 130])),
    "mushroom":     (np.array([10, 20,  80]),  np.array([30, 80,  200])),
    "basil":        (np.array([35, 80,  40]),  np.array([85, 255, 200])),
    "spinach":      (np.array([35, 80,  30]),  np.array([85, 255, 180])),
    "dough":        (np.array([15, 10,  140]), np.array([35, 80,  255])),
    "bread":        (np.array([15, 10,  140]), np.array([35, 80,  255])),
    "crust":        (np.array([15, 20,  100]), np.array([30, 100, 240])),
    "rice":         (np.array([15,  0,  180]), np.array([40, 40,  255])),
    "egg":          (np.array([15, 80,  180]), np.array([30, 200, 255])),
    "ham":          (np.array([0, 100,  120]), np.array([15, 220, 220])),
    "bacon":        (np.array([0,  80,  100]), np.array([15, 220, 220])),
    "beef":         (np.array([0,  60,   50]), np.array([20, 200, 180])),
    "chicken":      (np.array([15, 20,  140]), np.array([35, 100, 240])),
    "salami":       (np.array([0, 100,  80]),  np.array([15, 255, 200])),
    "pepper":       (np.array([35, 80,  40]),  np.array([85, 255, 200])),
    "onion":        (np.array([0,   0, 180]),  np.array([180, 50, 255])),
}


def _color_description_to_hsv(color_desc: str) -> tuple[np.ndarray, np.ndarray] | None:
    desc = color_desc.lower()
    if any(w in desc for w in ["red", "crimson", "scarlet"]):
        return np.array([0, 100, 80]), np.array([15, 255, 255])
    if "orange" in desc:
        return np.array([5, 100, 100]), np.array([20, 255, 255])
    if "yellow" in desc:
        return np.array([20, 60, 150]), np.array([35, 255, 255])
    if any(w in desc for w in ["beige", "pale", "tan", "cream", "light yellow"]):
        return np.array([15, 5, 130]), np.array([35, 70, 255])
    if any(w in desc for w in ["white", "ivory", "light"]):
        return np.array([0, 0, 180]), np.array([180, 50, 255])
    if "green" in desc:
        return np.array([35, 60, 40]), np.array([85, 255, 220])
    if any(w in desc for w in ["brown", "dark", "black"]):
        return np.array([0, 30, 20]), np.array([30, 180, 140])
    if any(w in desc for w in ["pink", "salmon", "rose"]):
        return np.array([0, 60, 140]), np.array([15, 180, 230])
    return None


def _segment_ingredient(
    hsv: np.ndarray,
    dish_mask: np.ndarray,
    ingredient_name: str,
    color_description: str,
) -> np.ndarray:
    H, W = hsv.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    lower, upper = None, None
    for key, (lo, hi) in _HSV_HINTS.items():
        if key in ingredient_name.lower():
            lower, upper = lo, hi
            break

    if lower is None:
        result = _color_description_to_hsv(color_description)
        if result:
            lower, upper = result

    if lower is not None and upper is not None:
        if lower[0] <= 10 and upper[0] <= 20:
            mask1 = cv2.inRange(hsv, lower, upper)
            lower2 = lower.copy(); lower2[0] = 165
            upper2 = upper.copy(); upper2[0] = 180
            mask2 = cv2.inRange(hsv, lower2, upper2)
            raw_mask = cv2.bitwise_or(mask1, mask2)
        else:
            raw_mask = cv2.inRange(hsv, lower, upper)

        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  kernel)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

        density = raw_mask.astype(np.float32) / 255.0
        density = cv2.GaussianBlur(density, (21, 21), 5)
        density[~dish_mask] = 0.0
        return density

    uniform = np.zeros((H, W), dtype=np.float32)
    uniform[dish_mask] = 0.5
    return uniform


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize_map(ingredient_map: np.ndarray, dish_mask: np.ndarray) -> np.ndarray:
    total = ingredient_map.sum(axis=2, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    ingredient_map = ingredient_map / total
    ingredient_map[~dish_mask] = 0.0
    return ingredient_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_dish(image_path: str) -> tuple[np.ndarray, dict[int, str]]:
    """
    Segments a dish photo into per-pixel ingredient maps.
    Single Gemini call gets both ingredients and dish boundary polygon.
    No perspective correction — Gemini handles tilted photos directly.
    """
    # 1. Load and resize
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot load image from '{image_path}'")

    H_orig, W_orig = img_bgr.shape[:2]
    scale = TARGET_SIZE / max(H_orig, W_orig)
    if scale < 1.0:
        new_w = int(W_orig * scale)
        new_h = int(H_orig * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    H, W = img_bgr.shape[:2]

    # Save resized image to a temp path so Gemini sees the same dimensions we work with
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    # 2. Single Gemini call: ingredients + polygon
    client = _init_model()
    try:
        ingredients, dish_mask = _call_gemini_combined(client, tmp_path, H, W)
    except Exception as e:
        print(f"[vision] Combined Gemini call failed ({e}), trying separate calls")
        # Retry with separate calls if combined fails
        try:
            ingredients, dish_mask = _call_gemini_combined(client, image_path, H, W)
        except Exception as e2:
            print(f"[vision] Second attempt also failed ({e2}), using fallback")
            dish_mask = _fallback_dish_mask(H, W)
            # Minimal default ingredients
            ingredients = [
                {"name": "dough", "color_description": "pale beige base"},
                {"name": "tomato_sauce", "color_description": "red sauce"},
                {"name": "cheese", "color_description": "white melted cheese"},
            ]

    K = len(ingredients)
    ingredient_labels: dict[int, str] = {i: ing["name"] for i, ing in enumerate(ingredients)}

    # 3. Per-ingredient HSV segmentation
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ingredient_map = np.zeros((H, W, K), dtype=np.float32)
    for k, ing in enumerate(ingredients):
        density = _segment_ingredient(
            hsv, dish_mask,
            ingredient_name=ing["name"],
            color_description=ing.get("color_description", ""),
        )
        ingredient_map[:, :, k] = density

    # 4. Fill unclaimed pixels with low-confidence base
    pixel_total = ingredient_map.sum(axis=2)
    unclaimed = (pixel_total < 0.05) & dish_mask
    ingredient_map[unclaimed, 0] = 0.3

    # 5. Dough as cumulative base — channel 0 fills entire dish before normalize
    dish_pixels = int(dish_mask.sum())
    if dish_pixels > 0:
        for k in range(K):
            channel = ingredient_map[:, :, k]
            coverage = float(((channel > 0.1) & dish_mask).sum()) / dish_pixels
            mean_val = float(channel[dish_mask].mean())
            name = ingredient_labels.get(k, f"channel_{k}")
            print(f"[vision] Channel {k} ({name}): coverage={coverage:.1%}, mean={mean_val:.3f}")

    ingredient_map[:, :, 0][dish_mask] = 1.0
    ingredient_map = _normalize_map(ingredient_map, dish_mask)

    # 6. Filter low-coverage channels (except base)
    if dish_pixels > 0 and K > 1:
        to_drop: list[int] = []
        for k in range(K):
            mean_k = float(ingredient_map[:, :, k][dish_mask].mean())
            if mean_k < MIN_CHANNEL_MEAN_COVERAGE:
                if k == 0:
                    print(f"[vision] Channel {k} ({ingredient_labels[k]}) preserved as base "
                          f"(mean={mean_k:.4f} below threshold)")
                    continue
                to_drop.append(k)
                print(f"[vision] Channel {k} ({ingredient_labels[k]}) dropped: "
                      f"mean={mean_k:.4f} < threshold")

        if to_drop:
            keep = [k for k in range(K) if k not in set(to_drop)]
            ingredient_map = ingredient_map[:, :, keep]
            ingredient_labels = {new_i: ingredient_labels[old_i] for new_i, old_i in enumerate(keep)}
            ingredient_map = _normalize_map(ingredient_map, dish_mask)
            K = ingredient_map.shape[-1]

    # 7. Validate
    assert ingredient_map.dtype == np.float32
    assert ingredient_map.shape == (H, W, K)
    assert ingredient_map.min() >= -1e-5
    assert ingredient_map.max() <= 1.0 + 1e-5

    return ingredient_map, ingredient_labels


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vision.py <path_to_image>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"[vision] Processing: {path}")
    imap, labels = segment_dish(path)
    print(f"[vision] Output shape: {imap.shape}")
    print(f"[vision] Ingredients: {labels}")
    print("[vision] Done")