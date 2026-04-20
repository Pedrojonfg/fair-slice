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

import cv2
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, Image as VertexImage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE = 500           # Resize longest edge to this before any processing
MAX_INGREDIENTS = 8         # Hard cap — group beyond this
MIN_INGREDIENTS = 2         # Gemini must return at least this many


# ---------------------------------------------------------------------------
# Vertex AI / Gemini helpers
# ---------------------------------------------------------------------------

def _init_vertex() -> GenerativeModel:
    """Initialise Vertex AI from environment variables and return the model."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not project:
        raise EnvironmentError(
            "GOOGLE_CLOUD_PROJECT environment variable is not set. "
            "See README section 6.3 for setup instructions."
        )

    vertexai.init(project=project, location=location)
    return GenerativeModel("gemini-1.5-flash")


def _call_gemini(model: GenerativeModel, image_path: str) -> list[dict]:
    """
    Ask Gemini to identify every ingredient in the dish photo.

    Returns a list of dicts:
        [
            {"name": "dough",      "color_description": "pale yellow, uniform base"},
            {"name": "tomato_sauce","color_description": "red, glossy patches"},
            ...
        ]
    The list is ordered with the base/dough ingredient first (channel 0).
    """
    vertex_image = VertexImage.load_from_file(image_path)

    prompt = """Analyze this dish photo carefully.

Identify every distinct ingredient or component visible, including:
- The base (bread, dough, rice, pastry…)
- Sauces or spreads
- Toppings, vegetables, meats, cheeses

Rules:
1. Group very similar ingredients together (e.g., all cheese → "mozzarella").
2. Return between 2 and 8 ingredients total.
3. Put the BASE ingredient (dough, bread, rice…) FIRST in the list.
4. For each ingredient, describe its visual color/appearance so it can be detected with OpenCV.

Return ONLY a valid JSON array, no extra text, no markdown fences.
Example format:
[
  {"name": "dough",       "color_description": "pale yellow-beige, covers entire base"},
  {"name": "tomato_sauce","color_description": "bright red, glossy, in patches"},
  {"name": "mozzarella",  "color_description": "white to light yellow, melted blobs"},
  {"name": "pepperoni",   "color_description": "dark reddish-brown small circles"}
]"""

    response = model.generate_content([vertex_image, prompt])
    raw = response.text.strip()

    # Strip accidental markdown fences if Gemini adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    ingredients = json.loads(raw)

    if not isinstance(ingredients, list) or len(ingredients) < MIN_INGREDIENTS:
        raise ValueError(
            f"Gemini returned fewer than {MIN_INGREDIENTS} ingredients: {ingredients}"
        )

    # Cap at MAX_INGREDIENTS
    return ingredients[:MAX_INGREDIENTS]


# ---------------------------------------------------------------------------
# Dish boundary detection
# ---------------------------------------------------------------------------

def _detect_dish_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask (H, W) — True for pixels that are part of the dish.

    Strategy:
      1. Try Hough Circle detection (great for pizzas, cakes, paella).
      2. Fallback: largest contour in a foreground/background separation.
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # --- Attempt 1: Hough circles ---
    min_r = int(min(H, W) * 0.25)
    max_r = int(min(H, W) * 0.55)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(H, W) // 2,
        param1=100,
        param2=40,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        cx, cy, r = np.round(circles[0][0]).astype(int)
        mask = np.zeros((H, W), dtype=bool)
        cv2.circle(mask.view(np.uint8), (cx, cy), r, 1, thickness=-1)
        # Accept if circle covers at least 20% of the image
        if mask.sum() > 0.20 * H * W:
            return mask

    # --- Fallback: grab-cut / largest contour ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Broad saturation threshold to find the dish vs plain background
    sat = hsv[:, :, 1]
    _, thresh = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Last resort: use the entire image
        return np.ones((H, W), dtype=bool)

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros((H, W), dtype=bool)
    cv2.drawContours(mask.view(np.uint8), [largest], -1, 1, thickness=-1)
    return mask


# ---------------------------------------------------------------------------
# Per-ingredient color segmentation
# ---------------------------------------------------------------------------

# Pre-built HSV ranges for common ingredients.
# Gemini's color_description is used as a guide; these are the fallback
# when no specific range can be inferred.
_HSV_HINTS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "tomato":       (np.array([0, 100, 80]),   np.array([15, 255, 255])),
    "sauce":        (np.array([0,  80, 60]),   np.array([20, 255, 220])),
    "pepperoni":    (np.array([0, 120, 60]),   np.array([12, 255, 200])),
    "mozzarella":   (np.array([15,  0, 160]),  np.array([40, 60,  255])),
    "cheese":       (np.array([15,  0, 150]),  np.array([40, 80,  255])),
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
    "beef":         (np.array([0,  60,   50]), np.array([20, 200, 180])),
    "chicken":      (np.array([15, 20,  140]), np.array([35, 100, 240])),
}


def _color_description_to_hsv(color_desc: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Very lightweight parser: look for color keywords in the description
    and map them to rough HSV ranges.
    """
    desc = color_desc.lower()
    # Red / tomato
    if any(w in desc for w in ["red", "crimson", "scarlet"]):
        return np.array([0, 100, 80]), np.array([15, 255, 255])
    # Orange / orange-red (pepperoni)
    if "orange" in desc:
        return np.array([5, 100, 100]), np.array([20, 255, 255])
    # Yellow
    if "yellow" in desc:
        return np.array([20, 60, 150]), np.array([35, 255, 255])
    # Beige / pale / tan (dough, crust)
    if any(w in desc for w in ["beige", "pale", "tan", "cream", "light yellow"]):
        return np.array([15, 5, 130]), np.array([35, 70, 255])
    # White / ivory (mozzarella, rice)
    if any(w in desc for w in ["white", "ivory", "light"]):
        return np.array([0, 0, 180]), np.array([180, 50, 255])
    # Green (basil, spinach, peppers)
    if "green" in desc:
        return np.array([35, 60, 40]), np.array([85, 255, 220])
    # Brown / dark (beef, mushroom, crust)
    if any(w in desc for w in ["brown", "dark", "black"]):
        return np.array([0, 30, 20]), np.array([30, 180, 140])
    # Pink / salmon (ham, prosciutto)
    if any(w in desc for w in ["pink", "salmon", "rose"]):
        return np.array([0, 60, 140]), np.array([15, 180, 230])
    return None


def _segment_ingredient(
    hsv: np.ndarray,
    dish_mask: np.ndarray,
    ingredient_name: str,
    color_description: str,
) -> np.ndarray:
    """
    Returns a float32 (H, W) array [0, 1] representing the density of the
    ingredient at each pixel.

    Steps:
      1. Try name-based HSV lookup.
      2. Try color_description parsing.
      3. Fallback: k-means cluster similarity.
    """
    H, W = hsv.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # --- Step 1: name-based hint ---
    lower, upper = None, None
    for key, (lo, hi) in _HSV_HINTS.items():
        if key in ingredient_name.lower():
            lower, upper = lo, hi
            break

    # --- Step 2: color description ---
    if lower is None:
        result = _color_description_to_hsv(color_description)
        if result:
            lower, upper = result

    if lower is not None and upper is not None:
        # Handle red hue wrap-around
        if lower[0] <= 10 and upper[0] <= 20:
            mask1 = cv2.inRange(hsv, lower, upper)
            lower2 = lower.copy(); lower2[0] = 165
            upper2 = upper.copy(); upper2[0] = 180
            mask2 = cv2.inRange(hsv, lower2, upper2)
            raw_mask = cv2.bitwise_or(mask1, mask2)
        else:
            raw_mask = cv2.inRange(hsv, lower, upper)

        # Clean up with morphology
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  kernel)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

        # Soft density: distance transform smooths harsh edges
        density = raw_mask.astype(np.float32) / 255.0
        density = cv2.GaussianBlur(density, (21, 21), 5)

        # Zero out outside dish
        density[~dish_mask] = 0.0
        return density

    # --- Step 3: fallback — use pixel value variance across the dish ---
    # Assign a uniform low-confidence density for unknown ingredients
    uniform = np.zeros((H, W), dtype=np.float32)
    uniform[dish_mask] = 0.5
    return uniform


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize_map(ingredient_map: np.ndarray, dish_mask: np.ndarray) -> np.ndarray:
    """
    Ensures channels sum to ~1.0 at each dish pixel.
    Pixels outside the dish remain 0.
    """
    total = ingredient_map.sum(axis=2, keepdims=True)            # (H, W, 1)

    # Avoid division by zero
    total = np.where(total > 0, total, 1.0)
    ingredient_map = ingredient_map / total

    # Enforce zeros outside dish
    ingredient_map[~dish_mask] = 0.0
    return ingredient_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_dish(image_path: str) -> tuple[np.ndarray, dict[int, str]]:
    """
    Segments a dish photo into per-pixel ingredient maps.

    Args:
        image_path: Path to the input image (JPG/PNG).

    Returns:
        ingredient_map: np.ndarray of shape (H, W, K), dtype float32.
                        Values in [0, 1]. Each channel k represents
                        the density of ingredient k at that pixel.
                        Channels sum to ~1.0 at each dish pixel.
        ingredient_labels: dict mapping channel index to ingredient name.
                           Example: {0: "dough", 1: "crust", 2: "pepperoni"}

    Raises:
        ValueError: If image cannot be loaded or is not a valid photo.
    """
    # ------------------------------------------------------------------
    # 1. Load & validate image
    # ------------------------------------------------------------------
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(
            f"Cannot load image from '{image_path}'. "
            "Make sure the file exists and is a valid JPG/PNG."
        )

    # ------------------------------------------------------------------
    # 2. Resize to TARGET_SIZE (keep aspect ratio, longest edge)
    # ------------------------------------------------------------------
    H_orig, W_orig = img_bgr.shape[:2]
    scale = TARGET_SIZE / max(H_orig, W_orig)
    if scale < 1.0:
        new_w = int(W_orig * scale)
        new_h = int(H_orig * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    H, W = img_bgr.shape[:2]

    # ------------------------------------------------------------------
    # 3. Detect dish boundary
    # ------------------------------------------------------------------
    dish_mask = _detect_dish_mask(img_bgr)   # bool (H, W)

    # ------------------------------------------------------------------
    # 4. Ask Gemini what ingredients are in the dish
    # ------------------------------------------------------------------
    model = _init_vertex()
    ingredients = _call_gemini(model, image_path)  # list[dict]

    K = len(ingredients)
    ingredient_labels: dict[int, str] = {i: ing["name"] for i, ing in enumerate(ingredients)}

    # ------------------------------------------------------------------
    # 5. Per-ingredient color segmentation (OpenCV HSV)
    # ------------------------------------------------------------------
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ingredient_map = np.zeros((H, W, K), dtype=np.float32)

    for k, ing in enumerate(ingredients):
        density = _segment_ingredient(
            hsv,
            dish_mask,
            ingredient_name=ing["name"],
            color_description=ing.get("color_description", ""),
        )
        ingredient_map[:, :, k] = density

    # ------------------------------------------------------------------
    # 6. Special handling: ensure "base" ingredient (channel 0) covers
    #    the whole dish — fill any pixel that has no detected ingredient
    # ------------------------------------------------------------------
    pixel_total = ingredient_map.sum(axis=2)             # (H, W)
    unclaimed = (pixel_total < 0.05) & dish_mask         # nearly zero inside dish
    ingredient_map[unclaimed, 0] = 1.0                   # assign to base channel

    # ------------------------------------------------------------------
    # 7. Normalize so channels sum to 1.0 at every dish pixel
    # ------------------------------------------------------------------
    ingredient_map = _normalize_map(ingredient_map, dish_mask)

    # ------------------------------------------------------------------
    # 8. Validate output before returning
    # ------------------------------------------------------------------
    assert ingredient_map.dtype == np.float32
    assert ingredient_map.shape == (H, W, K)
    assert ingredient_map.min() >= -1e-5
    assert ingredient_map.max() <= 1.0 + 1e-5

    dish_sums = ingredient_map[dish_mask].sum(axis=1)
    if not np.allclose(dish_sums, 1.0, atol=0.05):
        # Soft warning — don't crash, but log
        print(
            f"[vision] WARNING: {(np.abs(dish_sums - 1.0) > 0.05).sum()} pixels "
            "have channel sum outside [0.95, 1.05]."
        )

    return ingredient_map, ingredient_labels


# ---------------------------------------------------------------------------
# Quick smoke test (run directly: python vision.py <image_path>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vision.py <path_to_image>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"[vision] Processing: {path}")

    imap, labels = segment_dish(path)

    print(f"[vision] Output shape : {imap.shape}")
    print(f"[vision] Dtype        : {imap.dtype}")
    print(f"[vision] Value range  : [{imap.min():.4f}, {imap.max():.4f}]")
    print(f"[vision] Ingredients  : {labels}")

    # Channel-sum check on a random sample
    H, W, K = imap.shape
    sample_pixels = imap.reshape(-1, K)
    non_zero = sample_pixels[sample_pixels.sum(axis=1) > 0.01]
    sums = non_zero.sum(axis=1)
    print(f"[vision] Channel-sum  : mean={sums.mean():.4f}  std={sums.std():.4f}")
    print("[vision] Done ✓")
