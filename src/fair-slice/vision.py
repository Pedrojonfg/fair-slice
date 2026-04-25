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

# Perspective correction thresholds (minor_axis / major_axis ratio)
# A perfect top-down photo → ratio = 1.0
# A slightly tilted photo  → ratio ≈ 0.85–0.95  (correct it)
# Extremely tilted photo   → ratio < PERSPECTIVE_ABORT (refuse to process)
PERSPECTIVE_CORRECTION_THRESHOLD = 0.92   # below → apply correction
PERSPECTIVE_ABORT_THRESHOLD       = 0.40  # below → raise, result would be unreliable


# ---------------------------------------------------------------------------
# Vertex AI / Gemini helpers
# ---------------------------------------------------------------------------

def _init_model() -> genai.Client:
    # Ensure we always load the repo-root .env, regardless of current working directory.
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_AI_API_KEY not set in .env")
    return genai.Client(api_key=api_key)


def _call_gemini(client: genai.Client, image_path: str) -> list[dict]:
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
    image = Image.open(image_path)

    prompt = """Analyze this dish photo carefully.

Identify every distinct ingredient or component visible, including:
- The base (bread, dough, rice, pastry…)
- Sauces or spreads
- Toppings, vegetables, meats, cheeses

Rules:
1. Group very similar ingredients together (e.g., all cheese → "mozzarella").
2. Return between 2 and 8 ingredients total.
3. Put the BASE ingredient (dough, bread, rice…) FIRST in the list.
4. The pizza crust/border is ALSO an ingredient ("crust") and must be included if visible, so it can be allocated according to user preferences.
5. For each ingredient, describe its visual color/appearance so it can be detected with OpenCV.

Return ONLY a valid JSON array, no extra text, no markdown fences.
Example format:
[
  {"name": "dough",       "color_description": "pale yellow-beige, covers entire base"},
  {"name": "tomato_sauce","color_description": "bright red, glossy, in patches"},
  {"name": "mozzarella",  "color_description": "white to light yellow, melted blobs"},
  {"name": "pepperoni",   "color_description": "dark reddish-brown small circles"}
]"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, prompt],
    )
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
# Perspective correction
# ---------------------------------------------------------------------------

def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1e-3:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def correct_perspective(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detects whether the dish appears as an ellipse (tilted camera) and, if so,
    applies an affine transformation to restore the circular shape as if the
    photo had been taken perfectly overhead.

    The correction is based on fitting an ellipse to the dish contour and then
    scaling along the short axis until both axes are equal.

    Deformation is measured as:
        ratio = minor_axis / major_axis   (1.0 = perfect circle, 0.0 = edge-on)

    Decision table:
        ratio >= PERSPECTIVE_CORRECTION_THRESHOLD  → image returned unchanged
        PERSPECTIVE_ABORT_THRESHOLD <= ratio < PERSPECTIVE_CORRECTION_THRESHOLD
                                                   → correction applied
        ratio < PERSPECTIVE_ABORT_THRESHOLD        → ValueError raised
                                                     (tilt too extreme, result
                                                      would be meaningless)

    Args:
        img_bgr: BGR image as np.ndarray, already resized to TARGET_SIZE.

    Returns:
        Corrected BGR image as np.ndarray (same or different shape).

    Raises:
        ValueError: If the detected deformation is too extreme to correct.
    """
    H, W = img_bgr.shape[:2]

    # ------------------------------------------------------------------
    # 1. Isolate the dish contour to fit an ellipse
    # ------------------------------------------------------------------
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges   = cv2.Canny(blurred, 30, 100)

    # Dilate edges so fragmented arcs connect into a closed contour
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[vision] correct_perspective: no contours found, skipping correction.")
        return img_bgr

    # Keep only contours large enough to be the dish (> 10% of image area)
    min_area = 0.10 * H * W
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid:
        print("[vision] correct_perspective: no large contour found, skipping correction.")
        return img_bgr

    # Fit an ellipse to the largest valid contour
    # fitEllipse requires at least 5 points
    dish_contour = max(valid, key=cv2.contourArea)
    if len(dish_contour) < 5:
        print("[vision] correct_perspective: contour too small for ellipse fit, skipping.")
        return img_bgr

    circ = _circularity(dish_contour)
    print(f"[vision] Circularidad del contorno: {circ:.3f}")
    if circ > 0.80:
        print("[vision] Contorno suficientemente circular — saltando corrección de perspectiva.")
        return img_bgr

    (cx, cy), (axis_w, axis_h), angle_deg = cv2.fitEllipse(dish_contour)
    major = max(axis_w, axis_h)
    minor = min(axis_w, axis_h)

    # ------------------------------------------------------------------
    # 2. Evaluate deformation
    # ------------------------------------------------------------------
    if major < 1e-3:
        print("[vision] correct_perspective: degenerate ellipse, skipping correction.")
        return img_bgr

    ratio = minor / major
    print(f"[vision] Perspective ratio: {ratio:.3f}  (minor={minor:.1f}, major={major:.1f}, angle={angle_deg:.1f}°)")

    if ratio >= PERSPECTIVE_CORRECTION_THRESHOLD:
        print("[vision] Dish is close enough to circular — no correction needed.")
        return img_bgr

    if ratio < PERSPECTIVE_ABORT_THRESHOLD:
        raise ValueError(
            f"Perspective deformation is too extreme to correct reliably "
            f"(minor/major ratio = {ratio:.3f}, threshold = {PERSPECTIVE_ABORT_THRESHOLD}). "
            f"Please retake the photo from a more overhead angle. "
            f"Tip: hold the camera directly above the dish, roughly parallel to the table."
        )

    # ------------------------------------------------------------------
    # 3. Build the affine correction matrix
    #
    # Goal: scale the image along the ellipse's minor axis so that
    # minor → major (restoring a circle).
    #
    # Steps (all in matrix form so a single warpAffine call suffices):
    #   T1  translate center of ellipse to origin
    #   R1  rotate so major axis aligns with X
    #   S   scale Y by (major / minor)          ← the actual correction
    #   R2  rotate back
    #   T2  translate back to original center
    # ------------------------------------------------------------------
    scale_factor = major / minor  # how much to stretch the short axis

    # Rotation angle: OpenCV's fitEllipse returns the angle of the *wide* axis
    # relative to horizontal. We rotate so that axis becomes horizontal, then
    # scale vertically, then rotate back.
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # 2×3 affine matrices (homogeneous, last row implicit [0,0,1])
    T1 = np.array([[1, 0, -cx],
                   [0, 1, -cy]], dtype=np.float64)

    R1 = np.array([[cos_t,  sin_t, 0],
                   [-sin_t, cos_t, 0]], dtype=np.float64)

    S  = np.array([[1, 0,            0],
                   [0, scale_factor, 0]], dtype=np.float64)

    R2 = np.array([[cos_t, -sin_t, 0],
                   [sin_t,  cos_t, 0]], dtype=np.float64)

    # Chain: M = R2 · S · R1 · T1
    # We have to work in 3×3 for proper chaining, then drop the last row.
    def to3x3(m2x3):
        return np.vstack([m2x3, [0, 0, 1]])

    M3 = to3x3(R2) @ to3x3(S) @ to3x3(R1) @ to3x3(T1)

    # Translate so the corrected ellipse center stays in frame
    # After the transform, the center maps to M3 · [cx, cy, 1]
    new_cx = M3[0, 0] * cx + M3[0, 1] * cy + M3[0, 2]
    new_cy = M3[1, 0] * cx + M3[1, 1] * cy + M3[1, 2]

    # Compute output canvas size (the scale_factor makes the image taller)
    new_H = int(np.ceil(H * scale_factor))
    new_W = W

    # Shift so the dish center lands in the middle of the new canvas
    T2 = np.array([[1, 0, new_W / 2 - new_cx],
                   [0, 1, new_H / 2 - new_cy],
                   [0, 0, 1]], dtype=np.float64)

    M_final = (T2 @ M3)[:2]   # back to 2×3 for warpAffine

    # ------------------------------------------------------------------
    # 4. Apply the warp
    # ------------------------------------------------------------------
    corrected = cv2.warpAffine(
        img_bgr,
        M_final,
        (new_W, new_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Resize back to TARGET_SIZE so downstream code sees a consistent resolution
    scale_back = TARGET_SIZE / max(new_H, new_W)
    if scale_back < 1.0:
        corrected = cv2.resize(
            corrected,
            (int(new_W * scale_back), int(new_H * scale_back)),
            interpolation=cv2.INTER_AREA,
        )

    print(f"[vision] Perspective corrected: scale_factor={scale_factor:.3f}, "
          f"output shape={corrected.shape[:2]}")
    return corrected


# ---------------------------------------------------------------------------
# Dish boundary detection
# ---------------------------------------------------------------------------

def _detect_dish_mask(img_bgr: np.ndarray, model, image_path: str) -> np.ndarray:
    H, W = img_bgr.shape[:2]

    prompt = """Look at this image. Find the dish, pizza, or food item.
Return ONLY a JSON object with a polygon that traces the boundary of the dish:
{"points": [[x1_frac, y1_frac], [x2_frac, y2_frac], ...]}
Use 20 to 50 points, evenly spaced around the boundary. All values are fractions of image width (x) or height (y), between 0 and 1.
Trace the actual boundary of the food, not a bounding box.
Make sure the polygon includes the outer edge/border (e.g., pizza crust), not just the interior.
No markdown, no explanation, just the JSON."""

    try:
        from PIL import Image as PILImage

        pil_img = PILImage.open(image_path)
        response = model.models.generate_content(model="gemini-2.5-flash", contents=[pil_img, prompt])
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        points = data["points"]
        # Convertir fracciones a píxeles
        pts = np.array(
            [[int(p[0] * W), int(p[1] * H)] for p in points],
            dtype=np.int32,
        )
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        coverage = float(mask.sum()) / float(H * W)
        if len(pts) >= 50 and coverage > 0.95:
            raise ValueError(
                "Gemini devolvió un polígono que cubre prácticamente toda la imagen "
                "(no pudo trazar el borde real del plato/pizza)."
            )
        print(
            f"[vision] Gemini polygon mask: {len(pts)} points, "
            f"coverage={coverage:.3f}"
        )
        return mask.astype(bool)
    except Exception as e:
        print(f"[vision] Gemini dish detection failed ({e}), using full image")
        return np.ones((H, W), dtype=bool)


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
        ValueError: If the camera angle is so extreme that perspective
                    correction would produce unreliable results.
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
    # 3. Correct perspective distortion (tilted camera → ellipse → circle)
    #    Raises ValueError if the tilt is too extreme to recover from.
    # ------------------------------------------------------------------
    img_bgr = correct_perspective(img_bgr)
    H, W = img_bgr.shape[:2]   # dimensions may have changed after correction

    model = _init_model()

    # ------------------------------------------------------------------
    # 4. Detect dish boundary
    # ------------------------------------------------------------------
    dish_mask = _detect_dish_mask(img_bgr, model, image_path)   # bool (H, W)

    # ------------------------------------------------------------------
    # 5. Ask Gemini what ingredients are in the dish
    # ------------------------------------------------------------------
    ingredients = _call_gemini(model, image_path)  # list[dict]

    K = len(ingredients)
    ingredient_labels: dict[int, str] = {i: ing["name"] for i, ing in enumerate(ingredients)}

    # ------------------------------------------------------------------
    # 6. Per-ingredient color segmentation (OpenCV HSV)
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
    # 7. Special handling: ensure "base" ingredient (channel 0) covers
    #    the whole dish — fill any pixel that has no detected ingredient
    # ------------------------------------------------------------------
    pixel_total = ingredient_map.sum(axis=2)             # (H, W)
    unclaimed = (pixel_total < 0.05) & dish_mask         # nearly zero inside dish
    ingredient_map[unclaimed, 0] = 0.3                   # assign to base channel (low confidence)

    # ------------------------------------------------------------------
    # 8. Normalize so channels sum to 1.0 at every dish pixel
    # ------------------------------------------------------------------
    dish_pixels = int(dish_mask.sum())
    if dish_pixels > 0:
        for k in range(K):
            channel = ingredient_map[:, :, k]
            coverage = float(((channel > 0.1) & dish_mask).sum()) / dish_pixels
            mean_val = float(channel[dish_mask].mean())
            name = ingredient_labels.get(k, f"channel_{k}")
            print(f"[vision] Canal {k} ({name}): cobertura={coverage:.1%}, media={mean_val:.3f}")

    # Base ingredient is cumulative over the whole dish (before normalization).
    ingredient_map[:, :, 0][dish_mask] = 1.0
    ingredient_map = _normalize_map(ingredient_map, dish_mask)

    # ------------------------------------------------------------------
    # 9. Filter channels with insufficient mean coverage (after normalize)
    # ------------------------------------------------------------------
    if dish_pixels > 0 and K > 1:
        to_drop: list[int] = []
        for k in range(K):
            mean_k = float(ingredient_map[:, :, k][dish_mask].mean())
            if mean_k < MIN_CHANNEL_MEAN_COVERAGE:
                if k == 0:
                    # Base channel is never removed (README convention).
                    print(
                        f"[vision] Canal {k} ({ingredient_labels[k]}) marcado para eliminación "
                        f"pero preservado (base): media={mean_k:.4f} < umbral"
                    )
                    continue
                to_drop.append(k)
                print(
                    f"[vision] Canal {k} ({ingredient_labels[k]}) eliminado: "
                    f"media={mean_k:.4f} < umbral"
                )

        if to_drop:
            keep = [k for k in range(K) if k not in set(to_drop)]
            ingredient_map = ingredient_map[:, :, keep]
            ingredient_labels = {new_i: ingredient_labels[old_i] for new_i, old_i in enumerate(keep)}
            ingredient_map = _normalize_map(ingredient_map, dish_mask)
            K = ingredient_map.shape[-1]

    # ------------------------------------------------------------------
    # 10. Validate output before returning
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