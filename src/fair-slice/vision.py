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
import sys
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.cloud import secretmanager as gsecretmanager
from google.cloud import vision as gvision
from dotenv import load_dotenv
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE = 500           # Resize longest edge to this before any processing
MAX_INGREDIENTS = 8         # Hard cap — group beyond this
MIN_INGREDIENTS = 2         # Gemini must return at least this many
MIN_CHANNEL_MEAN_COVERAGE = 0.008  # canales con media < esto sobre dish_mask se eliminan


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def _init_model() -> genai.Client:
    env_key = os.environ.get("GOOGLE_AI_API_KEY")
    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    print(
        "[fairslice] _init_model env "
        f"GOOGLE_AI_API_KEY={'set' if env_key else 'missing'}"
        + (f"(len={len(env_key)})" if env_key else "")
        + f", GOOGLE_CLOUD_PROJECT={'set' if env_project else 'missing'}",
        file=sys.stderr,
        flush=True,
    )
    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        secret_name = os.environ.get("GOOGLE_AI_API_KEY_SECRET", "google-ai-api-key")
        if project_id:
            try:
                client = gsecretmanager.SecretManagerServiceClient()
                name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
                api_key = client.access_secret_version(name=name).payload.data.decode("utf-8").strip()
                print(
                    "[fairslice] _init_model loaded key from Secret Manager "
                    f"(secret={secret_name}, len={len(api_key)})",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as e:
                raise EnvironmentError(
                    "GOOGLE_AI_API_KEY not set (and Secret Manager fallback failed: "
                    f"{type(e).__name__}: {e})"
                ) from e
        if not api_key:
            raise EnvironmentError("GOOGLE_AI_API_KEY not set")
    return genai.Client(api_key=api_key)


def _detect_dish_mask_vision_api(img_bgr: np.ndarray, image_path: str) -> np.ndarray:
    """
    Uses Google Cloud Vision Object Localization to detect the dish boundary.
    Returns a bool mask (H, W). Falls back to center ellipse if detection fails.
    """
    H, W = img_bgr.shape[:2]

    try:
        client = gvision.ImageAnnotatorClient()
        with open(image_path, "rb") as f:
            content = f.read()
        image = gvision.Image(content=content)
        response = client.object_localization(image=image)

        # Look for food-related objects: pizza, food, dish, pastry, etc.
        FOOD_LABELS = {
            "pizza", "food", "dish", "cake", "pie", "pastry",
            "bread", "meal", "plate", "baked goods", "cuisine"
        }

        best = None
        best_score = 0.0
        for obj in response.localized_object_annotations:
            label = obj.name.lower()
            if any(food in label for food in FOOD_LABELS) and obj.score > best_score:
                best = obj
                best_score = obj.score

        # Fallback: if no food label found, use the highest-confidence object
        if best is None and response.localized_object_annotations:
            best = max(response.localized_object_annotations, key=lambda o: o.score)

        if best is None:
            raise ValueError("No objects detected by Vision API")

        # Convert normalized vertices to pixel coordinates
        pts = np.array(
            [[int(v.x * W), int(v.y * H)]
             for v in best.bounding_poly.normalized_vertices],
            dtype=np.int32,
        )
        print(f"[vision] Vision API bounding box raw: {pts.tolist()}")

        # Usar directamente el polígono de Vision API sin convertir a elipse
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        # Si el polígono es solo un rectángulo (4 puntos), añadir más puntos
        # interpolando el contorno para suavizar
        if len(pts) == 4:
            # Calcular centro y semiejes del bounding box
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            rx = int((x2 - x1) / 2 * 0.96)
            ry = int((y2 - y1) / 2 * 0.96)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)

        coverage = float(mask.sum()) / float(H * W)
        print(f"[vision] Google Vision mask: label='{best.name}' "
              f"score={best_score:.2f}, coverage={coverage:.3f}, "
              f"center=({cx},{cy}), rx={rx}, ry={ry}")

        if coverage < 0.10 or coverage > 0.85:
            raise ValueError(f"Mask coverage {coverage:.2%} out of expected range")

        return mask.astype(bool)

    except Exception as e:
        print(f"[vision] Google Vision detection failed ({e}), using center ellipse fallback")
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(mask, (W // 2, H // 2), (int(W * 0.42), int(H * 0.42)), 0, 0, 360, 1, -1)
        return mask.astype(bool)


def _call_gemini_ingredients(client: genai.Client, image_path: str) -> list[dict]:
    """Ask Gemini ONLY for ingredient identification. Dish boundary is handled by Vision API."""
    image = Image.open(image_path)
    prompt = """Analyze this dish photo carefully.
Identify every distinct ingredient or component visible:
- The base (bread, dough, rice, pastry…) MUST be FIRST
- Sauces, toppings, vegetables, meats, cheeses
- Group similar things (e.g., all cheese → "mozzarella")
- Return between 2 and 8 ingredients
- For each, describe its color/appearance for OpenCV HSV detection
- The pizza crust/border (the outer ring without toppings) is ALWAYS an ingredient called "crust". Include it even if it seems obvious. It will be used so every person gets a piece to hold.

Return ONLY a valid JSON array, no markdown, no extra text:
[
  {"name": "dough", "color_description": "pale yellow-beige base"},
  {"name": "tomato_sauce", "color_description": "bright red glossy patches"}
]"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, prompt],
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    ingredients = json.loads(raw)
    if not isinstance(ingredients, list) or len(ingredients) < MIN_INGREDIENTS:
        raise ValueError(f"Gemini returned fewer than {MIN_INGREDIENTS} ingredients")
    return ingredients[:MAX_INGREDIENTS]


def _fallback_dish_mask(H: int, W: int) -> np.ndarray:
    """Center ellipse fallback if Gemini fails."""
    print("[vision] Using fallback center ellipse mask")
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, (W // 2, H // 2), (int(W * 0.42), int(H * 0.42)), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _mask_from_image_foreground(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Deterministic dish candidate from the photo itself.

    This is intentionally tuned for the common demo case: a pizza/food item on a
    clean white or light background. The threshold finds non-background pixels,
    then morphology closes cheese/topping holes into one filled food silhouette.
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    non_background = (sat > 18) | (val < 245)
    non_background = non_background.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    non_background = cv2.morphologyEx(non_background, cv2.MORPH_CLOSE, kernel, iterations=2)
    non_background = cv2.morphologyEx(non_background, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(non_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.10 * H * W:
            continue
        M = cv2.moments(c)
        if M["m00"] < 1:
            continue
        cx = M["m10"] / M["m00"] / W
        cy = M["m01"] / M["m00"] / H
        if 0.10 < cx < 0.90 and 0.10 < cy < 0.90:
            valid.append(c)

    if not valid:
        return None

    best = max(valid, key=cv2.contourArea)
    candidate = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(candidate, [best], -1, 1, thickness=-1)
    candidate = _largest_component_mask(candidate)

    coverage = float(candidate.sum()) / float(H * W)
    if not 0.10 <= coverage <= 0.80:
        return None

    return candidate


def _refine_dish_mask(img_bgr: np.ndarray, dish_mask: np.ndarray) -> np.ndarray:
    image_mask = _mask_from_image_foreground(img_bgr)
    if image_mask is None:
        return dish_mask

    H, W = dish_mask.shape
    old_coverage = float(dish_mask.sum()) / float(H * W)
    new_coverage = float(image_mask.sum()) / float(H * W)
    overlap = float((dish_mask & image_mask).sum()) / float(max(1, image_mask.sum()))

    # Prefer the image-derived silhouette when Gemini/fallback spills outside the food.
    if overlap > 0.55 and (old_coverage > new_coverage * 1.08 or old_coverage > 0.55):
        print(
            f"[vision] Refined dish mask from image foreground: "
            f"{old_coverage:.3f} -> {new_coverage:.3f}"
        )
        return image_mask

    return dish_mask


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
    Google Vision detects the dish boundary; Gemini identifies ingredients.
    No perspective correction — detection handles tilted photos directly.
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

    # Guardar imagen resizada en temp para Vision API
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    # Detección de contorno: Google Vision (determinista, precisa)
    dish_mask = _detect_dish_mask_vision_api(img_bgr, tmp_path)

    # Identificación de ingredientes: Gemini (semántica)
    client = _init_model()
    ingredients = _call_gemini_ingredients(client, tmp_path)
    if not any("crust" in ing["name"].lower() for ing in ingredients):
        ingredients.append({"name": "crust", "color_description": "golden brown outer ring, thick edge"})
        if len(ingredients) > MAX_INGREDIENTS:
            ingredients = ingredients[:MAX_INGREDIENTS]

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

    # Crust geométrico: anillo exterior del dish_mask
    # Independiente del color — el crust es siempre el borde exterior
    crust_idx = next(
        (k for k, name in ingredient_labels.items() if "crust" in name.lower()),
        None
    )
    if crust_idx is not None:
        # Erosionar la máscara del plato para obtener el interior
        crust_width = max(8, int(min(H, W) * 0.04))  # ~4% del tamaño
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (crust_width * 2, crust_width * 2)
        )
        inner_mask = cv2.erode(
            dish_mask.astype(np.uint8), kernel, iterations=1
        ).astype(bool)
        outer_ring = dish_mask & ~inner_mask  # píxeles del borde exterior
        # Asignar densidad alta al crust solo en el anillo exterior
        ingredient_map[:, :, crust_idx] = np.where(outer_ring, 1.0, 0.0)
        print(f"[vision] Crust geométrico: {outer_ring.sum()} píxeles "
              f"({outer_ring.sum()/dish_mask.sum():.1%} del plato)")

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
                if "crust" in ingredient_labels[k].lower():
                    print(f"[vision] Canal {k} ({ingredient_labels[k]}) preservado (crust): mean={mean_k:.4f}")
                    continue  # nunca eliminar crust
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