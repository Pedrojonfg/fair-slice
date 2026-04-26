"""
Module: Visualization (Overlay & Output)
Owner: Quirce

Responsible for:
  - Taking the original photo + partition masks
  - Rendering colored overlays with cut lines
  - Generating the final output image + fairness stats

See README.md Section 4 for the full contract specification.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Color palette — up to 8 people (R, G, B, overlay_alpha)
# ---------------------------------------------------------------------------
PERSON_COLORS: list[tuple[int, int, int, int]] = [
    ( 78, 205, 196, 110),   # teal
    (255, 107, 107, 110),   # coral
    (149, 225, 211, 105),   # mint
    (255, 190,  60, 100),   # amber
    (108,  92, 231, 110),   # indigo
    (253, 150,  68, 110),   # orange
    ( 72, 219, 251, 110),   # sky
    (255, 159, 243, 110),   # lilac
]

BOUNDARY_THICKNESS = 3  # pixels

# ---------------------------------------------------------------------------
# Palette recommended for Streamlit overlay (RGBA, alpha=120)
# ---------------------------------------------------------------------------

PALETTE: list[tuple[int, int, int, int]] = [
    (255, 99, 71, 120),  # tomato
    (100, 149, 237, 120),  # cornflower blue
    (144, 238, 144, 120),  # light green
    (255, 215, 0, 120),  # gold
    (218, 112, 214, 120),  # orchid
    (255, 165, 0, 120),  # orange
    (135, 206, 235, 120),  # sky blue
    (240, 128, 128, 120),  # light coral
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_font(size: int):
    """Try common system font paths, fall back to the default bitmap font."""
    paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Windows/Fonts/arial.ttf",
        "/Windows/Fonts/segoeui.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    """Return (width, height) of a text string."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_text_centered(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    text: str,
    font,
    color: tuple,
    shadow_color: tuple = (0, 0, 0),
) -> None:
    """Draw text centred at (cx, cy) with a simple drop-shadow."""
    w, h = _text_size(draw, text, font)
    x, y = cx - w // 2, cy - h // 2
    # shadow
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        draw.text((x + dx, y + dy), text, fill=shadow_color, font=font)
    draw.text((x, y), text, fill=color, font=font)


def _foreground_clip_mask(base: Image.Image, size: tuple[int, int]) -> np.ndarray | None:
    W, H = size
    rgb = np.array(base.convert("RGB").resize((W, H), Image.LANCZOS))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask = ((sat > 18) | (val < 245)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    clip = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(clip, [max(valid, key=cv2.contourArea)], -1, 1, thickness=-1)
    coverage = float(clip.sum()) / float(H * W)
    if not 0.10 <= coverage <= 0.85:
        return None

    return clip.astype(bool)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_partition(
    image_path: str,
    masks: list[np.ndarray],
    scores: np.ndarray,
    ingredient_labels: dict[int, str],
    fairness: float,
) -> Image.Image:
    """
    Renders the partition overlay on the original dish photo.

    Args:
        image_path:        Path to the original photo.
        masks:             List of N boolean masks shape (H, W).
        scores:            (N, K) array of ingredient proportions per person.
        ingredient_labels: dict mapping channel index → ingredient name.
        fairness:          Overall fairness score in [0, 1].

    Returns:
        RGB PIL Image with:
          - Semi-transparent coloured regions (one colour per person)
          - White cut lines at region boundaries
          - Person number labels on each slice
          - Legend strip: fairness bar + per-person ingredient breakdown
    """
    N = len(masks)
    if N == 0:
        raise ValueError("masks list is empty")

    # ------------------------------------------------------------------
    # 1. Load base image and match it to the mask resolution
    # ------------------------------------------------------------------
    base = Image.open(image_path).convert("RGBA")

    mask_h, mask_w = masks[0].shape[:2]
    if base.size != (mask_w, mask_h):
        base = base.resize((mask_w, mask_h), Image.LANCZOS)

    W, H = base.size  # (width, height)

    # ------------------------------------------------------------------
    # 2. Build a (H, W) label map  — value = person index, -1 = background
    # ------------------------------------------------------------------
    label_map = np.full((H, W), -1, dtype=np.int16)
    for i, mask in enumerate(masks):
        m = np.asarray(mask, dtype=bool)
        if m.shape != (H, W):
            pil_m = Image.fromarray((m.astype(np.uint8) * 255)).resize(
                (W, H), Image.NEAREST
            )
            m = np.array(pil_m) > 127
        label_map[m] = i

    # ------------------------------------------------------------------
    # 3. Coloured overlay (one transparent layer per person)
    # ------------------------------------------------------------------
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for i in range(N):
        r, g, b, a = PERSON_COLORS[i % len(PERSON_COLORS)]
        region = np.zeros((H, W, 4), dtype=np.uint8)
        region[label_map == i] = [r, g, b, a]
        overlay = Image.alpha_composite(overlay, Image.fromarray(region))

    # ------------------------------------------------------------------
    # 4. Boundary lines — neon glow: soft outer halo + crisp 1px centre
    # ------------------------------------------------------------------
    boundary = np.zeros((H, W), dtype=bool)

    diff_h = label_map[:, :-1] != label_map[:, 1:]
    valid_h = (label_map[:, :-1] >= 0) | (label_map[:, 1:] >= 0)
    boundary[:, :-1] |= diff_h & valid_h
    boundary[:, 1:]  |= diff_h & valid_h

    diff_v = label_map[:-1, :] != label_map[1:, :]
    valid_v = (label_map[:-1, :] >= 0) | (label_map[1:, :] >= 0)
    boundary[:-1, :] |= diff_v & valid_v
    boundary[1:, :]  |= diff_v & valid_v

    def _expand(mask, times):
        m = mask.copy()
        for _ in range(times):
            m = (
                m
                | np.roll(m,  1, axis=0) | np.roll(m, -1, axis=0)
                | np.roll(m,  1, axis=1) | np.roll(m, -1, axis=1)
            )
        return m

    outer = _expand(boundary, 5)   # ~11px wide halo
    mid   = _expand(boundary, 2)   # ~5px inner glow
    inner = _expand(boundary, 1)   # ~3px tight glow

    bnd_arr = np.zeros((H, W, 4), dtype=np.uint8)
    bnd_arr[outer & ~mid]    = [255, 255, 255, 18]   # soft outer bloom
    bnd_arr[mid   & ~inner]  = [255, 255, 255, 55]   # inner glow
    bnd_arr[inner & ~boundary] = [255, 255, 255, 120] # tight halo
    bnd_arr[boundary]          = [255, 255, 255, 255] # crisp 1px centre line
    overlay = Image.alpha_composite(overlay, Image.fromarray(bnd_arr))

    # ------------------------------------------------------------------
    # 5. Composite overlay onto base image
    # ------------------------------------------------------------------
    result_rgb = Image.alpha_composite(base, overlay).convert("RGB")
    draw_dish = ImageDraw.Draw(result_rgb)

    # Person number labels at each slice centroid
    font_num = _load_font(32)
    for i in range(N):
        ys, xs = np.where(label_map == i)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        _draw_text_centered(
            draw_dish, cx, cy,
            str(i + 1),
            font_num,
            color=(255, 255, 255),
            shadow_color=(0, 0, 0),
        )

    return result_rgb


def render_overlay(image_path: str, masks: list[np.ndarray], n_people: int) -> Image.Image:
    base = Image.open(image_path).convert("RGBA")
    # Resize to match mask dimensions
    H, W = np.asarray(masks[0]).shape
    base = base.resize((W, H), Image.LANCZOS)
    clip_mask = _foreground_clip_mask(base, (W, H))

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    overlay_arr = np.array(overlay)

    aligned_masks: list[np.ndarray] = []
    for i, mask in enumerate(masks[:n_people]):
        m = np.asarray(mask, dtype=bool)
        if m.shape != (H, W):
            raise ValueError(f"Mask shape {m.shape} does not match expected {(H, W)}")
        if clip_mask is not None:
            m = m & clip_mask
        aligned_masks.append(m)

        color = PALETTE[i % len(PALETTE)]
        overlay_arr[m] = color  # solo píxeles donde mask es True

    overlay = Image.fromarray(overlay_arr, "RGBA")
    result = Image.alpha_composite(base, overlay)

    # Dibujar números centrados en cada región
    draw = ImageDraw.Draw(result)
    # Slightly bigger, image-relative numbers for readability.
    # 56px on a ~700px-wide image is ~8%; bump to ~9% while clamping extremes.
    font_size = int(np.clip(round(min(W, H) * 0.09), 44, 96))
    font = _load_font(font_size)
    for i, m in enumerate(aligned_masks):
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        text = str(i + 1)
        draw.text(
            (cx, cy),
            text,
            fill=(255, 255, 255, 255),
            anchor="mm",
            font=font,
            stroke_width=max(2, font_size // 18),
            stroke_fill=(0, 0, 0, 255),
        )

    return result.convert("RGB")


# ---------------------------------------------------------------------------
# Quick smoke test (python visualize.py <image_path>)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize.py <image_path> [n_people]")
        sys.exit(1)

    img_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    # Load image to get dimensions
    img = Image.open(img_path)
    H, W = img.height, img.width

    # Fake uniform masks for smoke-testing
    label_map = np.zeros((H, W), dtype=int)
    cx, cy = W // 2, H // 2
    for y in range(H):
        for x in range(W):
            angle = np.arctan2(y - cy, x - cx)
            label_map[y, x] = int((angle + np.pi) / (2 * np.pi) * n) % n

    fake_masks = [label_map == i for i in range(n)]
    fake_scores = np.ones((n, 3), dtype=np.float32) / n
    fake_labels = {0: "dough", 1: "sauce", 2: "pepperoni"}
    fake_fairness = 0.95

    out = render_partition(img_path, fake_masks, fake_scores, fake_labels, fake_fairness)
    out.save("visualize_test_output.png")
    print("[visualize] Saved to visualize_test_output.png")
