"""
FairSlice — FastAPI Backend
Owner: Quirce

Endpoints:
  POST /api/slice  — full pipeline: image → segment → partition → render
  GET  /health     — liveness check
  GET  /           — serves the PWA frontend (static files)
"""

import base64
import io
import os
import sys
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Make src/fair-slice importable regardless of where uvicorn is launched from
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "fair-slice")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Pipeline modules are imported lazily inside the endpoint so that the server
# can start and serve the UI even if optional dependencies (vertexai, cv2, scipy)
# are not yet installed.

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FairSlice",
    version="1.0.0",
    description="Mathematically fair dish division via computer vision.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# /api/slice  — main pipeline endpoint
# ---------------------------------------------------------------------------
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@app.post("/api/slice")
async def slice_dish(
    image: UploadFile = File(..., description="Dish photo (JPG / PNG / WebP)"),
    n_people: int = Form(..., ge=2, le=8, description="Number of people (2–8)"),
    mode: str = Form("free", description="'free' (Voronoi) or 'radial'"),
):
    """
    Full pipeline:
        uploaded photo
          → vision.segment_dish()     — per-pixel ingredient map
          → partition.compute_partition() — N fair regions
          → visualize.render_partition()  — overlay image
          → JSON response with base64 PNG + stats
    """
    # --- Validate inputs ---
    content_type = (image.content_type or "").lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            400,
            detail=(
                f"Unsupported image type '{content_type}'. "
                f"Accepted: {', '.join(sorted(ALLOWED_TYPES))}"
            ),
        )

    if mode not in ("free", "radial"):
        raise HTTPException(400, detail="mode must be 'free' or 'radial'")

    # --- Save upload to a temp file so OpenCV can read it ---
    suffix = ".png" if "png" in content_type else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        # Lazy imports so the server starts even without all ML deps installed
        from vision import segment_dish
        from partition import compute_partition
        from visualize import render_partition

        # Step 1 — Vision
        ingredient_map, labels = segment_dish(tmp_path)

        # Step 2 — Partition
        result = compute_partition(ingredient_map, n_people, mode=mode)

        # Step 3 — Visualise
        output_img = render_partition(
            image_path=tmp_path,
            masks=result["masks"],
            scores=result["scores"],
            ingredient_labels=labels,
            fairness=result["fairness"],
        )

        # Encode result as base64 PNG
        buf = io.BytesIO()
        output_img.save(buf, format="PNG", optimize=True)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return JSONResponse(
            {
                "result_image": img_b64,
                "fairness": float(result["fairness"]),
                "scores": result["scores"].tolist(),
                "ingredient_labels": {str(k): v for k, v in labels.items()},
            }
        )

    except NotImplementedError as e:
        raise HTTPException(501, detail=f"Module not yet implemented: {e}")
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Pipeline error: {type(e).__name__}: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "FairSlice"}


# ---------------------------------------------------------------------------
# Serve PWA frontend — mount LAST so API routes take priority
# ---------------------------------------------------------------------------
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
