# 🍕 FairSlice

**The mathematically fairest way to divide any dish.**

FairSlice takes a photo of a dish (pizza, cake, paella, etc.), identifies every ingredient using computer vision, and computes the optimal partition into N connected regions so that every person gets an equal share of *every single ingredient* — including bread, crust, and sauce.

Built for the **Tech Roulette Challenge** — Google Developer Group, IE University (Spring 2026).  
**Theme:** Cooking · **Suggested API:** Vertex AI





# PEDRO: Ni caso a Claude, no hace falta usar Docker, podemos deployear esto a google cloud sin necesidad de docker. La timeline es informativa
---

## Table of Contents

1. [How It Works](#1-how-it-works)
2. [Module 1: Vision — Ingredient Segmentation](#2-module-1-vision--ingredient-segmentation)
3. [Module 2: Partition — Fair Division Algorithm](#3-module-2-partition--fair-division-algorithm)
4. [Module 3: Visualization & UI](#4-module-3-visualization--ui)
5. [Pipeline Assembly](#5-pipeline-assembly)
6. [Development Setup](#6-development-setup)
7. [Deployment](#7-deployment)
8. [Timeline](#8-timeline)

---

## 1. How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  📷 Photo   │────▶│  🔬 Vision       │────▶│  🧮 Partition   │────▶│  🎨 Overlay  │
│  of a dish  │     │  segment_dish()  │     │  compute_part() │     │  render()    │
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                     Returns (H,W,K)           Returns N masks        Returns image
                     ingredient map            + fairness scores      with cut lines
```

**Example:** You photograph a pizza with uneven pepperoni distribution. FairSlice detects 5 ingredients (dough, crust, sauce, mozzarella, pepperoni), computes a Voronoi partition into 4 connected regions, and returns an image showing exactly where to cut — so nobody can complain they got less pepperoni.

---

## 2. Module 1: Vision — Ingredient Segmentation

**Owner:** Pablo 
**File:** `src/vision.py`  
**Dependencies:** `google-cloud-aiplatform`, `opencv-python-headless`, `numpy`

### 2.1 What this module does

Takes a photo of a dish and returns a **per-pixel ingredient density map**: a 3D numpy array where each "channel" represents one ingredient, and each pixel's value tells you how much of that ingredient is at that position.

### 2.2 Function signature (DO NOT CHANGE)

```python
def segment_dish(image_path: str) -> tuple[np.ndarray, dict[int, str]]:
```

### 2.3 Output specification

#### `ingredient_map: np.ndarray`

| Property   | Requirement                                                |
|------------|------------------------------------------------------------|
| Shape      | `(H, W, K)` where H, W = image dimensions, K = number of detected ingredients |
| Dtype      | `np.float32`                                               |
| Value range| `[0.0, 1.0]` at every position                            |
| Channel sum| `sum over k ≈ 1.0` for every pixel inside the dish         |
| Outside dish| `0.0` in all channels for pixels outside the dish boundary |
| Min K      | At least 2 (e.g., dough + one topping)                    |
| Max K      | No hard limit, but 3-8 is the realistic range              |

#### `ingredient_labels: dict[int, str]`

Maps each channel index to a human-readable ingredient name.

### 2.4 Concrete example

Imagine a 4x4 pixel "pizza" (absurdly small, just for illustration). It has 3 ingredients: dough (channel 0), sauce (channel 1), and pepperoni (channel 2).

```python
import numpy as np

# Shape: (4, 4, 3)
ingredient_map = np.array([
    # Row 0 (top edge — crust area)
    [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.8, 0.2, 0.0], [0.9, 0.1, 0.0]],
    # Row 1
    [[0.8, 0.2, 0.0], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3], [0.8, 0.2, 0.0]],
    # Row 2
    [[0.8, 0.2, 0.0], [0.3, 0.5, 0.2], [0.1, 0.2, 0.7], [0.8, 0.2, 0.0]],
    # Row 3 (bottom edge — crust area)
    [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.8, 0.2, 0.0], [0.9, 0.1, 0.0]],
], dtype=np.float32)

ingredient_labels = {
    0: "dough",
    1: "tomato_sauce",
    2: "pepperoni"
}

# Validation checks:
assert ingredient_map.shape == (4, 4, 3)
assert ingredient_map.dtype == np.float32
assert ingredient_map.min() >= 0.0
assert ingredient_map.max() <= 1.0
assert np.allclose(ingredient_map.sum(axis=2), 1.0, atol=0.05)
```

**How to read this:** At pixel (2, 3), the values are `[0.8, 0.2, 0.0]` — that's 80% dough, 20% sauce, 0% pepperoni. That's a crust pixel. At pixel (2, 2), the values are `[0.1, 0.2, 0.7]` — mostly pepperoni. The partition algorithm will try to ensure every person gets equal total pepperoni, equal total dough, etc.

### 2.5 Implementation strategy

There are two viable approaches. Pick whichever works faster.

#### Approach A: Gemini multimodal + color segmentation (RECOMMENDED for speed)

1. Send the photo to **Gemini via Vertex AI** with a prompt like:
   ```
   Analyze this photo of a dish. List every visible ingredient 
   and describe its color/appearance. Return as JSON:
   {"ingredients": [{"name": "pepperoni", "color_description": "dark red circles"}, ...]}
   ```
2. Use the Gemini response to know *what* ingredients exist.
3. Use **OpenCV color-based segmentation** (HSV thresholding, or k-means on pixel colors) to create the actual per-pixel map. Gemini tells you what to look for, OpenCV does the pixel-level work.
4. Optional: use **SAM (Segment Anything Model)** for cleaner boundaries.

```python
# Pseudo-code for Approach A
from google.cloud import aiplatform
import cv2
import numpy as np

def segment_dish(image_path: str) -> tuple[np.ndarray, dict[int, str]]:
    # 1. Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    
    # 2. Ask Gemini what ingredients are visible
    # (see Vertex AI docs for exact API call)
    ingredients = call_gemini_for_ingredients(image_path)
    # Returns something like: ["dough", "crust", "tomato_sauce", "mozzarella", "pepperoni"]
    
    K = len(ingredients)
    ingredient_map = np.zeros((H, W, K), dtype=np.float32)
    labels = {i: name for i, name in enumerate(ingredients)}
    
    # 3. For each ingredient, create a mask using color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for k, ingredient_name in enumerate(ingredients):
        mask = segment_by_color(hsv, ingredient_name)  # your logic here
        ingredient_map[:, :, k] = mask.astype(np.float32)
    
    # 4. Normalize channels to sum to 1
    total = ingredient_map.sum(axis=2, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    ingredient_map /= total
    
    return ingredient_map, labels
```

#### Approach B: Gemini with spatial grid (slower, more accurate)

1. Divide the image into a grid (e.g., 50x50 cells).
2. For each cell (or batch of cells), send the cropped region to Gemini and ask what ingredients are present and in what proportion.
3. Assemble into the full (H, W, K) map by interpolating between grid cells.

This is more accurate but much slower and burns more API credits. Only use if Approach A fails to segment properly.

### 2.6 Vertex AI setup

```python
# Authentication
# Set environment variable: GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json
# Or run: gcloud auth application-default login

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image

vertexai.init(project="YOUR_PROJECT_ID", location="us-central1")
model = GenerativeModel("gemini-1.5-flash")  # or gemini-1.5-pro

def call_gemini_for_ingredients(image_path: str) -> list[str]:
    image = Image.load_from_file(image_path)
    response = model.generate_content([
        image,
        """Analyze this dish photo. Identify every distinct ingredient 
        or component visible (including base/bread/dough, sauces, toppings).
        Return ONLY a JSON array of ingredient names, nothing else.
        Example: ["dough", "tomato_sauce", "mozzarella", "pepperoni", "olive"]"""
    ])
    import json
    return json.loads(response.text)
```

### 2.7 Testing without the real API

Use the mock data generator while developing:

```bash
python tests/generate_mock_data.py
```

This creates `tests/fixtures/mock_data/pizza_mock_500.npy` — a 500x500x5 synthetic ingredient map that you (and Pedro's partition module) can use immediately.

### 2.8 Key tips

- **Detect the dish boundary first.** Before segmenting ingredients, create a binary mask of "dish vs background" (table, plate, box). Set everything outside to zero. A circle detector (Hough transform) works well for pizzas and round dishes.
- **"Dough" or "base" is always channel 0.** The base ingredient that covers the entire dish surface should be channel 0. This is not a hard technical requirement, but it keeps things consistent.
- **Don't over-segment.** If there are 12 slightly different toppings, consider grouping similar ones. 3-6 ingredient channels is the sweet spot.
- **Resolution matters.** Resize the input image to ~500x500 before processing. The partition algorithm doesn't need 4K resolution and it'll run 100x faster.

---

## 3. Module 2: Partition — Fair Division Algorithm

**Owner:** Pedro  
**File:** `src/partition.py`  
**Dependencies:** `numpy`, `scipy`

### 3.1 What this module does

Given the ingredient map from the Vision module, computes the optimal division of the dish into N connected regions where each person gets the fairest possible share of every ingredient.

### 3.2 Function signature (DO NOT CHANGE)

```python
def compute_partition(
    ingredient_map: np.ndarray,
    n_people: int,
    mode: str = "free"
) -> dict:
```

### 3.3 Output specification

```python
{
    "masks": [              # list of N boolean arrays
        np.ndarray(H, W),  # True where person 0's share is
        np.ndarray(H, W),  # True where person 1's share is
        ...
    ],
    "scores": np.ndarray,   # shape (N, K) — proportion of each ingredient per person
    "fairness": float,      # 0.0 to 1.0 — overall fairness score
    "seeds": np.ndarray     # shape (N, 2) — Voronoi seed positions (for mode="free")
}
```

**Critical constraints on masks:**
- Every pixel that's part of the dish must belong to exactly one mask
- No overlaps: `sum(masks) == 1` for all dish pixels
- Each mask must be a single connected region (no fragments)
- Pixels outside the dish are False in all masks

### 3.4 Algorithm: Weighted Power Diagram (mode="free")

Core approach: iterative Lloyd's algorithm with per-ingredient density weighting.

1. Initialize N seed points (e.g., evenly spaced on the dish boundary)
2. Compute Power Diagram (weighted Voronoi) — each pixel assigned to nearest seed, adjusted by seed weights
3. For each region, compute the integral of each ingredient channel
4. Compute cost = variance of ingredient proportions across regions
5. Update seed positions (move toward weighted centroids) and weights
6. Repeat until convergence or max iterations
7. Return final masks

### 3.5 Algorithm: Optimal Radial (mode="radial")

Fallback approach. Faster, simpler, but less fair for asymmetric distributions.

1. Find optimal center point (cx, cy) by searching the dish area
2. Project ingredient density onto angular axis (integrate radially)
3. Find N-1 angles that equalize the cumulative density
4. Generate sector masks

### 3.6 Fairness metric

```python
# For each ingredient k, the ideal share per person is 1/N
# fairness = 1 - max deviation from ideal across all people and ingredients
ideal = 1.0 / n_people
max_deviation = max(abs(scores[i, k] - ideal) for all i, k)
fairness = 1.0 - (max_deviation / ideal)  # normalized to [0, 1]
```

---

## 4. Module 3: Visualization & UI

**Owner:** Quirce
**File:** `src/visualize.py` + `app/`  
**Dependencies:** `Pillow`, `FastAPI`, `React (frontend)`

### 4.1 Backend visualization

Takes masks + original photo → produces an output image with:
- Semi-transparent colored overlay (one color per region/person)
- White lines at region boundaries
- Per-person ingredient breakdown shown as a legend or sidebar

### 4.2 Frontend (PWA)

- Camera access to take a photo
- Number selector (how many people: 2-8)
- Mode toggle (radial / free)
- Display result with the overlay image
- Show fairness score and per-person breakdown

### 4.3 FastAPI endpoints

```python
POST /api/slice
  Body: multipart/form-data
    - image: file (JPG/PNG)
    - n_people: int (2-8)
    - mode: str ("free" | "radial")
  Response: JSON
    {
      "result_image": "base64-encoded PNG",
      "fairness": 0.94,
      "scores": [[0.26, 0.24, 0.25], [0.24, 0.26, 0.25], ...],
      "ingredient_labels": {"0": "dough", "1": "sauce", ...}
    }
```

---

## 5. Pipeline Assembly

All three modules connect through `src/main.py`. The contract is:

```
vision.segment_dish(path) → (ndarray[H,W,K], dict[int,str])
                                    │
                                    ▼
partition.compute_partition(map, N) → {"masks": [...], "scores": ndarray, "fairness": float}
                                    │
                                    ▼
visualize.render_partition(path, masks, scores, labels, fairness) → PIL.Image
```

**Integration testing:** Once any two modules are ready, test them together using:
```bash
python src/main.py tests/fixtures/sample_images/pizza_test.jpg 4
```

---

## 6. Development Setup

### 6.1 Clone and setup

```bash
git clone https://github.com/YOUR_USERNAME/fair-slice.git
cd fair-slice
chmod +x setup_project.sh
./setup_project.sh
```

### 6.2 Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 6.3 Google Cloud credentials

```bash
# Option 1: Service account key (recommended)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-key.json"

# Option 2: Default credentials
gcloud auth application-default login
```

Create a `.env` file (git-ignored):
```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

### 6.4 Generate mock data for testing

```bash
python tests/generate_mock_data.py
```

### 6.5 Run tests

```bash
pytest tests/ -v
```

### 6.6 Run the full pipeline locally

```bash
# Backend
cd fair-slice
uvicorn app.api:app --reload --port 8000

# Frontend (in another terminal)
cd app
npm install
npm run dev
```

---

## 7. Deployment

### 7.1 Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 7.2 Google Cloud Run

```bash
gcloud run deploy fairslice \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

This gives you a public URL → that's your "live link" for submission.

---

## 8. Timeline --- IGNORAR

| Day | Date | Focus | Milestone |
|-----|------|-------|-----------|
| 0 | Mon Apr 20 | Repo setup, contracts agreed, mock data ready | Everyone can run `pytest` locally |
| 1 | Tue Apr 21 | Vision: first segmentation working. Partition: radial mode working with mock data | Two modules produce valid output independently |
| 2 | Wed Apr 22 | Vision: iterate on accuracy. Partition: Voronoi mode working | Integration test: vision → partition works end-to-end |
| 3 | Thu Apr 23 | UI: frontend functional. Full pipeline connected | Photo → result visible in browser |
| 4 | Fri Apr 24 | Polish: edge cases, error handling, design. Deploy to Cloud Run | Live URL works |
| 5 | Sat Apr 25 | Final testing, README polish, pitch deck draft | **SUBMISSION at 20:00** |
| 6 | Sun Apr 26 | Pitch rehearsal, backup demo prep | Ready for Demo Day |
| 7 | Mon Apr 27 | **DEMO DAY** 🍕 | Bring the pizza |

---

## Team

| Role | Person | Module | Branch |
|------|--------|--------|--------|
| Math & Partition | Pedro | `src/partition.py` | `feat/partition` |
| Computer Vision | Pablo | `src/vision.py` | `feat/vision` |
| UI & Integration | Quirce | `src/visualize.py` + `app/` | `feat/ui` |