# 🍕 FairSlice

**FairSlice** is a web service that, from a photo of a dish (pizza, cake, paella...), tries to **identify ingredients** and calculate a **"fair" partition** for \(N\) people. It returns an image with colored areas (what each person gets) and distribution metrics.

The repo is prepared to run as a **FastAPI API** and serve a **static UI** (PWA) from `app/static/`.

---

## What it includes (high level)

- **HTTP API (FastAPI)**: `app/api.py`
  - `POST /api/slice`: complete pipeline (photo → vision → partition → render)
  - `GET /health`: liveness
  - `GET /`: static UI (if `app/static/` exists)
- **Python Pipeline**: modules in `src/fair-slice/`
  - `vision.py`: ingredient segmentation (Vertex AI / Gemini + OpenCV)
  - `partition.py`: partitioning algorithm (modes `free` and `radial`)
  - `visualize.py`: overlay render (PIL)
  - `main.py`: local pipeline runner

---

## Requirements

- **Python 3.11+** (recommended)
- **Google Cloud** credentials if you want to run `vision.segment_dish()` against Vertex AI
- Python dependencies installed from `requirements.txt`

---

## Installation (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration (Vertex AI / Gemini)

The vision part needs these environment variables:

- `GOOGLE_CLOUD_PROJECT` (**mandatory**)
- `GOOGLE_CLOUD_LOCATION` (optional, default `us-central1`)

Authentication (one of these options):

```bash
gcloud auth application-default login
```

or:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

---

## Run locally

### API + UI

```bash
uvicorn app.api:app --reload --port 8000
```

- UI: `http://localhost:8000/`
- Health: `http://localhost:8000/health`

### Call the `/api/slice` endpoint

Example with `curl` (multipart):

```bash
curl -sS -X POST "http://localhost:8000/api/slice" \
  -F "image=@tests/fixtures/sample_images/pizza_test.jpg" \
  -F "n_people=4" \
  -F "mode=free"
```

Response (JSON):

- `result_image`: PNG in base64
- `fairness`: float in \([0,1]\)
- `scores`: \(N \times K\) matrix with proportions per ingredient
- `ingredient_labels`: channel → ingredient mapping

---

## Run the pipeline as a script

```bash
python src/fair-slice/main.py tests/fixtures/sample_images/pizza_test.jpg 4
```

Generates `output.png` in the current directory.

---

## Tests

By default, `pytest` excludes tests marked as "slow".

```bash
pytest -v
```

To include the "slow" ones:

```bash
pytest -v -m slow
```

### Mock data (without Vertex AI)

If you don't want to depend on credentials/network, you can generate synthetic fixtures:

```bash
python tests/generate_mock_data.py
```

This creates:

- `tests/fixtures/mock_data/pizza_mock_500.npy` (ingredient map)
- `tests/fixtures/sample_images/pizza_test.jpg` (synthetic image)

---

## Smoke test with real images (script)

There's a runner designed to execute the pipeline on real photos (requires `vision` to work with credentials):

```bash
python tests/run_real_images_pipeline.py --people 3,4 --modes free,radial
```

By default, it looks for images in `tests/fixtures/real_images/` and writes outputs in `tests/fixtures/real_images_outputs/`.

---

## Deployment (Google Cloud Run)

The repo already includes `Procfile` with the startup command for the `PORT` port.

Example (build from source, without manual Docker):

```bash
gcloud run deploy fairslice \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=YOUR_PROJECT,GOOGLE_CLOUD_LOCATION=us-central1"
```

Notes:

- In Cloud Run, make sure the runtime has permissions for Vertex AI (service account with appropriate permissions).
- If you don't configure credentials/permissions, the `/api/slice` endpoint will fail in the vision phase.