#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: '${PYTHON_BIN}' not found. Set PYTHON_BIN=python3.11 (or similar)." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[FairSlice] Creating venv at ${VENV_DIR}/"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "[FairSlice] Activating venv"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[FairSlice] Installing/upgrading dependencies"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

# Ensure src modules import cleanly when launched from repo root
export PYTHONPATH="${ROOT_DIR}/src/fair-slice:${PYTHONPATH:-}"

APP_MODE="${APP_MODE:-streamlit}" # streamlit | api

if [[ "$APP_MODE" == "streamlit" ]]; then
  echo "[FairSlice] Launching Streamlit UI"
  echo "[FairSlice] URL: http://localhost:8501"
  exec streamlit run "src/fair-slice/app.py"
elif [[ "$APP_MODE" == "api" ]]; then
  PORT="${PORT:-8000}"
  echo "[FairSlice] Launching FastAPI API"
  echo "[FairSlice] URL: http://localhost:${PORT}"
  exec uvicorn app.api:app --reload --port "$PORT"
else
  echo "Error: APP_MODE must be 'streamlit' or 'api' (got: ${APP_MODE})" >&2
  exit 1
fi

