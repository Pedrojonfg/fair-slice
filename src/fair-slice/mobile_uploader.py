from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


_COMPONENT_ROOT = Path(__file__).with_name("mobile_uploader_component")

_mobile_uploader = components.declare_component(
    "mobile_uploader_component",
    path=str(_COMPONENT_ROOT),
)


def mobile_image_uploader(
    label: str = "Upload photo (optimized)",
    *,
    max_edge: int = 1600,
    target_format: str = "image/jpeg",
    quality: float = 0.82,
    max_bytes: int = 8_000_000,
    key: str = "mobile_uploader",
) -> tuple[bytes, str] | None:
    """
    Client-side resized/compressed image uploader.

    Returns (image_bytes, filename) or None.
    The goal is to keep uploads reliably under Cloud Run request limits.
    """
    if not _COMPONENT_ROOT.exists():
        st.error(f"Missing component files at {_COMPONENT_ROOT}")
        return None

    value: Any = _mobile_uploader(
        label=label,
        maxEdge=int(max_edge),
        targetFormat=str(target_format),
        quality=float(quality),
        maxBytes=int(max_bytes),
        key=key,
        default=None,
    )
    if not value:
        return None

    b64 = value.get("dataBase64")
    name = value.get("name") or "upload.jpg"
    if not isinstance(b64, str) or not b64:
        return None
    try:
        data = base64.b64decode(b64, validate=True)
    except Exception:
        # Some browsers omit padding; retry leniently.
        data = base64.b64decode(b64 + "===")
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        return None
    return bytes(data), str(name)

