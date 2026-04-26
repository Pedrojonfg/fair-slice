from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from mobile_uploader import mobile_image_uploader
from partition import _normalize_preferences, compute_partition
from preferences_ui import build_preference_matrix, uniform_preferences
from vision import segment_dish
from visualize import render_overlay


def _to_png_bytes(data: object) -> bytes | None:
    """
    Convert common image representations into PNG bytes.
    Returns None if conversion fails.
    """
    try:
        if data is None:
            return None
        if isinstance(data, (bytes, bytearray)):
            # Assume it's already encoded image bytes.
            return bytes(data)
        if isinstance(data, str) and data:
            img = Image.open(data)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
        if isinstance(data, Image.Image):
            img = data
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
        if isinstance(data, np.ndarray):
            arr = data
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
    except Exception:
        return None
    return None


def _placeholder_png_bytes(message: str = "Image unavailable") -> bytes:
    w, h = 1400, 900
    img = Image.new("RGB", (w, h), color=(20, 20, 35))
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, w - 1, h - 1), outline=(80, 80, 120), width=6)
        draw.text((60, 60), message, fill=(232, 232, 240))
    except Exception:
        pass
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _render_image_always(png_bytes: bytes | None, alt: str = "image") -> None:
    """
    Render an image in a way that cannot break as an external src.
    We embed the bytes as a data: URL to avoid Streamlit media endpoint issues.
    """
    import base64

    b = png_bytes if png_bytes else _placeholder_png_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    st.markdown(
        f"""
        <div class="fs-image-wrap">
          <img class="fs-image" alt="{alt}" src="data:image/png;base64,{b64}" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_design_tokens() -> None:
    css_path = Path(__file__).with_name("styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _init_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "upload"  # upload | prefs | result


def _clear_state() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def _sorted_ingredient_keys(labels: dict[int, str]) -> list[int]:
    return sorted(labels.keys(), key=int)


def _screen_upload() -> None:
    st.title("🍕 FairSlice")
    st.subheader("Fair splitting for any dish")

    st.caption(
        "Mobile-friendly uploader: your photo is resized/compressed on-device before upload, "
        "so even very large originals won’t fail."
    )
    optimized = mobile_image_uploader(
        "Upload photo (auto-optimized for reliability)",
        max_edge=1600,
        target_format="image/jpeg",
        quality=0.82,
        max_bytes=8_000_000,
        key="mobile_uploader_main",
    )

    with st.expander("Alternative upload methods", expanded=False):
        camera = st.camera_input("Take a photo")
        uploaded = st.file_uploader(
            "Upload a photo (JPG/PNG/WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
        )
    n_people = st.slider("How many people?", min_value=2, max_value=8, value=3)

    image_bytes: bytes | None = None
    suffix = ".jpg"
    if optimized is not None:
        image_bytes, name = optimized
        suffix = Path(name).suffix or ".jpg"
    else:
        chosen = camera or uploaded
        if chosen is not None:
            image_bytes = chosen.getbuffer().tobytes()
            suffix = Path(getattr(chosen, "name", "")).suffix or ".png"

    if image_bytes and st.button("Analyse dish"):
        raw_bytes = image_bytes

        # Normalize to PNG to avoid mobile codec issues (notably WEBP on older iOS/Safari).
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            image_bytes = buf.getvalue()
            suffix = ".png"
        except Exception:
            # If PIL can't decode it, fall back to the original bytes as-is.
            image_bytes = raw_bytes
            suffix = suffix or ".png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(image_bytes)
            image_path = f.name

        with st.spinner("Identifying ingredients..."):
            ingredient_map, labels = segment_dish(image_path)

        st.session_state.image_path = image_path
        st.session_state.image_bytes = image_bytes
        st.session_state.ingredient_map = ingredient_map
        st.session_state.labels = labels
        st.session_state.n_people = int(n_people)
        st.session_state.page = "prefs"
        st.rerun()


def _ensure_image_path() -> str:
    """
    Streamlit reruns + long computations can make temp file paths unreliable.
    Keep original bytes and recreate the temp file if missing.
    """
    image_path: str = st.session_state.image_path
    if Path(image_path).exists():
        return image_path

    image_bytes: bytes | None = st.session_state.get("image_bytes")
    if not image_bytes:
        return image_path

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
        f.write(image_bytes)
        image_path = f.name
    st.session_state.image_path = image_path
    return image_path


def _screen_preferences() -> None:
    image_path: str = _ensure_image_path()
    labels: dict[int, str] = st.session_state.labels
    n_people: int = int(st.session_state.n_people)

    st.title("Preferences")

    col1, col2 = st.columns([1, 1])
    with col1:
        image_bytes: bytes | None = st.session_state.get("image_bytes")
        png = _to_png_bytes(image_bytes) or _to_png_bytes(image_path)
        _render_image_always(png, alt="Uploaded photo")

    with col2:
        mode_label = st.selectbox(
            "Cut mode",
            options=["free", "radial"],
            index=0,
            format_func=lambda m: "Convex (free)" if m == "free" else "Radial",
        )
        equitable = st.checkbox("Equal split (no preferences)", value=False)

        ingredient_keys = _sorted_ingredient_keys(labels)
        K = len(ingredient_keys)

        if not equitable:
            tabs = st.tabs([f"Person {i + 1}" for i in range(n_people)])
            for i, tab in enumerate(tabs):
                with tab:
                    for k in ingredient_keys:
                        name = labels[k]
                        st.slider(
                            label=name,
                            min_value=0,
                            max_value=10,
                            value=5,
                            key=f"pref_{i}_{k}",
                        )

        if st.button("Calculate fair split"):
            if equitable:
                P = uniform_preferences(n_people, K)
            else:
                user_inputs: dict[int, list[float]] = {}
                for i in range(n_people):
                    user_inputs[i] = [float(st.session_state[f"pref_{i}_{k}"]) for k in ingredient_keys]
                P = build_preference_matrix(n_people, labels, user_inputs)

            P_norm = _normalize_preferences(P, n_people, K)

            with st.spinner("Calculating optimal partition..."):
                result = compute_partition(
                    st.session_state.ingredient_map,
                    n_people,
                    mode=mode_label,
                    preferences=P,
                )

            st.session_state.P_norm = P_norm
            st.session_state.result = result
            st.session_state.page = "result"
            st.rerun()


def _screen_result() -> None:
    image_path: str = _ensure_image_path()
    labels: dict[int, str] = st.session_state.labels
    n_people: int = int(st.session_state.n_people)

    result: dict = st.session_state.result
    masks = result["masks"]
    scores: np.ndarray = result["scores"]
    fairness: float = float(result["fairness"])

    P_norm: np.ndarray = st.session_state.P_norm

    st.title("Result")

    col1, col2 = st.columns([1, 1])
    with col1:
        dish_mask = st.session_state["ingredient_map"].sum(axis=-1) > 1e-3
        result["masks"] = [m & dish_mask for m in result["masks"]]
        masks = result["masks"]
        overlay = render_overlay(image_path, masks, n_people)
        overlay_png = _to_png_bytes(overlay) or _to_png_bytes(np.asarray(overlay, dtype=np.uint8))
        _render_image_always(overlay_png, alt="Proposed split")

    with col2:
        st.metric("Fairness Score", f"{fairness * 100:.0f}%")

        ingredient_keys = _sorted_ingredient_keys(labels)
        for i in range(n_people):
            with st.expander(f"Person {i + 1}", expanded=False):
                for k in ingredient_keys:
                    name = labels[k]
                    received = float(scores[i, k])
                    wanted = float(P_norm[i, k])
                    ok = received >= wanted - 0.05
                    color = "green" if ok else "red"
                    st.markdown(
                        f"<span style='color:{color}'><b>{name}</b>: "
                        f"{received * 100:.0f}% received / {wanted * 100:.0f}% target</span>",
                        unsafe_allow_html=True,
                    )
                    st.progress(max(0.0, min(1.0, received)))

        if st.button("New split"):
            _clear_state()
            st.session_state.page = "upload"
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="FairSlice", page_icon="🍕", layout="wide")
    _inject_design_tokens()
    _init_state()

    page = st.session_state.page
    if page == "upload":
        _screen_upload()
    elif page == "prefs":
        _screen_preferences()
    elif page == "result":
        _screen_result()
    else:
        st.session_state.page = "upload"
        st.rerun()


if __name__ == "__main__":
    main()

