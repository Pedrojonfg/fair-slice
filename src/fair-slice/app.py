from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from partition import _normalize_preferences, compute_partition
from preferences_ui import build_preference_matrix, uniform_preferences
from vision import segment_dish
from visualize import render_overlay


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

    uploaded = st.file_uploader("Upload a photo of the dish", type=["jpg", "jpeg", "png", "webp"])
    n_people = st.slider("How many people?", min_value=2, max_value=8, value=3)

    if uploaded and st.button("Analyse dish"):
        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.getbuffer())
            image_path = f.name

        with st.spinner("Identifying ingredients..."):
            ingredient_map, labels = segment_dish(image_path)

        st.session_state.image_path = image_path
        st.session_state.ingredient_map = ingredient_map
        st.session_state.labels = labels
        st.session_state.n_people = int(n_people)
        st.session_state.page = "prefs"
        st.rerun()


def _screen_preferences() -> None:
    image_path: str = st.session_state.image_path
    labels: dict[int, str] = st.session_state.labels
    n_people: int = int(st.session_state.n_people)

    st.title("Preferences")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image_path, caption="Uploaded photo", use_container_width=True)

    with col2:
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
                    mode="free",
                    preferences=P,
                )

            st.session_state.P_norm = P_norm
            st.session_state.result = result
            st.session_state.page = "result"
            st.rerun()


def _screen_result() -> None:
    image_path: str = st.session_state.image_path
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
        overlay = render_overlay(image_path, masks, n_people)
        st.image(overlay, caption="Proposed split", use_container_width=True)

    with col2:
        st.metric("Fairness Score", f"{fairness * 100:.0f}%")

        ingredient_keys = _sorted_ingredient_keys(labels)
        for i in range(n_people):
            with st.expander(f"Person {i + 1}", expanded=True):
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

