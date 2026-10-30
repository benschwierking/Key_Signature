"""Streamlit upload demo for the trained key-signature classifier."""

from __future__ import annotations

import tempfile
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from keyclf.features import FeatureConfig, extract_features
from keyclf.keys import KEY_NAMES, is_related


@st.cache_resource
def load_artifact(path: str = "models/model.joblib"):
    """Load and cache the exported estimator plus metadata."""
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        return artifact["model"], artifact
    return artifact, {}


def probability_24(estimator, vector: np.ndarray, metadata: dict) -> np.ndarray:
    """Map model probability columns to canonical key order."""
    names = metadata.get("feature_names")
    sample = pd.DataFrame([vector], columns=names) if names else vector.reshape(1, -1)
    raw = np.asarray(estimator.predict_proba(sample), dtype=float)[0]
    result = np.zeros(24, dtype=float)
    class_names = metadata.get("classes")
    for column, value in enumerate(estimator.classes_):
        key = KEY_NAMES.index(class_names[int(value)]) if class_names and isinstance(value, (int, np.integer)) else int(value)
        result[key] += raw[column]
    return result


def render_result(vector: np.ndarray, probability: np.ndarray) -> None:
    """Render the histogram, edge bins, and related-aware top predictions."""
    best = int(probability.argmax())
    first, last = int(vector[:12].argmax()), int(vector[12:24].argmax())
    st.subheader(f"Predicted key: {KEY_NAMES[best]}")
    st.metric("Confidence", f"{probability[best]:.1%}")
    left, right = st.columns(2)
    left.metric("First chroma bin", KEY_NAMES[first].split()[0])
    right.metric("Last chroma bin", KEY_NAMES[last].split()[0])
    chart_data = pd.DataFrame({"pitch class": [name.split()[0] for name in KEY_NAMES[:12]],
                               "energy": vector[24:36]})
    st.altair_chart(alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("pitch class:N", sort=None), y=alt.Y("energy:Q"), tooltip=["pitch class", "energy"]
    ).properties(title="Track-wide chroma histogram"), use_container_width=True)
    st.subheader("Top three")
    for rank, key in enumerate(np.argsort(probability)[::-1][:3], start=1):
        related = " — related key" if key != best and is_related(best, int(key)) else ""
        st.write(f"{rank}. **{KEY_NAMES[int(key)]}** — {probability[int(key)]:.1%}{related}")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="Key Signature Classifier", page_icon="🎼")
    st.title("Key Signature Classifier")
    st.caption("Chroma-based 24-class model with Krumhansl–Schmuckler interpretation")
    model_path = Path("models/model.joblib")
    if not model_path.exists():
        st.error("Train the model first with `python -m keyclf.train`.")
        return
    upload = st.file_uploader("Upload a song", type=["mp3", "mp4", "wav"])
    if upload is None:
        return
    estimator, metadata = load_artifact(str(model_path))
    feature_config = FeatureConfig(**metadata.get("feature_config", {}))
    suffix = Path(upload.name).suffix.lower() or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
        temporary.write(upload.getbuffer())
        temporary.flush()
        with st.spinner("Extracting chroma and predicting the key…"):
            vector = extract_features(temporary.name, feature_config)
            probability = probability_24(estimator, vector, metadata)
    render_result(vector, probability)


if __name__ == "__main__":
    main()
