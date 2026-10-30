# Deployment decision

## v1 choice: regular Streamlit

The app is served with regular, server-side Streamlit:

```bash
streamlit run keyclf/app.py
```

This preserves the exact tested Python feature path, including librosa chroma,
scikit-learn model loading, and the system ffmpeg fallback for MP4 decoding.

## Streamlit Lite feasibility check

Checked on 2026-10-29:

- Stlite runs Streamlit on the Pyodide WebAssembly runtime:
  https://stlite.net/
- Pyodide currently provides scikit-learn and soundfile, but its package list
  does not provide librosa or numba:
  https://pyodide.org/en/stable/usage/packages-in-pyodide.html
- The production feature extractor imports librosa, which depends on numba for
  the CQT path, and the upload contract also includes MP4/ffmpeg decoding.

Therefore the existing, acceptance-tested extractor cannot be installed as a
faithful browser-only Stlite app. A browser deployment would require a separate
WebAudio/WASM feature extractor and parity tests against `keyclf.features`,
which is outside v1. Regular Streamlit is the documented fallback authorized by
specification §5.
