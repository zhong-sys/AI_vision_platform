# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app (development)
streamlit run main.py --server.enableCORS false --server.enableXsrfProtection false

# Build distributable EXE
pyinstaller launcher.spec

# Install dependencies
pip install -r requirements.txt
```

- No test framework, linter, or formatter is configured in this project.
- The devcontainer auto-starts the app on port 8501.

## Architecture

Streamlit single-page app with client-side routing via `st.session_state.current_page`.

### Entry & routing

`main.py` — sets `st.set_page_config()`, renders sidebar navigation buttons (each sets `current_page` + calls `st.rerun()`), then dispatches to the correct module via `if/elif` on `current_page`. Uses `config.py` for color constants and API key retrieval from `secrets.toml`.

### Page modules (`pages_modules/`)

Each module exports a top-level `render_*()` or `show()` function called by `main.py`:

| Route | Module | Function |
|---|---|---|
| `classification` | `classification_lab.py` | `render_classification_lab()` |
| `regression` | `regression_lab.py` | `render_regression_lab()` |
| `clustering` | `clustering_lab.py` | `render_clustering_lab()` |
| `nn_basic` | `neural_vis_module.py` | `render_neural_network_viz()` |
| `nn_cnn` | `cnn_viz_module.py` | `nv_render_cnn_viz()` |
| `nn_rnn` | `rnn_viz_module.py` | `nv_render_rnn_viz()` |
| `nn_attention` | `transformer_viz_module.py` | `nv_render_transformer_viz()` |
| `llm_*` | `llm.py` | `show(model_preselected=...)` |
| `home` | `home.py` | `show()` |

Machine learning modules (classification/regression/clustering) are split into `_lab.py` (UI + state), `_viz.py` (plots), `_data.py` (datasets), `_models.py` / `_model_factory.py` (model wrappers), `_metrics.py`, and `_text.py` / `_teaching_text.py` (educational content). Neural network modules are self-contained in single files.

### NeuralVis desktop app (`pages_modules/NeuralVis/`)

Standalone PySide6 application (not used by the Streamlit app). Has its own architecture: `src/core/` (neural network, dataset, trainer) and `src/gui/` (PySide6 widgets). Entry point is `show.py`.

### LLM integration (`pages_modules/llm.py`)

Unified chat UI supporting DeepSeek, 智谱GLM, 通义千问, 文心一言 via OpenAI-compatible API. Provider base URLs and model names are hardcoded; API keys come from `st.secrets` via `config.get_api_key()`.

### Packaging

`launcher.py` starts Streamlit programmatically via `streamlit.web.cli.main()`. `launcher.spec` bundles everything (assets, pages_modules, utils) into `dist/launcher/launcher.exe`. The `resource_path()` helper (defined in `utils/helpers.py` and duplicated locally in some modules) resolves asset paths under `sys._MEIPASS` when running from the PyInstaller bundle.

### Key files

- `.streamlit/secrets.toml` — API keys (not committed to git)
- `assets/` — logo images
- `packages.txt` — system packages for Linux deployment (fonts-noto-cjk)
