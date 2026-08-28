"""Quick import smoke test for the production Streamlit page modules.

This deliberately imports page entry points without rendering a Streamlit
script, making it safe to run in CI or before starting the application.
"""

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PAGE_ENTRYPOINTS = {
    "home": ("pages_modules.home", "show"),
    "classification": ("pages_modules.classification_lab", "render_classification_lab"),
    "regression": ("pages_modules.regression_lab", "render_regression_lab"),
    "clustering": ("pages_modules.clustering_lab", "render_clustering_lab"),
    "nn_basic": ("pages_modules.neural_vis_module", "render_neural_network_viz"),
    "nn_cnn": ("pages_modules.cnn_viz_module", "nv_render_cnn_viz"),
    "nn_rnn": ("pages_modules.rnn_viz_module", "nv_render_rnn_viz"),
    "nn_attention": ("pages_modules.transformer_viz_module", "nv_render_transformer_viz"),
    "llm_deepseek": ("pages_modules.llm", "show"),
    "llm_zhipu": ("pages_modules.llm", "show"),
    "llm_qwen": ("pages_modules.llm", "show"),
    "llm_wenxin": ("pages_modules.llm", "show"),
}

SUPPORT_MODULES = (
    "app_constants",
    "config",
    "utils.helpers",
    "components.header",
    "components.sidebar",
    "components.footer",
    "components.page_header",
    "components.experiment_panel",
    "router",
)


def main():
    failures = []
    for module_name in SUPPORT_MODULES:
        try:
            importlib.import_module(module_name)
            print("[OK]   {}".format(module_name))
        except Exception as exc:
            failures.append((module_name, exc))
            print("[FAIL] {}".format(module_name), file=sys.stderr)

    for page_key, (module_name, entrypoint) in PAGE_ENTRYPOINTS.items():
        try:
            module = importlib.import_module(module_name)
            getattr(module, entrypoint)
            print("[OK]   {} -> {}.{}".format(page_key, module_name, entrypoint))
        except Exception as exc:
            failures.append((page_key, exc))
            print("[FAIL] {} -> {}.{}".format(page_key, module_name, entrypoint), file=sys.stderr)

    if failures:
        print("\n{} import smoke test(s) failed.".format(len(failures)), file=sys.stderr)
        for name, exc in failures:
            print("  {}: {}".format(name, exc), file=sys.stderr)
        return 1

    print("\nImport smoke test passed: {} modules/routes checked.".format(
        len(SUPPORT_MODULES) + len(PAGE_ENTRYPOINTS)
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
