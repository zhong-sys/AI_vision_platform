import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CORE_MODULES = (
    "app_constants",
    "config",
    "utils.helpers",
    "components.header",
    "components.sidebar",
    "components.footer",
    "components.page_header",
    "components.experiment_panel",
    "router",
    "pages_modules.home",
    "pages_modules.classification_lab",
    "pages_modules.regression_lab",
    "pages_modules.clustering_lab",
    "pages_modules.neural_vis_module",
    "pages_modules.cnn_viz_module",
    "pages_modules.rnn_viz_module",
    "pages_modules.transformer_viz_module",
    "pages_modules.llm",
)


class CoreImportTests(unittest.TestCase):
    def test_core_modules_import(self):
        for module_name in CORE_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
