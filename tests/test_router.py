import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_constants
import router


EXPECTED_PAGE_KEYS = (
    "home",
    "classification",
    "regression",
    "clustering",
    "nn_basic",
    "nn_cnn",
    "nn_rnn",
    "nn_attention",
    "llm_deepseek",
    "llm_zhipu",
    "llm_qwen",
    "llm_wenxin",
)


class RouterMappingTests(unittest.TestCase):
    def test_page_mapping_is_complete_and_ordered(self):
        self.assertEqual(tuple(app_constants.PAGE_KEYS), EXPECTED_PAGE_KEYS)
        self.assertEqual(tuple(router.PAGE_KEYS), EXPECTED_PAGE_KEYS)

    def test_llm_mapping_matches_existing_routes(self):
        self.assertEqual(
            router.LLM_MODEL_MAP,
            {
                "llm_deepseek": "DeepSeek",
                "llm_zhipu": "智谱GLM",
                "llm_qwen": "通义千问",
                "llm_wenxin": "文心一言",
            },
        )

    def test_each_non_home_route_has_a_display_name(self):
        self.assertEqual(
            set(app_constants.PAGE_DISPLAY_NAMES),
            set(EXPECTED_PAGE_KEYS) - {"home"},
        )


if __name__ == "__main__":
    unittest.main()
