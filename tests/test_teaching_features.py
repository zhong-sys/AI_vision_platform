import unittest

from pages_modules.classification_lab import CLASSIFICATION_PRESETS, _classification_preset_values
from pages_modules.clustering_lab import CLUSTERING_PRESETS, _clustering_preset_values
from pages_modules.cnn_viz_module import CNN_PRESETS, _cnn_preset_values
from pages_modules.neural_vis_module import NN_PRESETS, _nn_preset_values
from pages_modules.regression_lab import REGRESSION_PRESETS, _regression_preset_values


class TeachingFeatureTests(unittest.TestCase):
    def test_standard_presets_match_existing_defaults(self):
        classification = _classification_preset_values("svm", "标准示例")
        self.assertEqual(classification["n_neighbors"], 5)
        self.assertEqual(classification["kernel"], "linear")
        self.assertAlmostEqual(classification["noise"], 0.16)

        regression = _regression_preset_values("svr", "标准示例")
        self.assertAlmostEqual(regression["C"], 1.4)
        self.assertAlmostEqual(regression["epsilon"], 0.22)
        self.assertEqual(regression["kernel"], "linear")

        clustering = _clustering_preset_values("agg", "标准示例")
        self.assertEqual(clustering["n_clusters"], 3)
        self.assertEqual(clustering["linkage"], "single")

        self.assertEqual(_nn_preset_values("标准示例")["nv_max_epochs"], 100)
        self.assertEqual(_cnn_preset_values("标准示例")["pool_type"], "最大池化")

    def test_presets_have_expected_teaching_choices(self):
        expected = {"标准示例", "典型现象", "极端参数", "高噪声/复杂情况"}
        self.assertEqual(set(CLASSIFICATION_PRESETS), expected)
        self.assertEqual(set(REGRESSION_PRESETS), expected)
        self.assertEqual(set(CLUSTERING_PRESETS), expected)
        self.assertEqual(set(NN_PRESETS), {"标准示例", "典型现象", "极端参数", "快速观察"})
        self.assertEqual(set(CNN_PRESETS), {"标准示例", "边缘增强", "平滑纹理", "池化对照"})


if __name__ == "__main__":
    unittest.main()
