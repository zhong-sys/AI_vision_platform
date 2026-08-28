import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.helpers import resource_path


class ResourcePathTests(unittest.TestCase):
    def test_development_path_resolution(self):
        expected = os.path.join(os.path.abspath("."), "assets/logo.png")
        self.assertEqual(resource_path("assets/logo.png"), expected)
        self.assertTrue(Path(expected).is_file())

    def test_pyinstaller_path_resolution(self):
        with tempfile.TemporaryDirectory() as meipass:
            with patch.object(sys, "_MEIPASS", meipass, create=True):
                self.assertEqual(
                    resource_path("assets/logo.png"),
                    os.path.join(meipass, "assets/logo.png"),
                )


if __name__ == "__main__":
    unittest.main()
