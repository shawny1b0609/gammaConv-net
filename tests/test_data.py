import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from gammaconv.data import prepare, read_data, windows
from gammaconv.metrics import scores

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_release_data_integrity(self):
        manifest = json.loads((ROOT / "docs/data_check.json").read_text())
        for name, info in manifest.items():
            path = ROOT / "data" / name
            data = read_data(path)
            self.assertEqual(len(data), info["rows"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), info["sha256"]
            )
            np.testing.assert_array_equal(data.day, np.arange(1, len(data) + 1))

    def test_invalid_data_are_rejected_without_closing_gaps(self):
        for text in [
            "day,flow,rain\n1,2,0\n3,3,1\n",
            "day,flow,rain\n1,,0\n2,3,1\n",
            "day,flow,rain\n1,2,0\n2,3,-1\n",
            "day,flow,rain\n1.5,2,0\n2.5,3,1\n",
            "day,flow,rain\n1,2,0\n2,inf,1\n",
        ]:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "data.csv"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_data(path)

    def test_scaling_uses_training_prefix_only(self):
        data = pd.DataFrame(
            {
                "day": np.arange(1, 101),
                "flow": np.arange(100.0),
                "rain": np.arange(100.0),
            }
        )
        train, valid, scale, info = prepare(data, seq=5, split=0.8, fraction=0.5)
        self.assertEqual(scale["flow"], [0.0, 79.0])
        self.assertEqual(info["train_start_day"], 41)
        self.assertEqual(info["valid_first_target_day"], 86)
        self.assertGreater(float(valid[2][0, 0]), 1.0)
        np.testing.assert_allclose(train[1][0, :, 0], np.arange(40, 45) / 79)
        self.assertAlmostEqual(float(train[2][0, 0]), 45 / 79, places=6)

    def test_target_day_alignment_and_short_series(self):
        data = pd.DataFrame(
            {
                "day": [10, 11, 12, 13],
                "flow": [1.0, 2.0, 3.0, 4.0],
                "rain": [4.0, 3.0, 2.0, 1.0],
            }
        )
        rain, flow, target = windows(data, {"flow": [0.0, 4.0], "rain": [0.0, 4.0]}, 3)
        np.testing.assert_allclose(rain[0, :, 0], [1.0, 0.75, 0.5])
        np.testing.assert_allclose(flow[0, :, 0], [0.25, 0.5, 0.75])
        self.assertEqual(float(target[0, 0]), 1.0)
        with self.assertRaises(ValueError):
            windows(data, {"flow": [0.0, 4.0], "rain": [0.0, 4.0]}, 4)

    def test_metrics_have_defined_degenerate_behavior(self):
        self.assertEqual(scores([1, 2, 3], [1, 2, 3])["nse"], 1.0)
        self.assertIsNone(scores([1, 1], [1, 2])["nse"])
        self.assertAlmostEqual(scores([1, 2], [2, 3])["rmse"], 1.0)


if __name__ == "__main__":
    unittest.main()
