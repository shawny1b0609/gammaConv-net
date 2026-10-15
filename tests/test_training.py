import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from gammaconv.models import MODELS, make_model
from gammaconv.predict import forecast
from gammaconv.train import Config, clone_state, run


class TrainingTests(unittest.TestCase):
    def test_checkpoint_is_independent_of_live_cpu_parameters(self):
        model = make_model("dual", hidden=8)
        saved = clone_state(model)
        before = {k: v.clone() for k, v in saved.items()}
        with torch.no_grad():
            for param in model.parameters():
                param.add_(2.0)
        for key in saved:
            torch.testing.assert_close(saved[key], before[key])

    def test_training_and_reloaded_predictions_for_every_model(self):
        day = np.arange(1, 101)
        data = pd.DataFrame(
            {"day": day, "flow": 5.0 + np.sin(day / 7), "rain": 1.0 + np.cos(day / 5)}
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "toy.csv"
            data.to_csv(path, index=False)
            for name in MODELS:
                with self.subTest(model=name):
                    out = Path(folder) / name
                    record = run(
                        Config(
                            data=str(path),
                            out=str(out),
                            model=name,
                            epochs=3,
                            hidden=8,
                            seq=8,
                            lag=8,
                            split=0.7,
                            batch=32,
                            seed=7,
                        )
                    )
                    history = pd.read_csv(out / "history.csv")
                    self.assertEqual(
                        record["best_epoch"],
                        int(history.loc[history.rmse.idxmin(), "epoch"]),
                    )
                    self.assertAlmostEqual(
                        record["validation"]["rmse"], history.rmse.min(), places=5
                    )
                    pred = pd.read_csv(out / "predictions.csv")
                    loaded = forecast(out / "model.pt", data.iloc[70:])
                    np.testing.assert_array_equal(pred.day, loaded.day)
                    np.testing.assert_allclose(
                        pred.predicted, loaded.predicted, atol=1e-5
                    )
                    self.assertEqual(int(pred.day.iloc[0]), 79)
                    text = (out / "run.json").read_text()
                    self.assertNotIn(folder, text)
                    self.assertEqual(json.loads(text)["config"]["data"], "toy.csv")


if __name__ == "__main__":
    unittest.main()
