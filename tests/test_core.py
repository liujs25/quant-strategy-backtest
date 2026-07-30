import math
import unittest

import numpy as np
import pandas as pd

from src.backtest import BacktestEngine
from src.data_preprocessing import DataPreprocessor


class DataPreprocessorTests(unittest.TestCase):
    def test_windows_include_last_sample_and_are_globally_chronological(self):
        preprocessor = DataPreprocessor()
        preprocessor.processed_data = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-03",
                        "2024-01-05",
                        "2024-01-02",
                        "2024-01-04",
                        "2024-01-06",
                    ]
                ),
                "Name": ["A", "A", "A", "B", "B", "B"],
                "open": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
                "daily_return": [0.1, 0.3, 0.5, 0.2, 0.4, 0.6],
            }
        )

        X, y, features = preprocessor.prepare_features_target(
            target_column="daily_return", lookback=1, prediction_horizon=1
        )

        self.assertEqual(features, ["open"])
        self.assertEqual(X.shape, (4, 1, 1))
        np.testing.assert_allclose(X[:, 0, 0], [10.0, 20.0, 11.0, 21.0])
        np.testing.assert_allclose(y, [0.3, 0.4, 0.5, 0.6])

    def test_split_validation(self):
        preprocessor = DataPreprocessor()
        with self.assertRaises(ValueError):
            preprocessor.split_train_test(np.zeros((2, 1)), np.zeros(2), test_size=1)
        with self.assertRaises(ValueError):
            preprocessor.split_train_test(np.zeros((2, 1)), np.zeros(1))


class BacktestEngineTests(unittest.TestCase):
    def setUp(self):
        self.prices = np.array([100.0, 101.0, 102.0, 100.0, 103.0])
        self.predictions = np.array([0.0, 0.01, 0.01, -0.01, 0.01])

    def test_benchmark_metrics_require_real_benchmark(self):
        engine = BacktestEngine()
        results = engine.run_backtest(self.prices, self.predictions)

        self.assertTrue(math.isnan(results["alpha"]))
        self.assertTrue(math.isnan(results["beta"]))
        self.assertTrue(math.isnan(results["information_ratio"]))

        benchmark = np.array([0.0, 0.004, 0.003, -0.002, 0.005])
        results = engine.run_backtest(
            self.prices, self.predictions, benchmark_returns=benchmark
        )
        self.assertTrue(np.isfinite(results["alpha"]))
        self.assertTrue(np.isfinite(results["beta"]))

    def test_input_and_strategy_validation(self):
        engine = BacktestEngine()
        with self.assertRaises(ValueError):
            engine.run_backtest([100.0], [0.0])
        with self.assertRaises(ValueError):
            engine.run_backtest(self.prices, self.predictions, strategy_type="unknown")

    def test_repeated_buy_signal_does_not_pyramid_and_take_profit_closes(self):
        engine = BacktestEngine()
        results = engine.run_backtest(
            prices=[100.0, 101.0, 102.0, 107.0],
            predictions=[0.0, 0.01, 0.01, 0.0],
        )

        buy_trades = [trade for trade in engine.trade_log if trade["type"] == "BUY"]
        sell_trades = [trade for trade in engine.trade_log if trade["type"] == "SELL"]
        self.assertEqual(len(buy_trades), 1)
        self.assertEqual(len(sell_trades), 1)
        self.assertEqual(results["position_history"][-1], 0)
        self.assertGreater(sell_trades[0]["return"], 0)


class OptionalModelTests(unittest.TestCase):
    def test_position_encoding_and_predict_shape_when_ml_stack_is_available(self):
        try:
            import torch
            from src.models import LSTMModel, PositionalEncoding
        except ModuleNotFoundError as exc:
            self.skipTest(f"Optional ML dependency unavailable: {exc.name}")

        encoding = PositionalEncoding(d_model=4, max_len=8)
        encoded = encoding(torch.zeros(2, 3, 4))
        self.assertEqual(tuple(encoded.shape), (2, 3, 4))
        self.assertFalse(torch.allclose(encoded[0, 0], encoded[0, 1]))
        self.assertTrue(torch.allclose(encoded[0], encoded[1]))

        model = LSTMModel(input_size=4, hidden_size=8, num_layers=1)
        prediction = model.predict(np.zeros((2, 3, 4), dtype=np.float32))
        self.assertEqual(prediction.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
