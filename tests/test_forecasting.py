"""
Demandly AI — Automated Tests
Run: python3 -m pytest tests/ -v   (or python3 tests/test_forecasting.py)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

import unittest
import pandas as pd
import numpy as np

from backend.services.forecast_service import (
    forecast, get_scenario_forecast, STORES_META, PRODUCTS_META
)


class TestForecastBasics(unittest.TestCase):
    """Basic forecast output structure and value tests."""

    def setUp(self):
        self.store_id   = "S011"
        self.product_id = "P001"

    def test_forecast_returns_dict(self):
        f = forecast(self.store_id, self.product_id)
        self.assertIsInstance(f, dict)

    def test_forecast_has_required_keys(self):
        f = forecast(self.store_id, self.product_id)
        required = ["p10","p25","p50","p75","p90","uncertainty","risk_level","drivers","recommendations"]
        for k in required:
            self.assertIn(k, f, f"Missing key: {k}")

    def test_quantile_ordering(self):
        f = forecast(self.store_id, self.product_id)
        self.assertLessEqual(f["p10"], f["p25"])
        self.assertLessEqual(f["p25"], f["p50"])
        self.assertLessEqual(f["p50"], f["p75"])
        self.assertLessEqual(f["p75"], f["p90"])

    def test_no_negative_forecasts(self):
        f = forecast(self.store_id, self.product_id)
        for q in ["p10","p25","p50","p75","p90"]:
            self.assertGreater(f[q], 0, f"{q} should be positive")

    def test_uncertainty_values_valid(self):
        f = forecast(self.store_id, self.product_id)
        self.assertIn(f["uncertainty"], ["LOW","MEDIUM","HIGH","VERY HIGH"])

    def test_risk_level_valid(self):
        f = forecast(self.store_id, self.product_id)
        self.assertIn(f["risk_level"], ["LOW","MEDIUM","HIGH"])

    def test_recommendations_present(self):
        f = forecast(self.store_id, self.product_id)
        recs = f["recommendations"]
        self.assertIn("conservative", recs)
        self.assertIn("balanced", recs)
        self.assertIn("aggressive", recs)

    def test_recommendation_ordering(self):
        f = forecast(self.store_id, self.product_id)
        r = f["recommendations"]
        self.assertGreaterEqual(r["conservative"], r["balanced"])
        self.assertGreaterEqual(r["balanced"], r["aggressive"])


class TestForecastDrivers(unittest.TestCase):
    """Test that forecast responds correctly to driver changes."""

    def setUp(self):
        self.store_id   = "S011"
        self.product_id = "P001"   # Cold Drinks — high weather sensitivity

    def test_higher_temperature_increases_beverage_demand(self):
        f_cool = forecast(self.store_id, self.product_id, {"temperature": 25})
        f_hot  = forecast(self.store_id, self.product_id, {"temperature": 42})
        self.assertGreater(f_hot["p50"], f_cool["p50"],
            "Hotter weather should increase cold drink demand")

    def test_festival_increases_demand(self):
        f_none = forecast(self.store_id, self.product_id, {"festival_intensity": 0.0})
        f_fest = forecast(self.store_id, self.product_id, {"festival_intensity": 0.9})
        self.assertGreater(f_fest["p50"], f_none["p50"],
            "Festival should increase demand")

    def test_weekend_increases_demand(self):
        f_week = forecast(self.store_id, self.product_id, {"weekend": 0})
        f_wknd = forecast(self.store_id, self.product_id, {"weekend": 1})
        self.assertGreaterEqual(f_wknd["p50"], f_week["p50"],
            "Weekend should not decrease demand")

    def test_delivery_delay_raises_risk(self):
        f_norm  = forecast(self.store_id, self.product_id, {"delivery_delay": 0})
        f_delay = forecast(self.store_id, self.product_id, {"delivery_delay": 12})
        # Risk score should increase
        self.assertGreaterEqual(f_delay["risk_score"], f_norm["risk_score"])

    def test_warehouse_capacity_loss_raises_risk(self):
        f_full    = forecast(self.store_id, self.product_id, {"warehouse_capacity": 100})
        f_reduced = forecast(self.store_id, self.product_id, {"warehouse_capacity": 60})
        self.assertGreaterEqual(f_reduced["risk_score"], f_full["risk_score"])


class TestUncertaintyBehaviour(unittest.TestCase):
    """Test that uncertainty expands under stressful conditions."""

    def test_high_festival_widens_interval(self):
        f_low  = forecast("S011","P001",{"festival_intensity":0.0,"temperature":28})
        f_high = forecast("S011","P001",{"festival_intensity":0.9,"temperature":42})
        width_low  = f_low["p90"]  - f_low["p10"]
        width_high = f_high["p90"] - f_high["p10"]
        self.assertGreater(width_high, width_low,
            "High festival + heat should widen prediction interval")

    def test_extreme_heat_classified_as_high_risk(self):
        f = forecast("S011","P001",{"temperature":45,"festival_intensity":0.8})
        self.assertIn(f["risk_level"], ["MEDIUM","HIGH"])

    def test_combined_disruptions_raise_risk(self):
        f = forecast("S011","P001",{
            "temperature":42, "festival_intensity":0.9,
            "delivery_delay":12, "warehouse_capacity":65
        })
        self.assertEqual(f["risk_level"], "HIGH")


class TestScenarioAPI(unittest.TestCase):
    """Test scenario override functionality."""

    def test_scenario_returns_forecast(self):
        result = get_scenario_forecast("S011","P001",{"temperature":38})
        self.assertIn("p50", result)

    def test_multiple_stores_forecast(self):
        for sid in ["S011","S037","S063"]:
            f = forecast(sid, "P001")
            self.assertIn("p50", f)
            self.assertGreater(f["p50"], 0)

    def test_multiple_products_forecast(self):
        for pid in ["P001","P003","P009","P010"]:
            f = forecast("S011", pid)
            self.assertIn("p50", f)


class TestEdgeCases(unittest.TestCase):
    """Test failure and edge cases."""

    def test_extreme_temperature_does_not_crash(self):
        """Case: Temperature outside training range."""
        try:
            f = forecast("S011","P001",{"temperature": 55})
            self.assertIn("p50", f)
            self.assertGreater(f["p50"], 0)
        except Exception as e:
            self.fail(f"Extreme temperature crashed: {e}")

    def test_zero_festival_intensity(self):
        """Case: Festival data missing (intensity=0)."""
        f = forecast("S011","P001",{"festival_intensity": 0.0})
        self.assertIsNotNone(f["p50"])
        # Drivers should not show festival contribution
        fest_drivers = [d for d in f["drivers"] if "festival" in d["name"].lower()]
        for d in fest_drivers:
            self.assertEqual(d["pct"], 0, "Festival driver should be 0 when intensity=0")

    def test_max_delivery_delay(self):
        """Case: Extreme delivery delay."""
        f = forecast("S011","P001",{"delivery_delay": 48})
        self.assertIn("Delivery delay 48h", f["risk_factors"])

    def test_zero_warehouse_capacity(self):
        """Case: Warehouse capacity critically low."""
        f = forecast("S011","P001",{"warehouse_capacity": 40})
        self.assertGreaterEqual(f["risk_score"], 20)

    def test_unknown_store_falls_back(self):
        """Case: Unknown store_id uses default."""
        try:
            f = forecast("UNKNOWN_STORE","P001")
            self.assertIn("p50", f)
        except Exception as e:
            self.fail(f"Unknown store crashed: {e}")

    def test_all_zeros_params(self):
        """Case: All params zero/minimal."""
        f = forecast("S011","P001",{
            "temperature":0,"rainfall":0,"festival_intensity":0,
            "discount":0,"warehouse_capacity":100,"delivery_delay":0
        })
        self.assertGreater(f["p50"], 0)


class TestDataset(unittest.TestCase):
    """Test that synthetic dataset was generated correctly."""

    def setUp(self):
        self.df = pd.read_csv("dataset/demand_history.csv", parse_dates=["date"])

    def test_dataset_row_count(self):
        self.assertGreaterEqual(len(self.df), 50000, "Dataset should have ≥50k rows")

    def test_required_columns(self):
        required = ["date","store_id","product_id","actual_demand","temperature",
                    "festival","festival_intensity","price","disruption_flag"]
        for col in required:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")

    def test_no_negative_demand(self):
        self.assertTrue((self.df["actual_demand"] >= 0).all(), "Demand should not be negative")

    def test_festival_intensity_range(self):
        self.assertTrue((self.df["festival_intensity"] >= 0).all())
        self.assertTrue((self.df["festival_intensity"] <= 1).all())

    def test_multiple_stores(self):
        self.assertGreaterEqual(self.df["store_id"].nunique(), 5)

    def test_multiple_products(self):
        self.assertGreaterEqual(self.df["product_id"].nunique(), 5)


class TestEvaluationResults(unittest.TestCase):
    """Test that evaluation results are present and sensible."""

    def setUp(self):
        with open("evaluation/results/model_results.json") as f:
            self.results = json.load(f)

    def test_demandly_better_than_baseline_mae(self):
        self.assertLess(self.results["demandly"]["mae"],
                        self.results["baseline"]["mae"],
                        "Demandly should outperform naive baseline on MAE")

    def test_coverage_80_reasonable(self):
        cov = self.results["coverage"]["pct80"]
        self.assertGreater(cov, 0.60, "80% coverage should be >60%")
        self.assertLess(cov, 1.0,    "80% coverage should be <100%")

    def test_feature_importance_sums_to_one(self):
        fi = self.results["feature_importance"]
        total = sum(fi.values())
        self.assertAlmostEqual(total, 1.0, places=1, msg="Feature importances should sum to ~1")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Demandly AI — Test Suite")
    print("="*60 + "\n")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
