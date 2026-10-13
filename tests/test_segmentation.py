"""Unit tests for the customer segmentation project."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.customer_segmentation import (
    FEATURES,
    fit_segmentation,
    generate_customer_data,
    save_outputs,
)


class CustomerSegmentationTests(unittest.TestCase):
    def test_generated_data_is_reproducible_and_well_formed(self):
        first = generate_customer_data()
        second = generate_customer_data()
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 600)
        self.assertTrue(set(FEATURES).issubset(first.columns))
        self.assertFalse(first[FEATURES].isna().any().any())
        self.assertFalse(first["CustomerID"].duplicated().any())

    def test_fit_returns_valid_cluster_assignments(self):
        data = generate_customer_data(n_samples=180)
        assigned, profiles, selection, best_k = fit_segmentation(data, k_values=range(2, 6))
        self.assertIn("Cluster", assigned.columns)
        self.assertGreaterEqual(assigned["Cluster"].nunique(), 2)
        self.assertLessEqual(assigned["Cluster"].nunique(), 5)
        self.assertIn(best_k, selection["k"].tolist())
        self.assertEqual(len(profiles), assigned["Cluster"].nunique())
        self.assertTrue(selection["silhouette_score"].between(-1, 1).all())

    def test_missing_values_are_rejected(self):
        data = generate_customer_data(n_samples=30)
        data.loc[0, FEATURES[0]] = float("nan")
        with self.assertRaises(ValueError):
            fit_segmentation(data, k_values=range(2, 4))

    def test_outputs_are_created(self):
        data = generate_customer_data(n_samples=120)
        assigned, profiles, selection, _ = fit_segmentation(data, k_values=range(2, 5))
        with tempfile.TemporaryDirectory() as tmp:
            save_outputs(assigned, profiles, selection, tmp)
            expected = [
                "customer_segments.csv",
                "cluster_profiles.csv",
                "cluster_selection_metrics.csv",
                "elbow_curve.png",
                "silhouette_scores.png",
                "customer_segments_scatter.png",
            ]
            for filename in expected:
                self.assertTrue((Path(tmp) / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
