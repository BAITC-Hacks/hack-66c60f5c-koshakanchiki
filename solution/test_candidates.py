"""Boundary tests for public-data catalog and portable historical ranks."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import candidates


def tariffs():
    return pd.DataFrame({
        "tariff_plan_code": ["a", "b", "c", "d", "e"],
        "price_tariff": [100, 200, 150, 80, 300],
        "Data_in_PKG": [10, 20, 15, 2, 50],
        "Min_another_operator_in_PKG": [10, 20, 30, 20, 50],
        "Min_another_operator_and_city_in_PKG": [0, 0, 20, 0, 50],
    })


class CandidateTests(unittest.TestCase):
    def test_prefix_uses_id_order_and_preserves_baseline(self):
        profile = pd.DataFrame({
            "ID_NUMBER": [8, 2, 5, 3, 4],
            "current_tariff": ["a", "a", "a", None, "b"],
            "arpu_segment": ["HIGH", "HIGH", "HIGH", "LOW", None],
            "predicted_arpu": [10000000.0, 0.0, 7.0, 3.0, 2.0],
        })
        original = profile.copy(deep=True)
        stats = candidates.cell_stats(profile)
        self.assertEqual(set(stats), {("a", "HIGH")})
        self.assertEqual(stats[("a", "HIGH")]["N"], 3)
        np.testing.assert_array_equal(stats[("a", "HIGH")]["ids_sorted"], [2, 5, 8])
        np.testing.assert_array_equal(stats[("a", "HIGH")]["prefix_arpu"], [0, 0, 7, 10000007])
        pd.testing.assert_frame_equal(profile, original)
        self.assertEqual(candidates.cell_stats(profile.iloc[:0]), {})

    def test_history_boundaries_clipping_support_and_artifact(self):
        rows = [
            (100.0, 1000.0, "a", "b"),  # clipped to 3
            (100.0, 1000.0, "a", "b"),  # exact duplicate
            (1000.0, 0.0, "a", "b"),   # LOW inclusive; ratio -1
            (5000.0, 7500.0, "a", "c"), # MID inclusive
            (5000.1, 2500.05, "a", "d"), # HIGH; negative preserved
            (99.0, 9900.0, "a", "b"),  # excluded denominator
        ]
        with tempfile.TemporaryDirectory() as directory:
            source, artifact = Path(directory) / "history.csv", Path(directory) / "ranks.json"
            pd.DataFrame(rows, columns=["AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M",
                                       "tariff_plan_code_from", "tariff_plan_code_to"]).to_csv(source, index=False)
            with patch.object(candidates, "HISTORY_PATH", artifact):
                ranks = candidates.load_or_build_historical_rank(source, tariffs())
                self.assertAlmostEqual(ranks[("a", "LOW", "b")], 2 / 12 * 1.0)
                self.assertAlmostEqual(ranks[("a", "MID", "c")], 0.5 / 11)
                self.assertAlmostEqual(ranks[("a", "HIGH", "d")], -0.5 / 11)
                self.assertEqual(json.loads(artifact.read_text())["counts"]["eligible_rows"], 4)
                self.assertEqual(ranks.support[("a", "LOW", "b")], 2)
                original = artifact.read_bytes()
                self.assertEqual(candidates.load_or_build_historical_rank(source, tariffs()), ranks)
                self.assertEqual(artifact.read_bytes(), original)
                source.unlink()
                self.assertEqual(candidates.load_or_build_historical_rank(artifact, tariffs()), ranks)

    def test_missing_and_malformed_history_fall_back_to_no_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            source, artifact = Path(directory) / "history.csv", Path(directory) / "ranks.json"
            def assert_zero_ranks(path):
                ranks = candidates.load_or_build_historical_rank(path, tariffs())
                self.assertFalse(ranks.support)
                self.assertTrue(ranks)
                self.assertTrue(all(rank == 0.0 for rank in ranks.values()))

            with patch.object(candidates, "HISTORY_PATH", artifact):
                with self.assertLogs(candidates.LOGGER, level="WARNING"):
                    assert_zero_ranks(source)
                artifact.write_text("[]")
                with self.assertLogs(candidates.LOGGER, level="WARNING"):
                    assert_zero_ranks(artifact)
                source.write_text("wrong,schema\n1,2\n")
                with self.assertLogs(candidates.LOGGER, level="WARNING"):
                    assert_zero_ranks(source)

    def test_catalog_unique_diverse_deterministic_and_small_cells_allowed(self):
        profile = pd.DataFrame([
            {"ID_NUMBER": i, "current_tariff": source, "arpu_segment": segment,
             "predicted_arpu": 10000.0 if segment == "HIGH" else 100.0}
            for i, (source, segment) in enumerate(
                (source, segment) for source in ["a", "b", "c", "d", "e"]
                for segment in ["HIGH", "MID", "LOW"]
            )
        ])
        ranks = {(source, segment, "b" if source != "b" else "a"): 0.0
                 for source, segment in candidates.cell_stats(profile)}
        catalog = candidates.build_catalog(profile, tariffs(), ranks)
        keys = [(a["from_tariff"], a["arpu_segment"], a["target"]) for a in catalog]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertLessEqual(len(keys), 40)
        self.assertEqual(len({key[:2] for key in keys[:6]}), 6)
        self.assertTrue(any(key[1] != "HIGH" for key in keys[:8]))
        self.assertTrue(any(key not in ranks for key in keys[:8]))
        self.assertTrue(all(source != target for source, _, target in keys))
        self.assertEqual(catalog, candidates.build_catalog(
            profile.sample(frac=1, random_state=9), tariffs().iloc[::-1], ranks
        ))
        for cell in candidates.cell_stats(profile):
            self.assertLessEqual(sum(key[:2] == cell for key in keys), 4)
        self.assertTrue(candidates.build_catalog(profile, tariffs(), {}))
        self.assertEqual(candidates.build_catalog(profile.iloc[:0], tariffs(), ranks), [])


if __name__ == "__main__":
    unittest.main()
