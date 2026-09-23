"""Planner contract and resource regression checks (unittest, no extra deps)."""
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from planner import build_portfolio, simulate_plan, _expected_best
from beliefs import Beliefs
import planner


class KnownBeliefs:
    def __init__(self, values):
        self.values = values
        self.actions = tuple(sorted(values))
        self.observed = set(self.actions)

    def mean_ratio(self, key, channel):
        return self.values[key] * {"push": .5 / .65, "sms": 1., "digital_ads": .85 / .65}[channel]

    def sd_ratio(self, key, channel):
        return 0.

    def draws(self, n_draws, seed=42):
        return np.tile([self.values[key] for key in self.actions], (n_draws, 1))


def profile_and_stats():
    rows = []
    for t in range(3):
        for i in range(12):
            rows.append({
                "ID_NUMBER": t * 100 + i, "current_tariff": "tariff_%d" % t,
                "arpu_segment": "HIGH", "data_segment": ["NON_USER", "LITE", "HEAVY"][i % 3],
                "call_segment": ["LOW", "MEDIUM", "HIGH"][i % 3],
                "predicted_arpu": float(200 + i * 75 + t * 100),
            })
    profile = pd.DataFrame(rows).sample(frac=1., random_state=42)
    stats = {}
    for cell, part in profile.groupby(["current_tariff", "arpu_segment"]):
        part = part.sort_values("ID_NUMBER")
        stats[cell] = {"N": len(part), "A_sum": part.predicted_arpu.sum(),
                       "ids_sorted": part.ID_NUMBER.to_numpy(),
                       "prefix_arpu": np.r_[0., part.predicted_arpu.cumsum()]}
    return profile, stats


class PlannerTests(unittest.TestCase):
    def test_signed_expected_max_matches_event_enumeration(self):
        ratios = [np.array([-.4, .2]), np.array([-.1, -.3])]
        probs = [.25, .7]
        expected = np.zeros(2)
        for events in itertools.product([0, 1], repeat=2):
            probability = np.prod([p if seen else 1. - p for p, seen in zip(probs, events)])
            selected = [ratio for ratio, seen in zip(ratios, events) if seen]
            if selected:
                expected += probability * np.max(selected, axis=0)
        np.testing.assert_allclose(_expected_best(ratios, probs, 2), expected)

    def test_plan_is_deterministic_nonoverlapping_and_within_resources(self):
        profile, stats = profile_and_stats()
        beliefs = KnownBeliefs({(cell[0], cell[1], "tariff_9"): .2 for cell in stats})
        resources = {"remaining_budget": 180, "remaining_contacts": 25, "profile": profile}
        plan = build_portfolio(beliefs, stats, resources)
        self.assertTrue(plan)
        self.assertEqual(plan, build_portfolio(beliefs, stats, resources))
        net, detail = simulate_plan(profile, plan, resources, beliefs)
        self.assertGreater(net, 0)
        self.assertLessEqual(detail["total_contacts"], 25)
        self.assertLessEqual(detail["total_cost"], 180)
        self.assertEqual(detail["total_contacts"], detail["unique_customers_targeted"])
        self.assertLessEqual(len(plan), 10)
        self.assertTrue(all(row["channel"] != "call" for row in plan))
        self.assertTrue(all(len(row) == 7 for row in plan))

    def test_free_push_survives_zero_budget(self):
        profile, stats = profile_and_stats()
        beliefs = KnownBeliefs({(cell[0], cell[1], "tariff_9"): .1 for cell in stats})
        resources = {"remaining_budget": 0, "remaining_contacts": 20, "profile": profile}
        plan = build_portfolio(beliefs, stats, resources)
        self.assertTrue(plan)
        self.assertTrue(all(row["channel"] == "push" for row in plan))
        self.assertEqual(simulate_plan(profile, plan, resources, beliefs)[1]["total_cost"], 0)

    def test_losing_pilots_do_not_erase_profitable_final_plan(self):
        profile, stats = profile_and_stats()
        values = {(cell[0], cell[1], "tariff_9"): .1 for cell in stats}
        values[("tariff_0", "HIGH", "tariff_8")] = -50.
        beliefs = KnownBeliefs(values)
        resources = {
            "remaining_budget": 60, "remaining_contacts": 12, "profile": profile,
            "pilot_log": [{"action_key": ("tariff_0", "HIGH", "tariff_8"),
                           "channel": "sms", "actual_n": 6, "cost": 24}],
        }
        plan = build_portfolio(beliefs, stats, resources)
        self.assertTrue(plan)
        self.assertGreater(simulate_plan(profile, plan, resources, beliefs)[0], 0)

    def test_negative_effects_leave_fallback_to_agent(self):
        profile, stats = profile_and_stats()
        beliefs = KnownBeliefs({(cell[0], cell[1], "tariff_9"): -.1 for cell in stats})
        self.assertEqual(build_portfolio(beliefs, stats, {"profile": profile}), [])

    def test_empty_cells_and_zero_reach_are_safe(self):
        beliefs = KnownBeliefs({("tariff_0", "HIGH", "tariff_9"): .2})
        stats = {("tariff_0", "HIGH"): {"N": 0, "A_sum": 0, "ids_sorted": np.array([]),
                                           "prefix_arpu": np.array([0.])}}
        self.assertEqual(build_portfolio(beliefs, stats, {}), [])
        profile, stats = profile_and_stats()
        self.assertEqual(build_portfolio(beliefs, stats, {"remaining_contacts": 0}), [])

    def test_positive_prior_does_not_scale_unobserved_actions_at_k_zero(self):
        profile, stats = profile_and_stats()
        beliefs = Beliefs(prior_mu=.5)
        keys = [(cell[0], cell[1], "tariff_9") for cell in stats]
        beliefs.register(keys)
        self.assertEqual(build_portfolio(beliefs, stats, {"profile": profile}, k=0), [])
        beliefs.update(keys[0], "sms", 120, .3)
        plan = build_portfolio(beliefs, stats, {"profile": profile}, k=0)
        self.assertTrue(plan)
        self.assertTrue(all(row["filter_current_tariff"] == keys[0][0] for row in plan))

    def test_expired_deadline_returns_before_search(self):
        profile, stats = profile_and_stats()
        beliefs = KnownBeliefs({(cell[0], cell[1], "tariff_9"): .2 for cell in stats})
        with patch.object(beliefs, "draws", side_effect=AssertionError("late computation")):
            self.assertEqual(build_portfolio(beliefs, stats, {"deadline": -1}), [])

    def test_deadline_retains_best_completed_greedy_plan(self):
        profile, stats = profile_and_stats()
        beliefs = KnownBeliefs({(cell[0], cell[1], "tariff_9"): .2 for cell in stats})
        timed_out = [False]
        compatible = planner._compatible

        def expire_during_second_row(candidate, chosen):
            if chosen:
                timed_out[0] = True
            return compatible(candidate, chosen)

        resources = {"profile": profile, "deadline": 1}
        with patch("planner.time.monotonic", side_effect=lambda: 2 if timed_out[0] else 0), \
                patch("planner._compatible", side_effect=expire_during_second_row):
            plan = build_portfolio(beliefs, stats, resources)
        self.assertTrue(timed_out[0])
        self.assertTrue(plan)
        self.assertGreater(simulate_plan(profile, plan, {}, beliefs)[0], 0)


if __name__ == "__main__":
    unittest.main()
