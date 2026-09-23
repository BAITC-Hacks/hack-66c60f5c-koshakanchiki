"""Agent orchestration regressions using a public-API-only deterministic env."""

import math
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd

import agent as agent_module


class FakeEnv:
    """Small public environment with explicit resource accounting and faults."""

    def __init__(self, *, cells=8, customers_per_cell=200, response_mode=None,
                 ratios=None, missing_optional_segments=False):
        self.customer_profile = pd.DataFrame([
            {"ID_NUMBER": cell * customers_per_cell + customer,
             "current_tariff": f"t{cell}", "arpu_segment": "HIGH",
             "predicted_arpu": 1000.0,
             "data_segment": None if missing_optional_segments else "LITE",
             "call_segment": None if missing_optional_segments else "LOW"}
            for cell in range(cells) for customer in range(customers_per_cell)
        ])
        self.tariffs = pd.DataFrame({"tariff_plan_code": [f"t{i}" for i in range(cells + 1)]})
        self.remaining_contacts = 15000
        self.remaining_budget = 100000
        self.pilots_left = 20
        self.requests = []
        self.response_mode = response_mode
        self.ratios = [.8] + [-2.] * 20 if ratios is None else list(ratios)
        self.queue = [{"from_tariff": f"t{i}", "arpu_segment": "HIGH", "target": f"t{cells}"}
                      for i in range(cells)]

    def run_pilot(self, **request):
        requested = request["n_customers"]
        if not 10 <= requested <= 200 or request["channel"] != "sms":
            raise AssertionError("Invalid pilot request")
        self.requests.append(dict(request))
        audience = self.customer_profile[
            self.customer_profile.current_tariff.eq(request["filter_current_tariff"])
            & self.customer_profile.arpu_segment.eq(request["filter_arpu_segment"])
        ]
        actual_n = min(requested, len(audience), self.remaining_contacts, self.remaining_budget // 4)
        cost = 4 * actual_n
        self.remaining_contacts -= actual_n
        self.remaining_budget -= cost
        self.pilots_left -= 1
        ratio = self.ratios[min(len(self.requests) - 1, len(self.ratios) - 1)]
        if self.response_mode == "raise_after_charge":
            raise RuntimeError("pilot response lost after contact")
        if self.response_mode == "none":
            return None
        return {
            "n_customers": None if self.response_mode == "count_and_cost" else actual_n,
            "cost": None if self.response_mode == "count_and_cost" else cost,
            "observed_lift_ratio": ratio,
            "observed_lift_total": "unavailable" if self.response_mode == "total" else ratio * 1000 * actual_n,
        }


def simple_portfolio(beliefs, stats, resources, k=1.0):
    """Isolate orchestration from optimization: only positive cautious actions."""
    rows = []
    for key in sorted(beliefs.observed):
        if beliefs.mean_ratio(key, "sms") - beliefs.sd_ratio(key, "sms") > 0:
            rows.append({
                "campaign_name": "verified_positive", "filter_arpu_segment": key[1],
                "filter_data_segment": None, "filter_call_segment": None,
                "filter_current_tariff": key[0], "target_tariff": key[2], "channel": "sms",
            })
    return rows[:10]


class AgentTests(unittest.TestCase):
    def run_agent(self, env, instance=None, catalog_error=None):
        instance = agent_module.Agent() if instance is None else instance
        with ExitStack() as stack:
            stack.enter_context(patch.object(agent_module, "load_or_build_historical_rank", return_value={}))
            stack.enter_context(patch.object(agent_module, "build_catalog", return_value=env.queue,
                                            side_effect=catalog_error))
            planner = stack.enter_context(patch.object(agent_module, "build_portfolio", side_effect=simple_portfolio))
            plan = instance.act(env)
        return instance, plan, planner

    def assert_valid_nonempty(self, env, plan):
        self.assertTrue(1 <= len(plan) <= 10)
        contacts, budget = env.remaining_contacts, env.remaining_budget
        used = set()
        for row in plan:
            self.assertEqual(set(row), set(agent_module.FIELDS))
            self.assertIn(row["target_tariff"], set(env.tariffs.tariff_plan_code))
            audience = env.customer_profile
            for column in ["current_tariff", "arpu_segment", "data_segment", "call_segment"]:
                value = row["filter_" + column]
                if value is not None:
                    self.assertFalse(pd.isna(value))
                    if column == "current_tariff":
                        audience = audience[audience[column].isin(str(value).split(";"))]
                    else:
                        audience = audience[audience[column].eq(value)]
            audience = audience.sort_values("ID_NUMBER").iloc[:5000]
            cost = agent_module.COSTS[row["channel"]]
            n = min(len(audience), contacts, budget // cost if cost else contacts)
            self.assertGreater(n, 0)
            selected = set(audience.iloc[:n].ID_NUMBER)
            self.assertFalse(used & selected)
            used |= selected
            contacts -= n
            budget -= n * cost
        self.assertGreaterEqual(contacts, 0)
        self.assertGreaterEqual(budget, 0)

    def test_negative_refinement_drops_stale_paid_plan(self):
        env = FakeEnv()
        instance, plan, planner = self.run_agent(env)
        self.assertGreaterEqual(len(env.requests), 9)
        self.assertLess(instance.beliefs.posterior(("t0", "HIGH", "t8"))[0], 0)
        self.assertGreater(planner.call_count, 1)
        self.assert_valid_nonempty(env, plan)
        # Every measured action is now negative; the previous paid plan cannot
        # survive merely because the new optimizer returned an empty list.
        self.assertTrue(all(row["channel"] == "push" for row in plan))

    def test_catalog_failure_still_returns_nonempty_fallback(self):
        env = FakeEnv(cells=1, customers_per_cell=7)
        with self.assertLogs(agent_module.LOG, level="ERROR"):
            _, plan, _ = self.run_agent(env, catalog_error=RuntimeError("catalog unavailable"))
        self.assert_valid_nonempty(env, plan)
        self.assertEqual(env.requests, [])

    def test_malformed_response_keeps_resource_journal_numeric(self):
        for mode in ["none", "count_and_cost"]:
            with self.subTest(mode=mode):
                env = FakeEnv(cells=1, customers_per_cell=7, response_mode=mode, ratios=[.5])
                instance, plan, _ = self.run_agent(env)
                self.assert_valid_nonempty(env, plan)
                self.assertEqual(len(instance.pilot_log), len(env.requests))
                self.assertTrue(all(math.isfinite(record["actual_n"]) and math.isfinite(record["cost"])
                                    for record in instance.pilot_log))
                self.assertEqual(sum(r["actual_n"] for r in instance.pilot_log), 15000 - env.remaining_contacts)
                self.assertEqual(sum(r["cost"] for r in instance.pilot_log), 100000 - env.remaining_budget)

    def test_bad_optional_total_does_not_discard_valid_observation(self):
        env = FakeEnv(cells=1, customers_per_cell=200, response_mode="total", ratios=[.5])
        instance, plan, planner = self.run_agent(env)
        self.assertIn(("t0", "HIGH", "t1"), instance.beliefs.observed)
        self.assertGreater(planner.call_count, 0)
        self.assert_valid_nonempty(env, plan)
        self.assertEqual(plan[0]["channel"], "sms")
        self.assertNotIn("sample_arpu", instance.pilot_log[0])

    def test_pilot_exception_after_charge_keeps_cost_and_fallback(self):
        env = FakeEnv(cells=1, customers_per_cell=7, response_mode="raise_after_charge")
        with self.assertLogs(agent_module.LOG, level="WARNING"):
            instance, plan, _ = self.run_agent(env)
        self.assert_valid_nonempty(env, plan)
        self.assertEqual(len(instance.pilot_log), 1)
        self.assertIn("error", instance.pilot_log[0])
        self.assertEqual(instance.pilot_log[0]["actual_n"], 7)
        self.assertEqual(instance.pilot_log[0]["cost"], 28)
        self.assertFalse(instance.beliefs.observed)

    def test_small_actual_sample_none_filters_and_repeated_act(self):
        kwargs = dict(cells=1, customers_per_cell=7, ratios=[.5], missing_optional_segments=True)
        first_env = FakeEnv(**kwargs)
        instance, first_plan, _ = self.run_agent(first_env)
        first_state = {key: instance.beliefs.posterior(key) for key in instance.beliefs.actions}
        second_env = FakeEnv(**kwargs)
        instance, second_plan, _ = self.run_agent(second_env, instance=instance)
        self.assert_valid_nonempty(second_env, second_plan)
        self.assertEqual(first_plan, second_plan)
        self.assertEqual(first_state, {key: instance.beliefs.posterior(key) for key in instance.beliefs.actions})
        self.assertEqual(first_env.requests, second_env.requests)
        self.assertEqual(first_env.requests[0]["n_customers"], 10)
        self.assertEqual(instance.pilot_log[0]["actual_n"], 7)
        self.assertIsNone(second_plan[0]["filter_data_segment"])
        self.assertIsNone(second_plan[0]["filter_call_segment"])


if __name__ == "__main__":
    unittest.main()
