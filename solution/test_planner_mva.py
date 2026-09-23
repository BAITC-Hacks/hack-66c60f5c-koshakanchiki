"""Public planner contract compared with the independent official scorer."""
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

import scoring_core

import numpy as np
import pandas as pd

from planner_mva import simulate_plan, build_portfolio
from scoring_core import score_campaigns


def row(target="b", channel="sms", **filters):
    return dict(campaign_name="test", filter_arpu_segment=None,
                filter_data_segment=None, filter_call_segment=None,
                filter_current_tariff=None, target_tariff=target,
                channel=channel) | filters


class SimulatorTests(unittest.TestCase):
    def test_matches_official_score_with_negative_overlaps_and_conversion_cap(self):
        profile = pd.DataFrame({
            "ID_NUMBER": [4, 1, 3, 2], "current_tariff": ["a"] * 4,
            "arpu_segment": ["HIGH"] * 4, "data_segment": ["LITE"] * 4,
            "call_segment": ["LOW"] * 4,
            "predicted_arpu": [1000., 20., 400., 50.],
        })
        model = pd.DataFrame([
            dict(tariff_plan_code_from="a", arpu_segment="HIGH",
                 tariff_plan_code_to="b", arpu_change_pct=-.4, conversion_rate=.95),
            dict(tariff_plan_code_from="a", arpu_segment="HIGH",
                 tariff_plan_code_to="c", arpu_change_pct=-.1, conversion_rate=.95),
        ])
        rows = [row(), row("c", "call", filter_current_tariff=" a ; other ")]
        tariffs = pd.DataFrame({"tariff_plan_code": ["a", "b", "c"]})
        official = score_campaigns(pd.DataFrame(rows), profile, model, tariffs,
                                   profile.predicted_arpu.sum(), lambda *args: (0., 0.))
        ratios = {("a", "HIGH", r.tariff_plan_code_to): {
            "arpu_change_pct": r.arpu_change_pct, "conversion_rate": r.conversion_rate,
        } for r in model.itertuples()}
        net, detail = simulate_plan(profile, rows, {}, ratios)
        self.assertAlmostEqual(net, official["net_arpu_gain"], delta=1e-6)
        for field in ("total_contacts", "total_cost", "gross_arpu_lift", "unique_customers_targeted"):
            self.assertAlmostEqual(detail[field], official[field], delta=1e-6)
        self.assertEqual(detail["campaigns_detail"], official["campaigns_detail"])


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.profile = pd.DataFrame({
            "ID_NUMBER": [3, 1, 2, 6, 4, 5],
            "current_tariff": ["a"] * 3 + ["b"] * 3,
            "arpu_segment": ["HIGH"] * 6,
            "data_segment": ["LITE", "HEAVY", "LITE"] * 2,
            "call_segment": ["LOW"] * 6,
            "predicted_arpu": [1000., 500., 2000., 300., 1500., 800.],
        })
        self.stats = {}
        for key, part in self.profile.groupby(["current_tariff", "arpu_segment"]):
            part = part.sort_values("ID_NUMBER")
            self.stats[key] = dict(N=len(part), A_sum=float(part.predicted_arpu.sum()),
                ids_sorted=part.ID_NUMBER.to_numpy(),
                prefix_arpu=np.r_[0., np.cumsum(part.predicted_arpu)])
        self.resources = dict(profile=self.profile, remaining_contacts=6, remaining_budget=100)

    def test_spends_post_pilot_resources_only_on_measured_profitable_actions(self):
        profile = pd.DataFrame({
            "ID_NUMBER": range(8), "current_tariff": ["a"] * 4 + ["b"] * 4,
            "arpu_segment": ["HIGH"] * 8, "data_segment": ["LITE"] * 8,
            "call_segment": ["LOW"] * 8, "predicted_arpu": [1000.] * 8,
        })
        stats = {("a", "HIGH"): {"N": 4, "A_sum": 4000.,
                   "ids_sorted": np.arange(4), "prefix_arpu": np.arange(5) * 1000.},
                 ("b", "HIGH"): {"N": 4, "A_sum": 4000.,
                   "ids_sorted": np.arange(4, 8), "prefix_arpu": np.arange(5) * 1000.}}
        beliefs = {("a", "HIGH", "c"): {"mu": .2, "var": .0001, "n": 120},
                   ("b", "HIGH", "c"): {"mu": 5., "var": .0001, "n": 0}}
        resources = dict(profile=profile, remaining_budget=16, remaining_contacts=8)
        plan = build_portfolio(beliefs, stats, resources)
        self.assertTrue(plan)
        self.assertEqual({r["filter_current_tariff"] for r in plan}, {"a"})
        self.assertNotIn("call", [r["channel"] for r in plan])
        net, detail = simulate_plan(profile, plan, resources, {("a", "HIGH", "c"): .19})
        self.assertGreater(net, 0)
        self.assertLessEqual(detail["total_cost"], 16)
        self.assertLessEqual(detail["total_contacts"], 8)

    def test_object_contract_matches_dict_without_double_scaling(self):
        states = {("a", "HIGH", "c"): dict(mu=.2, var=.0001, n=120),
                  ("b", "HIGH", "c"): dict(mu=.1, var=.0004, n=180)}

        class PublicBeliefs:
            def mean_ratio(self, action, channel):
                return states[action]["mu"] * {"push": .5, "sms": .65, "digital_ads": .85}[channel] / .65

            def sd_ratio(self, action, channel):
                return np.sqrt(states[action]["var"]) * {"push": .5, "sms": .65, "digital_ads": .85}[channel] / .65

        expected = build_portfolio(states, self.stats, self.resources)
        actual = build_portfolio(PublicBeliefs(), self.stats,
                                 dict(self.resources, piloted_actions=list(states)))
        self.assertEqual(actual, expected)
        public = PublicBeliefs()
        public.observed = set(states)
        self.assertEqual(build_portfolio(public, self.stats, self.resources), expected)

    def test_stats_only_preserves_id_prefix_and_zero_budget_uses_push(self):
        states = {("a", "HIGH", "c"): dict(mu=.2, var=.0001, n=120)}
        resources = dict(remaining_contacts=2, remaining_budget=0)
        plan = build_portfolio(states, self.stats, resources)
        self.assertEqual(plan[0]["channel"], "push")
        _, detail = simulate_plan(self.profile, plan, resources, {("a", "HIGH", "c"): .19})
        np.testing.assert_array_equal(detail["selected_ids"][0], [1, 2])
        self.assertAlmostEqual(detail["gross_arpu_lift"], .19 * .5 / .65 * 2500)

    def test_high_mean_with_uncertainty_is_rejected_when_measured_winner_exists(self):
        states = {("a", "HIGH", "c"): dict(mu=.5, var=1., n=10),
                  ("b", "HIGH", "c"): dict(mu=.1, var=.0001, n=120)}
        plan = build_portfolio(states, self.stats, self.resources)
        self.assertEqual({r["filter_current_tariff"] for r in plan}, {"b"})

    def test_sunk_negative_pilots_do_not_replace_profitable_multirow_plan(self):
        states = {("a", "HIGH", "c"): dict(mu=.2, var=.0001, n=120),
                  ("b", "HIGH", "c"): dict(mu=.1, var=.0001, n=120)}
        resources = dict(self.resources, ledger=[{"cost": 99999, "observed_lift_ratio": -100}])
        plan = build_portfolio(states, self.stats, resources)
        self.assertGreaterEqual(len(plan), 2)
        _, detail = simulate_plan(self.profile, plan, resources,
                                  {a: s["mu"] - np.sqrt(s["var"]) for a, s in states.items()})
        gains = [r["gross_lift"] - r["cost"] for r in detail["campaigns_detail"]]
        self.assertEqual(gains, sorted(gains, reverse=True))
        self.assertEqual(detail["total_contacts"], detail["unique_customers_targeted"])
        self.assertTrue(all(value is None or isinstance(value, str)
                            for row_ in plan for value in row_.values()))
        self.assertEqual(plan, build_portfolio(states, self.stats, resources))

    def test_negative_world_returns_one_filterable_fallback(self):
        states = {("a", "HIGH", "c"): dict(mu=-.2, var=.01, n=120)}
        plan = build_portfolio(states, self.stats, self.resources)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["channel"], "push")
        net, detail = simulate_plan(self.profile, plan, self.resources, {("a", "HIGH", "c"): -.3})
        self.assertLess(net, 0)
        np.testing.assert_array_equal(detail["selected_ids"][0], [1])

    def test_empty_cell_and_exhausted_reach_return_no_campaign(self):
        states = {("missing", "HIGH", "c"): dict(mu=.5, var=.01, n=120)}
        self.assertEqual(build_portfolio(states, self.stats, self.resources), [])
        self.assertEqual(build_portfolio(states, self.stats,
                                        dict(self.resources, remaining_contacts=0)), [])


# Independent official-scorer boundary cases.
def campaign(target="tariff_20", channel="sms", **filters):
    return {
        "campaign_name": "edge",
        "filter_arpu_segment": None,
        "filter_data_segment": None,
        "filter_call_segment": None,
        "filter_current_tariff": None,
        "target_tariff": target,
        "channel": channel,
        **filters,
    }


def profile_fixture(n=12):
    ids = np.arange(1, n + 1)
    profile = pd.DataFrame({
        "ID_NUMBER": ids,
        "current_tariff": np.where(ids % 2, "tariff_1", "tariff_2"),
        "arpu_segment": np.where(ids % 3, "HIGH", "MID"),
        "data_segment": np.array(["LITE", "HEAVY", "NON_USER"])[ids % 3],
        "call_segment": np.where(ids % 2, "LOW", "MEDIUM"),
        "predicted_arpu": ids.astype(float) * 137.0 + 0.125,
    })
    return profile.sample(frac=1, random_state=31).reset_index(drop=True)


def effect_fixture(profile, negative=False):
    cells = profile[["current_tariff", "arpu_segment"]].drop_duplicates()
    records = []
    for number, cell in enumerate(cells.itertuples(index=False)):
        for target, effect in (("tariff_20", -.4 if negative else .28),
                               ("tariff_21", -.1 if negative else .13)):
            records.append({
                "tariff_plan_code_from": cell.current_tariff,
                "arpu_segment": cell.arpu_segment,
                "tariff_plan_code_to": target,
                "arpu_change_pct": effect,
                "conversion_rate": .55 + .02 * (number % 5),
            })
    return pd.DataFrame(records)


class SimulatorEdgeTests(unittest.TestCase):
    def compare(self, profile, rows, *, contacts=15000, budget=100000, negative=False):
        model = effect_fixture(profile, negative=negative)
        ratios = {
            (r.tariff_plan_code_from, r.arpu_segment, r.tariff_plan_code_to): {
                "arpu_change_pct": r.arpu_change_pct,
                "conversion_rate": r.conversion_rate,
            }
            for r in model.itertuples(index=False)
        }
        tariffs = pd.DataFrame({
            "tariff_plan_code": sorted(set(profile.current_tariff) | {"tariff_20", "tariff_21"})
        })

        def unexpected_fallback(*args):
            self.fail(f"Test fixture omitted an effect: {args[:3]}")

        with ExitStack() as stack:
            stack.enter_context(patch.object(scoring_core, "MAX_TOTAL_CONTACTS", contacts))
            stack.enter_context(patch.object(scoring_core, "TOTAL_BUDGET", budget))
            official = scoring_core.score_campaigns(
                pd.DataFrame(rows), profile, model, tariffs,
                float(profile.predicted_arpu.sum()), unexpected_fallback,
            )
        net, actual = simulate_plan(
            profile, rows,
            {"remaining_contacts": contacts, "remaining_budget": budget}, ratios,
        )
        self.assertAlmostEqual(net, official["net_arpu_gain"], delta=1e-6)
        for field in ("gross_arpu_lift", "net_arpu_gain", "total_cost", "total_contacts",
                      "unique_customers_targeted"):
            self.assertAlmostEqual(actual[field], official[field], delta=1e-6, msg=field)
        self.assertEqual(len(actual["campaigns_detail"]), len(official["campaigns_detail"]))
        for index, (got, expected) in enumerate(zip(actual["campaigns_detail"], official["campaigns_detail"])):
            self.assertEqual(set(got), set(expected))
            for field in expected:
                with self.subTest(row=index, field=field):
                    if field in {"cost", "gross_lift"}:
                        self.assertAlmostEqual(got[field], expected[field], delta=1e-6)
                    else:
                        self.assertEqual(got[field], expected[field])
        self.assertEqual(actual["remaining_contacts"], contacts - official["total_contacts"])
        self.assertAlmostEqual(actual["remaining_budget"], budget - official["total_cost"], delta=1e-6)
        return net, actual

    def test_cap5000_is_applied_before_resources_and_uses_sorted_ids(self):
        profile = profile_fixture(5011)
        _, detail = self.compare(profile, [campaign(channel="push")])
        self.assertEqual(detail["total_contacts"], 5000)
        self.assertTrue(detail["campaigns_detail"][0]["capped_at_campaign_limit"])
        np.testing.assert_array_equal(detail["selected_ids"][0], np.arange(1, 5001))

    def test_row_order_changes_reach_allocation(self):
        profile = profile_fixture()
        first = campaign(filter_current_tariff="tariff_1")
        second = campaign(target="tariff_21", filter_current_tariff="tariff_2")
        net_forward, forward = self.compare(profile, [first, second], contacts=8)
        net_reverse, reverse = self.compare(profile, [second, first], contacts=8)
        self.assertEqual([r["n_contacts"] for r in forward["campaigns_detail"]], [6, 2])
        self.assertEqual([r["n_contacts"] for r in reverse["campaigns_detail"]], [6, 2])
        self.assertNotAlmostEqual(net_forward, net_reverse)
        self.assertTrue(forward["campaigns_detail"][1]["capped_at_reach_budget"])

    def test_money_floor_then_push_uses_contact_remainder(self):
        rows = [campaign(), campaign(channel="digital_ads"), campaign(channel="push")]
        _, detail = self.compare(profile_fixture(), rows, contacts=5, budget=11)
        self.assertEqual([r["n_contacts"] for r in detail["campaigns_detail"]], [2, 0, 3])
        self.assertEqual(detail["total_cost"], 8)
        self.assertEqual(detail["remaining_budget"], 3)
        self.assertTrue(detail["campaigns_detail"][0]["capped_at_money_budget"])
        self.assertTrue(detail["campaigns_detail"][1]["capped_at_money_budget"])

    def test_zero_money_keeps_push_available(self):
        # The official report divides by TOTAL_BUDGET; preserve its positive
        # denominator while making a preceding explicit pilot spend all money.
        profile = profile_fixture()
        paid = campaign(explicit_ids=[1], channel="sms")
        push = campaign(channel="push")
        _, detail = self.compare(profile, [paid, campaign(), push], contacts=5, budget=4)
        self.assertEqual([r["n_contacts"] for r in detail["campaigns_detail"]], [1, 0, 4])
        _, zero = simulate_plan(
            profile, [campaign(), push],
            {"remaining_contacts": 4, "remaining_budget": 0}, {},
        )
        self.assertEqual([r["n_contacts"] for r in zero["campaigns_detail"]], [0, 4])
        self.assertEqual(zero["total_cost"], 0)

    def test_none_nan_and_empty_filters(self):
        rows = [
            campaign(filter_arpu_segment=None, filter_data_segment=np.nan),
            campaign(filter_current_tariff=""),
            campaign(filter_data_segment=""),
            campaign(filter_call_segment="UNKNOWN"),
        ]
        _, detail = self.compare(profile_fixture(), rows)
        self.assertEqual([r["n_contacts"] for r in detail["campaigns_detail"]], [12, 0, 0, 0])

    def test_empty_cell_and_zero_contact_balance(self):
        rows = [campaign(filter_current_tariff="tariff_missing"), campaign(channel="push")]
        _, detail = self.compare(profile_fixture(), rows, contacts=0)
        self.assertEqual(detail["total_contacts"], 0)
        self.assertEqual(detail["net_arpu_gain"], 0)

    def test_actual_id_prefix_preserves_outlier_baseline(self):
        profile = profile_fixture()
        profile.loc[profile.ID_NUMBER == 1, "predicted_arpu"] = 100_000_000.0
        _, detail = self.compare(profile, [campaign()], contacts=2)
        np.testing.assert_array_equal(detail["selected_ids"][0], [1, 2])
        self.assertGreater(detail["gross_arpu_lift"], 9_000_000)

    def test_negative_repeats_pay_twice_and_retain_negative_max(self):
        profile = profile_fixture()
        rows = [campaign(), campaign(target="tariff_21")]
        net, detail = self.compare(profile, rows, negative=True)
        _, one = self.compare(profile, [rows[1]], negative=True)
        self.assertLess(net, 0)
        self.assertLess(detail["gross_arpu_lift"], 0)
        self.assertAlmostEqual(detail["gross_arpu_lift"], one["gross_arpu_lift"], delta=1e-6)
        self.assertEqual(detail["total_contacts"], 24)
        self.assertEqual(detail["unique_customers_targeted"], 12)
        self.assertEqual(detail["total_cost"], 96)

    def test_explicit_ids_override_filters_for_pilot_parity_only(self):
        profile = profile_fixture()
        rows = [campaign(explicit_ids=[9, 3, 3, 999], filter_current_tariff="missing",
                         filter_arpu_segment="NOT_A_SEGMENT")]
        _, detail = self.compare(profile, rows)
        np.testing.assert_array_equal(detail["selected_ids"][0], [3, 9])
        self.assertEqual(detail["total_contacts"], 2)
        _, empty_explicit = self.compare(profile, [campaign(explicit_ids=[], filter_current_tariff="missing")])
        self.assertEqual(empty_explicit["total_contacts"], 0)

    def test_public_profile_multiple_tariffs_and_data_call_cells(self):
        path = Path(__file__).resolve().parent / "customer_profile.csv"
        if not path.exists():
            self.skipTest("Local public customer_profile.csv is not installed")
        columns = ["ID_NUMBER", "current_tariff", "arpu_segment", "data_segment",
                   "call_segment", "predicted_arpu"]
        profile = pd.read_csv(path, usecols=columns).dropna(subset=columns[1:3])
        self.assertGreater(profile.current_tariff.nunique(), 1)
        self.assertGreater(profile.data_segment.nunique(), 1)
        rows = [
            campaign(filter_current_tariff=" tariff_8 ; tariff_10 ", filter_arpu_segment="HIGH",
                     filter_data_segment="HEAVY"),
            campaign(target="tariff_21", channel="digital_ads", filter_arpu_segment="HIGH",
                     filter_call_segment="MEDIUM"),
            campaign(channel="push", filter_current_tariff="tariff_4", filter_data_segment="LITE"),
        ]
        _, detail = self.compare(profile, rows, contacts=3800, budget=14003)
        self.assertGreater(detail["total_contacts"], 0)
        self.assertGreater(detail["total_cost"], 0)


if __name__ == "__main__":
    unittest.main()
