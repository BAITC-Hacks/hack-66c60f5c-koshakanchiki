"""Small reproducible checks of public mechanics on invented, non-secret data.

Run from any directory: python research_mechanics_checks.py
No competition agent is implemented. Starter files are imported, never changed.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from environment import make_environment, _apply_filters
from scoring_core import CHANNELS, TOTAL_BUDGET, MAX_TOTAL_CONTACTS, apply_filters, score_campaign, score_campaigns
from mock_environment import _mock_fallback


ROOT = Path(__file__).resolve().parent


def make_profile(n=2):
    return pd.DataFrame({
        "ID_NUMBER": np.arange(n), "current_tariff": ["a"] * n,
        "arpu_segment": ["MID"] * n, "data_segment": ["LITE"] * n,
        "call_segment": ["LOW"] * n, "predicted_arpu": [100.0] * n,
    })


TARIFFS = pd.DataFrame({"tariff_plan_code": ["a", "b", "positive", "negative", "zero"],
                        "price_tariff": [100., 200., 300., 50., 100.]})
MODEL = pd.DataFrame([
    {"tariff_plan_code_from": source, "arpu_segment": "MID", "tariff_plan_code_to": target,
     "arpu_change_pct": delta, "conversion_rate": 1.0}
    for source in ["a", "b"]
    for target, delta in [("positive", 1.0 if source == "a" else -.8), ("negative", -.2), ("zero", 0.)]
])


def fallback(*args):
    return 0., 0.


def score(profile, campaigns):
    return score_campaigns(pd.DataFrame(campaigns), profile, MODEL, TARIFFS,
                           float(profile.predicted_arpu.sum()), fallback)


def campaign(target="positive", channel="push", **kwargs):
    return {"target_tariff": target, "channel": channel, **kwargs}


def environment(profile, seed=42, contacts=MAX_TOTAL_CONTACTS, budget=TOTAL_BUDGET):
    return make_environment(profile, MODEL, TARIFFS, CHANNELS, budget, contacts, fallback, seed)


def main():
    records = []

    def check(name, condition, **details):
        assert bool(condition), (name, details)
        records.append({"name": name, "passed": True, **details})

    p = make_profile(2)
    p["predicted_arpu"] = [100., 10000.]
    p["data_segment"] = ["LITE", "HEAVY"]
    scored = score_campaign(p, "positive", MODEL, TARIFFS, 0., "push", fallback)
    ratios = scored.expected_lift_per_customer / scored.predicted_arpu
    check("data_segment_does_not_change_ratio_within_effect_cell", np.allclose(ratios, [.5, .5]),
          lift_ratios=ratios.tolist(), absolute_lifts=scored.expected_lift_per_customer.tolist())

    lite = score(p, [campaign(filter_data_segment="LITE")])
    heavy = score(p, [campaign(filter_data_segment="HEAVY")])
    check("data_filter_changes_arpu_sum_and_gross_lift", lite["gross_arpu_lift"] == 50. and heavy["gross_arpu_lift"] == 5000.,
          lite_gross=lite["gross_arpu_lift"], heavy_gross=heavy["gross_arpu_lift"])

    mixed = p.copy()
    mixed.loc[1, "current_tariff"] = "b"
    env, internals = environment(mixed)
    observed = env.run_pilot("positive", "push", n_customers=10)
    uniform = mixed.assign(predicted_arpu=100.)
    env_uniform, _ = environment(uniform)
    observed_uniform = env_uniform.run_pilot("positive", "push", n_customers=10)
    true_ratios = np.array([.5, -.4])
    check("pilot_ratio_is_unweighted_by_arpu", observed["observed_lift_ratio"] == observed_uniform["observed_lift_ratio"],
          observed_ratio=observed["observed_lift_ratio"], true_unweighted_ratio=float(true_ratios.mean()),
          true_arpu_weighted_ratio=float(np.dot(true_ratios, mixed.predicted_arpu) / mixed.predicted_arpu.sum()))
    true_gross = score(mixed, [campaign()])["gross_arpu_lift"]
    implied_total = float(true_ratios.mean() * mixed.predicted_arpu.sum())
    check("mixed_pilot_mean_times_arpu_can_have_wrong_monetary_sign", implied_total > 0 and true_gross < 0,
          noiseless_mean_times_arpu=implied_total, actual_gross=true_gross,
          reported_observed_lift_total=observed["observed_lift_total"])
    check("pilot_actual_sample_can_be_below_requested_minimum", observed["n_customers"] == 2,
          requested=10, actual=observed["n_customers"])

    negative = score(make_profile(), [campaign("negative")])
    check("negative_lift_not_clamped_against_no_action", np.isclose(negative["gross_arpu_lift"], -20.),
          gross=negative["gross_arpu_lift"], negative_share_pct=negative["risk_score_pct"])
    repeated = score(make_profile(), [campaign(channel="sms"), campaign(channel="sms")])
    check("duplicate_contacts_cost_money_and_reach_without_double_lift",
          repeated["total_contacts"] == 4 and repeated["unique_customers_targeted"] == 2
          and repeated["total_cost"] == 16 and np.isclose(repeated["gross_arpu_lift"], 130.),
          contacts=repeated["total_contacts"], unique=repeated["unique_customers_targeted"],
          cost=repeated["total_cost"], gross=repeated["gross_arpu_lift"])
    improved = score(make_profile(), [campaign("negative"), campaign("positive")])
    check("later_better_campaign_replaces_negative_for_same_customer", np.isclose(improved["gross_arpu_lift"], 100.),
          gross=improved["gross_arpu_lift"], contacts=improved["total_contacts"])

    big = make_profile(5001)
    big.loc[5000, "predicted_arpu"] = 1000000.
    forward = score(big, [campaign()])
    backward = score(big.iloc[::-1], [campaign()])
    check("campaign_cap_uses_smallest_ids_not_highest_arpu", forward["gross_arpu_lift"] == backward["gross_arpu_lift"] == 250000.,
          contacts=forward["total_contacts"], gross=forward["gross_arpu_lift"],
          highest_id_arpu=float(big.loc[5000, "predicted_arpu"]))
    broad_repeat = score(big, [campaign(), campaign(), campaign()])
    check("repeating_broad_filter_does_not_page_to_new_customers", broad_repeat["total_contacts"] == 15000
          and broad_repeat["unique_customers_targeted"] == 5000,
          contacts=broad_repeat["total_contacts"], unique=broad_repeat["unique_customers_targeted"])
    expensive_first = score(big, [campaign(channel="call"), campaign(channel="sms")])
    cheap_first = score(big, [campaign(channel="sms"), campaign(channel="call")])
    check("order_changes_budget_allocation", [c["n_contacts"] for c in expensive_first["campaigns_detail"]] == [625, 0]
          and [c["n_contacts"] for c in cheap_first["campaigns_detail"]] == [5000, 500],
          call_then_sms_contacts=[c["n_contacts"] for c in expensive_first["campaigns_detail"]],
          sms_then_call_contacts=[c["n_contacts"] for c in cheap_first["campaigns_detail"]])

    explicit = apply_filters(p, pd.Series(campaign(filter_data_segment="IMPOSSIBLE", explicit_ids=[1])))
    empty_explicit = apply_filters(p, pd.Series(campaign(explicit_ids=[])))
    check("nonempty_explicit_ids_override_filters_empty_list_does_not", explicit.ID_NUMBER.tolist() == [1]
          and len(empty_explicit) == 2, explicit_ids_result=explicit.ID_NUMBER.tolist(), empty_list_count=len(empty_explicit))

    small = make_profile(10)
    env, internals = environment(small)
    env.run_pilot("positive", "push", 10)
    env.run_pilot("positive", "push", 10)
    executed = internals.executed_pilot_campaigns()
    pilot_score = score(small, executed)
    check("pilots_can_recontact_same_people", pilot_score["total_contacts"] == 20
          and pilot_score["unique_customers_targeted"] == 10,
          contacts=pilot_score["total_contacts"], unique=pilot_score["unique_customers_targeted"])

    env1, internals1 = environment(make_profile(100), seed=1)
    env2, internals2 = environment(make_profile(100), seed=2)
    r1 = env1.run_pilot("positive", "sms", 20)
    r2 = env2.run_pilot("positive", "sms", 20)
    s1 = score(make_profile(100), internals1.executed_pilot_campaigns())
    s2 = score(make_profile(100), internals2.executed_pilot_campaigns())
    check("seeds_change_sample_and_noise_not_fixed_effect", r1["observed_lift_ratio"] != r2["observed_lift_ratio"]
          and s1["gross_arpu_lift"] == s2["gross_arpu_lift"],
          observed_ratios=[r1["observed_lift_ratio"], r2["observed_lift_ratio"]],
          true_gross_values=[s1["gross_arpu_lift"], s2["gross_arpu_lift"]])

    missing = make_profile(3)
    missing.loc[0, "current_tariff"] = np.nan
    missing.loc[1, "arpu_segment"] = np.nan
    missing.loc[2, "data_segment"] = np.nan
    scored_missing = score_campaign(missing, "positive", MODEL, TARIFFS, .2, "push", _mock_fallback)
    check("mock_fallback_handles_nan_tariff_and_nan_arpu_in_object_columns",
          len(scored_missing) == 3 and np.isfinite(scored_missing.expected_lift_per_customer).all(),
          per_customer_lifts=scored_missing.expected_lift_per_customer.tolist())
    selected_known = apply_filters(missing, pd.Series(campaign(filter_current_tariff="a", filter_arpu_segment="MID")))
    nan_filter = apply_filters(missing, pd.Series(campaign(filter_arpu_segment=np.nan)))
    check("missing_values_excluded_by_equality_filters_nan_filter_means_no_filter",
          selected_known.ID_NUMBER.tolist() == [2] and len(nan_filter) == 3,
          selected_known_ids=selected_known.ID_NUMBER.tolist(), nan_filter_count=len(nan_filter))

    pdna_profile = make_profile(3).astype({"arpu_segment": "object"})
    try:
        pdna_count = len(_apply_filters(pdna_profile, {"filter_arpu_segment": pd.NA}))
        pdna_result = f"selected_rows={pdna_count}"
    except Exception as error:
        pdna_count = None
        pdna_result = f"{type(error).__name__}: {error}"
    check("pd_na_filter_is_not_equivalent_to_none_in_pilot_filter", pdna_count != 3
          and len(_apply_filters(pdna_profile, {"filter_arpu_segment": None})) == 3,
          pd_na_outcome=pdna_result, scoring_filter_rows=len(apply_filters(pdna_profile, pd.Series({"filter_arpu_segment": pd.NA}))),
          note="Outcome depends on pandas dtype/version; use None for absent API filters.")

    def strict_fallback(current, target, segment, tariffs, conversion):
        return {"MID": (.1, .5)}[segment]

    try:
        score_campaign(missing.iloc[[1]], "positive", MODEL, TARIFFS, .2, "push", strict_fallback)
        strict_error = None
    except Exception as error:
        strict_error = type(error).__name__
    check("fallback_errors_propagate_from_scorer", strict_error == "KeyError", error_type=strict_error,
          note="Invented strict fallback, not a claim about hidden evaluator.")

    zero_profile = make_profile(3)
    zero_profile["predicted_arpu"] = [0., 0., 100.]
    zero_profile["data_segment"] = ["LITE", "LITE", "HEAVY"]
    zero = score(zero_profile, [campaign(channel="sms", filter_data_segment="LITE")])
    check("zero_arpu_contacts_can_only_spend_budget_for_fixed_ratio", zero["gross_arpu_lift"] == 0
          and zero["net_arpu_gain"] == -8., gross=zero["gross_arpu_lift"], net=zero["net_arpu_gain"])

    result = {"scope": "public mechanics; invented data and impact models; no hidden-state access",
              "python_libraries": {"pandas": pd.__version__, "numpy": np.__version__},
              "checks_passed": len(records), "checks": records}
    output = ROOT / "research_tables" / "mechanics_checks.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
