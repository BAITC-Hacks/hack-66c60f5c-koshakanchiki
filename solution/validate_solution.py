"""Independent delivery checks; this module is never imported by the agent.

The harness owns its synthetic effect models and evaluator-side pilot records.
Only the documented environment API is passed to Agent.act.  Synthetic worlds
are correctness smoke tests, not evidence of superiority on hidden effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from environment import make_environment
from make_submission import CAMPAIGN_COLUMNS
from scoring_core import CHANNELS, apply_filters, score_campaigns, validate_strategy


ROOT = Path(__file__).resolve().parent
FILTERS = CAMPAIGN_COLUMNS[1:5]
PUBLIC_ATTRIBUTES = {
    "customer_profile", "tariffs", "channels", "total_budget",
    "max_total_contacts", "remaining_budget", "remaining_contacts",
    "pilots_left", "pilot_history",
}


def close(actual, expected, label="value"):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-7), (
        f"{label}: {actual!r} != {expected!r}"
    )


def row(target="tariff_b", channel="sms", current="tariff_a", segment="HIGH", name="test"):
    return {
        "campaign_name": name,
        "filter_arpu_segment": segment,
        "filter_data_segment": None,
        "filter_call_segment": None,
        "filter_current_tariff": current,
        "target_tariff": target,
        "channel": channel,
    }


def tiny_profile(n=7):
    # Reverse ID order and nonconstant baseline expose prefix/mean mistakes.
    return pd.DataFrame({
        "ID_NUMBER": np.arange(n, 0, -1),
        "current_tariff": ["tariff_a"] * n,
        "arpu_segment": ["HIGH"] * n,
        "data_segment": ["HEAVY"] * n,
        "call_segment": ["LOW"] * n,
        "predicted_arpu": np.arange(n, 0, -1, dtype=float) * 1000,
    })


def tiny_tariffs():
    return pd.DataFrame({
        "tariff_plan_code": ["tariff_a", "tariff_b", "tariff_c"],
        "price_tariff": [1000, 2000, 3000],
        "Data_in_PKG": [1, 2, 3],
        "Min_another_operator_in_PKG": [10, 20, 30],
        "Min_another_operator_and_city_in_PKG": [5, 10, 15],
    })


def model_from_ratios(ratios):
    # q=.5 stays below saturation for every channel; ratio keys use SMS scale.
    return pd.DataFrame([
        {"tariff_plan_code_from": key[0], "arpu_segment": key[1],
         "tariff_plan_code_to": key[2], "arpu_change_pct": ratio / (.65 * .5),
         "conversion_rate": .5}
        for key, ratio in sorted(ratios.items())
    ])


def zero_fallback(*_args):
    return 0.0, .5


class AuditedEnvironment:
    """Expose precisely the public API and inspect requests outside the agent."""

    def __init__(self, env):
        self._env = env
        self.requests = []

    def __getattr__(self, name):
        if name in PUBLIC_ATTRIBUTES:
            return getattr(self._env, name)
        raise AttributeError(f"Not part of the public environment API: {name}")

    def run_pilot(self, *args, **kwargs):
        assert not args, "Pilot requests must use named arguments for auditing"
        requested = kwargs.get("n_customers", 100)
        assert 10 <= requested <= 200, f"Pilot request outside 10..200: {requested}"
        assert kwargs.get("channel") != "call", "Call is disabled in this version"
        result = self._env.run_pilot(**kwargs)
        self.requests.append({"request": dict(kwargs), "result": dict(result)})
        return result


def assert_rows(rows, env):
    assert isinstance(rows, list) and 1 <= len(rows) <= 10, "Need 1..10 final rows"
    for campaign in rows:
        assert isinstance(campaign, dict) and set(campaign) == set(CAMPAIGN_COLUMNS)
        assert isinstance(campaign["campaign_name"], str) and campaign["campaign_name"]
        assert campaign["channel"] in {"push", "sms", "digital_ads"}
        for name in FILTERS:
            assert campaign[name] is None or isinstance(campaign[name], str), (
                f"{name} must contain a string or Python None, not NaN/NA"
            )
        assert len(apply_filters(env.customer_profile, pd.Series(campaign))) > 0, (
            f"Empty final audience: {campaign}"
        )
    validate_strategy(pd.DataFrame(rows), env.tariffs)

    # Every returned row must actually execute, even after preceding rows.
    contacts, money = env.remaining_contacts, env.remaining_budget
    final_contacts, final_cost = 0, 0
    for campaign in rows:
        audience = apply_filters(env.customer_profile, pd.Series(campaign))
        count = min(len(audience), 5000, contacts)
        cost = env.channels[campaign["channel"]]["cost_per_contact"]
        if cost:
            count = min(count, int(money // cost))
        assert count > 0, "A final row is exhausted by preceding rows"
        contacts -= count
        money -= count * cost
        final_contacts += count
        final_cost += count * cost
    assert contacts >= 0 and money >= 0
    return {"final_contacts": final_contacts, "final_cost": final_cost}


def check_beliefs():
    from beliefs import Beliefs

    key = ("tariff_a", "HIGH", "tariff_b")
    for channel in ("push", "sms", "digital_ads"):
        beliefs = Beliefs()
        beliefs.register([key])
        n, observed = 7, -.24  # actual_n may be below the legal request minimum.
        h = CHANNELS[channel]["conversion_multiplier"] / .65
        noise_var = .804 ** 2 / n
        expected_var = 1 / (1 / .25 ** 2 + h ** 2 / noise_var)
        expected_mu = expected_var * h * observed / noise_var
        assert beliefs.update(key, channel, n, observed)
        mu, var = beliefs.posterior(key)
        close(mu, expected_mu, f"{channel} posterior mean")
        close(var, expected_var, f"{channel} posterior variance")
        assert mu < 0, "Negative observations must not be clipped"
        close(beliefs.mean_ratio(key, channel), h * mu)
        close(beliefs.sd_ratio(key, channel), h * math.sqrt(var))
        state = beliefs.posterior(key)
        for bad_n, bad_ratio in ((0, .1), (-1, .1), (120, None), (120, float("nan")),
                                 (120, float("inf")), (120, "not a number")):
            assert not beliefs.update(key, "sms", bad_n, bad_ratio)
            assert beliefs.posterior(key) == state, "Invalid response changed posterior"

    beliefs = Beliefs()
    second = ("tariff_a", "MID", "tariff_c")
    beliefs.register([second, key])
    assert tuple(beliefs.actions) == tuple(sorted([key, second]))
    before = beliefs.draws(64, seed=42)
    assert before.shape == (64, 2) and np.isfinite(before).all()
    np.testing.assert_array_equal(before, beliefs.draws(64, seed=42))
    means_before = np.array([beliefs.posterior(k)[0] for k in beliefs.actions])
    sds_before = np.sqrt([beliefs.posterior(k)[1] for k in beliefs.actions])
    z_before = (before - means_before) / sds_before
    beliefs.update(key, "sms", 120, .15)
    after = beliefs.draws(64, seed=42)
    means_after = np.array([beliefs.posterior(k)[0] for k in beliefs.actions])
    sds_after = np.sqrt([beliefs.posterior(k)[1] for k in beliefs.actions])
    np.testing.assert_allclose((after - means_after) / sds_after, z_before,
                               rtol=1e-12, atol=1e-12)
    return {"channels": 3, "draws": list(before.shape), "actual_n": 7}


def check_cells():
    from candidates import cell_stats

    profile = tiny_profile()
    profile.loc[profile.ID_NUMBER == 1, "predicted_arpu"] = 0
    profile.loc[profile.ID_NUMBER == 7, "predicted_arpu"] = 1e9
    before = profile.copy(deep=True)
    stats = cell_stats(profile)
    cell = stats[("tariff_a", "HIGH")]
    assert cell["N"] == 7
    np.testing.assert_array_equal(cell["ids_sorted"], np.arange(1, 8))
    expected = np.r_[0., profile.sort_values("ID_NUMBER").predicted_arpu.cumsum()]
    np.testing.assert_array_equal(cell["prefix_arpu"], expected)
    close(cell["A_sum"], expected[-1], "No baseline clipping")
    pd.testing.assert_frame_equal(profile, before)
    assert cell_stats(profile.iloc[:0]) == {}
    return {"empty_cell": "ok", "baseline_outlier": "preserved"}


def check_simulator():
    from planner import simulate_plan

    profile = tiny_profile(5207)
    ratios = {("tariff_a", "HIGH", "tariff_b"): -.1,
              ("tariff_a", "HIGH", "tariff_c"): -.2}
    campaigns = [row(channel="push", name="negative_push"),
                 row(target="tariff_c", channel="sms", name="worse_repeat")]
    model = model_from_ratios(ratios)
    scored = score_campaigns(pd.DataFrame(campaigns), profile, model, tiny_tariffs(),
                             profile.predicted_arpu.sum(), zero_fallback)
    net, detail = simulate_plan(profile, campaigns,
                                {"remaining_budget": 100000, "remaining_contacts": 15000}, ratios)
    close(net, scored["net_arpu_gain"], "Simulator vs official net")
    for key in ("gross_arpu_lift", "total_cost", "total_contacts", "unique_customers_targeted"):
        close(detail[key], scored[key], key)
    assert detail["gross_arpu_lift"] < 0, "Max over contacts must not add imaginary zero"
    assert detail["campaigns_detail"][0]["ids"] == list(range(1, 5001))

    small = tiny_profile()
    positive = {("tariff_a", "HIGH", "tariff_b"): .13}
    limited = {"remaining_budget": 9, "remaining_contacts": 4}
    net, detail = simulate_plan(small, [row(), row(channel="push", name="free")], limited, positive)
    # First two IDs receive both campaigns: dedup keeps SMS .13, contacts still cost.
    close(net, .13 * 3000 - 8)
    assert detail["total_contacts"] == 4 and detail["unique_customers_targeted"] == 2
    free_net, free_detail = simulate_plan(small, [row(channel="push")],
                                          {"remaining_budget": 0, "remaining_contacts": 3}, positive)
    close(free_net, .1 * 6000)
    assert free_detail["total_contacts"] == 3 and free_detail["total_cost"] == 0
    empty = row(segment="LOW")
    empty_net, empty_detail = simulate_plan(small, [empty], limited, positive)
    assert empty_net == 0 and empty_detail["total_contacts"] == 0
    return {"official_match": True, "cap": 5000, "negative_max": "preserved", "free_budget": "ok"}


def check_small_pilot():
    from beliefs import Beliefs

    key = ("tariff_a", "HIGH", "tariff_b")
    env, _ = make_environment(tiny_profile(), model_from_ratios({key: .1}), tiny_tariffs(),
                              CHANNELS, 100000, 15000, zero_fallback, seed=42)
    request = row()
    request.pop("campaign_name")
    result = env.run_pilot(n_customers=10, **request)
    assert result["n_customers"] == 7
    assert env.remaining_contacts == 14993 and env.remaining_budget == 99972
    beliefs = Beliefs()
    assert beliefs.update(key, "sms", result["n_customers"], result["observed_lift_ratio"])
    before = (env.remaining_contacts, env.remaining_budget, env.pilots_left)
    try:
        env.run_pilot(n_customers=10, **{**request, "filter_arpu_segment": "LOW"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("Empty pilot audience should fail")
    assert before == (env.remaining_contacts, env.remaining_budget, env.pilots_left)
    return {"requested": 10, "actual": 7, "empty_pilot": "no resource mutation"}


def run_agent(env, agent=None, require_pilot=True):
    from agent import Agent

    proxy = AuditedEnvironment(env)
    profile_before = env.customer_profile.copy(deep=True)
    started = time.perf_counter()
    rows = (Agent() if agent is None else agent).act(proxy)
    elapsed = time.perf_counter() - started
    assert elapsed < 240, f"Agent exceeded internal return deadline: {elapsed:.2f}s"
    details = assert_rows(rows, env)
    pd.testing.assert_frame_equal(env.customer_profile, profile_before)
    if require_pilot:
        assert proxy.requests, "Working environment must receive a successful pilot"
    assert len(proxy.requests) <= 16
    pilot_contacts = sum(item["result"]["n_customers"] for item in proxy.requests)
    pilot_cost = sum(item["result"]["cost"] for item in proxy.requests)
    assert pilot_contacts <= 2480 and pilot_cost <= 9920
    assert pilot_contacts + details["final_contacts"] <= env.max_total_contacts
    assert pilot_cost + details["final_cost"] <= env.total_budget
    return rows, {**details, "pilot_calls": len(proxy.requests), "pilot_contacts": pilot_contacts,
                  "pilot_cost": pilot_cost, "elapsed_seconds": elapsed}


def check_agent_contract():
    from agent import Agent
    from mock_environment import make_mock_env

    same_agent = Agent()
    env, _ = make_mock_env(seed=42)
    first, details = run_agent(env, same_agent)
    env_again, _ = make_mock_env(seed=42)
    second, repeated = run_agent(env_again, same_agent)
    assert first == second, "Same Agent instance retained state across environments"
    for key in ("pilot_calls", "pilot_contacts", "pilot_cost"):
        assert details[key] == repeated[key]
    return {**details, "repeat_same_instance": True, "final_rows": len(first)}


def check_agent_boundaries():
    ratios = {("tariff_a", "HIGH", "tariff_b"): .13,
              ("tariff_a", "HIGH", "tariff_c"): .08}
    model = model_from_ratios(ratios)
    details = {}
    for case, budget in (("actual_n_below_ten", 100000), ("zero_budget", 0),
                         ("pilot_failure", 100000), ("invalid_observation", 100000)):
        env, _ = make_environment(tiny_profile(), model, tiny_tariffs(), CHANNELS,
                                  budget, 15000, zero_fallback, seed=42)
        if case == "pilot_failure":
            def failing(**_kwargs):
                raise RuntimeError("Controlled validation failure before pilot execution")
            env.run_pilot = failing
        elif case == "invalid_observation":
            actual_pilot = env.run_pilot

            def invalid(**kwargs):
                response = actual_pilot(**kwargs)
                return {**response, "observed_lift_ratio": float("nan")}
            env.run_pilot = invalid
        campaigns, metrics = run_agent(env, require_pilot=case not in {"zero_budget", "pilot_failure"})
        if case == "actual_n_below_ten":
            assert metrics["pilot_calls"] > 0
            assert metrics["pilot_contacts"] == 7 * metrics["pilot_calls"]
        if case in {"zero_budget", "pilot_failure", "invalid_observation"}:
            assert all(campaign["channel"] == "push" for campaign in campaigns)
        details[case] = metrics
    return details


def check_export():
    environment = dict(os.environ)
    environment.pop("OPENAI_API_KEY", None)
    command = [sys.executable, "make_submission.py"]
    payloads = []
    for _ in range(2):
        subprocess.run(command, cwd=ROOT, env=environment, check=True,
                       capture_output=True, text=True, timeout=240)
        payloads.append((ROOT / "submission.csv").read_bytes())
    assert payloads[0] == payloads[1], "Two clean-process seed42 exports differ"
    exported = pd.read_csv(ROOT / "submission.csv")
    assert list(exported.columns) == CAMPAIGN_COLUMNS and 1 <= len(exported) <= 10
    validate_strategy(exported, pd.read_csv(ROOT / "data/dict_tariff.csv"))
    return {"rows": len(exported), "sha256": hashlib.sha256(payloads[0]).hexdigest(),
            "clean_processes": 2, "llm_key": "absent"}


def check_worlds():
    # Fixed before any run: six families, world_seed=100, noise_seed=1000,1001.
    profile = pd.read_csv(ROOT / "customer_profile.csv")
    tariffs = pd.read_csv(ROOT / "data/dict_tariff.csv")
    cells = sorted(map(tuple, profile[["current_tariff", "arpu_segment"]].dropna().drop_duplicates().values))
    keys = [(current, segment, target) for current, segment in cells
            for target in sorted(tariffs.tariff_plan_code) if target != current]
    history = pd.read_csv(ROOT / "data/change_tariff.csv")
    history = history[history.AVG_ARPU_PREV_3M >= 100].copy()
    history["segment"] = pd.cut(history.AVG_ARPU_PREV_3M,
                                  [-np.inf, 1000, 5000, np.inf], labels=["LOW", "MID", "HIGH"])
    supported = set(map(tuple, history[["tariff_plan_code_from", "segment", "tariff_plan_code_to"]].values))
    rng = np.random.default_rng(100)
    jitter = rng.uniform(size=len(keys))
    ratios_by_family = {
        "rare_winners": np.where(jitter < .08, .35, -.025),
        "close_effects": .08 + (jitter - .5) * .01,
        "weak_positive": .012 + jitter * .008,
        "all_negative": -.08 - jitter * .08,
        "history_misleading": np.array([-.07 if key in supported else .22 for key in keys]),
        "outside_high": np.array([-.05 if key[1] == "HIGH" else .30 for key in keys]),
    }
    records = []
    for family, values in ratios_by_family.items():
        model = model_from_ratios(dict(zip(keys, values)))
        for seed in (1000, 1001):
            env, evaluator = make_environment(profile, model, tariffs, CHANNELS, 100000, 15000,
                                               zero_fallback, seed=seed)
            campaigns, details = run_agent(env)
            all_rows = pd.DataFrame(evaluator.executed_pilot_campaigns() + campaigns)
            scored = score_campaigns(all_rows, profile, model, tariffs, profile.predicted_arpu.sum(),
                                     zero_fallback, team_id=family)
            assert scored["total_contacts"] <= 15000 and scored["total_cost"] <= 100000
            assert scored["n_campaigns"] <= 26
            if family == "all_negative":
                assert scored["net_arpu_gain"] < 0, "Negative world unexpectedly escaped required harm"
            record = {"family": family, "world_seed": 100, "noise_seed": seed,
                      "net_arpu_gain": scored["net_arpu_gain"], **details}
            records.append(record)
            print(f"    {family:20} seed={seed} net={scored['net_arpu_gain']:,.0f}", flush=True)
    return {"purpose": "correctness smoke test; no paired superiority claim", "runs": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units-only", action="store_true", help="Skip complete Agent runs and CSV export")
    parser.add_argument("--skip-export", action="store_true", help="Do not regenerate submission.csv")
    parser.add_argument("--worlds", action="store_true", help="Also run 12 predefined synthetic smoke cases")
    parser.add_argument("--json", type=Path, help="Write factual results to this JSON file")
    args = parser.parse_args()
    os.chdir(ROOT)
    checks = [("beliefs", check_beliefs), ("cells", check_cells),
              ("simulator", check_simulator), ("small_pilot", check_small_pilot)]
    if not args.units_only:
        checks.append(("agent_contract", check_agent_contract))
        checks.append(("agent_boundaries", check_agent_boundaries))
        if not args.skip_export:
            checks.append(("export", check_export))
        if args.worlds:
            checks.append(("synthetic_worlds", check_worlds))
    def source_hashes():
        names = ("agent.py", "beliefs.py", "candidates.py", "planner.py",
                 "historical_candidates.json", "validate_solution.py")
        return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in names if (ROOT / name).exists()}

    result = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
              "source_sha256": source_hashes(), "checks": {}, "ok": True}
    for name, function in checks:
        started = time.perf_counter()
        try:
            details = function()
            result["checks"][name] = {"ok": True, "details": details,
                                      "elapsed_seconds": time.perf_counter() - started}
            print(f"PASS {name}", flush=True)
        except Exception as error:
            result["ok"] = False
            result["checks"][name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            print(f"FAIL {name}: {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
    after = source_hashes()
    result["source_unchanged"] = after == result["source_sha256"]
    if not result["source_unchanged"]:
        result["ok"] = False
        result["source_sha256_after"] = after
        print("FAIL: source files changed during validation; repeat on a stable version", flush=True)
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'PASS' if result['ok'] else 'FAIL'}: {len(checks)} check groups", flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
