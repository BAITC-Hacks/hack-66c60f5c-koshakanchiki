"""Frozen, paired comparison of the saved MVA and the current agent.

Truth belongs only to this harness.  Both policies receive identical public
data/catalogues and paired pilot randomness; models are never passed to them.
The scope is chosen from timing, or fixed at 600 pairs with --full, before
inspecting any performance outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from agent import Agent
from environment import make_environment
from scoring_core import CHANNELS, score_campaigns
from validate_solution import ROOT, model_from_ratios, run_agent, zero_fallback


FAMILIES = ("rare_winners", "close_effects", "weak_positive", "all_negative",
            "history_misleading", "outside_high")
SOURCE_FILES = ("agent.py", "beliefs.py", "candidates.py", "planner.py",
                "historical_candidates.json", "validate_solution.py", "compare_policies.py")


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def prepare_data():
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
    return profile, tariffs, keys, supported


def make_world(family, world_seed, keys, supported):
    jitter = np.random.default_rng(world_seed).uniform(size=len(keys))
    values = {
        "rare_winners": np.where(jitter < .08, .35, -.025),
        "close_effects": .08 + (jitter - .5) * .01,
        "weak_positive": .012 + jitter * .008,
        "all_negative": -.08 - jitter * .08,
        "history_misleading": np.array([-.07 if key in supported else .22 for key in keys]),
        "outside_high": np.array([-.05 if key[1] == "HIGH" else .30 for key in keys]),
    }[family]
    # Independently generated world variants retain the specified family.
    # The last two families also vary with world_seed, rather than five copies.
    if family == "history_misleading":
        values = values + (jitter - .5) * .02
    elif family == "outside_high":
        values = values + (jitter - .5) * .03
    return model_from_ratios(dict(zip(keys, values)))


def run_policy(policy_class, profile, tariffs, model, noise_seed):
    env, evaluator = make_environment(profile, model, tariffs, CHANNELS, 100000, 15000,
                                      zero_fallback, seed=noise_seed)
    agent = policy_class()
    rows, metrics = run_agent(env, agent)
    all_rows = pd.DataFrame(evaluator.executed_pilot_campaigns() + rows)
    score = score_campaigns(all_rows, profile, model, tariffs, profile.predicted_arpu.sum(),
                            zero_fallback)
    assert score["total_contacts"] <= 15000 and score["total_cost"] <= 100000
    assert len(score["campaigns_detail"]) <= 26
    log = [{"action_key": record["action_key"], "actual_n": record["actual_n"],
            "observed_ratio": record["observed_ratio"], "cost": record["cost"]}
           for record in agent.pilot_log]
    return {**metrics, "net_arpu_gain": score["net_arpu_gain"],
            "gross_arpu_lift": score["gross_arpu_lift"], "total_cost": score["total_cost"],
            "total_contacts": score["total_contacts"],
            "unique_customers": score["unique_customers_targeted"],
            "final_campaigns": rows, "pilot_log": log}, [r["request"] for r in agent.pilot_log]


def run_pair(v1, profile, tariffs, model, noise_seed):
    first, first_requests = run_policy(v1, profile, tariffs, model, noise_seed)
    second, second_requests = run_policy(Agent, profile, tariffs, model, noise_seed)
    assert first_requests == second_requests, "Policies used different pilot actions; noise is not paired"
    assert first["pilot_log"] == second["pilot_log"], "Paired pilot observations differ"
    return {"v1_mva": first, "v2_planner": second,
            "difference": second["net_arpu_gain"] - first["net_arpu_gain"]}


def metrics(records, policy):
    nets = np.array([record[policy]["net_arpu_gain"] for record in records], dtype=float)
    times = np.array([record[policy]["elapsed_seconds"] for record in records], dtype=float)
    worst_n = max(1, int(np.ceil(len(nets) * .1)))
    return {"runs": len(nets), "mean_net": float(nets.mean()), "median_net": float(np.median(nets)),
            "p10_net": float(np.quantile(nets, .1)), "cvar10_net": float(np.sort(nets)[:worst_n].mean()),
            "negative_rate": float((nets < 0).mean()), "time_p50": float(np.quantile(times, .5)),
            "time_p95": float(np.quantile(times, .95)), "time_max": float(times.max())}


def summarize(records):
    world_keys = sorted({(record["family"], record["world_seed"]) for record in records})
    world_differences = np.array([
        np.mean([r["difference"] for r in records if (r["family"], r["world_seed"]) == key])
        for key in world_keys
    ])
    rng = np.random.default_rng(42)
    sampled = rng.integers(0, len(world_keys), size=(10000, len(world_keys)))
    means = world_differences[sampled].mean(axis=1)
    interval = np.quantile(means, [.025, .975])
    families = {}
    for family in FAMILIES:
        group = [r for r in records if r["family"] == family]
        first, second = metrics(group, "v1_mva"), metrics(group, "v2_planner")
        tolerance = .02 * max(100000, abs(first["mean_net"]))
        tails_ok = (second["p10_net"] - first["p10_net"] >= -tolerance and
                    second["cvar10_net"] - first["cvar10_net"] >= -tolerance and
                    second["negative_rate"] - first["negative_rate"] <= .02 + 1e-12)
        families[family] = {"v1_mva": first, "v2_planner": second,
                            "mean_difference": second["mean_net"] - first["mean_net"],
                            "tail_tolerance": tolerance, "tails_acceptable": tails_ok}
    first, second = metrics(records, "v1_mva"), metrics(records, "v2_planner")
    performance_ok = float(interval[0]) > 0
    tails_ok = all(family["tails_acceptable"] for family in families.values())
    time_ok = second["time_p95"] <= 60 and second["time_max"] <= 240
    return {"v1_mva": first, "v2_planner": second, "families": families,
            "world_count": len(world_keys), "mean_paired_difference": float(world_differences.mean()),
            "bootstrap_unit": "family/world_seed mean (equal weight per world)",
            "bootstrap_repetitions": 10000, "bootstrap_seed": 42,
            "paired_difference_95pct_ci": interval.tolist(),
            "positive_lower_bound": performance_ok, "tails_acceptable": tails_ok,
            "time_acceptable": time_ok, "performance_acceptance": performance_ok and tails_ok and time_ok}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="a5084f3", help="Immutable MVA commit")
    parser.add_argument("--world-start", type=int, default=100, help="First world seed; freeze before running")
    parser.add_argument("--timing-world", type=int, help="Timing-only world, default: world-start minus one")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Describe control/bugfix effects; does not select or tune a policy")
    parser.add_argument("--candidate-label", default="posterior_planner_v2",
                        help="Description of the current Agent; source SHA identifies its exact code")
    parser.add_argument("--json", type=Path, default=ROOT / "comparison_results.json")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--full", action="store_true",
                        help="Force all 600 paired runs regardless of the timing estimate")
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.disable(logging.WARNING)  # Repeated exclusion diagnostics add no information here.
    initial_hashes = hashes()
    revision = subprocess.check_output(["git", "rev-parse", args.baseline], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(["git", "show", f"{revision}:solution/agent.py"], cwd=ROOT)
    profile, tariffs, keys, supported = prepare_data()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hack-frozen-mva-") as temporary:
        directory = Path(temporary)
        module_path = directory / "frozen_agent.py"
        module_path.write_bytes(source)
        (directory / "data").symlink_to(ROOT / "data", target_is_directory=True)
        spec = importlib.util.spec_from_file_location("frozen_mva_agent", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Only elapsed time is inspected. The estimate world is outside holdout.
        timing_started = time.perf_counter()
        timing_world = args.world_start - 1 if args.timing_world is None else args.timing_world
        run_pair(module.Agent, profile, tariffs,
                 make_world("close_effects", timing_world, keys, supported), noise_seed=999)
        pair_seconds = time.perf_counter() - timing_started
        estimated_full_seconds = pair_seconds * 600
        reduced = not args.full and estimated_full_seconds > 600
        world_seeds = list(range(args.world_start, args.world_start + (3 if reduced else 5)))
        noise_seeds = list(range(1000, 1010 if reduced else 1020))
        scope = {"families": list(FAMILIES), "world_seeds": world_seeds, "noise_seeds": noise_seeds,
                 "pairs": len(FAMILIES) * len(world_seeds) * len(noise_seeds),
                 "timing_pair_seconds": pair_seconds, "estimated_full_seconds": estimated_full_seconds,
                 "timing_world": timing_world,
                 "reduced_for_runtime": reduced,
                 "selection_rule": ("600 pairs explicitly requested with --full" if args.full else
                                    "180 pairs if timing-only estimate of 600 pairs exceeds 600 seconds; otherwise 600")}
        print("SCOPE " + json.dumps(scope), flush=True)
        if args.estimate_only:
            return 0
        result = {"status": "running", "scope": scope, "source_sha256": initial_hashes,
                  "purpose": "diagnostic_control_bugfixes" if args.diagnostic else "challenger_acceptance",
                  "candidate_label": args.candidate_label,
                  "policy_field_note": "v2_planner is the current Agent identified by candidate_label and SHA",
                  "baseline_commit": revision, "baseline_agent_sha256": hashlib.sha256(source).hexdigest(),
                  "shared_current_modules": ["beliefs.py", "candidates.py", "planner.py",
                                             "historical_candidates.json"],
                  "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                  "agent_seed": 42, "pilot_requests_and_observations_equal": True, "records": []}
        # Persist the chosen scope before examining any holdout result.
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for family in FAMILIES:
            for world_seed in world_seeds:
                model = make_world(family, world_seed, keys, supported)
                for noise_seed in noise_seeds:
                    pair = run_pair(module.Agent, profile, tariffs, model, noise_seed)
                    result["records"].append({"family": family, "world_seed": world_seed,
                                               "noise_seed": noise_seed, **pair})
                print(f"PROGRESS {family} world={world_seed}: {len(result['records'])}/{scope['pairs']} pairs; "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
        result["summary"] = summarize(result["records"])
        if args.diagnostic:
            result["summary"]["interpretation"] = (
                "Diagnostic of the final control with required correctness fixes. "
                "A zero interval is compatible with unchanged control; performance_acceptance "
                "is the original challenger statistic, not a new policy-selection decision."
            )
        result["source_unchanged"] = hashes() == initial_hashes
        result["elapsed_seconds"] = time.perf_counter() - started
        result["status"] = "complete" if result["source_unchanged"] else "invalid_source_changed"
        if not result["source_unchanged"]:
            result["source_sha256_after"] = hashes()
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("RESULT " + json.dumps({"status": result["status"], **result["summary"]}), flush=True)
        return 0 if result["source_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
