"""Synthetic whole-block portfolio benchmark; no hackathon package imports/data.

All blocks can be expressed by disjoint tariff/segment filters. Every block
has <=5000 customers; no action is fractional, so row order cannot truncate it.
The objective is known illustrative net, not a prediction of leaderboard score.
"""
import itertools
import json
import random
import statistics
import time
from pathlib import Path

CONTACTS = 12520
BUDGET = 90080
ROWS = 10
CHANNELS = [("skip", 0.0, 0), ("push", .50, 0),
            ("sms", .65, 4), ("digital_ads", .85, 22)]


def instance(seed):
    rng = random.Random(seed)
    blocks = []
    for idx in range(8):
        n = rng.randrange(5, 46) * 100
        arpu = rng.uniform(400, 12000)
        u = rng.uniform(.001, .16)
        options = [(0., 0, 0, 0)]
        for _, mult, cost in CHANNELS[1:]:
            options.append((n * (arpu * u * mult - cost), n, n * cost, 1))
        blocks.append(options)
    return blocks


def score(blocks, state):
    v = 0.
    n = b = k = 0
    for options, choice in zip(blocks, state):
        dv, dn, db, dk = options[choice]
        v += dv
        n += dn
        b += db
        k += dk
    return v if n <= CONTACTS and b <= BUDGET and k <= ROWS else -float("inf")


def sms_first(blocks):
    state = [0] * len(blocks)
    order = sorted(range(len(blocks)), key=lambda j: blocks[j][2][0] / blocks[j][2][1], reverse=True)
    for j in order:
        trial = state.copy()
        trial[j] = 2
        if score(blocks, trial) > score(blocks, state):
            state = trial
    for j in order:
        if state[j]:
            trial = state.copy()
            trial[j] = 3
            if score(blocks, trial) > score(blocks, state):
                state = trial
    return tuple(state)


def greedy(blocks, criterion):
    state = (0,) * len(blocks)
    while True:
        options = []
        for j, choices in enumerate(blocks):
            if state[j]:
                continue
            for channel in (1, 2, 3):
                v, n, b, k = choices[channel]
                trial = tuple(channel if i == j else ch for i, ch in enumerate(state))
                if score(blocks, trial) > score(blocks, state):
                    denominator = [1, n, n / CONTACTS + b / BUDGET + k / ROWS][criterion]
                    options.append((v / denominator, trial))
        if not options:
            return state
        state = max(options)[1]


def improve(blocks, initial):
    state = initial
    for _ in range(20):
        best = state
        best_value = score(blocks, state)
        # Up to three block-status changes includes a 1-out/2-in exchange,
        # channel replacement, deletion, and a small rebalancing of channels.
        for width in (1, 2, 3):
            for positions in itertools.combinations(range(len(blocks)), width):
                choices = [[c for c in range(4) if c != state[p]] for p in positions]
                for replacement in itertools.product(*choices):
                    trial = list(state)
                    for p, c in zip(positions, replacement):
                        trial[p] = c
                    value = score(blocks, trial)
                    if value > best_value + 1e-8:
                        best, best_value = tuple(trial), value
        if best == state:
            break
        state = best
    return state


def main():
    started = time.perf_counter()
    rows = []
    for seed in range(20):
        blocks = instance(seed)
        baseline = sms_first(blocks)
        starts = [baseline] + [greedy(blocks, criterion) for criterion in range(3)]
        multi = max(starts, key=lambda s: score(blocks, s))
        local = improve(blocks, multi)
        exact = max(itertools.product(range(4), repeat=len(blocks)), key=lambda s: score(blocks, s))
        values = {name: score(blocks, plan) for name, plan in
                  [("sms_first", baseline), ("multistart", multi), ("local", local), ("exact", exact)]}
        rows.append({"seed": seed, **values, "states": {"sms_first": baseline, "local": local, "exact": exact}})
    summary = {}
    for name in ("sms_first", "multistart", "local"):
        ratios = [r[name] / r["exact"] for r in rows]
        summary[name] = {"mean_pct_of_exact": statistics.mean(ratios) * 100,
                         "worst_pct_of_exact": min(ratios) * 100,
                         "matches_exact": sum(abs(r[name] - r["exact"]) < 1e-7 for r in rows)}
    output = {"seeds": "0..19", "blocks": 8, "choices_per_block": 4,
              "enumerated_assignments_per_seed": 4 ** 8,
              "remaining_contacts": CONTACTS, "remaining_budget": BUDGET,
              "max_rows": ROWS, "elapsed_seconds": time.perf_counter() - started,
              "summary": summary, "cases": rows}
    destination = Path(__file__).with_name("portfolio_toy_results.json")
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
