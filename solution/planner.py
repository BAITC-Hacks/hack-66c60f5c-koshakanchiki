"""Public-data campaign planning with exact final-contact accounting.

The simulator returns the incremental net of the supplied final rows. Resources
are *remaining* after pilots; it never charges pilot costs twice. The optimiser
uses public pilot counts to integrate their unknown overlap in expectation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

COST = {"push": 0.0, "sms": 4.0, "digital_ads": 22.0, "call": 160.0}
MULT = {"push": 0.50 / 0.65, "sms": 1.0, "digital_ads": 0.85 / 0.65, "call": 1.20 / 0.65}
FINAL_CHANNELS = ("push", "sms", "digital_ads")
FIELDS = (
    "campaign_name", "filter_arpu_segment", "filter_data_segment",
    "filter_call_segment", "filter_current_tariff", "target_tariff", "channel",
)


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(pd.notna(value))
    except (TypeError, ValueError):
        return False


def _limits(resources: Mapping) -> tuple[float, int, int, int]:
    budget = max(0.0, float(resources.get("remaining_budget", 100000)))
    contacts = max(0, int(resources.get("remaining_contacts", 15000)))
    rows = max(0, min(10, int(resources.get("max_campaigns", 10))))
    cap = max(0, min(5000, int(resources.get("max_customers_per_campaign", 5000))))
    return budget, contacts, rows, cap


def _filter(profile: pd.DataFrame, row: Mapping) -> pd.DataFrame:
    result = profile
    for name in ("arpu_segment", "data_segment", "call_segment"):
        value = row.get("filter_" + name)
        if _present(value):
            result = result[result[name] == value]
    current = row.get("filter_current_tariff")
    if _present(current):
        tariffs = [part.strip() for part in str(current).split(";") if part.strip()]
        result = result[result["current_tariff"].isin(tariffs)]
    return result.sort_values("ID_NUMBER", kind="stable")


def _ratio(source: Any, key: tuple, channel: str) -> float:
    if hasattr(source, "mean_ratio"):
        value = source.mean_ratio(key, channel)
    else:
        # Explicit channel estimates take precedence; plain action keys are SMS.
        value = source.get((key, channel), None)
        if value is None:
            value = source.get((*key, channel), None)
        if value is None:
            value = float(source.get(key, 0.0)) * MULT[channel]
    value = float(value)
    return value if math.isfinite(value) else 0.0


def simulate_plan(profile, rows, resources, ratios_or_beliefs):
    """Apply public filters/caps in order and deduplicate *signed* monetary lift.

    ``ratios_or_beliefs`` may be Beliefs or a mapping from action triples to SMS
    ratios. A ``(action_key, channel)`` mapping key supplies a channel ratio.
    Unknown pilot IDs are not invented here. The returned detail contains exact
    final IDs, expenditures and per-ID best lift for audit against the scorer.
    """
    budget, reach, max_rows, cap = _limits(resources)
    total_cost = 0.0
    total_contacts = 0
    best = {}
    details = []
    for index, row in enumerate(list(rows)[:max_rows]):
        channel = row.get("channel")
        if channel not in COST:
            raise ValueError("Unknown channel: %r" % channel)
        segment = _filter(profile, row)
        capped_campaign = len(segment) > cap
        segment = segment.iloc[:cap]
        capped_reach = len(segment) > reach
        segment = segment.iloc[:reach]
        price = COST[channel]
        affordable = max(0, int(budget // price)) if price else len(segment)
        capped_money = len(segment) > affordable
        segment = segment.iloc[:affordable]
        count = len(segment)
        cost = count * price
        budget -= cost
        reach -= count
        total_cost += cost
        total_contacts += count
        lifts = []
        for current, arpu, ident, baseline in segment[
            ["current_tariff", "arpu_segment", "ID_NUMBER", "predicted_arpu"]
        ].itertuples(index=False, name=None):
            ratio = _ratio(ratios_or_beliefs, (current, arpu, row["target_tariff"]), channel)
            lift = ratio * float(baseline)
            if not math.isfinite(lift):
                lift = 0.0
            lifts.append(lift)
            # A first negative contact must stay negative (no implicit zero arm).
            if ident not in best or lift > best[ident]:
                best[ident] = lift
        details.append({
            "name": row.get("campaign_name", "campaign_%d" % index),
            "channel": channel, "cost": cost, "n_contacts": count,
            "gross_lift": float(sum(lifts)),
            "n_negative": sum(value < 0 for value in lifts),
            "ids": segment["ID_NUMBER"].tolist(),
            "capped_at_campaign_limit": capped_campaign,
            "capped_at_reach_budget": capped_reach,
            "capped_at_money_budget": capped_money,
        })
    gross = float(sum(best.values()))
    net = gross - total_cost
    return net, {
        "gross_arpu_lift": gross, "total_cost": total_cost,
        "total_contacts": total_contacts, "unique_customers_targeted": len(best),
        "net_arpu_gain": net, "remaining_budget": budget,
        "remaining_contacts": reach, "campaigns_detail": details,
        "best_lift_by_id": best,
    }


@dataclass(frozen=True, eq=False)
class _Audience:
    cell: tuple
    data: Any
    calls: Any
    prefix: np.ndarray

    @property
    def size(self):
        return len(self.prefix) - 1


@dataclass(frozen=True, eq=False)
class _Option:
    action: tuple
    audience: _Audience
    channel: str
    delta: np.ndarray
    cautious_ratio: float
    full_net: float

    @property
    def key(self):
        return (*self.action, str(self.audience.data or ""),
                str(self.audience.calls or ""), self.channel)


def _audiences(stats, profile, cap, expired=lambda: False):
    """Only exportable filters; no arbitrary ID exclusions or final n field."""
    result = {}
    for cell in sorted(stats, key=lambda x: tuple(map(str, x))):
        if expired():
            break
        stat = stats[cell]
        prefix = np.asarray(stat["prefix_arpu"], dtype=float)
        # The contracted prefix includes a leading zero.
        if len(prefix) != int(stat["N"]) + 1:
            raise ValueError("prefix_arpu must have N+1 entries, starting at zero")
        result[cell] = [_Audience(cell, None, None, prefix[:cap + 1])]
    if not result or expired() or profile is None or not {"data_segment", "call_segment"}.issubset(profile.columns):
        return result
    # Replanning is frequent during exploration; construct subgroup prefixes
    # only for cells with an observed action that passed the cautious cutoff.
    cell_index = pd.MultiIndex.from_frame(profile[["current_tariff", "arpu_segment"]])
    eligible = profile.loc[cell_index.isin(result)]
    grouped = eligible.groupby(
        ["current_tariff", "arpu_segment"], sort=True, observed=True)
    for cell, group in grouped:
        if expired():
            break
        if cell not in result:
            continue
        group = group.sort_values("ID_NUMBER", kind="stable")
        data_values = sorted(group["data_segment"].dropna().unique(), key=str)
        call_values = sorted(group["call_segment"].dropna().unique(), key=str)
        seen = {tuple(group.iloc[:cap]["ID_NUMBER"])}
        for data in [None] + data_values:
            for calls in [None] + call_values:
                if data is None and calls is None:
                    continue
                part = group
                if data is not None:
                    part = part[part["data_segment"] == data]
                if calls is not None:
                    part = part[part["call_segment"] == calls]
                part = part.iloc[:cap]
                ids = tuple(part["ID_NUMBER"])
                if not ids or ids in seen:
                    continue
                seen.add(ids)
                values = pd.to_numeric(part["predicted_arpu"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                prefix = np.r_[0.0, np.cumsum(values)]
                result[cell].append(_Audience(cell, data, calls, prefix))
    return result


def _expected_best(ratios, probabilities, n_draws):
    """Expected max of contacted effects, without adding a fictitious zero."""
    if not ratios:
        return np.zeros(n_draws, dtype=float)
    matrix = np.asarray(ratios, dtype=float).T
    order = np.argsort(-matrix, axis=1, kind="stable")
    ordered_ratio = np.take_along_axis(matrix, order, axis=1)
    probs = np.broadcast_to(np.asarray(probabilities), matrix.shape)
    ordered_prob = np.take_along_axis(probs, order, axis=1)
    missed = np.cumprod(1.0 - ordered_prob, axis=1)
    before = np.c_[np.ones(n_draws), missed[:, :-1]]
    return np.sum(ordered_ratio * ordered_prob * before, axis=1)


def _pilot_model(beliefs, stats, resources, theta, lookup):
    """Integrate random pilot membership using only recorded public counts."""
    n_draws = theta.shape[0]
    groups = {}
    pilot_cost = 0.0
    for entry in resources.get("pilot_log", []):
        key = entry.get("action_key")
        if key is None:
            continue
        key = tuple(key)
        cell = key[:2]
        actual = int(entry.get("actual_n", entry.get("result", {}).get("n_customers", 0)) or 0)
        channel = entry.get("channel", "sms")
        if actual <= 0 or cell not in stats or channel not in MULT:
            continue
        pilot_cost += float(entry.get("cost", actual * COST[channel]))
        if key not in lookup:
            continue
        count = int(stats[cell]["N"])
        if count <= 0:
            continue
        group = groups.setdefault(cell, {})
        pair = (key, channel)
        group[pair] = group.get(pair, 1.0) * (1.0 - min(actual, count) / count)
    baseline = np.full(n_draws, -pilot_cost, dtype=float)
    cell_models = {}
    for cell, group in groups.items():
        ratios = [theta[:, lookup[key]] * MULT[channel] for key, channel in group]
        probs = [1.0 - never for never in group.values()]
        old = _expected_best(ratios, probs, n_draws)
        baseline += old * float(stats[cell]["A_sum"])
        cell_models[cell] = (ratios, probs, old)
    return baseline, cell_models


def _compatible(candidate, chosen):
    a = candidate.audience
    for other in chosen:
        b = other.audience
        if a.cell != b.cell:
            continue
        # Multiple subgroups may use different channels, but one target per cell.
        if candidate.action != other.action:
            return False
        data_overlap = a.data is None or b.data is None or a.data == b.data
        call_overlap = a.calls is None or b.calls is None or a.calls == b.calls
        if data_overlap and call_overlap:
            return False
    return True


def build_portfolio(beliefs, cell_stats, resources, k=1.0):
    """Return cautious profitable rows, using four starts and bounded swaps.

    Whole cells and data/call subgroups share the same action posterior. Exact
    prefix ARPU is used for every budget/reach truncation. All candidate plans
    use the same 64 posterior draws and expected pilot overlap. Only observed
    actions are eligible, even with k=0 or a positive custom prior. No call.
    """
    deadline = resources.get("deadline")
    deadline = math.inf if deadline is None else float(deadline)

    def expired():
        return time.monotonic() >= deadline

    budget, reach, max_rows, cap = _limits(resources)
    if expired() or not reach or not max_rows or not cap or not cell_stats:
        return []
    actions = tuple(sorted(beliefs.actions, key=lambda a: tuple(map(str, a))))
    if not actions:
        return []
    observed = getattr(beliefs, "observed", None)
    if observed is None:
        observed = {
            tuple(entry["action_key"]) for entry in resources.get("pilot_log", [])
            if entry.get("action_key") and int(entry.get("actual_n", 0) or 0) > 0
        }
    eligible_actions = [
        key for key in actions
        if key in observed and key[:2] in cell_stats and
        float(beliefs.mean_ratio(key, "sms")) - k * float(beliefs.sd_ratio(key, "sms")) > 0
    ]
    if not eligible_actions or expired():
        return []
    lookup = {key: i for i, key in enumerate(actions)}
    n_draws = 64
    # Beliefs.draws columns follow Beliefs.actions; remap explicitly if needed.
    raw = np.asarray(beliefs.draws(n_draws, seed=42), dtype=float)
    native_actions = tuple(beliefs.actions)
    native_index = {key: i for i, key in enumerate(native_actions)}
    theta = raw[:, [native_index[key] for key in actions]]
    if theta.shape != (n_draws, len(actions)) or not np.isfinite(theta).all():
        raise ValueError("Beliefs.draws returned invalid posterior samples")
    baseline, pilot_models = _pilot_model(beliefs, cell_stats, resources, theta, lookup)
    eligible_cells = {key[:2] for key in eligible_actions}
    eligible_stats = {cell: cell_stats[cell] for cell in eligible_cells}
    audiences = _audiences(eligible_stats, resources.get("profile"), cap, expired)
    options = []
    for key in eligible_actions:
        if expired():
            break
        if key[:2] not in audiences:
            continue
        for channel in FINAL_CHANNELS:
            price = COST[channel]
            caution = float(beliefs.mean_ratio(key, channel)) - k * float(beliefs.sd_ratio(key, channel))
            if not math.isfinite(caution) or caution <= 0:
                continue
            ratio_draws = theta[:, lookup[key]] * MULT[channel]
            if key[:2] in pilot_models:
                pilot_ratios, probs, old = pilot_models[key[:2]]
                delta = _expected_best(pilot_ratios + [ratio_draws], probs + [1.0], n_draws) - old
            else:
                delta = ratio_draws
            for audience in audiences[key[:2]]:
                count = min(audience.size, reach)
                if price:
                    count = min(count, int(budget // price))
                if count <= 0:
                    continue
                cautious = caution * audience.prefix[count] - count * price
                if cautious <= 0:
                    continue
                options.append(_Option(key, audience, channel, delta, caution, float(cautious)))
    if not options:
        return []
    options.sort(key=lambda opt: (-opt.full_net, opt.key))

    def evaluate(plan):
        # Sort by complete-prefix cautious net; recompute every affected prefix.
        ordered = sorted(plan, key=lambda opt: (-opt.full_net, opt.key))
        left_money, left_reach = budget, reach
        samples = baseline.copy()
        executed = []
        cost = 0.0
        for option in ordered[:max_rows]:
            price = COST[option.channel]
            count = min(option.audience.size, left_reach)
            if price:
                count = min(count, int(left_money // price))
            if count <= 0:
                continue
            amount = float(option.audience.prefix[count])
            # Truncation can change average ARPU; check the real prefix again.
            if option.cautious_ratio * amount - count * price <= 0:
                return -math.inf, [], 0, 0.0
            charge = count * price
            samples += option.delta * amount - charge
            cost += charge
            left_money -= charge
            left_reach -= count
            executed.append(option)
        utility = float(samples.mean() - k * samples.std())
        return utility, executed, reach - left_reach, cost

    def signature(plan):
        return tuple(option.key for option in plan)

    base_utility = float(baseline.mean() - k * baseline.std())
    best_value, best_plan = base_utility, []
    # The SMS start is subsequently upgraded by the same neighbourhood search.
    for mode in range(4):
        if expired():
            break
        chosen = []
        utility = base_utility
        for _ in range(max_rows):
            if expired():
                break
            winner = None
            winner_rank = -math.inf
            _, _, old_used, old_expense = evaluate(chosen)
            for option in options:
                if expired():
                    break
                if mode == 3 and option.channel != "sms":
                    continue
                if not _compatible(option, chosen):
                    continue
                value, result, used, expense = evaluate(chosen + [option])
                gain = value - utility
                if gain <= 1e-8 or len(result) <= len(chosen):
                    continue
                additional_n = max(1, used - old_used)
                additional_cost = max(0.0, expense - old_expense)
                denominator = 1.0
                if mode == 1:
                    denominator = additional_n
                elif mode == 2:
                    denominator = additional_n / max(reach, 1) + 1.0 / max_rows
                    if budget:
                        denominator += additional_cost / budget
                rank = gain / denominator
                if rank > winner_rank + 1e-8:
                    winner = (value, result)
                    winner_rank = rank
            if winner is None:
                break
            utility, chosen = winner
        if utility > best_value + 1e-8 or (
            utility == best_value and chosen and (not best_plan or signature(chosen) < signature(best_plan))
        ):
            best_value, best_plan = utility, chosen
    if not best_plan:
        return []

    # Bounded deterministic neighbourhood: remove a row or replace one row.
    # Rank a short list but retain every selected option's same-audience channels.
    shortlist = options[:72]
    for candidate in options:
        if expired():
            break
        if any(candidate.audience is old.audience for old in best_plan) and candidate not in shortlist:
            shortlist.append(candidate)
    attempts = 0
    for _round in range(2):
        if expired():
            break
        improved = False
        # Evaluate the neighbourhood of one fixed incumbent, then begin the
        # next round from its best improvement. best_value never decreases.
        original = list(best_plan)
        for removed in range(len(original)):
            if expired():
                break
            rest = original[:removed] + original[removed + 1:]
            if rest:
                value, trial, _, _ = evaluate(rest)
                if value > best_value + 1e-8:
                    best_value, best_plan = value, trial
                    improved = True
            for candidate in shortlist:
                attempts += 1
                if attempts > 768 or expired():
                    break
                if not _compatible(candidate, rest):
                    continue
                value, trial, _, _ = evaluate(rest + [candidate])
                if value > best_value + 1e-8:
                    best_value, best_plan = value, trial
                    improved = True
            if attempts > 768:
                break
        if not improved or attempts > 768:
            break

    _, best_plan, _, _ = evaluate(best_plan)
    result = []
    for index, option in enumerate(best_plan, 1):
        current, segment, target = option.action
        result.append({
            "campaign_name": "campaign_%02d_%s_%s" % (index, current, segment),
            "filter_arpu_segment": segment,
            "filter_data_segment": option.audience.data,
            "filter_call_segment": option.audience.calls,
            "filter_current_tariff": current,
            "target_tariff": target,
            "channel": option.channel,
        })
    return result
