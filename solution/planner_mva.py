"""Resource-exact campaign simulation and deterministic cautious portfolios.

Only public profile data and pilot estimates enter this module. Scalar ratios
are on the SMS scale; channel-specific quadruple keys are already scaled.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import time

import numpy as np
import pandas as pd

CHANNELS = {
    "push": {"cost_per_contact": 0, "conversion_multiplier": .50},
    "sms": {"cost_per_contact": 4, "conversion_multiplier": .65},
    "digital_ads": {"cost_per_contact": 22, "conversion_multiplier": .85},
    "call": {"cost_per_contact": 160, "conversion_multiplier": 1.20},
}
CAMPAIGN_COLUMNS = (
    "campaign_name", "filter_arpu_segment", "filter_data_segment",
    "filter_call_segment", "filter_current_tariff", "target_tariff", "channel",
)
FILTERS = (
    ("arpu_segment", "filter_arpu_segment"),
    ("data_segment", "filter_data_segment"),
    ("call_segment", "filter_call_segment"),
)


def _resource(resources, name, default):
    if isinstance(resources, Mapping):
        return resources.get(name, default)
    return getattr(resources, name, default)


def _value(value):
    return None if value is None or pd.isna(value) else value


def _ratio(ratios, action, channel, channels):
    """Raw d/q supports exact conversion saturation; SMS alone cannot infer call."""
    if callable(ratios):
        return float(ratios(action, channel))
    if (*action, channel) in ratios:
        return float(ratios[(*action, channel)])
    estimate = ratios.get(action, 0.0)
    multiplier = channels[channel]["conversion_multiplier"]
    if isinstance(estimate, Mapping):
        return float(estimate["arpu_change_pct"]) * min(
            float(estimate["conversion_rate"]) * multiplier, 1.0)
    if channel == "call" and action in ratios:
        raise ValueError("Call requires a channel-specific ratio or raw conversion parameters")
    return float(estimate) * multiplier / channels["sms"]["conversion_multiplier"]


class _Simulator:
    """One-call cache: filter masks are reusable across full-plan evaluations."""

    def __init__(self, profile, resources, ratios):
        self.profile = profile.reset_index(drop=True)
        self.resources = resources
        self.ratios = ratios
        self.channels = _resource(resources, "channels", CHANNELS)
        self.id_codes, self.unique_ids = pd.factorize(self.profile["ID_NUMBER"], sort=True)
        self.cache = {}
        self.lift_cache = {}

    def selection(self, row):
        explicit = row.get("explicit_ids")
        has_explicit = isinstance(explicit, (list, tuple, set, np.ndarray)) and len(explicit) > 0
        signature = (("ids", tuple(sorted(explicit))) if has_explicit else
                     tuple(_value(row.get(key)) for _, key in FILTERS) +
                     (_value(row.get("filter_current_tariff")),))
        if signature not in self.cache:
            segment = self.profile
            if has_explicit:
                segment = segment[segment["ID_NUMBER"].isin(explicit)]
            else:
                for column, key in FILTERS:
                    value = _value(row.get(key))
                    if value is not None:
                        segment = segment[segment[column] == value]
                tariff = _value(row.get("filter_current_tariff"))
                if tariff is not None:
                    wanted = [part.strip() for part in str(tariff).split(";") if part.strip()]
                    segment = segment[segment["current_tariff"].isin(wanted)]
            self.cache[signature] = segment.sort_values("ID_NUMBER").index.to_numpy()
        return signature, self.cache[signature]

    def lifts(self, row, signature, positions):
        key = (signature, row["target_tariff"], row["channel"])
        if key not in self.lift_cache:
            segment = self.profile.iloc[positions[:5000]]
            result = np.zeros(len(segment), dtype=float)
            for cell, offsets in segment.groupby(["current_tariff", "arpu_segment"],
                                                 dropna=False).indices.items():
                action = (*cell, row["target_tariff"])
                ratio = _ratio(self.ratios, action, row["channel"], self.channels)
                values = segment.iloc[offsets]["predicted_arpu"].to_numpy(dtype=float) * ratio
                result[offsets] = np.where(np.isnan(values), 0.0, values)
            self.lift_cache[key] = result
        return self.lift_cache[key]

    def run(self, rows):
        budget = float(_resource(self.resources, "remaining_budget", 100000))
        contacts = int(_resource(self.resources, "remaining_contacts", 15000))
        total_cost, total_contacts = 0.0, 0
        best = np.full(len(self.unique_ids), -np.inf)
        details, selected_ids, selected_positions = [], [], []
        for index, row in enumerate(rows):
            channel = row["channel"]
            cost = self.channels[channel]["cost_per_contact"]
            signature, positions = self.selection(row)
            n = len(positions)
            campaign_cap = n > 5000
            n = min(n, 5000)
            reach_cap = n > contacts
            n = min(n, max(contacts, 0))
            money_cap = cost > 0 and n > int(budget // cost)
            if cost > 0:
                n = min(n, max(int(budget // cost), 0))
            campaign_cost = n * cost
            contacts -= n
            budget -= campaign_cost
            total_contacts += n
            total_cost += campaign_cost
            positions = positions[:n]
            lifts = self.lifts(row, signature, self.cache[signature])[:n]
            codes = self.id_codes[positions]
            valid = codes >= 0  # official groupby excludes missing IDs
            np.maximum.at(best, codes[valid], lifts[valid])
            selected_ids.append(self.profile.iloc[positions]["ID_NUMBER"].to_numpy())
            selected_positions.append(positions)
            details.append({
                "name": row.get("campaign_name", f"campaign_{index}"),
                "channel": channel, "cost": campaign_cost, "n_contacts": n,
                "gross_lift": float(lifts.sum()), "n_negative": int((lifts < 0).sum()),
                "capped_at_campaign_limit": campaign_cap,
                "capped_at_reach_budget": reach_cap,
                "capped_at_money_budget": money_cap,
            })
        contacted = best != -np.inf
        gross = float(best[contacted].sum())
        net = gross - total_cost
        return net, {
            "net_arpu_gain": net, "gross_arpu_lift": gross,
            "total_cost": total_cost, "total_contacts": total_contacts,
            "unique_customers_targeted": int(contacted.sum()),
            "campaigns_detail": details, "selected_ids": selected_ids,
            "selected_positions": selected_positions,
            "remaining_budget": budget, "remaining_contacts": contacts,
        }


def simulate_plan(profile, rows, resources, ratios):
    """Return (net, detail), applying official filters, ID caps and max dedup.

    resources contains remaining_budget/remaining_contacts AFTER pilots, with
    optional channels. No pilot costs are subtracted here. Ratios map
    (from_tariff, arpu_segment, target) to an SMS ratio or to raw
    {arpu_change_pct, conversion_rate}; a four-tuple ending in channel provides
    an already scaled ratio. Missing estimates are zero. Call needs raw or
    explicit channel ratios because its saturation is not identifiable by SMS.
    """
    return _Simulator(profile, resources, ratios).run(rows)


def _profile_from_stats(stats):
    """Whole cells can be planned without the optional original profile."""
    frames = []
    for (tariff, segment), cell in sorted(stats.items()):
        ids = np.asarray(cell["ids_sorted"])
        prefix = np.asarray(cell["prefix_arpu"], dtype=float)
        if len(prefix) != len(ids) + 1 or prefix[0] != 0:
            raise ValueError("prefix_arpu must have N+1 entries, starting with 0")
        frames.append(pd.DataFrame({
            "ID_NUMBER": ids, "current_tariff": tariff, "arpu_segment": segment,
            "data_segment": None, "call_segment": None,
            "predicted_arpu": np.diff(prefix),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
        "ID_NUMBER", "current_tariff", "arpu_segment", "data_segment",
        "call_segment", "predicted_arpu",
    ])


def _estimates(beliefs, resources, k, channels):
    states = beliefs if isinstance(beliefs, Mapping) else getattr(beliefs, "states", None)
    if isinstance(states, Mapping):
        measured = [a for a, s in states.items() if s["n"] > 0]
        all_actions = list(states)
    else:
        measured = _resource(resources, "piloted_actions", getattr(beliefs, "observed", None))
        if measured is None:
            raise ValueError("Beliefs object needs resources['piloted_actions'], .observed or .states")
        measured = [tuple(a) for a in measured]
        all_actions = measured
    ratios = {}
    for action in sorted(set(all_actions)):
        for channel in ("push", "sms", "digital_ads"):
            if isinstance(beliefs, Mapping):
                state = beliefs[action]
                mu, var = float(state["mu"]), float(state["var"])
                if not math.isfinite(mu) or not math.isfinite(var) or var < 0:
                    raise ValueError(f"Invalid posterior for {action!r}")
                scale = channels[channel]["conversion_multiplier"] / channels["sms"]["conversion_multiplier"]
                mean, sd = mu * scale, math.sqrt(var) * scale
            else:
                mean = float(beliefs.mean_ratio(action, channel))
                sd = float(beliefs.sd_ratio(action, channel))
                if not math.isfinite(mean) or not math.isfinite(sd) or sd < 0:
                    raise ValueError(f"Invalid channel posterior for {action!r}")
            # Unmeasured actions may only serve as a broad-prior fallback.
            if action not in measured:
                mean = 0.
                sd = .25 * channels[channel]["conversion_multiplier"] / channels["sms"]["conversion_multiplier"]
            ratios[(*action, channel)] = mean - k * sd
    return sorted(set(measured)), sorted(set(all_actions)), ratios


def _rows_for_actions(profile, actions, subgroups=True):
    for action in actions:
        source, segment, target = action
        if source == target:
            continue
        cell = profile[(profile.current_tariff == source) & (profile.arpu_segment == segment)]
        if cell.empty:
            continue
        data_values = [None]
        call_values = [None]
        if subgroups:
            data_values += sorted(set(cell.data_segment.dropna()) & {"NON_USER", "LITE", "HEAVY"})
            call_values += sorted(set(cell.call_segment.dropna()) & {"LOW", "MEDIUM", "HIGH"})
        seen = set()
        for data in data_values:
            for call in call_values:
                part = cell
                if data is not None:
                    part = part[part.data_segment == data]
                if call is not None:
                    part = part[part.call_segment == call]
                mask = frozenset(part.index)
                if not mask or mask in seen:
                    continue
                seen.add(mask)
                row = dict(zip(CAMPAIGN_COLUMNS, (
                    "candidate", segment, data, call, source, target, "push")))
                yield action, row, mask


def _sort_and_check(simulator, rows):
    """Sorting can change capped prefixes, so re-evaluate until order is valid."""
    seen = set()
    while rows:
        _, detail = simulator.run(rows)
        gains = [r["gross_lift"] - r["cost"] for r in detail["campaigns_detail"]]
        kept = [i for i, r in enumerate(detail["campaigns_detail"])
                if r["n_contacts"] > 0 and gains[i] > 0]
        order = sorted(kept, key=lambda i: (-gains[i], i))
        if order == list(range(len(rows))):
            return rows
        signature = tuple(tuple(row.get(key) for key in CAMPAIGN_COLUMNS[1:]) for row in rows)
        if signature in seen:
            # Rare cap-induced ordering cycle: retain the stronger rows.
            order = order[:-1]
            seen.clear()
        else:
            seen.add(signature)
        rows = [rows[i] for i in order]
    return []


def build_portfolio(beliefs, cell_stats, resources, k=1.0):
    """Build 1..10 cautious campaigns using only public pilot posteriors.

    FROZEN BELIEFS CONTRACT (two supported forms):
      * dict[(from_tariff, arpu_segment, target)] =
        {"mu": float, "var": float, "n": int}. mu/var are on the SMS scale;
        n is the number of valid observed customers. Only n>0 may be scaled.
      * object with mean_ratio(action_key, channel) and
        sd_ratio(action_key, channel), both ALREADY scaled to that channel.
        Supply measured action keys in resources["piloted_actions"], or expose
        the public iterable .observed. A public .states mapping with the dict
        schema above may supply those keys instead.
        The methods alone cannot enumerate which actions have been piloted.

    cell_stats maps (from_tariff, arpu_segment) to N, A_sum, ids_sorted and
    prefix_arpu (length N+1, first entry 0). resources contains
    remaining_budget/remaining_contacts AFTER pilots and optional channels,
    profile, deadline (time.monotonic). profile enables data/call subgroups;
    without it the exact whole-cell ID prefixes are reconstructed from stats.
    All state is local to this call. Inputs are never mutated.

    For each actual affordable ID prefix and channel, cautious_net is
    (mean_ratio - k*sd_ratio)*sum(prefix ARPU) - n*cost. Choose the channel
    with maximum positive cautious_net, then greedily select by net/contact.
    Final masks do not overlap and use one target per cell. Sort by final net
    and re-simulate after sorting. Call is disabled. Pilot costs are already
    sunk and do not veto profitable additions; pilot overlap uncertainty is
    not modeled by this MVA objective.

    If no positive measured action exists, return the least harmful single
    filterable push/SMS/ads row as the mandatory fallback (may be negative).
    With no audience, action keys or contacts, return [] so the caller can
    handle an impossible final campaign explicitly.
    """
    if not math.isfinite(k) or k < 0:
        raise ValueError("k must be finite and nonnegative")
    channels = _resource(resources, "channels", CHANNELS)
    budget = float(_resource(resources, "remaining_budget", 100000))
    contacts = int(_resource(resources, "remaining_contacts", 15000))
    if contacts <= 0:
        return []
    profile = _resource(resources, "profile", None)
    if profile is None:
        profile = _profile_from_stats(cell_stats)
    else:
        profile = profile.reset_index(drop=True)
    if profile.empty:
        return []
    measured, all_actions, ratios = _estimates(beliefs, resources, k, channels)
    base_resources = dict(remaining_budget=budget, remaining_contacts=contacts, channels=channels)
    simulator = _Simulator(profile, base_resources, ratios)
    subgroups = _resource(resources, "subgroup_limit", 1) != 0
    pool = list(_rows_for_actions(profile, measured, subgroups))
    deadline = float(_resource(resources, "deadline", math.inf))
    selected, used, targets = [], set(), {}
    while len(selected) < 10 and contacts > 0:
        best = None
        simulator.resources = dict(remaining_budget=budget, remaining_contacts=contacts, channels=channels)
        for action, template, mask in pool:
            if time.monotonic() >= deadline and best is not None:
                break
            if mask & used or (action[:2] in targets and targets[action[:2]] != action[2]):
                continue
            channel_best = None
            for channel in ("push", "sms", "digital_ads"):
                row = {**template, "channel": channel}
                net, detail = simulator.run([row])
                n = detail["total_contacts"]
                if n > 0 and net > 0 and (channel_best is None or net > channel_best[0]):
                    channel_best = (net, n, row, detail)
            if channel_best is None:
                if time.monotonic() >= deadline:
                    break
                continue
            net, n, row, detail = channel_best
            rank = (net / n, net)
            if best is None or rank > best[0]:
                best = (rank, action, row, mask, detail)
        if best is None:
            break
        _, action, row, mask, detail = best
        selected.append(row)
        used.update(mask)
        targets[action[:2]] = action[2]
        budget, contacts = detail["remaining_budget"], detail["remaining_contacts"]
        if time.monotonic() >= deadline:
            break
    simulator.resources = base_resources
    selected = _sort_and_check(simulator, selected)
    if not selected:
        # A sunk loss never triggers this path if a useful multirow plan exists.
        fallback = None
        for _, template, _ in _rows_for_actions(profile, all_actions, subgroups):
            for channel in ("push", "sms", "digital_ads"):
                row = {**template, "channel": channel}
                net, detail = simulator.run([row])
                if detail["total_contacts"] and (fallback is None or net > fallback[0]):
                    fallback = (net, row)
            if fallback is not None and time.monotonic() >= deadline:
                break
        selected = [fallback[1]] if fallback else []
    return [{**row, "campaign_name": f"campaign_{i}"}
            for i, row in enumerate(selected, 1)]
