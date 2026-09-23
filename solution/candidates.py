"""Public-data candidates; no environment internals or posterior updates.

Action keys are ``(from_tariff, arpu_segment, target)``. Historical ranks are
ordering hints only. ``cell_stats(profile).report`` describes excluded profiles;
``prefix_arpu[k]`` is the baseline sum of the first k customers in ID order.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
SEGMENTS = ("LOW", "MID", "HIGH")
HISTORY_PATH = Path(__file__).with_name("historical_candidates.json")
HISTORY_COLUMNS = (
    "tariff_plan_code_from", "tariff_plan_code_to",
    "AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M",
)
FORMULA = "rank = support / (support + 10) * median; median = median(clip(AVG_ARPU_NEXT_3M / AVG_ARPU_PREV_3M - 1, -1, 3))"
METHOD = {
    "deduplication": "exact duplicates across all source columns",
    "filter": "AVG_ARPU_PREV_3M >= 100",
    "segments": {"bins": ["-inf", 1000, 5000, "inf"],
                 "labels": list(SEGMENTS), "right": True},
    "smoothing_support": 10,
    "formula": FORMULA,
}
SCHEMA = {
    "action_key": ["from_tariff", "arpu_segment", "target"],
    "aggregate": {"from_tariff": "string", "arpu_segment": "LOW|MID|HIGH",
                  "target": "string", "support": "positive integer",
                  "median": "float in [-1, 3]", "rank": "float"},
}
TARIFF_FEATURES = (
    "price_tariff", "Data_in_PKG", "Min_another_operator_in_PKG",
    "Min_another_operator_and_city_in_PKG",
)
BASES = ("historical", "nearest_more_expensive", "nearest_not_more_expensive",
         "unsupported_farthest")


class HistoricalRank(dict):
    """dict[action_key, float], with support kept distinct from rank == 0."""

    def __init__(self, values=(), *, support=None, report=None):
        super().__init__(values)
        self.support = {} if support is None else support
        self.report = {} if report is None else report


class CellStats(dict):
    """dict[(from_tariff, segment), stats], plus an aggregate exclusion report."""

    def __init__(self, values=(), *, report=None):
        super().__init__(values)
        self.report = {} if report is None else report


def _tariff_codes(tariffs):
    values = tariffs["tariff_plan_code"] if isinstance(tariffs, pd.DataFrame) else tariffs
    return sorted({str(value) for value in values if pd.notna(value)})


def _zero_rank(tariffs, reason):
    codes = _tariff_codes(tariffs)
    return HistoricalRank(
        {(source, segment, target): 0.0
         for source in codes for segment in SEGMENTS for target in codes
         if source != target},
        report={"fallback": reason},
    )


def _read_artifact(path, tariffs, expected_hash=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("historical artifact must be an object")
    if (payload.get("schema_version") != 1 or payload.get("schema") != SCHEMA
            or payload.get("method") != METHOD):
        raise ValueError("incompatible historical schema or formula")
    digest = payload.get("source_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("invalid source SHA-256")
    if expected_hash is not None and digest != expected_hash:
        raise ValueError("source SHA-256 changed")
    rows = payload.get("aggregates")
    if not isinstance(rows, list):
        raise ValueError("missing historical aggregates")
    result = _zero_rank(tariffs, None)
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(SCHEMA["aggregate"]):
            raise ValueError("invalid aggregate columns")
        source, segment, target = (row[name] for name in SCHEMA["action_key"])
        support, median, rank = row["support"], row["median"], row["rank"]
        if (not isinstance(source, str) or not source
                or not isinstance(target, str) or not target
                or segment not in SEGMENTS
                or type(support) is not int or support <= 0
                or type(median) not in (int, float) or not math.isfinite(median)
                or not -1 <= median <= 3
                or type(rank) not in (int, float) or not math.isfinite(rank)
                or not math.isclose(rank, support / (support + 10) * median,
                                    rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("invalid historical aggregate")
        key = (source, segment, target)
        if key in result.support:
            raise ValueError("duplicate historical action")
        result[key] = float(rank)
        result.support[key] = support
    result.report = {"fallback": None, "source_sha256": digest,
                     "counts": payload.get("counts", {})}
    return result


def load_or_build_historical_rank(change_tariff_path, tariffs, *, cache_path=None):
    """Read/build ranks and persist audited aggregates beside this module.

    Pass a CSV path to build (or reuse a cache with the same source hash), or
    pass the JSON artifact itself for offline use without the source CSV.
    A missing input or invalid input schema returns all-zero ranks and logs a
    fallback. A corrupt/stale cache is rebuilt if its CSV source is available.
    ``cache_path`` is an optional override for isolated builds/tests.
    """
    source = Path(change_tariff_path)
    destination = HISTORY_PATH if cache_path is None else Path(cache_path)
    try:
        if source.suffix.lower() == ".json":
            return _read_artifact(source, tariffs)
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if destination.is_file():
            try:
                return _read_artifact(destination, tariffs, digest)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                LOGGER.warning("Historical cache rejected; rebuilding: %s", exc)
        frame = pd.read_csv(source)
        if not set(HISTORY_COLUMNS).issubset(frame.columns):
            raise ValueError("history is missing required columns")
        raw_count = len(frame)
        frame = frame.drop_duplicates().copy()
        deduplicated_count = len(frame)
        for column in HISTORY_COLUMNS[2:]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise ValueError(f"nonfinite historical ARPU in {column}")
        if frame[list(HISTORY_COLUMNS[:2])].isna().any().any():
            raise ValueError("missing historical tariff keys")
        frame = frame.loc[frame["AVG_ARPU_PREV_3M"] >= 100].copy()
        frame["arpu_segment"] = pd.cut(
            frame["AVG_ARPU_PREV_3M"], [-np.inf, 1000, 5000, np.inf],
            labels=list(SEGMENTS),
        )
        frame["relative_change"] = (
            frame["AVG_ARPU_NEXT_3M"] / frame["AVG_ARPU_PREV_3M"] - 1
        ).clip(-1, 3)
        grouped = frame.groupby(
            ["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"],
            observed=True, sort=True,
        )["relative_change"].agg(support="size", median="median")
        rows = []
        result = _zero_rank(tariffs, None)
        for (from_tariff, segment, target), row in grouped.iterrows():
            support, median = int(row["support"]), float(row["median"])
            rank = support / (support + 10) * median
            key = (str(from_tariff), str(segment), str(target))
            rows.append(dict(zip(SCHEMA["action_key"], key),
                             support=support, median=median, rank=rank))
            result[key] = rank
            result.support[key] = support
        counts = {"source_rows": raw_count,
                  "exact_duplicates_removed": raw_count - deduplicated_count,
                  "rows_below_100_removed": deduplicated_count - len(frame),
                  "eligible_rows": len(frame), "aggregate_count": len(rows)}
        payload = {"schema_version": 1, "schema": SCHEMA, "method": METHOD,
                   "source": source.name, "source_sha256": digest,
                   "source_columns": list(pd.read_csv(source, nrows=0).columns),
                   "counts": counts, "aggregates": rows}
        result.report = {"fallback": None, "source_sha256": digest, "counts": counts}
        try:
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                              allow_nan=False) + "\n", encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Could not persist historical aggregates: %s", exc)
        return result
    except (OSError, ValueError, TypeError, KeyError) as exc:
        LOGGER.warning("Historical fallback: zero prior for all actions (%s): %s", source, exc)
        return _zero_rank(tariffs, str(exc))


def cell_stats(profile):
    """Build exact ID-prefix sums without editing the input or filtering size.

    Missing current_tariff/arpu_segment rows are excluded and counted in
    ``result.report``; missing data/call segments do not exclude a customer.
    Arrays contain a leading zero: ``prefix_arpu.shape == (N + 1,)``.
    """
    required = ["current_tariff", "arpu_segment", "ID_NUMBER", "predicted_arpu"]
    missing = set(required) - set(profile.columns)
    if missing:
        raise ValueError(f"Profile is missing columns: {sorted(missing)}")
    key_columns = required[:2]
    missing_keys = profile[key_columns].isna() | profile[key_columns].eq("")
    excluded = missing_keys.any(axis=1)
    report = {
        "total_profiles": len(profile),
        "included_profiles": int((~excluded).sum()),
        "excluded_missing_keys": int(excluded.sum()),
        "missing_current_tariff": int(missing_keys["current_tariff"].sum()),
        "missing_arpu_segment": int(missing_keys["arpu_segment"].sum()),
        "excluded_A_sum": float(profile.loc[excluded, "predicted_arpu"].sum()),
    }
    if report["excluded_missing_keys"]:
        LOGGER.warning("Profile exclusion report (no broad filters): %s", report)
    valid = profile.loc[~excluded, required]
    if not valid["arpu_segment"].isin(SEGMENTS).all():
        raise ValueError("Unknown profile ARPU segment")
    if valid["ID_NUMBER"].isna().any():
        raise ValueError("Missing ID_NUMBER in a valid cell")
    if not np.isfinite(valid["predicted_arpu"].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite predicted_arpu in a valid cell")
    result = CellStats(report=report)
    for key, rows in valid.groupby(key_columns, observed=True, sort=True):
        ordered = rows.sort_values("ID_NUMBER", kind="stable")
        arpu = ordered["predicted_arpu"].to_numpy(dtype=float, copy=True)
        prefix = np.empty(len(arpu) + 1, dtype=float)
        prefix[0] = 0.0
        np.cumsum(arpu, out=prefix[1:])
        result[key] = {"N": len(rows), "A_sum": float(arpu.sum()),
                       "ids_sorted": ordered["ID_NUMBER"].to_numpy(copy=True),
                       "prefix_arpu": prefix}
    return result


def _has_history(hist_rank, key):
    # A sparse ordinary dict denotes observed actions, including zero ranks.
    # The loader's dense HistoricalRank uses explicit support instead.
    support = getattr(hist_rank, "support", None)
    return support.get(key, 0) > 0 if support is not None else key in hist_rank


def _tariff_geometry(tariffs):
    required = {"tariff_plan_code", *TARIFF_FEATURES}
    if not required.issubset(tariffs.columns):
        raise ValueError(f"Tariffs are missing columns: {sorted(required - set(tariffs.columns))}")
    frame = tariffs.loc[:, ["tariff_plan_code", *TARIFF_FEATURES]].copy()
    if frame["tariff_plan_code"].isna().any():
        raise ValueError("Missing tariff code")
    frame["tariff_plan_code"] = frame["tariff_plan_code"].astype(str)
    if frame["tariff_plan_code"].duplicated().any():
        raise ValueError("Duplicate tariff code")
    frame = frame.sort_values("tariff_plan_code", kind="stable")
    codes = frame["tariff_plan_code"].tolist()
    values = frame[list(TARIFF_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Tariff features must be finite and nonnegative")
    if not codes:
        return codes, values, np.empty((0, 0))
    logged = np.log1p(values)
    active = np.ptp(logged, axis=0) > 0
    standardized = logged[:, active]
    if active.any():
        standardized = (standardized - standardized.mean(axis=0)) / standardized.std(axis=0)
    distances = np.linalg.norm(standardized[:, None, :] - standardized[None, :, :], axis=2)
    return codes, values, distances


def _reserve_overview(queue, hist_rank):
    """Keep the best six, then promote two distinct diversity actions."""
    selected = list(queue[:6])
    remaining = list(queue[6:])

    def key(action):
        return action["from_tariff"], action["arpu_segment"], action["target"]

    def candidates(predicate):
        used_cells = {(a["from_tariff"], a["arpu_segment"]) for a in selected}
        eligible = [a for a in remaining if predicate(a)]
        # Prefer a new cell; within either group retain the original queue order.
        return sorted(eligible, key=lambda a: (a["from_tariff"], a["arpu_segment"]) in used_cells)

    nonhigh = candidates(lambda a: a["arpu_segment"] != "HIGH")
    unknown = candidates(lambda a: not _has_history(hist_rank, key(a)))
    # Avoid consuming the sole unknown action in the non-HIGH slot if another
    # non-HIGH action can occupy it. Each reserve must be a different action.
    outside = next((a for a in nonhigh if any(key(b) != key(a) for b in unknown)),
                   nonhigh[0] if nonhigh else None)
    if outside is not None:
        selected.append(outside)
        remaining.remove(outside)
    unknown = candidates(lambda a: not _has_history(hist_rank, key(a)))
    if unknown:
        selected.append(unknown[0])
        remaining.remove(unknown[0])
    return (selected + remaining)[:40]


def build_catalog(profile, tariffs, hist_rank):
    """Return up to 40 distinct actions, with at most four per cell.

    Cells are ordered by descending A_sum (ties by cell key). The ten most
    valuable cells are visited once before their second/third/fourth targets,
    then other cells fill available slots. Within a cell, a target already
    selected for an earlier basis is skipped for the next eligible target.
    The first six therefore favor breadth across valuable cells;
    slots seven/eight reserve a non-HIGH and an unsupported action when possible.
    Reserves are selected before truncating the full queue to 40.

    Pass the rank dict returned by load_or_build_historical_rank directly to
    preserve its support metadata. For a sparse ordinary dict, key membership
    denotes support, even when its rank is zero. History never removes a cell.
    """
    cells = cell_stats(profile)
    codes, values, distances = _tariff_geometry(tariffs)
    if len(codes) < 2:
        return []
    positions = {code: index for index, code in enumerate(codes)}
    lanes = [[] for _ in BASES]
    for (source, segment), stats in sorted(cells.items(), key=lambda item: (-item[1]["A_sum"], item[0])):
        if source not in positions:
            LOGGER.warning("Skipping cell with tariff absent from dictionary: %s", (source, segment))
            continue
        current = positions[source]
        others = [j for j, code in enumerate(codes) if code != source]
        historical = [j for j in others if _has_history(hist_rank, (source, segment, codes[j]))]
        expensive = [j for j in others if values[j, 0] > values[current, 0]]
        cheaper = [j for j in others if values[j, 0] <= values[current, 0]
                   and np.any(values[j, 1:] != values[current, 1:])]
        unseen = [j for j in others if not _has_history(hist_rank, (source, segment, codes[j]))]
        options = [
            sorted(historical, key=lambda j: (-hist_rank.get((source, segment, codes[j]), 0.0), codes[j])),
            sorted(expensive, key=lambda j: (distances[current, j], codes[j])),
            sorted(cheaper, key=lambda j: (distances[current, j], codes[j])),
            sorted(unseen, key=lambda j: (-distances[current, j], codes[j])),
        ]
        used = set()
        for index, ordered in enumerate(options):
            target = next((codes[j] for j in ordered if codes[j] not in used), None)
            if target is None:
                continue
            used.add(target)
            lanes[index].append({"from_tariff": source, "arpu_segment": segment,
                                 "target": target, "basis": BASES[index],
                                 "prior_rank": float(hist_rank.get((source, segment, target), 0.0))})
    ordered_cells = [cell for cell in sorted(cells, key=lambda cell: (-cells[cell]["A_sum"], cell))
                     if cell[0] in positions]
    choices = {cell: [] for cell in ordered_cells}
    for lane in lanes:
        for action in lane:
            choices[(action["from_tariff"], action["arpu_segment"])].append(action)

    def round_robin(cell_list):
        return [choices[cell][index] for index in range(len(BASES))
                for cell in cell_list if index < len(choices[cell])]

    queue = round_robin(ordered_cells[:10]) + round_robin(ordered_cells[10:])
    return _reserve_overview(queue, hist_rank)


def _main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path,
                        default=Path(__file__).parent / "data/change_tariff.csv")
    parser.add_argument("--tariffs", type=Path,
                        default=Path(__file__).parent / "data/dict_tariff.csv")
    parser.add_argument("--profile", type=Path,
                        default=Path(__file__).parent / "customer_profile.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    tariffs = pd.read_csv(args.tariffs)
    ranks = load_or_build_historical_rank(args.history, tariffs)
    print("Top 20 historical transitions (from, segment -> target):")
    for key in sorted(ranks.support, key=lambda key: (-ranks[key], key))[:20]:
        print(f"{key[0]:>10} {key[1]:>4} -> {key[2]:<10} "
              f"support={ranks.support[key]:5d} rank={ranks[key]:.8f}")
    profile = pd.read_csv(args.profile)
    cells = cell_stats(profile)
    print(json.dumps({"history": ranks.report, "profile": cells.report,
                      "cells": len(cells)}, indent=2))
    catalog = build_catalog(profile, tariffs, ranks)
    keys = [(a["from_tariff"], a["arpu_segment"], a["target"]) for a in catalog]
    assert catalog, "Empty catalog"
    assert len(keys) == len(set(keys)) <= 40
    assert all(source != target for source, _, target in keys)
    assert all(sum(k[:2] == key for k in keys) <= 4 for key in cells)
    print("First eight catalog actions:")
    print(json.dumps(catalog[:8], indent=2))
    print(f"Catalog: {len(catalog)} actions; "
          f"non-HIGH in first eight: {any(k[1] != 'HIGH' for k in keys[:8])}; "
          f"unsupported in first eight: {any(not _has_history(ranks, k) for k in keys[:8])}")


if __name__ == "__main__":
    _main()
