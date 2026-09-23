#!/usr/bin/env python3
"""Read-only audit of public Beeline CSVs; emit aggregate counts, never IDs.

Example:
    python priors_audit.py --data-dir /path/to/beeline_case_participants \
        --output priors_audit_results.json

Cleaning matches 03-priors-and-validation.md:
* Deduplicate exact change_tariff rows across every source column.
* Keep previous ARPU > 0 for relative-change support.
* Map historical previous ARPU: LOW < 1000; MID 1000..5000; HIGH > 5000.
* Current cells require nonempty current_tariff and arpu_segment.
* Candidate actions exclude target == current tariff.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def historical_segment(previous_arpu):
    if previous_arpu < 1000:
        return "LOW"
    if previous_arpu <= 5000:
        return "MID"
    return "HIGH"


def arpu_sum(rows):
    # Three decimals match the report; source values are otherwise unchanged.
    return round(sum(float(row["predicted_arpu"]) for row in rows), 3)


def audit(data_dir):
    profiles = read_csv(data_dir / "customer_profile.csv")
    changes = read_csv(data_dir / "data" / "change_tariff.csv")
    tariff_rows = read_csv(data_dir / "tariff_dictionary.csv")
    tariffs = {row["tariff_plan_code"] for row in tariff_rows}

    seen = set()
    unique_changes = []
    for row in changes:
        key = tuple(row.items())
        if key not in seen:
            seen.add(key)
            unique_changes.append(row)

    positive_changes = [
        row for row in unique_changes
        if float(row["AVG_ARPU_PREV_3M"]) > 0
    ]
    support = Counter(
        (
            row["tariff_plan_code_from"],
            historical_segment(float(row["AVG_ARPU_PREV_3M"])),
            row["tariff_plan_code_to"],
        )
        for row in positive_changes
    )
    cells = defaultdict(list)
    for row in profiles:
        if row["current_tariff"] and row["arpu_segment"]:
            if row["current_tariff"] not in tariffs:
                raise ValueError("A current tariff is absent from the dictionary")
            if row["arpu_segment"] not in {"LOW", "MID", "HIGH"}:
                raise ValueError("Unexpected current ARPU segment")
            cells[(row["current_tariff"], row["arpu_segment"])].append(row)

    actions = [
        (current, segment, target)
        for current, segment in cells
        for target in sorted(tariffs)
        if target != current
    ]
    targets_by_cell = {
        key: [target for target in sorted(tariffs)
              if target != key[0] and support[(*key, target)] > 0]
        for key in cells
    }

    def summarize_cells(predicate):
        selected = [key for key in cells if predicate(key)]
        rows = [row for key in selected for row in cells[key]]
        return {
            "cells": len(selected),
            "profiles": len(rows),
            "predicted_arpu_sum": arpu_sum(rows),
        }

    supported = sum(support[action] > 0 for action in actions)
    well_supported = sum(support[action] >= 10 for action in actions)
    valid_rows = [row for rows in cells.values() for row in rows]
    return {
        "method": {
            "source_files": [
                "customer_profile.csv", "data/change_tariff.csv",
                "tariff_dictionary.csv",
            ],
            "history_deduplication": "exact duplicate over all columns",
            "history_support_filter": "AVG_ARPU_PREV_3M > 0",
            "historical_segment_mapping": {
                "LOW": "previous ARPU < 1000",
                "MID": "1000 <= previous ARPU <= 5000",
                "HIGH": "previous ARPU > 5000",
            },
            "current_cells": "nonempty current_tariff and arpu_segment",
            "candidate_rule": "every dictionary target except current tariff",
            "arpu_aggregation": "sum original predicted_arpu, rounded to 3 decimals",
        },
        "history": {
            "raw_transition_rows": len(changes),
            "exact_duplicate_rows_removed": len(changes) - len(unique_changes),
            "deduplicated_transition_rows": len(unique_changes),
            "positive_previous_arpu_rows": len(positive_changes),
            "nonpositive_previous_arpu_rows_excluded":
                len(unique_changes) - len(positive_changes),
            "observed_target_tariffs": len({
                row["tariff_plan_code_to"] for row in unique_changes
            }),
        },
        "current_audience": {
            "all_profiles": len(profiles),
            "all_predicted_arpu_sum": arpu_sum(profiles),
            "valid_current_arpu_cells": len(cells),
            "valid_profiles": len(valid_rows),
            "valid_predicted_arpu_sum": arpu_sum(valid_rows),
            "profiles_missing_cell_key": len(profiles) - len(valid_rows),
            "dictionary_tariffs": len(tariffs),
        },
        "coverage": {
            "possible_nonself_actions": len(actions),
            "historically_supported_actions": supported,
            "historically_supported_actions_pct": round(100 * supported / len(actions), 2),
            "actions_with_at_least_10_observations": well_supported,
            "actions_with_at_least_10_observations_pct":
                round(100 * well_supported / len(actions), 2),
            "no_historically_supported_target": summarize_cells(
                lambda key: not targets_by_cell[key]
            ),
            "cells_over_5000_profiles": summarize_cells(
                lambda key: len(cells[key]) > 5000
            ),
            "cells_under_150_profiles": summarize_cells(
                lambda key: len(cells[key]) < 150
            ),
            "cells_under_10_profiles": summarize_cells(
                lambda key: len(cells[key]) < 10
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write aggregate JSON; otherwise stdout")
    args = parser.parse_args()
    result = json.dumps(audit(args.data_dir), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
