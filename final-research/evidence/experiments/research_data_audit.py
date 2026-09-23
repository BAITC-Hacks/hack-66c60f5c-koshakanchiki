"""Reproduce the research tables. Reads only participant-supplied CSV files.

Run: python research_data_audit.py
Requires pandas and numpy. Does not implement or run a competition agent.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research_tables"


def main():
    OUT.mkdir(exist_ok=True)
    paths = ["customer_profile.csv", "data/change_tariff.csv", "data/traffic.csv",
             "data/arpu_monthly.csv", "data/dict_tariff.csv", "tariff_dictionary.csv"]
    frames = {name: pd.read_csv(ROOT / name) for name in paths}
    p, h = frames[paths[0]], frames[paths[1]]
    inventory = []
    for name, df in frames.items():
        inventory.append({"file": name, "rows": len(df), "columns": len(df.columns),
                          "unique_ids": int(df.ID_NUMBER.nunique()) if "ID_NUMBER" in df else None,
                          "duplicate_rows": int(df.duplicated().sum()),
                          "null_cells": int(df.isna().sum().sum()),
                          "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()})
    pd.DataFrame(inventory).to_csv(OUT / "file_inventory.csv", index=False)
    p.describe(include="all").to_csv(OUT / "profile_describe.csv")
    p.isna().sum().rename("missing").to_csv(OUT / "profile_missing.csv")
    for cols, name in [(["arpu_segment"], "arpu_segments"),
                       (["current_tariff", "arpu_segment"], "audience_cells"),
                       (["current_tariff", "arpu_segment", "data_segment", "call_segment"], "audience_subcells")]:
        summary = p.groupby(cols, observed=True).agg(
            n=("ID_NUMBER", "size"), arpu_sum=("predicted_arpu", "sum"),
            arpu_mean=("predicted_arpu", "mean"), arpu_median=("predicted_arpu", "median"))
        summary.sort_values("arpu_sum", ascending=False).to_csv(OUT / f"{name}.csv")
    h = h.copy()
    h["arpu_segment"] = pd.cut(h.AVG_ARPU_PREV_3M, [-np.inf, 1000, 5000, np.inf],
                                labels=["LOW", "MID", "HIGH"])
    h["ratio_raw"] = np.where(h.AVG_ARPU_PREV_3M > 0,
                              h.AVG_ARPU_NEXT_3M / h.AVG_ARPU_PREV_3M - 1, np.nan)
    h["ratio_clipped"] = h.ratio_raw.clip(-1, 3)
    usable = h[h.AVG_ARPU_PREV_3M >= 100]
    transitions = usable.groupby(["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"], observed=True).agg(
        n=("ID_NUMBER", "size"), ratio_mean=("ratio_clipped", "mean"),
        ratio_median=("ratio_clipped", "median"), ratio_std=("ratio_clipped", "std"))
    transitions["conditional_destination_share"] = transitions.n / transitions.groupby(level=[0, 1]).n.transform("sum")
    transitions.to_csv(OUT / "history_transitions.csv")
    sizes = transitions.n
    overlap = []
    for name, df in frames.items():
        if "ID_NUMBER" in df:
            ids = set(df.ID_NUMBER)
            overlap.append({"file": name, "unique_ids": len(ids),
                            "ids_in_customer_profile": len(ids & set(p.ID_NUMBER))})
    pd.DataFrame(overlap).to_csv(OUT / "id_overlap.csv", index=False)
    dates = {}
    for name, df in frames.items():
        for col in ["TIME_KEY", "time_key"]:
            if col in df:
                dates[name] = df[col].value_counts().sort_index().to_dict()
    tariffs = frames["data/dict_tariff.csv"]
    dictionary = frames["tariff_dictionary.csv"]
    joined = tariffs.merge(dictionary, on="tariff_plan_code", suffixes=("_data", "_dictionary"))
    mismatches = []
    for col in tariffs.columns.drop("tariff_plan_code"):
        bad = joined[~np.isclose(joined[col + "_data"], joined[col + "_dictionary"], equal_nan=True)]
        mismatches.extend({"tariff": r["tariff_plan_code"], "field": col,
                           "data_value": r[col + "_data"], "dictionary_value": r[col + "_dictionary"]}
                          for _, r in bad.iterrows())
    pd.DataFrame(mismatches, columns=["tariff", "field", "data_value", "dictionary_value"]).to_csv(
        OUT / "tariff_dictionary_differences.csv", index=False)
    audit = {
        "baseline_total_arpu": float(p.predicted_arpu.sum()),
        "profile_id_duplicates": int(p.ID_NUMBER.duplicated().sum()),
        "profile_nonpositive_predicted_arpu": int((p.predicted_arpu <= 0).sum()),
        "profile_nonfinite_predicted_arpu": int((~np.isfinite(p.predicted_arpu)).sum()),
        "profile_missing_core_keys": int(p[["current_tariff", "arpu_segment"]].isna().any(axis=1).sum()),
        "profile_zero_predicted_arpu": int((p.predicted_arpu == 0).sum()),
        "profile_negative_predicted_arpu": int((p.predicted_arpu < 0).sum()),
        "high_segment_baseline_share": float(p.loc[p.arpu_segment == "HIGH", "predicted_arpu"].sum() / p.predicted_arpu.sum()),
        "top_four_cells_baseline_share": float(p.groupby(["current_tariff", "arpu_segment"]).predicted_arpu.sum().nlargest(4).sum() / p.predicted_arpu.sum()),
        "profile_sorted_by_id": bool(p.ID_NUMBER.is_monotonic_increasing),
        "history_unique_ids": int(h.ID_NUMBER.nunique()),
        "history_same_tariff_rows": int((h.tariff_plan_code_from == h.tariff_plan_code_to).sum()),
        "history_pre_arpu_below_100": int((h.AVG_ARPU_PREV_3M < 100).sum()),
        "history_pre_arpu_nonpositive": int((h.AVG_ARPU_PREV_3M <= 0).sum()),
        "history_raw_ratio_above_3": int((h.ratio_raw > 3).sum()),
        "history_ratio_quantiles_positive_denominator": h.ratio_raw.quantile([0, .01, .1, .5, .9, .99, 1]).to_dict(),
        "history_usable_rows": len(usable),
        "history_observed_transition_cells": len(transitions),
        "history_cells_n_lt_5": int((sizes < 5).sum()),
        "history_cells_n_lt_10": int((sizes < 10).sum()),
        "history_transition_n_quantiles": sizes.quantile([0, .25, .5, .75, .9, 1]).to_dict(),
        "audience_tariff_arpu_cells": int(p.groupby(["current_tariff", "arpu_segment"]).ngroups),
        "audience_full_subcells": int(p.groupby(["current_tariff", "arpu_segment", "data_segment", "call_segment"]).ngroups),
        "arpu_boundaries": {str(x): int((p.ARPU_3m_avg == x).sum()) for x in [1000, 5000]},
        "data_segment_counts": p.data_segment.value_counts().to_dict(),
        "call_segment_counts": p.call_segment.value_counts().to_dict(),
        "profile_lte_greater_than_data_rows": int((p.LTE_DATA_VOLUME > p.DATA_VOLUME).sum()),
        "tariff_dictionary_difference_count": len(mismatches),
        "dates": dates,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    noise = []
    for n in [10, 30, 50, 75, 100, 150, 200]:
        se = .804 / math.sqrt(n)
        noise.append({"n": n, "se": se, "normal_95_halfwidth": 1.96 * se,
                      "p_observed_negative_if_true_0_1": .5 * math.erfc(.1 / se / math.sqrt(2)),
                      "p_observed_negative_if_true_0_2": .5 * math.erfc(.2 / se / math.sqrt(2)),
                      **{f"cost_{channel}": n * cost for channel, cost in
                         [("push", 0), ("sms", 4), ("digital_ads", 22), ("call", 160)]}})
    pd.DataFrame(noise).to_csv(OUT / "pilot_noise_and_cost.csv", index=False)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(pd.read_csv(OUT / "arpu_segments.csv").to_string(index=False))
    print(pd.read_csv(OUT / "id_overlap.csv").to_string(index=False))


if __name__ == "__main__":
    main()
