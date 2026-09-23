# Safe fix #7: read-only history at runtime

Branch: `miras/safe-fixes`. Baseline: `32b07172248cfb04057c797025eeffdacec7c6a1`.

Only production change: `Agent.act` always passes the packaged `historical_candidates.json` to the existing loader. Its JSON branch reads and validates the artifact; missing or corrupt artifacts use the existing zero-rank fallback without rebuilding or writing files. Raw CSV changes require an offline artifact rebuild before shipping.

No changes to k, pilot schedule, audience selection, partial-fit handling, or planner logic. Fixes #4 and #3 were not implemented. No merge or push was performed.

## Regression checks

- Before the fix, the new real `Agent.act` test detected an attempted JSON write after the raw CSV changed, despite a read-only filesystem.
- After the fix, runtime tests passed for a valid packaged artifact, a missing artifact and a corrupt artifact; no raw-history reads or writes were attempted. Pilots and a nonempty plan survived.
- Passed: 2 runtime-history tests, 6 existing agent tests and 4 existing candidate tests.

## Reproduce the paired comparison

```bash
cd solution
../.venv/bin/python compare_policies.py --baseline 32b07172248cfb04057c797025eeffdacec7c6a1 \
  --world-start 300 --timing-world 299 --full --diagnostic \
  --candidate-label readonly_history --json safe_history_comparison_results.json
```

`--full` was added only to the comparison harness to force 600 pairs; otherwise the timing estimate would have reduced this run to 180. Default harness behavior is unchanged.

Scope: 6 families × 5 world seeds (300–304) × 20 noise seeds (1000–1019) = **600 paired runs**, covering **30 distinct synthetic worlds**. Timing used a separate world 299 / noise 999.

Elapsed: 485.708 seconds. Status: `complete`. Source hashes unchanged: `True`.

The old agent is loaded from the frozen baseline commit. Shared beliefs, candidates, planner and historical artifact are unchanged from that baseline. The changed production module is fully frozen on the old side. Pilot requests and observations were asserted equal for every pair.

## Tails: before → after

| Family | P10 net | CVaR10 net | Negative runs |
|---|---:|---:|---:|
| rare_winners | 2,343,098.14 → 2,343,098.14 | 2,019,890.43 → 2,019,890.43 | 0% → 0% |
| close_effects | 4,929,246.63 → 4,929,246.63 | 4,115,305.96 → 4,115,305.96 | 0% → 0% |
| weak_positive | 354,597.76 → 354,597.76 | 286,423.38 → 286,423.38 | 0% → 0% |
| all_negative | -1,427,811.60 → -1,427,811.60 | -1,540,922.08 → -1,540,922.08 | 100% → 100% |
| history_misleading | 390,453.64 → 390,453.64 | 86,260.18 → 86,260.18 | 4% → 4% |
| outside_high | 4,457,616.19 → 4,457,616.19 | 4,254,662.52 → 4,254,662.52 | 0% → 0% |

**600/600 net results and final campaign lists are exactly equal.** Mean paired difference: **0.0**; world-bootstrap 95% CI: **[0.0, 0.0]**. No tail worsened, including under a strict zero-degradation rule. The fix is retained locally for review.

The JSON field `performance_acceptance` remains false because the original harness requires a strictly positive improvement. That superiority criterion is not the acceptance gate for this read-only safety fix; exact invariance passes the requested no-regression check.

Full per-pair plans, pilot logs, costs, contacts, source hashes and metrics: [safe_history_comparison_results.json](safe_history_comparison_results.json).

No separate mock-family benchmark was run in this step: the requested comparison covers the six synthetic families above. These results establish unchanged behavior on this frozen test set.
