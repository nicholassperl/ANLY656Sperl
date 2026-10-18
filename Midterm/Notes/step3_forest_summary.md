# Random Forest Summary

## Kitchen sink
- Full-data perfect accuracy; 70/30 val MISC 2.5% (severe 25.3%)

## Feature selection (exam-allowed)
- Threshold 99% cumulative DT importance → **9 features** (28 → 9)
- All 8,500 cases kept for optimization

## Tuned forest
- **entropy, n_estimators=50, max_depth=7, max_features=6, min_samples_leaf=39, min_samples_split=78**
- Grid: 2500 combos, ~17 minutes

## Validation
| Metric | Train | Validation |
|--------|------:|-----------:|
| Hold-out accuracy | 97.71% | **97.80%** |
| Hold-out MISC | 2.3% | **2.2%** |
| MISC ratio | — | **0.96** (no overfit) |
| Best CV (9-fold) val MISC | — | 2.51% (ratio 1.19) |

## Importance (causes)
1. `total_sleep_hours` (~82%)
2. `morning_alarm_snoozes` (~14%)
3. `sleep_latency_min` (~3%)
4. Smaller: bedtime phone minutes, deep sleep %

## Artifacts
`forest.py`, `forest_output.txt`, `forest_feature_importance.png`
