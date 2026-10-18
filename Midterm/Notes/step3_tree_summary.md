# Decision Tree Summary

## Kitchen sink
- Full-data perfect fit (depth 14, 152 leaves) — classic overfit
- 70/30 hold-out: train MISC 0%, val MISC 3.8% (ratio ∞)

## Tuned tree (grid search)
- **criterion=gini, max_depth=7, min_samples_leaf=20, min_samples_split=20**
- 37 leaves when fit to all cases
- Hold-out: train acc 97.70%, val acc **97.14%**, MISC ratio **1.24**
- Best CV K: **4-fold** (val MISC 2.71%, ratio 1.21)

## Feature importance (causes)
1. `total_sleep_hours` (~87%)
2. `morning_alarm_snoozes` (~10%)
3. `sleep_latency_min` (~2%)
4. Smaller: deep sleep %, caffeine, brightness, blue-light filter

## Artifacts
`tree.py`, `tree_output.txt`, `sleep_deficit_tree.png/.pdf`, `sleep_deficit_tree_matplotlib.png`
