# Step 3–7 — Logistic Regression Summary

## Model selection
- **Best model: Multinomial** (97.89% full-data accuracy) beat One-vs-Rest (87.88%)
- Multinomial recovers rare `severe` class much better (86.6% vs 62.8% OvR sensitivity on full data)

## Validation
| Metric | Train (70%) | Validation (30%) |
|--------|------------:|-----------------:|
| Accuracy | 98.02% | 97.53% |
| MISC | 2.0% | 2.5% |
| Overfit ratio (MISC) | — | **1.25** (slightly above 1.2 rule) |

10-fold CV: test MISC ≈ 2.5%, mean overfit ratio ≈ 1.20

## Per-class validation sensitivity
| Class | Val sensitivity | Most often mistaken for |
|-------|----------------:|-------------------------|
| 0-none | 99.3% | mild |
| mild | 99.0% | moderate |
| moderate | 98.4% | severe |
| severe | 83.5% | moderate (16.5%) |

## Causes signal (interval coefs, severe class)
Large effects toward severe: higher `sleep_latency_min`, more `morning_alarm_snoozes`; lower `total_sleep_hours` (very large negative coef for severe/moderate vs strong positive for 0-none/mild).

## Artifacts
- `log_reg.py`, `log_reg_output.txt`
- `confusion_all_features.png`, `confusion_ovr.png`, `confusion_heatmap_holdout.png`
