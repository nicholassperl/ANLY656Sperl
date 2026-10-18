# Step 2 — Data & RIE

## Data
- File: `Data/sleep_rating.csv`
- Cases: 8,500 | Raw columns: 16
- Target: `sleep_deficit` (`DT.String`) — `0-none`, `mild`, `moderate`, `severe`

## Data map (shared across programs)
- 10 Interval (includes `morning_alarm_snoozes`)
- 1 Binary (`blue_light_filter_active`)
- 4 Nominal (`gender`, `occupation`, `favorite_bedtime_app`, `clock_type`) → one-hot
- 1 Excluded String target
- No Ignore/ID columns
- RIE found **0 missing** and **0 outliers** within map bounds

## Encoded predictors
28 columns after one-hot (all three programs agree on column names).

## RIE settings by program
| Program | `interval_scale` | Notes |
|---------|------------------|--------|
| `log_reg.py` | `"std"` | Matches nominal logistic template |
| `tree.py` | `None` | Matches nominal tree template |
| `forest.py` | `None` | Matches nominal forest template |

## Target distribution
| Class | N | % |
|-------|--:|--:|
| moderate | 4462 | 52.5% |
| mild | 2004 | 23.6% |
| 0-none | 1387 | 16.3% |
| severe | 647 | 7.6% |

## Next (Step 3)
Hyperparameter / model optimization — start with `log_reg.py` (multinomial vs OvR), then tree grid, then forest grid.
