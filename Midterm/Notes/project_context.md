# ANLY 656 Midterm — Project Context

## What / Why / How
- **What:** Predict nominal `sleep_deficit` (`0-none`, `mild`, `moderate`, `severe`) from `Data/sleep_rating.csv` (8,500 rows, 16 columns). Deliver `log_reg.py`, `tree.py`, `forest.py` + outputs + `midterm_report.pdf`.
- **Why:** Help identify causes of sleep deficit among software engineers so interventions can be targeted.
- **How:** Step-by-step with Cursor; student runs code in Spyder/`anly656`. Follow `Cursor_Project_Checklist`.

## Foundation completed
1. Statement of Intent: `Notes/Initial_Prompt.txt`
2. Project folder: `Midterm/` with `Templates/`, `Data/`, `Tools/`, `Notes/`
3. Data map draft/review: `Data/sleep_rating_data_map.py` (from `Tools/Data_Map_Tool.py`)

## Class distribution (imbalanced)
| Class | Count | Pct |
|-------|------:|----:|
| moderate | 4462 | 52.5% |
| mild | 2004 | 23.6% |
| 0-none | 1387 | 16.3% |
| severe | 647 | 7.6% |

## Templates to follow
- Logistic: `Templates/Logistic_Reg/nominal_logistic.py` (std scaling; multinomial + OvR; hold-out + CV)
- Tree: `Templates/Decision_Tree/Nominal_Target/Nominal_DecisionTree_Cellphone.py` (no interval scale; kitchen sink → grid → hold-out/CV)
- Forest: `Templates/Random_Forest/Nominal_Forest/Nominal_RandomForest_Cellphone.py` (optional feature selection; grid on `n_estimators`/depth/leaf)

## Pending decisions before Step 2 coding
1. Keep `morning_alarm_snoozes` as Nominal (0–7) or treat as Interval?
2. Confirm target as `DT.String` (template style) vs `DT.Nominal` + `no_encode`.
3. Build order: `log_reg.py` → `tree.py` → `forest.py`?
