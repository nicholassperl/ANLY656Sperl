# ANLY 656 Midterm — Project Context

## What / Why / How
- **What:** Predict and address causes of sleep deficit among software engineers using `Data/sleep_rating.csv` (nominal target `sleep_deficit`: `0-none`, `mild`, `moderate`, `severe`). Success = deliverables in the midterm Word doc.
- **Why:** For the ANLY 656 midterm assignment.
- **How:** Agent may run each step, but must check with the student before the next step. Follow `Cursor_Project_Checklist` and class templates.

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

## Data map decisions
1. `morning_alarm_snoozes` = **Interval** (count).
2. `sleep_deficit` = **DT.String** (template style).
3. Build order (default): `log_reg.py` → `tree.py` → `forest.py`.
