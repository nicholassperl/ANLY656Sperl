# Step 1 — Template & AdvancedAnalytics Review

## Working agreement (How)
Agent may run each step; check with student before advancing.

## Templates reviewed (nominal target)

### Logistic — `Templates/Logistic_Reg/nominal_logistic.py`
- RIE with `interval_scale="std"`, one-hot binary/nominal, target kept via `no_encode` + concat back
- Fit multinomial (`C=np.inf`) and One-vs-Rest; pick higher accuracy
- Stratified 70/30 hold-out + k-fold CV (MISC / overfit ratio > 1.2 flag)
- Helpers: `logreg.display_split_metrics`, confusion heatmaps, per-class sensitivity

### Tree — `Templates/Decision_Tree/Nominal_Target/Nominal_DecisionTree_Cellphone.py`
- RIE with `interval_scale=None` (no std scaling)
- Target as `DT.String`, concat back after RIE
- Kitchen-sink tree → grid search (depth/leaf/split/criterion) → hold-out → CV
- `tree_classifier.display_metrics` / `display_split_metrics` / `display_importance`
- Save tree plot (first 3 levels)

### Forest — `Templates/Random_Forest/Nominal_Forest/Nominal_RandomForest_Cellphone.py`
- RIE with `interval_scale=None`
- Kitchen-sink RF → optional feature selection (`FEATURE_THRESHOLD`) → grid search
- Exam allows feature selection for forest only; do not shrink the case count
- Hold-out + CV; `forest_classifier` metrics/importance

## AdvancedAnalytics focus
- `ReplaceImputeEncode` / `DT`: Interval, Binary, Nominal, String; outliers → missing → impute (except target rules)
- `Regression.logreg`, `Tree.tree_classifier`, `Forest.forest_classifier` for metrics and importance

## Fit to midterm data
- Same pattern as cellphone templates, swap data/target to `sleep_rating.csv` / `sleep_deficit`
- Target is 4-class and imbalanced (severe ~7.6%) → keep stratified splits
- No Ignore/ID columns
