#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANLY 656 Midterm — Random Forest (Nominal Target)
Data: Data/sleep_rating.csv
Target: sleep_deficit (Nominal / String labels)
Data map drafted by Tools/Data_Map_Tool.py and reviewed for midterm.
Based on Templates/Random_Forest/Nominal_Forest/Nominal_RandomForest_Cellphone.py

Exam note: feature selection allowed for forest; do not reduce the case count.
"""
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

RANDOM_STATE        = 31415
FEATURE_THRESHOLD   = 0.99  # Proportion of features to select for optimization
OPTIMIZATION_SAMPLE = 1.00  # Use all cases (exam: do not reduce dataset)

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy  as np
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Forest              import forest_classifier

from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics         import accuracy_score
import warnings, time, math

def print_boundary(lbl, b_width=60, boundary=True):
    print("")
    margin = b_width - len(lbl) - 2
    lmargin = int(margin / 2)
    rmargin = lmargin
    if lmargin + rmargin < margin:
        lmargin += 1
    if boundary:
        print(f"{TEAL}", "=" * b_width, f"{RESET}")
    print(f"{GREEN}", lmargin * "*", lbl, rmargin * "*", f"{RESET}")
    if boundary:
        print(f"{TEAL}", "=" * b_width, f"{RESET}")

def print_acc_ratio(scores, n):
    n_folds     = len(scores["train_score"])
    train_misc  = (1.0 - scores["train_score"])
    train_smisc = 2.0 * (1.0 - scores["train_score"]).std()
    val_misc    = (1.0 - scores["test_score"])
    val_smisc   = 2.0 * (1.0 - scores["test_score"]).std()
    ratio_misc  = np.zeros(n_folds)
    for i in range(0, n_folds):
        if train_misc[i] > 0:
            ratio_misc[i] = val_misc[i] / train_misc[i]
        elif val_misc[i] > 0:
            ratio_misc[i] = np.inf
        else:
            ratio_misc[i] = 1.0
    try:
        s_ratio = 2.0 * ratio_misc.std()
    except Exception:
        s_ratio = np.nan
    train_misc  = train_misc.mean()
    val_misc    = val_misc.mean()
    ratio       = val_misc / train_misc if train_misc > 0 else np.inf
    print(f"{TEAL}\n")
    print(f" ====== {n_folds:.0f}-Fold Cross Validation =======")
    print(f"  Train Avg. MISC..... {train_misc:.4f} +/-{train_smisc:.4f}")
    print(f"  Test  Avg. MISC..... {val_misc:.4f} +/-{val_smisc:.4f}")
    if s_ratio == np.nan or s_ratio == np.inf:
        print(f"  Mean Misc Ratio..... {ratio:.4f}")
    else:
        print(f"  Mean Misc Ratio..... {ratio:.4f} +/-{s_ratio:.4f}")
    print(" ", 39 * "=", f"{RESET}")
    n_v = n * (1.0 / n_folds)
    n_t = n - n_v
    print(f"Equivalent to {n_folds:.0f} splits each with " +
          f"{n_t:.0f}/{n_v:.0f} Cases")

def print_summary(train_acc, val_acc):
    train_misc = 1.0 - train_acc
    val_misc   = 1.0 - val_acc
    ratio_acc  = train_acc / val_acc if val_acc > 0 else np.inf
    if train_misc > 0:
        ratio_misc = val_misc / train_misc
    elif val_misc > 0:
        ratio_misc = np.inf
    else:
        ratio_misc = 1.0
    print(f"{GREEN}{'TRAIN':>28s} {'VALIDATION':>11s} {'RATIO':>7s}")
    color = GREEN if ratio_acc < 1.2 else RED
    print(f"{GREEN} {'ACCURACY':.<20s}{GOLD}{train_acc:>7.4f}",
          f"  {val_acc:>7.4f}   {color}{ratio_acc:>7.4f}{RESET}")
    color = GREEN if ratio_misc < 1.2 else RED
    print(f"{GREEN} {'MISCLASSIFICATION':.<20s}{GOLD}{train_misc:>7.4f}",
          f"  {val_misc:>7.4f}   {color}{ratio_misc:>7.4f}{RESET}")
    print(f"{TEAL}", "-" * 47, f"{RESET}")

def tree_selection(X, y, threshold=0.9):
    if threshold >= 1.0:
        return X
    dt_selector = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X, y)
    feature_importance = dt_selector.feature_importances_
    feature_name       = X.columns

    importance_df = pd.DataFrame({
        'feature': feature_name,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()

    print(f"\n{GREEN}            Feature Importance Analysis{GOLD}")
    print(f"{'=' * 51}")
    print(f"{'Feature':<27} {'Importance':<12} {'Cumulative':<12}")
    print(f"{'-' * 51}")
    lne = False
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:.<29} {row['importance']:<12.4f} ",
              f"{row['cumulative_importance']:<12.4f}")
        if row['cumulative_importance'] > threshold and not lne:
            print(f"{RED}{25 * '- '}{GOLD}")
            lne = True
    print(f"{'=' * 51}")

    selected_mask = importance_df['cumulative_importance'] >= threshold
    if selected_mask.any():
        first_idx = selected_mask.idxmax()
        selected_features = importance_df.loc[:first_idx, 'feature'].tolist()
        threshold_reached = importance_df.loc[first_idx, 'cumulative_importance']
    else:
        selected_features = importance_df['feature'].tolist()
        threshold_reached = importance_df['cumulative_importance'].max()
        print(f"\n{RED}No feature combination reaches {threshold:.0%} importance. ",
              f"Using all features (cumulative: {threshold_reached:.1%})")

    print(f"\n{GREEN}Selected {len(selected_features)} features accounting for ",
          f"{threshold_reached:.0%} of importance")
    X_selected = X[selected_features]
    print(f"{GREEN}Feature selection complete - reduction: ",
          f"{RED}{X.shape[1]} -→ {X_selected.shape[1]} {GREEN}features{RESET}")
    return X_selected

""" ======================================================================= """
lbl = "Step 1: Reading Data"
print_boundary(lbl)
df = pd.read_csv("Data/sleep_rating.csv")
print(f"{GOLD}Data loaded: {df.shape[0]} ",
      f"observations and {df.shape[1]} columns.{RESET}")
print(f"\n{GOLD}First 5 rows of the data:{RESET}")
print(df.head())

lbl = "Step 2: Data Map and ReplaceImputeEncode (RIE) Processing"
print_boundary(lbl)

data_map = {
    'age':                       [DT.Interval, (17.99, 65.01)],
    'gender':                    [DT.Nominal, ('Female', 'Male', 'Non-Binary')],
    'bedtime_phone_minutes':     [DT.Interval, (0.99, 180.01)],
    'screen_brightness_pct':     [DT.Interval, (9.99, 100.01)],
    'blue_light_filter_active':  [DT.Binary, (0, 1)],
    'caffeine_post_5pm_mg':      [DT.Interval, (-0.01, 250.01)],
    'physical_activity_min':     [DT.Interval, (-0.01, 112.01)],
    'sleep_latency_min':         [DT.Interval, (5.99, 123.31)],
    'total_sleep_hours':         [DT.Interval, (3.19, 9.81)],
    'deep_sleep_pct':            [DT.Interval, (8.09, 28.01)],
    'rem_sleep_pct':             [DT.Interval, (9.59, 27.01)],
    'morning_alarm_snoozes':     [DT.Interval, (-0.01, 7.01)],
    'sleep_deficit':             [DT.String, ('0-none', 'mild', 'moderate', 'severe')],
    'occupation':                [DT.Nominal, (1, 2, 3, 4, 5)],
    'favorite_bedtime_app':      [DT.Nominal, (1, 2, 3, 4, 5, 6)],
    'clock_type':                [DT.Nominal, (1, 2, 3)],
}

print(f"{GOLD}")
print(15 * "=", "DATA MAP", 15 * "=")
lk = len(max(data_map, key=len)) + 1
ignored = 0
for col, (dt_type, valid_values) in data_map.items():
    if dt_type.name == "ID" or dt_type.name == "Ignore":
        ignored += 1
    print(f"  {TEAL}{col:.<{lk}s} {GOLD}{dt_type.name:9s}{GREEN}{valid_values}")
print(f"{GOLD} === Data Map has{RED}", len(data_map) - ignored,
      f"{GOLD}attribute columns", 3 * "=", f"{RESET}")

target = "sleep_deficit"
print(f"{GOLD}Target variable: {target}")
if data_map[target][0] == DT.String:
    target_ = None
else:
    target_ = [target]

rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale=None,
                          no_impute=target_,
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          drop=False,
                          display=True)

encoded_df = rie.fit_transform(df)
print(f"\n{RED}encoded_df:{GOLD}:",
      f"{encoded_df.shape[0]} cases and",
      f"{encoded_df.shape[1]} columns,\n",
      "               including targets.")
print(f"\n{GOLD}Preprocessing complete. Ready for next step.{RESET}")

# Step 3: Kitchen Sink Random Forest Evaluation
lbl = "Step 3: Kitchen Sink Random Forest (Default Parameters)"
print_boundary(lbl)

y = df[target]
X = encoded_df

print(f"{GOLD}Fitting kitchen sink random forest using entire dataset")
kitchen_sink_forest = RandomForestClassifier(random_state=RANDOM_STATE)
kitchen_sink_forest = kitchen_sink_forest.fit(X, y)
forest_classifier.display_metrics(kitchen_sink_forest, X, y)

lbl = "70/30 Holdout Validation of Kitchen Sink Forest"
print_boundary(lbl)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                  stratify=y,
                                                  random_state=RANDOM_STATE)

kitchen_sink_forest_cv = RandomForestClassifier(random_state=RANDOM_STATE)
kitchen_sink_forest_cv = kitchen_sink_forest_cv.fit(X_train, y_train)

print(f"{GOLD}")
forest_classifier.display_split_metrics(kitchen_sink_forest_cv,
                                        X_train, y_train, X_val, y_val)

train_pred = kitchen_sink_forest_cv.predict(X_train)
val_pred   = kitchen_sink_forest_cv.predict(X_val)
train_acc  = accuracy_score(y_train, train_pred)
val_acc    = accuracy_score(y_val, val_pred)

lbl = "Kitchen Sink 70/30 Validation"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

print(f"{GOLD}\nTop 10 Feature Importance (from training data):")
forest_classifier.display_importance(kitchen_sink_forest_cv, X.columns,
                                     top=10, plot=False)
print(f"{RESET}")

# Step 4.1: Feature Selection using Decision Tree Importance
lbl = "Step 4.1: Feature Selection using Decision Tree Importance"
print_boundary(lbl)

if FEATURE_THRESHOLD < 1.0:
    X_selected = tree_selection(X, y, FEATURE_THRESHOLD)
else:
    X_selected = X
    print(f"{GREEN}")
    print(f"Feature Threshold = 100%, using all features.{RESET}")
selected_features = X_selected.columns

# Step 4.2: Case Reduction - Construct Stratified Random Sample
lbl = "Step 4.2: Case Reduction using Stratified Random Sample"
print_boundary(lbl)
train_size = OPTIMIZATION_SAMPLE
print(f"{GREEN}")
if train_size >= 1:
    print(f"Optimization Sample = 100%, using all cases.{RESET}")
else:
    c = round(train_size * X.shape[0])
    print(f"Optimization Sample = {train_size:0.1%}, using {c} cases.{RESET}")

if train_size < 1.0:
    X_train, X_val, y_train, y_val = \
                   train_test_split(X_selected, y, train_size=train_size,
                                    stratify=y, random_state=RANDOM_STATE)
else:
    X_train = X_selected
    y_train = y

# Step 4.3: Random Forest Hyperparameter Optimization
lbl = "Step 4.3: Random Forest Hyperparameter Optimization"
print_boundary(lbl)

param_grid = {
    'n_estimators':      [50, 100],
    'criterion':         ['gini', 'entropy'],
    'max_depth':         [5, 7, 9, 11, 13],
    'min_samples_split': [74, 78, 82, 86, 90],
    'min_samples_leaf':  [37, 39, 41, 43, 45],
    'max_features':      [3, 5, 7, 9, 11, 13]
}
k     = X_selected.shape[1]
m     = math.floor(np.sqrt(k))
steps = math.ceil((k - m) / 6) if k > m else 1
maxfeatures = list(range(m, max(k - 1, m + 1), max(steps, 1)))
if not maxfeatures:
    maxfeatures = [m]
param_grid['max_features'] = maxfeatures

m  = math.floor(0.005 * y.shape[0]) - (y.nunique() - 1)
m  = max(m, 2)
ks = min(k, m, 5)
param_grid['max_depth']         = list(range(ks, min(ks + 10, 20), 2))
param_grid['min_samples_leaf']  = list(range(m, m + 10, 2))
param_grid['min_samples_split'] = list(range(2 * m, 2 * (m + 10), 4))

lbl = "Hyperparameters"
print_boundary(lbl, 47)
for parm in param_grid:
    print(f"{GREEN} {parm:.<20s}{GOLD}{param_grid[parm][0:]}{RESET}")

total_combinations = 1
for param_list in param_grid.values():
    total_combinations *= len(param_list)
total_fits = total_combinations * 4

njobs = -1
print(f"\n{GOLD}Grid Search: {total_combinations} parameter combinations with")
print(f"4-fold CV requires {total_fits} total fits.\n")
t_size = 100 * train_size
t = 100 * FEATURE_THRESHOLD
print(f"Hyperparameter optimization uses only the top {t}% of",
      f"\nfeatures and {t_size}% of the dataset.\n")
print(f"{GOLD}Setting {RED}n_jobs={njobs}{GOLD} for faster calculations",
      f"by using all available computer cores.{RESET}")

start_time = time.time()
print(f"\n{GREEN}Starting grid search...{RESET}")
rf = RandomForestClassifier(random_state=RANDOM_STATE)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=4, scoring='accuracy', return_train_score=True,
                           n_jobs=njobs).fit(X_train, y_train)
end_time = time.time()
elapsed_time = end_time - start_time
minutes = math.floor(elapsed_time / 60)
seconds = elapsed_time - 60 * minutes
print(f"{GOLD}Grid search completed in {minutes:.0f} min. {seconds:.0f} sec.")
print(f"Average time per parameter combination ",
      f"{elapsed_time / total_combinations:.2f} seconds{RESET}")

lbl = "Optimum Hyperparameters"
print_boundary(lbl, 47)
max_param_len = max(len(str(val)) for val in grid_search.best_params_.values())
sze = max_param_len + 3
for parm in grid_search.best_params_:
    parameter = str(grid_search.best_params_[parm])
    print(f"{GREEN}            {parm:.<20s}{GOLD}{parameter:>{sze}s}{RESET}")

best_idx  = np.argmin(grid_search.cv_results_['rank_test_score'])
val_acc   = grid_search.cv_results_['mean_test_score'][best_idx]
train_acc = grid_search.cv_results_['mean_train_score'][best_idx]

lbl = "Optimum Forest Performance Metrics"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

lbl = "Optimum Random Forest Importance"
print_boundary(lbl, 47)

opt_forest = grid_search.best_estimator_
importance = opt_forest.feature_importances_
feature    = X_train.columns
data       = {'feature': feature, 'importance': importance}
imp_df     = pd.DataFrame(data)
imp_df     = imp_df.sort_values(by='importance', ascending=False)
imp_df['cumulative'] = imp_df['importance'].cumsum()
print(f"{GOLD}")
print(imp_df.to_string(index=False))

# Step 5: Holdout Validation (70/30 split) - Selected Features
lbl = "Step 5: Holdout Validation (Selected Features)"
print_boundary(lbl)

Xt, Xv, yt, yv = train_test_split(X_selected, y, test_size=0.3,
                                  stratify=y, random_state=RANDOM_STATE)
hold_out_forest = opt_forest
hold_out_forest = hold_out_forest.fit(Xt, yt)
print(f"{GOLD}")
forest_classifier.display_split_metrics(hold_out_forest, Xt, yt, Xv, yv)
print(f"{RESET}")

train_pred = hold_out_forest.predict(Xt)
val_pred   = hold_out_forest.predict(Xv)
train_acc  = accuracy_score(yt, train_pred)
val_acc    = accuracy_score(yv, val_pred)

lbl = "Holdout Validation Performance Summary (Selected Features)"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

print(f"{GOLD}\nFeature Importance (optimized model - selected features):")
forest_classifier.display_importance(hold_out_forest, selected_features,
                                     top='all', plot=False)

# Save importance bar chart without interactive show()
imp_plot = pd.DataFrame({
    'feature': list(selected_features),
    'importance': hold_out_forest.feature_importances_
}).sort_values('importance', ascending=True)
ax = imp_plot.set_index('feature').plot(kind='barh', figsize=(8, 10), fontsize=12)
ax.set_ylabel("Features")
ax.set_title("Random Forest Feature Importance")
import matplotlib.pyplot as plt
plt.tight_layout()
plt.savefig("forest_feature_importance.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"{GOLD}Importance plot saved to {TEAL}forest_feature_importance.png{RESET}")

# Step 6: K-Fold Cross Validation - Selected Features
lbl = "Step 6: K-Fold Cross-Validation (Selected Features)"
print_boundary(lbl)

warnings.filterwarnings('ignore', category=RuntimeWarning)
n            = X_selected.shape[0]
best_val_acc = 0
for k in range(2, 11):
    scores = cross_validate(opt_forest, X_selected, y,
                            scoring='accuracy', cv=k, return_train_score=True)
    train_acc = scores["train_score"].mean()
    val_acc   = scores["test_score"].mean()

    print_acc_ratio(scores, n)
    if val_acc > best_val_acc:
        best_k = k
        best_train_acc = train_acc
        best_val_acc   = val_acc

print(f"\n{GOLD} Best K :",
      f"{RED}{best_k}-Fold{GOLD}")
lbl = "K-Fold Cross-Validation Performance Summary (Selected Features)"
print_boundary(lbl, 47)
print_summary(best_train_acc, best_val_acc)
lbl = "Random Forest Analysis Complete"
print_boundary(lbl)
