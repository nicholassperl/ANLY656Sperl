#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANLY 656 Midterm — Decision Tree (Nominal Target)
Data: Data/sleep_rating.csv
Target: sleep_deficit (Nominal / String labels)
Data map drafted by Tools/Data_Map_Tool.py and reviewed for midterm.
Based on Templates/Decision_Tree/Nominal_Target/Nominal_DecisionTree_Cellphone.py
"""
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

max_ratio = 1.2

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy  as np
import graphviz
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Tree                import tree_classifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.base            import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics         import accuracy_score
from sklearn.tree            import plot_tree, export_graphviz
from matplotlib              import pyplot as plt
import time, math

def print_boundary(lbl, b_width=60):
    print("")
    margin = b_width - len(lbl) - 2
    lmargin = int(margin / 2)
    rmargin = lmargin
    if lmargin + rmargin < margin:
        lmargin += 1
    print(f"{TEAL}", "=" * b_width, f"{RESET}")
    print(f"{GREEN}", lmargin * "*", lbl, rmargin * "*", f"{RESET}")
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
    if np.all(np.isfinite(ratio_misc)):
        s_ratio = 2.0 * ratio_misc.std()
    else:
        s_ratio = np.nan
    train_misc  = train_misc.mean()
    val_misc    = val_misc.mean()
    ratio       = val_misc / train_misc if train_misc > 0 else np.inf
    print(f"{TEAL}\n")
    print(f" ====== {n_folds:.0f}-Fold Cross Validation =======")
    print(f"  Train Avg. MISC..... {train_misc:.4f} +/-{train_smisc:.4f}")
    print(f"  Test  Avg. MISC..... {val_misc:.4f} +/-{val_smisc:.4f}")
    if np.isfinite(s_ratio):
        print(f"  Mean Misc Ratio..... {ratio:.4f} +/-{s_ratio:.4f}")
    else:
        print(f"  Mean Misc Ratio..... {ratio:.4f}")
    print(" ", 39 * "=", f"{RESET}")
    n_v = n * (1.0 / n_folds)
    n_t = n - n_v
    print(f"Equivalent to {n_folds:.0f} splits each with " +
          f"{n_t:.0f}/{n_v:.0f} Cases")

def ratio_color(ratio):
    return RED if ratio > max_ratio else GREEN

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
    print(f"{GREEN} {'ACCURACY':.<20s}{GOLD}{train_acc:>7.4f}",
          f"  {val_acc:>7.4f}   {ratio_color(ratio_acc)}{ratio_acc:>7.4f}{RESET}")
    print(f"{GREEN} {'MISCLASSIFICATION':.<20s}{GOLD}{train_misc:>7.4f}",
          f"  {val_misc:>7.4f}   {ratio_color(ratio_misc)}{ratio_misc:>7.4f}{RESET}")
    print(f"{TEAL}", "-" * 47, f"{RESET}")

def misc_ratio(train_misc, val_misc):
    if train_misc > 0:
        return val_misc / train_misc
    return np.inf if val_misc > 0 else 1.0

def print_misc_row(label, n_val, t, v, ks_t, ks_v):
    ratio    = misc_ratio(t, v)
    ks_ratio = misc_ratio(ks_t, ks_v)
    c1 = ratio_color(ratio)
    c2 = ratio_color(ks_ratio)
    print(f" {TEAL}{label:.<15s}{GREEN}{n_val:6d}",
          f"{GOLD}{t:7.1%} {v:7.1%} {c1}{ratio:7.2f}",
          f"  {GOLD}{ks_t:7.1%} {ks_v:7.1%} {c2}{ks_ratio:7.2f}{RESET}")

# Step 1: Read the data
lbl = "Step 1: Reading Sleep Rating Data"
print_boundary(lbl)
data_file = "sleep_rating.csv"
df        = pd.read_csv("Data/" + data_file)
print(f"{GOLD}Data loaded: {RED}{df.shape[0]}{GOLD} observations",
      f"{RED}{df.shape[1]} {GOLD}columns.{RESET}")

# Step 2: Create data map and apply ReplaceImputeEncode
lbl = "Step 2: ReplaceImputeEncode (RIE) Processing"
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
print(f"{GOLD}")
rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale=None,
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          drop=False,
                          display=True)

encoded_df = rie.fit_transform(df)
encoded_df = pd.concat([df[target], encoded_df], axis=1)
print(f"\n{RED}encoded_df    {RESET}:",
      f"{encoded_df.shape[0]} cases and",
      f"{encoded_df.shape[1]} columns,\n",
      "               including target.")

print(f"\n{GOLD}Target {RED}'{target}'{GOLD} class counts:")
counts = encoded_df[target].value_counts()
for cls, n_cls in counts.items():
    print(f"  {TEAL}{cls:.<15s}{GREEN}{n_cls:6d}  {n_cls / len(encoded_df):6.1%}")
print(f"{RESET}")

# Step 3: Kitchen Sink Decision Tree Evaluation
lbl = "Step 3: Kitchen Sink Decision Tree (Default Parameters)"
print_boundary(lbl)

y = encoded_df[target]
X = encoded_df.drop(target, axis=1)

print(f"{GOLD}Fitting kitchen sink tree using entire dataset")
kitchen_sink_tree = DecisionTreeClassifier(random_state=12345)
kitchen_sink_tree = kitchen_sink_tree.fit(X, y)
tree_classifier.display_metrics(kitchen_sink_tree, X, y)
print(f"{GOLD}Tree grew to depth {RED}{kitchen_sink_tree.get_depth()}{GOLD}",
      f"with {RED}{kitchen_sink_tree.get_n_leaves()}{GOLD} leaves",
      f"for {RED}{X.shape[0]}{GOLD} cases.")
print(f"{RED}'Overfitting?'{RESET}")

lbl = "70/30 Holdout Validation of Kitchen Sink Tree"
print_boundary(lbl)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                  stratify=y,
                                                  random_state=12345)

kitchen_sink_tree_cv = DecisionTreeClassifier(random_state=12345)
kitchen_sink_tree_cv = kitchen_sink_tree_cv.fit(X_train, y_train)

print(f"{GOLD}")
tree_classifier.display_split_metrics(kitchen_sink_tree_cv,
                                      X_train, y_train, X_val, y_val)

train_pred = kitchen_sink_tree_cv.predict(X_train)
val_pred   = kitchen_sink_tree_cv.predict(X_val)
train_acc  = accuracy_score(y_train, train_pred)
val_acc    = accuracy_score(y_val, val_pred)

lbl = "Kitchen Sink 70/30 Validation"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

print(f"{GOLD}\nTop 10 Feature Importance (from training data):")
tree_classifier.display_importance(kitchen_sink_tree_cv, X.columns,
                                   plot=False, top=10)
print(f"{RESET}")

# Step 4: Decision Tree Hyperparameter Optimization
lbl = "Step 4: Decision Tree Hyperparameter Optimization"
print_boundary(lbl)

param_grid = {
    'criterion':         ['gini', 'entropy'],
    'max_depth':         [5, 6, 7],
    'min_samples_split': [20, 27, 34, 41, 48, 55],
    'min_samples_leaf':  [20, 22, 24, 26, 28]
}
k  = X.shape[1]
m  = math.floor(0.005 * X.shape[0] * 0.7)
m  = min(m, 20)
ks = min(k, m, 5)
param_grid['max_depth']         = list(range(ks, min(ks + 3, 20), 1))
param_grid['min_samples_leaf']  = list(range(m, m + 10, 2))
param_grid['min_samples_split'] = list(range(m, 3 * m, math.ceil(2 * m / 6)))

n_folds = 4
lbl = "Hyperparameters"
print_boundary(lbl, 47)
for parm in param_grid:
    print(f"{GREEN} {parm:.<20s}{GOLD}{param_grid[parm]}{RESET}")

total_combinations = 1
for param_list in param_grid.values():
    total_combinations *= len(param_list)
total_fits = total_combinations * n_folds

njobs = -1
print(f"\n{GOLD}Grid Search: {total_combinations} parameter combinations")
print(f"Using a {n_folds}-fold CV requires {total_fits} total fits")
print(f"Parallel processing with {GOLD}n_jobs={njobs}",
      f"{RED}for maximum speed")

start_time = time.time()
print(f"\n{GOLD}Starting grid search...{RESET}")
dtree       = DecisionTreeClassifier(random_state=12345)
grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid,
                           cv=StratifiedKFold(n_splits=n_folds, shuffle=True,
                                              random_state=12345),
                           scoring='accuracy',
                           return_train_score=True, n_jobs=njobs).fit(X, y)
end_time = time.time()
elapsed_time = end_time - start_time
minutes = math.floor(elapsed_time / 60)
seconds = elapsed_time - 60 * minutes
print(f"{GOLD}Grid search completed in {minutes:.0f} min. {seconds:.0f} sec.")
print(f"Average time per parameter combination ",
      f"{elapsed_time / total_combinations:.2f} seconds{RESET}")

lbl = "Optimum Decision Tree Hyperparameters"
print_boundary(lbl, 47)
max_param_len = max(len(str(val)) for val in grid_search.best_params_.values())
sze = max_param_len + 3
for parm in grid_search.best_params_:
    parameter = str(grid_search.best_params_[parm])
    print(f"{GREEN}            {parm:.<20s}{GOLD}{parameter:>{sze}s}{RESET}")

best_idx  = np.argmin(grid_search.cv_results_['rank_test_score'])
val_acc   = grid_search.cv_results_['mean_test_score'][best_idx]
train_acc = grid_search.cv_results_['mean_train_score'][best_idx]

lbl = "Optimum Tree Performance Metrics"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

best_dtree = grid_search.best_estimator_
print(f"{GOLD}Optimum tree (refit to all {X.shape[0]} cases) has depth",
      f"{RED}{best_dtree.get_depth()}{GOLD} and",
      f"{RED}{best_dtree.get_n_leaves()}{GOLD} leaves.{RESET}")

n_features = X.shape[1]
if n_features > 10:
    lbl   = "Optimum Tree - Top 10 Features by Importance"
    n_top = 10
else:
    lbl   = "Optimum Tree - Features by Importance"
    n_top = n_features
print_boundary(lbl, 47)
print(f"{GOLD}")
tree_classifier.display_importance(best_dtree, X.columns,
                                   plot=False, top=n_top)
print(f"{RESET}")

# Step 5: Holdout Validation (70/30 split)
lbl = "Step 5: Holdout Validation (70/30 split)"
print_boundary(lbl)

Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3,
                                  stratify=y, random_state=12345)
hold_out_tree = clone(best_dtree)
hold_out_tree = hold_out_tree.fit(Xt, yt)
print(f"{GOLD}")
tree_classifier.display_split_metrics(hold_out_tree, Xt, yt, Xv, yv)
print(f"{RESET}")

train_pred = hold_out_tree.predict(Xt)
val_pred   = hold_out_tree.predict(Xv)
train_acc  = accuracy_score(yt, train_pred)
val_acc    = accuracy_score(yv, val_pred)

lbl = "Holdout Validation Performance Summary"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

ks_train_pred = kitchen_sink_tree_cv.predict(Xt)
ks_val_pred   = kitchen_sink_tree_cv.predict(Xv)

print(f"\n{GOLD}{24 * ' '}OPTIMUM TREE{12 * ' '}KITCHEN SINK TREE{RESET}")
print(f"{GOLD} CLASS.......... N VAL   TRAIN     VAL   RATIO",
      f"    TRAIN     VAL   RATIO{RESET}")
for cls in hold_out_tree.classes_:
    t    = (train_pred[yt == cls] != cls).mean()
    v    = (val_pred[yv == cls] != cls).mean()
    ks_t = (ks_train_pred[yt == cls] != cls).mean()
    ks_v = (ks_val_pred[yv == cls] != cls).mean()
    print_misc_row(cls, (yv == cls).sum(), t, v, ks_t, ks_v)
print_misc_row("ALL CLASSES", len(yv),
               (train_pred != yt).mean(), (val_pred != yv).mean(),
               (ks_train_pred != yt).mean(), (ks_val_pred != yv).mean())
print("")

# Step 6: K-Fold Cross Validation
lbl = "Step 6: K-Fold Cross-Validation"
print_boundary(lbl)

n             = X.shape[0]
best_val_misc = np.inf
for k in range(2, 11):
    cv_k   = StratifiedKFold(n_splits=k, shuffle=True, random_state=12345)
    scores = cross_validate(best_dtree, X, y, scoring='accuracy',
                            cv=cv_k, return_train_score=True)
    train_misc = 1.0 - scores["train_score"].mean()
    val_misc   = 1.0 - scores["test_score"].mean()

    print_acc_ratio(scores, n)
    if val_misc < best_val_misc:
        best_k          = k
        best_train_misc = train_misc
        best_val_misc   = val_misc

print(f"\n{GOLD} Best K (lowest Test Avg. MISC) :",
      f"{RED}{best_k}-Fold{GOLD}")
lbl = "K-Fold Cross-Validation Performance Summary"
print_boundary(lbl, 47)
print_summary(1.0 - best_train_misc, 1.0 - best_val_misc)

# Step 7: Display and Save Decision Tree
lbl = "Step 7: Display and Save Decision Tree"
print_boundary(lbl)

features = X.columns
classes  = [str(c) for c in best_dtree.classes_]
print(f"{GOLD}Optimum tree: depth {RED}{best_dtree.get_depth()}{GOLD},",
      f"{RED}{best_dtree.get_n_leaves()}{GOLD} leaves.",
      f"Displaying the first {RED}3{GOLD} levels.")
print(f"{GOLD}Class order in each node: {TEAL}{classes}{RESET}")

fig = plt.figure(figsize=(28, 10))
plot_tree(best_dtree, feature_names=features, class_names=classes,
          fontsize=9, filled=True, impurity=False,
          proportion=True, precision=2, max_depth=3)
plt.savefig("sleep_deficit_tree_matplotlib.png", dpi=200, bbox_inches='tight')
plt.close()

dot_data = export_graphviz(best_dtree, feature_names=list(features),
                           class_names=classes, filled=True,
                           impurity=False, proportion=True,
                           precision=3, max_depth=3)
graph = graphviz.Source(dot_data)
with open("sleep_deficit_tree.png", "wb") as f:
    f.write(graph.pipe(format="png"))
graph.render("sleep_deficit_tree", format="pdf", cleanup=True)
print(f"{GOLD}Tree saved to {TEAL}sleep_deficit_tree.png{GOLD},",
      f"{TEAL}sleep_deficit_tree.pdf{GOLD}, and",
      f"{TEAL}sleep_deficit_tree_matplotlib.png{RESET}")

lbl = "Sleep Deficit Decision Tree Analysis Complete"
print_boundary(lbl)
