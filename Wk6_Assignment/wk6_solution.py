# -*- coding: utf-8 -*-
"""
Wk6 Assignment: Credit Default Decision Tree
Data map drafted by Tools/Data_Map_Tool.py from Data/CreditDefaultData.csv
Target: Default (Binary 0/1)
Reviewed: Customer=Ignore (ID), Education=Nominal (kept).
Based on Templates/Decision_Tree/Binary_Target/DecisionTree_Forgery.py
"""
# ANSI color codes
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

# Overfitting rule used throughout: a tree passes when its validation MISC
# is at most max_ratio times its training MISC, OR at most max_gap above
# its training MISC.
max_ratio = 1.2
max_gap   = 0.005

import pandas as pd
import numpy as np
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Tree import tree_classifier
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
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
    n_folds = len(scores["train_score"])
    train_misc = (1.0 - scores["train_score"])
    train_smisc = 2.0 * (1.0 - scores["train_score"]).std()
    val_misc = (1.0 - scores["test_score"])
    val_smisc = 2.0 * (1.0 - scores["test_score"]).std()
    ratio_misc = np.zeros(n_folds)
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
    train_misc = train_misc.mean()
    val_misc = val_misc.mean()
    ratio = val_misc / train_misc if train_misc > 0 else np.inf
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

def passes_rule(train_misc, val_misc):
    return (val_misc <= max_ratio * train_misc) | \
           (val_misc - train_misc <= max_gap)

def print_summary(train_acc, val_acc):
    train_misc = 1.0 - train_acc
    val_misc = 1.0 - val_acc
    ratio_acc = train_acc / val_acc if val_acc > 0 else np.inf
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
    color = GREEN if passes_rule(train_misc, val_misc) else RED
    print(f"{GREEN} {'MISCLASSIFICATION':.<20s}{GOLD}{train_misc:>7.4f}",
          f"  {val_misc:>7.4f}   {color}{ratio_misc:>7.4f}{RESET}")
    print(f"{TEAL}", "-" * 47, f"{RESET}")

def make_grid(leaf_sizes):
    return [{'criterion': criteria,
             'max_depth': depths,
             'min_samples_leaf': [leaf],
             'min_samples_split': [2 * leaf]} for leaf in leaf_sizes]

def run_grid(leaf_sizes):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=12345)
    dtree = DecisionTreeClassifier(random_state=12345)
    grid_search = GridSearchCV(estimator=dtree, param_grid=make_grid(leaf_sizes),
                               cv=cv, scoring='accuracy',
                               return_train_score=True, refit=False,
                               n_jobs=njobs).fit(X, y)
    params = grid_search.cv_results_['params']
    results = pd.DataFrame(params)
    results['max_depth'] = pd.Series([p['max_depth'] for p in params],
                                     dtype=object)
    results['train_misc'] = 1.0 - grid_search.cv_results_['mean_train_score']
    results['val_misc'] = 1.0 - grid_search.cv_results_['mean_test_score']
    train_misc = results['train_misc']
    val_misc = results['val_misc']
    results['ratio'] = np.where(train_misc > 0,
                                val_misc / train_misc.where(train_misc > 0, 1),
                                np.inf)
    results['passed'] = passes_rule(train_misc, val_misc)
    return results

def select_tree(results):
    passed = results[results['passed']]
    if passed.empty:
        print(f"{RED}No tree passes the overfitting rule.",
              f"Selecting the tree with the smallest ratio.{RESET}")
        return results['ratio'].idxmin()
    return passed['val_misc'].idxmin()

def print_leaf_scan(results, selected):
    print(f"{GREEN} {'LEAF':>5s} {'SPLIT':>6s} {'% SMALLEST':>11s}",
          f"{'CRITERION':>9s} {'DEPTH':>5s}",
          f"{'TRAIN MISC':>10s} {'VAL MISC':>9s} {'RATIO':>7s}{RESET}")
    for leaf, rows in results.groupby('min_samples_leaf'):
        passed = rows[rows['passed']]
        if passed.empty:
            idx = rows['val_misc'].idxmin()
        else:
            idx = passed['val_misc'].idxmin()
        row = rows.loc[idx]
        color = GREEN if row['passed'] else RED
        mark = "  <-- selected" if idx == selected else ""
        print(f"{TEAL} {leaf:5d} {2 * leaf:6d} {leaf / n_min_fold:11.1%}",
              f"{row['criterion']:>9s} {str(row['max_depth']):>5s}",
              f"{GOLD}{row['train_misc']:10.4f} {row['val_misc']:9.4f}",
              f"{color}{row['ratio']:7.4f}{RED}{mark}{RESET}")

def misc_ratio(train_misc, val_misc):
    if train_misc > 0:
        return val_misc / train_misc
    return np.inf if val_misc > 0 else 1.0

def print_misc_row(label, n_val, t, v, ks_t, ks_v):
    ratio = misc_ratio(t, v)
    ks_ratio = misc_ratio(ks_t, ks_v)
    c1 = GREEN if passes_rule(t, v) else RED
    c2 = GREEN if passes_rule(ks_t, ks_v) else RED
    print(f" {TEAL}{str(label):.<15s}{GREEN}{n_val:6d}",
          f"{GOLD}{t:7.1%} {v:7.1%} {c1}{ratio:7.2f}",
          f"  {GOLD}{ks_t:7.1%} {ks_v:7.1%} {c2}{ks_ratio:7.2f}{RESET}")

data_map = {
    'Customer':        [DT.Ignore, (0.99, 29998.01)],
    'Default':         [DT.Binary, (0, 1)],
    'card_class':      [DT.Nominal, (1, 2, 3)],
    'Gender':          [DT.Binary, (1.0, 2.0)],
    'Education':       [DT.Nominal, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)],
    'Marital_Status':  [DT.Nominal, (0, 1, 2, 3)],
    'Age':             [DT.Interval, (20.99, 75.01)],
    'Credit_Limit':    [DT.Interval, (299.99, 27400.01)],
    'Jun_Status':      [DT.Interval, (-2.01, 8.01)],
    'May_Status':      [DT.Interval, (-2.01, 7.01)],
    'Apr_Status':      [DT.Interval, (-2.01, 8.01)],
    'Mar_Status':      [DT.Interval, (-2.01, 8.01)],
    'Feb_Status':      [DT.Interval, (-2.01, 8.01)],
    'Jan_Status':      [DT.Interval, (-2.01, 8.01)],
    'Jun_Bill':        [DT.Interval, (-492.01, 20166.18)],
    'May_Bill':        [DT.Interval, (-2309.4, 19896.72)],
    'Apr_Bill':        [DT.Interval, (-5378.44, 19800.82)],
    'Mar_Bill':        [DT.Interval, (-1731.08, 18148.99)],
    'Feb_Bill':        [DT.Interval, (-1812.85, 18737.51)],
    'Jan_Bill':        [DT.Interval, (-11614.43, 16990.6)],
    'Jun_Payment':     [DT.Interval, (-0.01, 16872.85)],
    'May_Payment':     [DT.Interval, (-0.01, 41966.21)],
    'Apr_Payment':     [DT.Interval, (-0.01, 17381.44)],
    'Mar_Payment':     [DT.Interval, (-0.01, 9092.15)],
    'Feb_Payment':     [DT.Interval, (-0.01, 11319.59)],
    'Jan_Payment':     [DT.Interval, (-0.01, 15150.64)],
    'Jun_PayPercent':  [DT.Interval, (-0.01, 1.01)],
    'May_PayPercent':  [DT.Interval, (-0.01, 1.01)],
    'Apr_PayPercent':  [DT.Interval, (-0.01, 1.01)],
    'Mar_PayPercent':  [DT.Interval, (-0.01, 1.01)],
    'Feb_PayPercent':  [DT.Interval, (-0.01, 1.01)],
    'Jan_PayPercent':  [DT.Interval, (-0.01, 1.01)],
}

# Step 1: Read the data
lbl = "Step 1: Reading Credit Default Data"
print_boundary(lbl)
df = pd.read_csv("Data/CreditDefaultData.csv")
print(f"{GOLD}Data loaded: {df.shape[0]} observations and {df.shape[1]} columns.{RESET}")

# Step 2: Create data map and apply ReplaceImputeEncode
lbl = "Step 2: ReplaceImputeEncode (RIE) Processing"
print_boundary(lbl)

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

target = "Default"
print(f"{GOLD}")
rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale=None,
                          no_impute=[target],
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          drop=False,
                          display=True)

encoded_df = rie.fit_transform(df)
print(f"\n{RED}encoded_df    {RESET}:",
      f"{encoded_df.shape[0]} cases and",
      f"{encoded_df.shape[1]} columns,\n",
      "               including target.")

print(f"\n{GOLD}Target {RED}'{target}'{GOLD} class counts:")
counts = encoded_df[target].value_counts()
for cls, n_cls in counts.items():
    print(f"  {TEAL}{str(cls):.<15s}{GREEN}{n_cls:6d}",
          f"{n_cls / len(encoded_df):6.1%}")
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
val_pred = kitchen_sink_tree_cv.predict(X_val)
train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)

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

# Leaf sizes based on smallest class (here balanced: 4000 each).
# Kitchen sink depth was 33, so depth grid covers moderate depths + None.
n_folds = 4
min_class = y.value_counts().idxmin()
n_min = y.value_counts().min()
n_min_fold = n_min * (n_folds - 1) / n_folds

leaf_pcts = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
leaf_sizes = sorted(set(max(2, round(p * n_min_fold)) for p in leaf_pcts))
criteria = ['gini', 'entropy']
depths = [4, 5, 6, 8, 10, 12, 15, 20, None]

lbl = "Hyperparameters"
print_boundary(lbl, 47)
print(f"{GREEN} {'criterion':.<20s}{GOLD}{criteria}{RESET}")
print(f"{GREEN} {'max_depth':.<20s}{GOLD}{depths}{RESET}")
print(f"{GREEN} {'min_samples_leaf':.<20s}{GOLD}{leaf_sizes}{RESET}")
split_sizes = [2 * leaf for leaf in leaf_sizes]
print(f"{GREEN} {'min_samples_split':.<20s}{GOLD}{split_sizes}{RESET}")

print(f"\n{GOLD}Smallest class {RED}'{min_class}'{GOLD}: {RED}{n_min}{GOLD}",
      f"cases, about {RED}{n_min_fold:.0f}{GOLD} in each training fold")
print(f"{GOLD}Leaf sizes are {RED}{leaf_pcts[0]:.0%}{GOLD} to",
      f"{RED}{leaf_pcts[-1]:.0%}{GOLD} of {RED}{n_min_fold:.0f}{RESET}")

total_combinations = len(criteria) * len(depths) * len(leaf_sizes)
total_fits = total_combinations * n_folds

njobs = -1
print(f"\n{GOLD}Grid Search: {total_combinations} parameter combinations")
print(f"Using a {n_folds}-fold CV requires {total_fits} total fits")
print(f"Parallel processing with {GOLD}n_jobs={njobs}",
      f"{RED}for maximum speed")

start_time = time.time()
print(f"\n{GOLD}Starting grid search...{RESET}")
results = run_grid(leaf_sizes)
end_time = time.time()
elapsed_time = end_time - start_time
minutes = math.floor(elapsed_time / 60)
seconds = elapsed_time - 60 * minutes
print(f"{GOLD}Grid search completed in {minutes:.0f} min. {seconds:.0f} sec.")
print(f"Average time per parameter combination ",
      f"{elapsed_time / total_combinations:.2f} seconds{RESET}")

best_idx = results['val_misc'].idxmin()
lbl = "Lowest Val MISC (Ignoring Overfitting)"
print_boundary(lbl, 47)
for parm in ['criterion', 'max_depth', 'min_samples_leaf', 'min_samples_split']:
    parameter = str(results.loc[best_idx, parm])
    print(f"{GREEN}            {parm:.<20s}{GOLD}{parameter:>9s}{RESET}")
print_summary(1.0 - results.loc[best_idx, 'train_misc'],
              1.0 - results.loc[best_idx, 'val_misc'])

lbl = "Pass 1: Coarse Leaf Size Scan"
print_boundary(lbl, 47)
selected = select_tree(results)
print_leaf_scan(results, selected)

leaf_sel = results.loc[selected, 'min_samples_leaf']
k = leaf_sizes.index(leaf_sel)
if k > 0:
    leaf_lo = leaf_sizes[k - 1]
else:
    leaf_lo = max(2, leaf_sizes[0] // 2)
leaf_hi = leaf_sizes[min(k + 1, len(leaf_sizes) - 1)]
fine_sizes = [leaf for leaf in range(leaf_lo, leaf_hi + 1)
              if leaf not in leaf_sizes]
if fine_sizes:
    lbl = "Pass 2: Fine Leaf Size Scan"
    print_boundary(lbl, 47)
    n_fine = len(criteria) * len(depths) * len(fine_sizes)
    print(f"{GOLD}Leaf sizes {RED}{leaf_lo}{GOLD} to {RED}{leaf_hi}{GOLD}:",
          f"{n_fine} new combinations,",
          f"{n_fine * n_folds} fits{RESET}")
    results = pd.concat([results, run_grid(fine_sizes)], ignore_index=True)
    selected = select_tree(results)
    fine = results[results['min_samples_leaf'].between(leaf_lo, leaf_hi)]
    print_leaf_scan(fine, selected)

lbl = "Optimum Hyperparameters"
print_boundary(lbl, 47)
best_params = results.loc[selected, ['criterion', 'max_depth',
                                     'min_samples_leaf',
                                     'min_samples_split']].to_dict()
for parm in best_params:
    parameter = str(best_params[parm])
    print(f"{GREEN}            {parm:.<20s}{GOLD}{parameter:>9s}{RESET}")

val_acc = 1.0 - results.loc[selected, 'val_misc']
train_acc = 1.0 - results.loc[selected, 'train_misc']

lbl = "Optimum Tree Performance Metrics"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

best_dtree = DecisionTreeClassifier(**best_params, random_state=12345)
best_dtree = best_dtree.fit(X, y)
print(f"{GOLD}Optimum tree (refit to all {X.shape[0]} cases) has depth",
      f"{RED}{best_dtree.get_depth()}{GOLD} and",
      f"{RED}{best_dtree.get_n_leaves()}{GOLD} leaves.{RESET}")

n_features = X.shape[1]
if n_features > 10:
    lbl = "Optimum Tree - Top 10 Features by Importance"
    n_top = 10
else:
    lbl = "Optimum Tree - Features by Importance"
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
val_pred = hold_out_tree.predict(Xv)
train_acc = accuracy_score(yt, train_pred)
val_acc = accuracy_score(yv, val_pred)

lbl = "Holdout Validation Performance Summary"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

ks_train_pred = kitchen_sink_tree_cv.predict(Xt)
ks_val_pred = kitchen_sink_tree_cv.predict(Xv)

print(f"\n{GOLD}{24 * ' '}OPTIMUM TREE{12 * ' '}KITCHEN SINK TREE{RESET}")
print(f"{GOLD} CLASS.......... N VAL   TRAIN     VAL   RATIO",
      f"    TRAIN     VAL   RATIO{RESET}")
for cls in hold_out_tree.classes_:
    t = (train_pred[yt == cls] != cls).mean()
    v = (val_pred[yv == cls] != cls).mean()
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

n = X.shape[0]
best_val_misc = np.inf
for k in range(2, 11):
    cv_k = StratifiedKFold(n_splits=k, shuffle=True, random_state=12345)
    scores = cross_validate(best_dtree, X, y, scoring='accuracy',
                            cv=cv_k, return_train_score=True)
    train_misc = 1.0 - scores["train_score"].mean()
    val_misc = 1.0 - scores["test_score"].mean()

    print_acc_ratio(scores, n)
    if val_misc < best_val_misc:
        best_k = k
        best_train_misc = train_misc
        best_val_misc = val_misc

print(f"\n{GOLD} Best K (lowest Test Avg. MISC) :",
      f"{RED}{best_k}-Fold{GOLD}")
lbl = "K-Fold Cross-Validation Performance Summary"
print_boundary(lbl, 47)
print_summary(1.0 - best_train_misc, 1.0 - best_val_misc)

# Step 7: Display and Save Decision Tree
lbl = "Step 7: Display and Save Decision Tree"
print_boundary(lbl)

features = X.columns
classes = ['No Default', 'Default']
print(f"{GOLD}Optimum tree: depth {RED}{best_dtree.get_depth()}{GOLD},",
      f"{RED}{best_dtree.get_n_leaves()}{GOLD} leaves.",
      f"Displaying up to {RED}3{GOLD} levels.")
print(f"{GOLD}Class order in each node: {TEAL}{classes}{RESET}")

fig = plt.figure(figsize=(26, 10))
plot_tree(best_dtree, feature_names=features, class_names=classes,
          fontsize=10, filled=True, impurity=False,
          proportion=True, precision=2, max_depth=3)
plt.tight_layout()
fig.savefig("CreditDefault_Tree.png", dpi=150, bbox_inches='tight')
fig.savefig("CreditDefault_Tree.pdf", bbox_inches='tight')
plt.close(fig)

# Also save DOT source for graphviz viewers
dot_data = export_graphviz(best_dtree, feature_names=features,
                           class_names=classes, filled=True,
                           impurity=False, proportion=True,
                           precision=3, max_depth=3)
with open("CreditDefault_Tree", "w") as f:
    f.write(dot_data)

print(f"{GOLD}Tree saved to {TEAL}CreditDefault_Tree.png{GOLD},",
      f"{TEAL}CreditDefault_Tree.pdf{GOLD}, and",
      f"{TEAL}CreditDefault_Tree{RESET} (DOT)")

lbl = "Credit Default Decision Tree Analysis Complete"
print_boundary(lbl)
