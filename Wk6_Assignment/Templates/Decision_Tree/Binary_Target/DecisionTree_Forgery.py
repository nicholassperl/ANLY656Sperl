#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@purpose: Bank check signature forgery detection.  Step-by-step fitting
          and validation of a decision tree for a binary target using
          all attributes.
@data:    banknote_authentication.csv (1,372 rows x 5 columns)
@author:  EJones
@email:   ejones@tamu.edu
"""
# ANSI color codes - to print in color, the package colorama must be installed
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

# Overfitting rule used throughout: a tree passes when its validation MISC
# is at most max_ratio times its training MISC, OR at most max_gap above
# its training MISC.  The gap matters when MISC is near zero: 0.4% vs 0.0%
# is an infinite ratio, but only a 0.4 percentage point difference.
max_ratio = 1.2
max_gap   = 0.005

# Import required packages
import pandas as pd
import numpy  as np
import graphviz
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Tree                import tree_classifier
from matplotlib              import pyplot as plt
from sklearn.tree            import DecisionTreeClassifier, plot_tree
from sklearn.tree            import export_graphviz
from sklearn.base            import clone
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics         import accuracy_score
import time, math

def print_boundary(lbl, b_width=60):
    print("")
    margin = b_width - len(lbl) - 2
    lmargin = int(margin/2)
    rmargin = lmargin
    if lmargin+rmargin < margin:
        lmargin += 1
    print(f"{TEAL}", "="*b_width, f"{RESET}")
    print(f"{GREEN}", lmargin*"*", lbl, rmargin*"*", f"{RESET}")
    print(f"{TEAL}", "="*b_width, f"{RESET}")

def print_acc_ratio(scores, n):
    n_folds     = len(scores["train_score"])
    train_misc  = (1.0 - scores["train_score"])
    train_smisc = 2.0*(1.0 - scores["train_score"]).std()
    val_misc    = (1.0 - scores["test_score"])
    val_smisc   = 2.0*(1.0 - scores["test_score"]).std()
    ratio_misc  = np.zeros(n_folds)
    for i in range(0, n_folds):
        if train_misc[i]>0:
            ratio_misc[i] = val_misc[i]  / train_misc[i]
        elif val_misc[i]>0:
            ratio_misc[i] = np.inf
        else:
            ratio_misc[i] = 1.0
    # The spread of the ratios is only defined when every fold's ratio is
    # finite (training MISC > 0 in every fold)
    if np.all(np.isfinite(ratio_misc)):
        s_ratio = 2.0*ratio_misc.std()
    else:
        s_ratio = np.nan
    train_misc  = train_misc.mean()
    val_misc    = val_misc.mean()
    ratio       = val_misc/train_misc if train_misc>0 else np.inf
    print(f"{TEAL}\n")
    print(f" ====== {n_folds:.0f}-Fold Cross Validation =======")
    print(f"  Train Avg. MISC..... {train_misc:.4f} +/-{train_smisc:.4f}")
    print(f"  Test  Avg. MISC..... {val_misc:.4f} +/-{val_smisc:.4f}")
    if np.isfinite(s_ratio):
        print(f"  Mean Misc Ratio..... {ratio:.4f} +/-{s_ratio:.4f}")
    else:
        print(f"  Mean Misc Ratio..... {ratio:.4f}")
    print(" ", 39*"=", f"{RESET}")
    n_v = n*(1.0/n_folds)
    n_t = n - n_v
    print(f"Equivalent to {n_folds:.0f} splits each with "+
          f"{n_t:.0f}/{n_v:.0f} Cases")

def passes_rule(train_misc, val_misc):
    # Works for single values and for pandas columns
    return (val_misc <= max_ratio*train_misc) | \
           (val_misc - train_misc <= max_gap)

def print_summary(train_acc, val_acc):
    train_misc = 1.0 - train_acc
    val_misc   = 1.0 - val_acc
    ratio_acc  = train_acc / val_acc    if val_acc>0    else np.inf
    if train_misc>0:
        ratio_misc = val_misc  / train_misc 
    elif val_misc>0:
        ratio_misc = np.inf
    else:
        ratio_misc = 1.0
    #print accuracy and misclassification summary for train/validation
    print(f"{GREEN}{'TRAIN':>28s} {'VALIDATION':>11s} {'RATIO':>7s}")
    if ratio_acc < 1.2:
        color = GREEN
    else:
        color = RED
    print(f"{GREEN} {'ACCURACY':.<20s}{GOLD}{train_acc:>7.4f}",
      f"  {val_acc:>7.4f}   {color}{ratio_acc:>7.4f}{RESET}")

    if passes_rule(train_misc, val_misc):
        color = GREEN
    else:
        color = RED
    print(f"{GREEN} {'MISCLASSIFICATION':.<20s}{GOLD}{train_misc:>7.4f}",
      f"  {val_misc:>7.4f}   {color}{ratio_misc:>7.4f}{RESET}")
    print(f"{TEAL}","-"*47, f"{RESET}")

def make_grid(leaf_sizes):
    # A list of grids, one per leaf size, so that each leaf size is
    # paired only with its own split size (2 x leaf)
    return [{'criterion':         criteria,
             'max_depth':         depths,
             'min_samples_leaf':  [leaf],
             'min_samples_split': [2*leaf]} for leaf in leaf_sizes]

def run_grid(leaf_sizes):
    # Grid search over the leaf sizes; returns one row per combination
    # Stratified folds keep each class's proportion in every fold, and
    # shuffle=True keeps any ordering in the file from affecting the folds.
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=12345)
    dtree = DecisionTreeClassifier(random_state=12345)
    grid_search = GridSearchCV(estimator=dtree, param_grid=make_grid(leaf_sizes),
                               cv=cv, scoring='accuracy',
                               return_train_score=True, refit=False,
                               n_jobs=njobs).fit(X, y)
    params  = grid_search.cv_results_['params']
    results = pd.DataFrame(params)
    # keep max_depth as integers and None (not floats and NaN)
    results['max_depth'] = pd.Series([p['max_depth'] for p in params],
                                     dtype=object)
    # MISC = 1 - accuracy, averaged over the folds
    results['train_misc'] = 1.0 - grid_search.cv_results_['mean_train_score']
    results['val_misc']   = 1.0 - grid_search.cv_results_['mean_test_score']
    train_misc = results['train_misc']
    val_misc   = results['val_misc']
    results['ratio'] = np.where(train_misc > 0,
                                val_misc/train_misc.where(train_misc > 0, 1),
                                np.inf)
    results['passed'] = passes_rule(train_misc, val_misc)
    return results

def select_tree(results):
    # Lowest validation MISC among trees that pass the overfitting rule.
    # Ties keep the first (gini before entropy, shallower depth first).
    passed = results[results['passed']]
    if passed.empty:
        print(f"{RED}No tree passes the overfitting rule.",
              f"Selecting the tree with the smallest ratio.{RESET}")
        return results['ratio'].idxmin()
    return passed['val_misc'].idxmin()

def print_leaf_scan(results, selected):
    # For each leaf size, the tree the rule would select at that leaf:
    # the lowest validation MISC among trees that pass the rule, or, if
    # none pass, the lowest validation MISC (ratio shown in red)
    print(f"{GREEN} {'LEAF':>5s} {'SPLIT':>6s} {'% SMALLEST':>11s}",
          f"{'CRITERION':>9s} {'DEPTH':>5s}",
          f"{'TRAIN MISC':>10s} {'VAL MISC':>9s} {'RATIO':>7s}{RESET}")
    for leaf, rows in results.groupby('min_samples_leaf'):
        passed = rows[rows['passed']]
        if passed.empty:
            idx = rows['val_misc'].idxmin()
        else:
            idx = passed['val_misc'].idxmin()
        row   = rows.loc[idx]
        color = GREEN if row['passed'] else RED
        mark  = "  <-- selected" if idx == selected else ""
        print(f"{TEAL} {leaf:5d} {2*leaf:6d} {leaf/n_min_fold:11.1%}",
              f"{row['criterion']:>9s} {str(row['max_depth']):>5s}",
              f"{GOLD}{row['train_misc']:10.4f} {row['val_misc']:9.4f}",
              f"{color}{row['ratio']:7.4f}{RED}{mark}{RESET}")

def misc_ratio(train_misc, val_misc):
    if train_misc > 0:
        return val_misc/train_misc
    return np.inf if val_misc > 0 else 1.0

def print_misc_row(label, n_val, t, v, ks_t, ks_v):
    ratio    = misc_ratio(t, v)
    ks_ratio = misc_ratio(ks_t, ks_v)
    c1 = GREEN if passes_rule(t, v)       else RED
    c2 = GREEN if passes_rule(ks_t, ks_v) else RED
    print(f" {TEAL}{str(label):.<15s}{GREEN}{n_val:6d}",
          f"{GOLD}{t:7.1%} {v:7.1%} {c1}{ratio:7.2f}",
          f"  {GOLD}{ks_t:7.1%} {ks_v:7.1%} {c2}{ks_ratio:7.2f}{RESET}")
		
# Step 1: Read the data
lbl = "Step 1: Reading Banknote Authentication Data"
print_boundary(lbl)
df = pd.read_csv("../../data/banknote_authentication.csv")
print(f"{GOLD}Data loaded: {df.shape[0]} observations and {df.shape[1]} columns.{RESET}")

# Step 2: Create data map and apply ReplaceImputeEncode
lbl = "Step 2: ReplaceImputeEncode (RIE) Processing"
print_boundary(lbl)
# Create data map based on data dictionary
data_map = {
    'variance':  [DT.Interval, (-7.0521, 6.8348)],
    'skewness':  [DT.Interval, (-13.7831, 12.9616)],
    'kurtosis':  [DT.Interval, (-5.2961, 17.9374)],
    'entropy':   [DT.Interval, (-8.5582, 2.4595)],
    'forged':    [DT.Binary, ('no', 'yes')],
}

print(f"{GOLD}")
print(15*"=", "DATA MAP", 15*"=")
lk = len(max(data_map, key=len)) + 1
ignored = 0
for col, (dt_type, valid_values) in data_map.items():
    if dt_type.name == "ID" or dt_type.name=="Ignore":
        ignored += 1
    print(f"  {TEAL}{col:.<{lk}s} {GOLD}{dt_type.name:9s}{GREEN}{valid_values}")
print(f"{GOLD} === Data Map has{RED}", len(data_map)-ignored,
      f"{GOLD}attribute columns", 3*"=",f"{RESET}")

# Set target variable
target = "forged"
print(f"{GOLD}")
# Apply ReplaceImputeEncode preprocessing
rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale=None, # No Interval Scaling
                          no_impute=[target],  # Do not impute target
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          drop=False, # Keep all columns
                          display=True)

# Transform the data
encoded_df = rie.fit_transform(df)
print(f"\n{RED}encoded_df    {RESET}:",
      f"{encoded_df.shape[0]} cases and",
      f"{encoded_df.shape[1]} columns,\n",
      "               including target.")

# Target class distribution
print(f"\n{GOLD}Target {RED}'{target}'{GOLD} class counts:")
counts = encoded_df[target].value_counts()
for cls, n_cls in counts.items():
    print(f"  {TEAL}{str(cls):.<15s}{GREEN}{n_cls:6d}",
          f"{n_cls/len(encoded_df):6.1%}")
print(f"{RESET}")

# Step 3: Kitchen Sink Decision Tree Evaluation
lbl = "Step 3: Kitchen Sink Decision Tree (Default Parameters)"
print_boundary(lbl)

y = encoded_df[target]
X = encoded_df.drop(target, axis=1)

# First show the overfitting on full data
# Default parameters: no limit on depth, and leaves can hold a single case.
# random_state only fixes how ties between equally good splits are broken.
print(f"{GOLD}Fitting kitchen sink tree using entire dataset")
kitchen_sink_tree = DecisionTreeClassifier(random_state=12345)
kitchen_sink_tree = kitchen_sink_tree.fit(X, y)
tree_classifier.display_metrics(kitchen_sink_tree, X, y)
print(f"{GOLD}Tree grew to depth {RED}{kitchen_sink_tree.get_depth()}{GOLD}",
      f"with {RED}{kitchen_sink_tree.get_n_leaves()}{GOLD} leaves",
      f"for {RED}{X.shape[0]}{GOLD} cases.")
print(f"{RED}'Overfitting?'{RESET}")
# Now evaluate with proper holdout validation
lbl = "70/30 Holdout Validation of Kitchen Sink Tree"
print_boundary(lbl)

# Split the data 70/30, stratified so each class keeps its proportion.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                  stratify=y,
                                                  random_state=12345)

# Fit kitchen sink tree on training data only
kitchen_sink_tree_cv = DecisionTreeClassifier(random_state=12345)
kitchen_sink_tree_cv = kitchen_sink_tree_cv.fit(X_train, y_train)

# Evaluate using AdvancedAnalytics display_split_metrics
print(f"{GOLD}")
tree_classifier.display_split_metrics(kitchen_sink_tree_cv,
                                      X_train, y_train, X_val, y_val)

# Calculate and display accuracy ratio and misclassification ratio
train_pred = kitchen_sink_tree_cv.predict(X_train)
val_pred   = kitchen_sink_tree_cv.predict(X_val)
train_acc  = accuracy_score(y_train, train_pred)
val_acc    = accuracy_score(y_val, val_pred)

lbl = "Kitchen Sink 70/30 Validation"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc) #train/validation summary

# Show feature importance from the properly trained model
print(f"{GOLD}\nTop 10 Feature Importance (from training data):")
tree_classifier.display_importance(kitchen_sink_tree_cv, X.columns, 
                                   plot=False, top=10)
print(f"{RESET}")
# Step 4: Decision Tree Hyperparameter Optimization
lbl = "Step 4: Decision Tree Hyperparameter Optimization"
print_boundary(lbl)

# How the leaf and split sizes are selected (rule of thumb)
# ----------------------------------------------------------
# 1. Base min_samples_leaf on the SMALLEST class, not on the total n.
#    Every class needs room for several leaves of its own.  If leaves
#    are too large, a small class cannot be separated from the others
#    and its misclassification rate grows, even when overall accuracy
#    looks fine.
# 2. Here the smallest class is 1 (forged) with 610 cases.  With
#    4-fold cross-validation each training fold holds 3/4 of the data,
#    or about 458 forged cases.  A leaf of about 2% to 5% of that
#    count (9 to 23 cases) is a reasonable target.  The leaf sizes are
#    computed as 1% to 15% of that count, so the grid brackets this
#    range from both sides and adapts to any dataset.
# 3. min_samples_split is always set to 2 x min_samples_leaf.  A split
#    creates two children, each of which must hold at least
#    min_samples_leaf cases, so any split size smaller than 2 x leaf
#    has no effect.  Tying the two avoids fitting duplicate trees.
# 4. max_depth: the kitchen sink tree grew to depth 7, so the grid
#    covers depths from 2 to 8, plus no limit (None).
#
# How the optimum tree is selected (overfitting rule)
# ---------------------------------------------------
# Trees are compared on MISC (misclassification rate), the classification
# counterpart of ASE for interval targets.  The tree with the lowest
# validation MISC usually has small leaves and overfits: its validation
# MISC is much larger than its training MISC.  Minimizing the MISC ratio
# alone is not the goal either, since a tree with one split has a ratio
# near 1.0.  The rule used here is:
#     Select the tree with the LOWEST validation MISC among the trees
#     whose MISC ratio (validation MISC / training MISC) <= 1.2,
#     or whose validation MISC is at most 0.01 above its training MISC
# The second condition only matters when MISC is near zero, where the
# ratio is unstable (see max_ratio and max_gap at the top of the file).
# Pass 1 (coarse) applies the rule to the leaf sizes from item 2.
# Pass 2 (fine) tries every leaf size between the coarse leaf sizes just
# below and just above the one selected in Pass 1, and applies the rule
# again to all the trees from both passes.  If Pass 1 selects the smallest
# coarse leaf size, Pass 2 starts at half that size, so the search is not
# cut off at the edge of the grid.
# Both leaf size and depth control overfitting: a tree with small leaves
# can still pass the rule if its depth is limited.  The rule weighs
# every leaf size and depth combination together.
n_folds    = 4
min_class  = y.value_counts().idxmin()
n_min      = y.value_counts().min()
n_min_fold = n_min*(n_folds-1)/n_folds

leaf_pcts  = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
leaf_sizes = sorted(set(max(2, round(p*n_min_fold)) for p in leaf_pcts))
criteria   = ['gini', 'entropy']
depths     = [4, 5, 6, 8, 10, 12, None]

lbl = "Hyperparameters"
print_boundary(lbl, 47)
print(f"{GREEN} {'criterion':.<20s}{GOLD}{criteria}{RESET}")
print(f"{GREEN} {'max_depth':.<20s}{GOLD}{depths}{RESET}")
print(f"{GREEN} {'min_samples_leaf':.<20s}{GOLD}{leaf_sizes}{RESET}")
split_sizes = [2*leaf for leaf in leaf_sizes]
print(f"{GREEN} {'min_samples_split':.<20s}{GOLD}{split_sizes}{RESET}")

print(f"\n{GOLD}Smallest class {RED}'{min_class}'{GOLD}: {RED}{n_min}{GOLD}",
      f"cases, about {RED}{n_min_fold:.0f}{GOLD} in each training fold")
print(f"{GOLD}Leaf sizes are {RED}{leaf_pcts[0]:.0%}{GOLD} to",
      f"{RED}{leaf_pcts[-1]:.0%}{GOLD} of {RED}{n_min_fold:.0f}{RESET}")

"""
=============================================== 
 *************** Hyperparameters *************** 
 =============================================== 
 criterion...........['gini', 'entropy']
 max_depth...........[4, 5, 6, 8, 10, 12, None]
 min_samples_leaf....[5, 9, 14, 23, 34, 46, 69]
 min_samples_split...[10, 18, 28, 46, 68, 92, 138]
 """
 
# Calculate and display grid search information
total_combinations = len(criteria)*len(depths)*len(leaf_sizes)
total_fits = total_combinations * n_folds

njobs = -1
print(f"\n{GOLD}Grid Search: {total_combinations} parameter combinations")
print(f"Using a {n_folds}-fold CV requires {total_fits} total fits")
print(f"Parallel processing with {GOLD}n_jobs={njobs}",
      f"{RED}for maximum speed")

# Pass 1: coarse scan of leaf sizes
start_time = time.time()
print(f"\n{GOLD}Starting grid search...{RESET}")
results = run_grid(leaf_sizes)
end_time = time.time()
elapsed_time = end_time - start_time
minutes = math.floor(elapsed_time/60)
seconds = elapsed_time - 60*minutes
minutes = math.floor(elapsed_time/60)
seconds = elapsed_time - 60*minutes
print(f"{GOLD}Grid search completed in {minutes:.0f} min. {seconds:.0f} sec.")
print(f"Average time per parameter combination ", 
      f"{elapsed_time/total_combinations:.2f} seconds{RESET}")

# The tree with the lowest validation MISC, ignoring the overfitting rule
best_idx  = results['val_misc'].idxmin()
lbl = "Lowest Val MISC (Ignoring Overfitting)"
print_boundary(lbl, 47)
for parm in ['criterion', 'max_depth', 'min_samples_leaf',
             'min_samples_split']:
    parameter = str(results.loc[best_idx, parm])
    print(f"{GREEN}            {parm:.<20s}{GOLD}{parameter:>9s}{RESET}")
print_summary(1.0 - results.loc[best_idx, 'train_misc'],
              1.0 - results.loc[best_idx, 'val_misc'])

lbl = "Pass 1: Coarse Leaf Size Scan"
print_boundary(lbl, 47)
selected = select_tree(results)
print_leaf_scan(results, selected)

# Pass 2: fine scan of every leaf size between the coarse leaf sizes
# just below and just above the leaf selected in Pass 1.  If the smallest
# coarse leaf was selected, the scan starts at half of it (minimum 2).
leaf_sel   = results.loc[selected, 'min_samples_leaf']
k          = leaf_sizes.index(leaf_sel)
if k > 0:
    leaf_lo = leaf_sizes[k-1]
else:
    leaf_lo = max(2, leaf_sizes[0]//2)
leaf_hi    = leaf_sizes[min(k+1, len(leaf_sizes)-1)]
fine_sizes = [leaf for leaf in range(leaf_lo, leaf_hi+1)
              if leaf not in leaf_sizes]
if fine_sizes:
    lbl = "Pass 2: Fine Leaf Size Scan"
    print_boundary(lbl, 47)
    n_fine = len(criteria)*len(depths)*len(fine_sizes)
    print(f"{GOLD}Leaf sizes {RED}{leaf_lo}{GOLD} to {RED}{leaf_hi}{GOLD}:",
          f"{n_fine} new combinations,",
          f"{n_fine*n_folds} fits{RESET}")
    results  = pd.concat([results, run_grid(fine_sizes)], ignore_index=True)
    selected = select_tree(results)
    fine     = results[results['min_samples_leaf'].between(leaf_lo, leaf_hi)]
    print_leaf_scan(fine, selected)

lbl = "Optimum Hyperparameters"
print_boundary(lbl, 47)
best_params = results.loc[selected, ['criterion', 'max_depth',
                                     'min_samples_leaf',
                                     'min_samples_split']].to_dict()
for parm in best_params:
    parameter = str(best_params[parm])
    print(f"{GREEN}            {parm:.<20s}{GOLD}{parameter:>9s}{RESET}")

val_acc   = 1.0 - results.loc[selected, 'val_misc']
train_acc = 1.0 - results.loc[selected, 'train_misc']
"""
 =============================================== 
 *********** Optimum Hyperparameters *********** 
 =============================================== 
            criterion...........  entropy
            max_depth...........        5
            min_samples_leaf....       34
            min_samples_split...       68

 =============================================== 
 ******* Optimum Tree Performance Metrics ****** 
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ACCURACY............ 0.9560    0.9475    1.0090
 MISCLASSIFICATION... 0.0440    0.0525    1.1934
 ----------------------------------------------- 
Optimum tree (refit to all 1372 cases) has depth 5 and 14 leaves.
"""
lbl = "Optimum Tree Performance Metrics"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

# Refit the selected tree to all cases
best_dtree = DecisionTreeClassifier(**best_params, random_state=12345)
best_dtree = best_dtree.fit(X, y)
print(f"{GOLD}Optimum tree (refit to all {X.shape[0]} cases) has depth",
      f"{RED}{best_dtree.get_depth()}{GOLD} and",
      f"{RED}{best_dtree.get_n_leaves()}{GOLD} leaves.{RESET}")

# Feature importance for the optimum tree (top 10 when there are more)
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

# Step 5: Holdout Validation (70/30 split)
lbl = "Step 5: Holdout Validation (70/30 split)"
print_boundary(lbl)

# Same 70/30 split as Step 3 (stratified, random_state=12345), so the
# optimum tree and the kitchen sink tree are validated on the same cases.
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3,
                                  stratify=y, random_state=12345)
# Train an unfitted copy of the optimum tree on the training set.
# clone() is used so best_dtree keeps its fit to all cases for Step 7.
hold_out_tree = clone(best_dtree)
hold_out_tree = hold_out_tree.fit(Xt, yt)
print(f"{GOLD}")
tree_classifier.display_split_metrics(hold_out_tree, Xt, yt, Xv, yv)
print(f"{RESET}")

# Calculate final performance metrics
train_pred = hold_out_tree.predict(Xt)
val_pred   = hold_out_tree.predict(Xv)
train_acc  = accuracy_score(yt, train_pred)
val_acc    = accuracy_score(yv, val_pred)
"""
Model Metrics..........       Training     Validation
Observations...........            960            412
Features...............              4              4
Maximum Tree Depth.....              5              5
Minimum Leaf Size......             34             34
Minimum split Size.....             68             68
Mean Absolute Error....         0.0802         0.0913
Avg Squared Error......         0.0401         0.0536
Accuracy...............         0.9406         0.9078
Precision..................     0.9363         0.8756
Recall (Sensitivity).......     0.9297         0.9235
Specificity................     0.9493         0.8952
F1-score...................     0.9330         0.8989
Total Misclassifications...         57             38
MISC (Misclassification)...       5.9%           9.2%
     class 0...............       5.1%          10.5%
     class 1...............       7.0%           7.7%
	 
 =============================================== 
 **** Holdout Validation Performance Summary *** 
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ACCURACY............ 0.9406    0.9078    1.0362
 MISCLASSIFICATION... 0.0594    0.0922    1.5534
 ----------------------------------------------- 
 
 est K (lowest Test Avg. MISC) : 10-Fold

 =============================================== 
 * K-Fold Cross-Validation Performance Summary * 
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ACCURACY............ 0.9700    0.9657    1.0044
 MISCLASSIFICATION... 0.0300    0.0343    1.1408
 ----------------------------------------------- 
"""

lbl = "Holdout Validation Performance Summary"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

# MISC by class: the percent of each class's cases that are
# misclassified.  Small classes are where a tree is most likely to hold
# up in training but fail validation.  The kitchen sink tree from Step 3
# was fit to the same training cases and is validated on the same cases.
ks_train_pred = kitchen_sink_tree_cv.predict(Xt)
ks_val_pred   = kitchen_sink_tree_cv.predict(Xv)

print(f"\n{GOLD}{24*' '}OPTIMUM TREE{12*' '}KITCHEN SINK TREE{RESET}")
print(f"{GOLD} CLASS.......... N VAL   TRAIN     VAL   RATIO",
      f"    TRAIN     VAL   RATIO{RESET}")
for cls in hold_out_tree.classes_:
    t    = (train_pred[yt == cls]    != cls).mean()
    v    = (val_pred[yv == cls]      != cls).mean()
    ks_t = (ks_train_pred[yt == cls] != cls).mean()
    ks_v = (ks_val_pred[yv == cls]   != cls).mean()
    print_misc_row(cls, (yv == cls).sum(), t, v, ks_t, ks_v)
print_misc_row("ALL CLASSES", len(yv),
               (train_pred != yt).mean(), (val_pred != yv).mean(),
               (ks_train_pred != yt).mean(), (ks_val_pred != yv).mean())
print("")

# Step 6: K-Fold Cross Validation
lbl = "Step 6: K-Fold Cross-Validation"
print_boundary(lbl)

# Each k uses stratified folds (every fold keeps each class's proportion),
# shuffled with the same random_state as Step 4.  cross_validate fits an
# unfitted copy of best_dtree in each fold, so best_dtree is unchanged.
n             = X.shape[0]
best_val_misc = np.inf
for k in range(2, 11):  # Test 2-fold through 10-fold CV
    cv_k   = StratifiedKFold(n_splits=k, shuffle=True, random_state=12345)
    scores = cross_validate(best_dtree, X, y, scoring='accuracy',
                            cv=cv_k, return_train_score=True)
    # Calculate metrics
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

# best_dtree is the optimum tree from Step 4, fit to all cases.  At most
# the first 3 levels are drawn.  Each node shows the proportion of its
# cases in each class, in the order of classes below (0, then 1).
features = X.columns
classes  = ['Not Forged', 'Forged']
print(f"{GOLD}Optimum tree: depth {RED}{best_dtree.get_depth()}{GOLD},",
      f"{RED}{best_dtree.get_n_leaves()}{GOLD} leaves.",
      f"Displaying up to {RED}3{GOLD} levels.")
print(f"{GOLD}Class order in each node: {TEAL}{classes}{RESET}")

fig      = plt.figure(figsize=(26, 10))
myplot   = plot_tree(best_dtree, feature_names=features, class_names=classes,
                     fontsize=10, filled=True, impurity=False,
                     proportion=True, precision=2, max_depth=3)
plt.show()

dot_data = export_graphviz(best_dtree, feature_names=features,
                           class_names=classes, filled=True,
                           impurity=False, proportion=True,
                           precision=3, max_depth=3)
# Save the tree as a PNG image, then as a PDF opened in the viewer
graph    = graphviz.Source(dot_data)
with open("Forged_Check_Tree.png", "wb") as f:
    f.write(graph.pipe(format="png"))
graph.view("Forged_Check_Tree")
print(f"{GOLD}Tree saved to {TEAL}Forged_Check_Tree.png{GOLD} and",
      f"{TEAL}Forged_Check_Tree.pdf{RESET}")

lbl = "Bank Check Signature Forgery Analysis Complete"
print_boundary(lbl)
