#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@purpose: Step-by-step fitting and validation of a decision tree for
          a nominal target using all attributes.  
@data:    CellphoneActivity_StratifiedRS.csv
@Date:    Oct 2026
@Author:  eJones
@Email:   ejones@tamu.edu
"""
# ANSI color codes - to print in color, the package colorama must be installed
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

# Ratios above max_ratio in summary tables indicate possible overfitting.
max_ratio = 1.2

# Import required packages
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

def ratio_color(ratio):
    # Ratios above max_ratio indicate overfitting and are shown in red.
    return RED if ratio > max_ratio else GREEN

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
    print(f"{GREEN} {'ACCURACY':.<20s}{GOLD}{train_acc:>7.4f}",
      f"  {val_acc:>7.4f}   {ratio_color(ratio_acc)}{ratio_acc:>7.4f}{RESET}")
    print(f"{GREEN} {'MISCLASSIFICATION':.<20s}{GOLD}{train_misc:>7.4f}",
      f"  {val_misc:>7.4f}   {ratio_color(ratio_misc)}{ratio_misc:>7.4f}{RESET}")
    print(f"{TEAL}","-"*47, f"{RESET}")

def misc_ratio(train_misc, val_misc):
    if train_misc > 0:
        return val_misc/train_misc
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
lbl = "Step 1: Reading Cellphone Activity Data"
print_boundary(lbl)
data_file = "CellphoneActivity_StratifiedRS.csv" #Reduced Dataset
df        = pd.read_csv("../../data/"+data_file)
print(f"{GOLD}Data loaded: {RED}{df.shape[0]}{GOLD} observations",
      f"{RED}{df.shape[1]} {GOLD}columns.{RESET}")

# Step 2: Create data map and apply ReplaceImputeEncode
lbl = "Step 2: ReplaceImputeEncode (RIE) Processing"
print_boundary(lbl)
# Data map from nominal_logistic.py.  The nominal target 'activity' is
# mapped as DT.String so RIE leaves it out of the encoded dataframe.  It
# is added back after RIE as a single column of class labels.
data_map = {
    'activity':[DT.String, ('sitting',  'sittingdown',
                             'standing', 'standingup', 'walking')],
    'user':    [DT.Nominal, ('wallace', 'debora', 'katia', 'jose_carlos')],
    'gender':  [DT.Binary, ('Man', 'Woman')],
    'age':     [DT.Interval,(20,     80)],
    'height':  [DT.Interval,(1.5,  1.75)],
    'weight':  [DT.Interval,(50,     85)],
    'BMI':     [DT.Interval,(20,     30)],
    'x1':      [DT.Interval,(-750, +750)],
    'y1':      [DT.Interval,(-750, +750)],
    'z1':      [DT.Interval,(-750, +750)],
    'x2':      [DT.Interval,(-750, +750)],
    'y2':      [DT.Interval,(-750, +750)],
    'z2':      [DT.Interval,(-750, +750)],
    'x3':      [DT.Interval,(-750, +750)],
    'y3':      [DT.Interval,(-750, +750)],
    'z3':      [DT.Interval,(-750, +750)],
    'x4':      [DT.Interval,(-750, +750)],
    'y4':      [DT.Interval,(-750, +750)],
    'z4':      [DT.Interval,(-750, +750)]
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
target = "activity"
print(f"{GOLD}")
# Apply ReplaceImputeEncode preprocessing
rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale=None, # No Interval Scaling
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          drop=False, # Keep all columns
                          display=True)

# Transform the data
encoded_df = rie.fit_transform(df)
encoded_df = pd.concat([df[target], encoded_df], axis=1) #Insert Target back
print(f"\n{RED}encoded_df    {RESET}:",
      f"{encoded_df.shape[0]} cases and",
      f"{encoded_df.shape[1]} columns,\n",
      "               including target.")

# Target class distribution
print(f"\n{GOLD}Target {RED}'{target}'{GOLD} class counts:")
counts = encoded_df[target].value_counts()
for cls, n_cls in counts.items():
    print(f"  {TEAL}{cls:.<15s}{GREEN}{n_cls:6d}  {n_cls/len(encoded_df):6.1%}")
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
# Same split as nominal_logistic.py (random_state=12345).
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

# Grid built from data dimensions (IntervalDecisionTree_Template approach).
# Shallower trees and larger leaves/splits reduce overfitting vs kitchen sink.
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
param_grid['min_samples_split'] = list(range(m, 3*m, math.ceil(2*m/6)))

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
minutes = math.floor(elapsed_time/60)
seconds = elapsed_time - 60*minutes
print(f"{GOLD}Grid search completed in {minutes:.0f} min. {seconds:.0f} sec.")
print(f"Average time per parameter combination ",
      f"{elapsed_time/total_combinations:.2f} seconds{RESET}")

lbl = "Optimum Decision Tree Hyperparameters"
print_boundary(lbl, 47)
max_param_len = max(len(str(val)) for val in grid_search.best_params_.values())
sze = max_param_len + 3
for parm in grid_search.best_params_:
    parameter = str(grid_search.best_params_[parm])
    print(f"{GREEN}            {parm:.<20s}{GOLD}{parameter:>{sze}s}{RESET}")

best_idx   = np.argmin(grid_search.cv_results_['rank_test_score'])
val_acc    = grid_search.cv_results_['mean_test_score'][best_idx]
train_acc  = grid_search.cv_results_['mean_train_score'][best_idx]

lbl = "Optimum Tree Performance Metrics"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)
"""
Grid Search: 180 parameter combinations
Using a 4-fold CV requires 720 total fits
Parallel processing with n_jobs=-1 for maximum speed

Starting grid search...
Grid search completed in 0 min. 4 sec.
Average time per parameter combination  0.02 seconds

 =============================================== 
 **** Optimum Decision Tree Hyperparameters **** 
 =============================================== 
            criterion...........   gini
            max_depth...........      7
            min_samples_leaf....     20
            min_samples_split...     20

 =============================================== 
 ******* Optimum Tree Performance Metrics ****** 
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ACCURACY............ 0.8882    0.8790    1.0105
 MISCLASSIFICATION... 0.1118    0.1210    1.0825
 ----------------------------------------------- 
"""
best_dtree = grid_search.best_estimator_
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
print(f"{RESET}")


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
"""
Model Metrics..........       Training     Validation
Observations...........           5796           2485
Features...............             21             21
Maximum Tree Depth.....              7              7
Minimum Leaf Size......             20             20
Minimum split Size.....             20             20
Avg Squared Error......         0.0328         0.0310
Root ASE...............         0.1812         0.1762
Accuracy...............         0.8873         0.8922
Precision..............         0.8577         0.8605
Recall (Sensitivity)...         0.8088         0.8117
F1-score...............         0.8280         0.8312
Total Misclassifications...        653            268
MISC (Misclassification)...      11.3%          10.8%
     class sitting.........       1.4%           0.5%
     class sittingdown.....      31.9%          32.8%
     class standing........       3.9%           4.1%
     class standingup......      42.1%          41.4%
     class walking.........      16.4%          15.4%
	 
 =============================================== 
 **** Holdout Validation Performance Summary *** 
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ACCURACY............ 0.8873    0.8922    0.9946
 MISCLASSIFICATION... 0.1127    0.1078    0.9572
 ----------------------------------------------- 
"""
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

"""
 Best K (lowest Test Avg. MISC) : 7-Fold

 =============================================== 
 * K-Fold Cross-Validation Performance Summary * 
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ACCURACY............ 0.8906    0.8820    1.0097
 MISCLASSIFICATION... 0.1094    0.1180    1.0782
 ----------------------------------------------- 

 ============================================================ 
 ********** Step 7: Display and Save Decision Tree ********** 
 ============================================================ 
Optimum tree: depth 7, 42 leaves. Displaying the first 3 levels.
Class order in each node: ['sitting', 'sittingdown', 'standing', 'standingup', 'walking']
"""
# Step 7: Display and Save Decision Tree
lbl = "Step 7: Display and Save Decision Tree"
print_boundary(lbl)

# best_dtree is the optimum tree from Step 4, fit to all cases.  It has
# many levels, so only the first 3 levels are drawn.  Each node shows the
# proportion of its cases in each class, in the order of classes below.
features = X.columns
classes  = list(best_dtree.classes_)
print(f"{GOLD}Optimum tree: depth {RED}{best_dtree.get_depth()}{GOLD},",
      f"{RED}{best_dtree.get_n_leaves()}{GOLD} leaves.",
      f"Displaying the first {RED}3{GOLD} levels.")
print(f"{GOLD}Class order in each node: {TEAL}{classes}{RESET}")

fig      = plt.figure(figsize=(28, 10))
myplot   = plot_tree(best_dtree, feature_names=features, class_names=classes,
                     fontsize=9, filled=True, impurity=False,
                     proportion=True, precision=2, max_depth=3)
plt.show()

dot_data = export_graphviz(best_dtree, feature_names=features,
                           class_names=classes, filled=True,
                           impurity=False, proportion=True,
                           precision=3, max_depth=3)
# Save the tree as a PNG image, then as a PDF opened in the viewer
graph    = graphviz.Source(dot_data)
with open("nominal_decisiontree_cellphone.png", "wb") as f:
    f.write(graph.pipe(format="png"))
graph.view("nominal_decisiontree_cellphone")
print(f"{GOLD}Tree saved to {TEAL}nominal_decisiontree_cellphone.png{GOLD} and",
      f"{TEAL}nominal_decisiontree_cellphone.pdf{RESET}")

lbl = "Cellphone Activity Decision Tree Analysis Complete"
print_boundary(lbl)
