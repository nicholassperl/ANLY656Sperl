# -*- coding: utf-8 -*-
"""
Wk7 Assignment: Cellphone Activity Random Forest
Data: Data/CellphoneActivity.csv (165,633 readings)
Target: activity (nominal String: sitting, sittingdown, standing,
        standingup, walking)
Predictors: 1 binary (gender), 1 nominal (user), 16 interval
            (age, height, weight, BMI, x1-z4)
Based on Templates/Random_Forest/Nominal_Forest/Nominal_RandomForest_Cellphone.py
Steps complete: (1) data map, (2) ReplaceImputeEncode, (3) kitchen-sink RF.
"""
# ANSI color codes for output formatting
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

RANDOM_STATE        = 31415
FEATURE_THRESHOLD   = 0.99  # Proportion of features to select for optimization
OPTIMIZATION_SAMPLE = 0.1   # Stratified sample fraction for hyperparameter search

import pandas as pd
import numpy as np
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Forest import forest_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def print_boundary(lbl, b_width=60, boundary=True):
    """Print formatted section boundary with label"""
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
    if ratio_acc < 1.2:
        color = GREEN
    else:
        color = RED
    print(f"{GREEN} {'ACCURACY':.<20s}{GOLD}{train_acc:>7.4f}",
          f"  {val_acc:>7.4f}   {color}{ratio_acc:>7.4f}{RESET}")
    if ratio_misc < 1.2:
        color = GREEN
    else:
        color = RED
    print(f"{GREEN} {'MISCLASSIFICATION':.<20s}{GOLD}{train_misc:>7.4f}",
          f"  {val_misc:>7.4f}   {color}{ratio_misc:>7.4f}{RESET}")
    print(f"{TEAL}", "-" * 47, f"{RESET}")


# Data map: attribute types and allowed values for ReplaceImputeEncode.
# Target activity uses DT.String so RIE leaves it as labels (not one-hot).
# Interval bounds follow the Nominal Random Forest cellphone template
# (wide enough for observed sensor ranges; demographics cover the 4 users).
data_map = {
    'activity': [DT.String, ('sitting', 'sittingdown',
                             'standing', 'standingup', 'walking')],
    'user':     [DT.Nominal, ('wallace', 'debora', 'katia', 'jose_carlos')],
    'gender':   [DT.Binary, ('Man', 'Woman')],
    'age':      [DT.Interval, (20, 80)],
    'height':   [DT.Interval, (1.5, 1.75)],
    'weight':   [DT.Interval, (50, 85)],
    'BMI':      [DT.Interval, (20, 30)],
    'x1':       [DT.Interval, (-750, +750)],
    'y1':       [DT.Interval, (-750, +750)],
    'z1':       [DT.Interval, (-750, +750)],
    'x2':       [DT.Interval, (-750, +750)],
    'y2':       [DT.Interval, (-750, +750)],
    'z2':       [DT.Interval, (-750, +750)],
    'x3':       [DT.Interval, (-750, +750)],
    'y3':       [DT.Interval, (-750, +750)],
    'z3':       [DT.Interval, (-750, +750)],
    'x4':       [DT.Interval, (-750, +750)],
    'y4':       [DT.Interval, (-750, +750)],
    'z4':       [DT.Interval, (-750, +750)],
}

# Step 1: Read the data and display the data map (no RIE yet)
lbl = "Step 1: Reading Data and Data Map"
print_boundary(lbl)
df = pd.read_csv("Data/CellphoneActivity.csv")

print(f"{GOLD}Data loaded: {df.shape[0]} ",
      f"observations and {df.shape[1]} columns.{RESET}")
print(f"\n{GOLD}First 5 rows of the data:{RESET}")
print(df.head())

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
print(f"{GOLD}Target: activity (DT.String / nominal labels).{RESET}")

# Step 2: Apply ReplaceImputeEncode (RIE)
lbl = "Step 2: ReplaceImputeEncode (RIE) Processing"
print_boundary(lbl)

target = "activity"
print(f"{GOLD}Target variable: {target}")
if data_map[target][0] == DT.String:
    target_ = None
else:
    target_ = [target]

rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale=None,  # No standardization
                          no_impute=target_,    # Don't impute target
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          drop=False,           # Keep all encoded columns
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
# X = encoded_df.drop(target, axis=1)

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
val_pred = kitchen_sink_forest_cv.predict(X_val)
train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)

lbl = "Kitchen Sink 70/30 Validation"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

print(f"{GOLD}\nTop 10 Feature Importance (from training data):")
forest_classifier.display_importance(kitchen_sink_forest_cv, X.columns,
                                     top=10, plot=True)
print(f"{RESET}")
print(f"{GOLD}Step 3 complete. Stopping before feature selection / tuning.{RESET}")
