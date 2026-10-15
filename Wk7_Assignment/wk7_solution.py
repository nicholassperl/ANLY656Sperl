# -*- coding: utf-8 -*-
"""
Wk7 Assignment: Cellphone Activity Random Forest
Data: Data/CellphoneActivity.csv (165,633 readings)
Target: activity (nominal String: sitting, sittingdown, standing,
        standingup, walking)
Predictors: 1 binary (gender), 1 nominal (user), 16 interval
            (age, height, weight, BMI, x1-z4)
Based on Templates/Random_Forest/Nominal_Forest/Nominal_RandomForest_Cellphone.py
Steps complete: (1) data map, (2) ReplaceImputeEncode.
"""
# ANSI color codes for output formatting
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

RANDOM_STATE        = 31415
FEATURE_THRESHOLD   = 0.99  # Proportion of features to select for optimization
OPTIMIZATION_SAMPLE = 0.1   # Stratified sample fraction for hyperparameter search

import pandas as pd
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode


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
print(f"{GOLD}Step 2 complete. Stopping before kitchen-sink forest.{RESET}")
