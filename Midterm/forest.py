#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANLY 656 Midterm — Random Forest (Nominal Target)
Data: Data/sleep_rating.csv
Target: sleep_deficit (Nominal / String labels)
Data map drafted by Tools/Data_Map_Tool.py and reviewed for midterm.
Based on Templates/Random_Forest/Nominal_Forest/Nominal_RandomForest_Cellphone.py

Step 2 (current): read data, print data map, RIE encode.
Later steps: kitchen sink, feature selection (allowed), grid search, validation.
"""
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

import pandas as pd
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode

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

lbl = "Step 1: Reading Data"
print_boundary(lbl)

data_file = "sleep_rating.csv"
df        = pd.read_csv("Data/" + data_file)
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

# Forest template: no interval standardization; do not impute target.
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
      f"{encoded_df.shape[1]} columns",
      "(predictors; target kept as String labels from df).")

print(f"\n{GOLD}Target {RED}'{target}'{GOLD} class counts:")
counts = df[target].value_counts()
for cls, n_cls in counts.items():
    print(f"  {TEAL}{cls:.<15s}{GREEN}{n_cls:6d}  {n_cls / len(df):6.1%}")
print(f"{RESET}")

print(f"\n{GOLD}Predictor columns ({RED}{encoded_df.shape[1]}{GOLD}):")
print(list(encoded_df.columns))
print(f"{RESET}")

print(f"\n{GOLD}Preprocessing complete. Ready for next step.{RESET}")
lbl = "Step 2 Complete — Data Map and RIE Ready (forest)"
print_boundary(lbl)
