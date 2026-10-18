#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANLY 656 Midterm — Multinomial / One-vs-Rest Logistic Regression
Data: Data/sleep_rating.csv
Target: sleep_deficit (Nominal / String labels)
Data map drafted by Tools/Data_Map_Tool.py and reviewed for midterm.
Based on Templates/Logistic_Reg/nominal_logistic.py

Step 2 (current): read data, print data map, RIE encode.
Later steps: model fit, validation, interpretation.
"""
# ANSI color codes - to print in color, the package colorama must be installed
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

import pandas as pd
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode

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

""" ======================================================================= """
lbl = "Step 1: Reading Data and Preparing Data Map"
print_boundary(lbl)

data_file = "sleep_rating.csv"
df        = pd.read_csv("Data/" + data_file)
print(f"{GOLD}Data loaded: {RED}{df.shape[0]}{GOLD} observations",
      f"{RED}{df.shape[1]} {GOLD}columns.{RESET}")

# Target is nominal. After RIE (DT.String / no_encode), it is added back
# as a single column of class labels.
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

print(f"{GOLD}", 15 * "=", "DATA MAP", 15 * "=")
lk = len(max(data_map, key=len)) + 1
ignored = 0
for col, (dt_type, valid_values) in data_map.items():
    if dt_type.name == "ID" or dt_type.name == "Ignore":
        ignored += 1
    print(f"  {TEAL}{col:.<{lk}s} {GOLD}{dt_type.name:9s}{GREEN}{valid_values}")
print(f"{GOLD} === Data Map has{RED}", len(data_map) - ignored,
      f"{GOLD}attribute columns", 3 * "=", f"{RESET}")

# Attributes constant within each level of a nominal add nothing beyond it.
nominal = [col for col, (dt_type, _) in data_map.items()
           if dt_type.name == "Nominal"]
for nom in nominal:
    others = [col for col, (dt_type, _) in data_map.items()
              if col != nom and dt_type.name in ("Interval", "Binary",
                                                 "Nominal")]
    constant = [col for col in others
                if df.groupby(nom)[col].nunique().max() == 1]
    if constant:
        print(f"{RED}NOTE: {', '.join(constant)} take one value per level",
              f"of '{nom}'")
        print(f"      ({df[nom].nunique()} levels). They add nothing",
              f"beyond '{nom}', and their")
        print(f"      coefficients (and '{nom}') should not be interpreted.",
              f"{RESET}")

lbl = "Step 2: ReplaceImputeEncode (RIE) Processing"
print_boundary(lbl)

target = "sleep_deficit"
print(f"{GOLD}Target variable: {RED}{target}{RESET}")
print(f"{GOLD}")
# Logistic template: standardize intervals; keep all one-hot columns.
rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale="std",
                          no_encode=[target],
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          display=True)

encoded_df = rie.fit_transform(df)
encoded_df = pd.concat([df[target], encoded_df], axis=1)
print(f"{RESET}")
print(f"{RED}encoded_df{GOLD} created from {TEAL}{data_file}{GOLD} containing",
      f"{TEAL}{encoded_df.shape[0]} {GOLD}cases & {TEAL}{encoded_df.shape[1]}",
      f" {GOLD}columns, including the target {RED}'{target}'{RESET}")

print(f"\n{GOLD}Target {RED}'{target}'{GOLD} class counts:")
counts = encoded_df[target].value_counts()
for cls, n_cls in counts.items():
    print(f"  {TEAL}{cls:.<15s}{GREEN}{n_cls:6d}  {n_cls / len(encoded_df):6.1%}")
print(f"{RESET}")

print(f"\n{GOLD}Predictor columns ({RED}{encoded_df.shape[1] - 1}{GOLD}):")
print(list(encoded_df.drop(columns=[target]).columns))
print(f"{RESET}")

lbl = "Step 2 Complete — Data Map and RIE Ready (log_reg)"
print_boundary(lbl)
