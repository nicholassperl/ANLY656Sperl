# -*- coding: utf-8 -*-
"""
Wk6 Assignment: Credit Default Decision Tree
Data map drafted by Tools/Data_Map_Tool.py from Data/CreditDefaultData.csv
Target: Default (Binary 0/1)
Reviewed: Customer=Ignore (ID), Education=Nominal (kept).
"""
# ANSI color codes
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
                          interval_scale=None,  # No Interval Scaling
                          no_impute=[target],   # Do not impute target
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          drop=False,           # Keep all columns
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
