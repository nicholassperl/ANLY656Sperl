#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stratified random sample. Rename the input and output filenames
and adapt this script for your ML analysis.
"""
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from sklearn.model_selection import train_test_split
import pandas as pd

data_map = {
    'user':      [DT.Nominal, ('debora', 'jose_carlos', 'katia', 'wallace')],
    'gender':    [DT.Binary, ('Man', 'Woman')],
    'age':       [DT.Nominal, (28, 31, 46, 75)],
    'height':    [DT.Nominal, (1.58, 1.62, 1.67, 1.71)],
    'weight':    [DT.Nominal, (55, 67, 75, 83)],
    'BMI':       [DT.Nominal, (22.0, 24.0, 28.4, 28.6)],
    'x1':        [DT.Interval, (-306.01, 509.01)],
    'y1':        [DT.Interval, (-271.01, 533.01)],
    'z1':        [DT.Interval, (-603.01, 411.01)],
    'x2':        [DT.Interval, (-494.01, 473.01)],
    'y2':        [DT.Interval, (-517.01, 295.01)],
    'z2':        [DT.Interval, (-617.01, 122.01)],
    'x3':        [DT.Interval, (-499.01, 507.01)],
    'y3':        [DT.Interval, (-506.01, 517.01)],
    'z3':        [DT.Interval, (-613.01, 410.01)],
    'x4':        [DT.Interval, (-702.01, -12.99)],
    'y4':        [DT.Interval, (-526.01, 86.01)],
    'z4':        [DT.Interval, (-537.01, -42.99)],
    'activity':  [DT.Nominal, ('sitting', 'sittingdown', 'standing', 'standingup', 'walking')],
}

size      = 0.05  # Proportion of all data sampled
file_in   = "input_file.csv"
file_out  = "output_file.csv"
#--------- Create Random Stratified Sample -----------------------------------
# Nominal and Binary features from the data map
categorical_list = ['user', 'gender', 'age', 'height', 'weight', 'BMI', 'activity']

df  = pd.read_csv(file_in, index_col=None)
if size >= 1:
    Xt = df
else:
    Xt, Xv = train_test_split(df, train_size=size,
                              stratify=df[categorical_list],
                              random_state=12345)
Xt.to_csv(file_out, index=False)

