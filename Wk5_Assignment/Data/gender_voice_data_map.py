# 3,168 rows x 21 columns   (pandas 3.0.6, max_n=10, max_s=30)
# source: Data/gender_voice_data.csv
# Drafted with Tools/Data_Map_Tool.py, then reviewed against
# Gender_Voice_Data_Dictionary.pdf: data types confirmed, bounds replaced
# with the dictionary's bounds. No missing values in the file.

from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode

data_map = {
    'meanfreq':  [DT.Interval, (0, 1.0)],
    'sd':        [DT.Interval, (0, 1.0)],
    'median':    [DT.Interval, (0, 1.0)],
    'Q25':       [DT.Interval, (0, 1.0)],
    'Q75':       [DT.Interval, (0, 1.0)],
    'IQR':       [DT.Interval, (0, 1.0)],
    'skew':      [DT.Interval, (0, 35.0)],
    'kurt':      [DT.Interval, (0, 1310.0)],
    'spent':     [DT.Interval, (0, 1.0)],
    'sfm':       [DT.Interval, (0, 1.0)],
    'mode':      [DT.Interval, (0, 1.0)],
    'centroid':  [DT.Interval, (0, 1.0)],
    'meanfun':   [DT.Interval, (0, 1.0)],
    'minfun':    [DT.Interval, (0, 1.0)],
    'maxfun':    [DT.Interval, (0, 1.0)],
    'meandom':   [DT.Interval, (0, 3.0)],
    'mindom':    [DT.Interval, (0, 1.0)],
    'maxdom':    [DT.Interval, (0, 22.0)],
    'dfrange':   [DT.Interval, (0, 22.0)],
    'modindx':   [DT.Interval, (0, 1.0)],
    'gender':    [DT.Binary, ('female', 'male')],
}
