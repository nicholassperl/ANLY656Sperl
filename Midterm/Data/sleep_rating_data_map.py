# Drafted by Tools/Data_Map_Tool.py from Data/sleep_rating.csv
# Reviewed for ANLY 656 Midterm (nominal target sleep_deficit).
# No ID/Ignore columns (per exam). Target typed as DT.String so RIE
# leaves class labels out of the encoded frame (matches class templates).
# Question pending: morning_alarm_snoozes as Nominal (0-7) vs Interval.
from AdvancedAnalytics.ReplaceImputeEncode import DT

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
    'morning_alarm_snoozes':     [DT.Nominal, (0, 1, 2, 3, 4, 5, 6, 7)],
    'sleep_deficit':             [DT.String, ('0-none', 'mild', 'moderate', 'severe')],
    'occupation':                [DT.Nominal, (1, 2, 3, 4, 5)],
    'favorite_bedtime_app':      [DT.Nominal, (1, 2, 3, 4, 5, 6)],
    'clock_type':                [DT.Nominal, (1, 2, 3)],
}
