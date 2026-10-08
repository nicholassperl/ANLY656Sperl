# 8,000 rows x 32 columns   (pandas 3.0.6, max_n=10, max_s=30)
# source: /workspace/Wk6_Assignment/Data/CreditDefaultData.csv
#
# column          dtype        nulls        unique
#   Customer       int64        -            8000
#   Default        int64        -            2
#   card_class     int64        -            3
#   Gender         float64      832 (10.4%)  2
#   Education      float64      1182 (14.78%) 7
#   Marital_Status int64        -            4
#   Age            float64      1592 (19.9%) 53
#   Credit_Limit   int64        -            66
#   Jun_Status     int64        -            11
#   May_Status     int64        -            10
#   Apr_Status     int64        -            11
#   Mar_Status     int64        -            11
#   Feb_Status     int64        -            10
#   Jan_Status     int64        -            10
#   Jun_Bill       float64      -            6796
#   May_Bill       float64      -            6747
#   Apr_Bill       float64      -            6614
#   Mar_Bill       float64      -            6519
#   Feb_Bill       float64      -            6405
#   Jan_Bill       float64      -            6247
#   Jun_Payment    float64      -            2790
#   May_Payment    float64      -            2796
#   Apr_Payment    float64      -            2677
#   Mar_Payment    float64      -            2537
#   Feb_Payment    float64      -            2505
#   Jan_Payment    float64      -            2510
#   Jun_PayPercent float64      -            2160
#   May_PayPercent float64      -            2128
#   Apr_PayPercent float64      -            1952
#   Mar_PayPercent float64      -            1868
#   Feb_PayPercent float64      -            1826
#   Jan_PayPercent float64      -            1852
#
# first rows:
#   Customer | Default | card_class | Gender | Education | Marital_Status | Age | Credit_Limit | Jun_Status | May_Status | Apr_Status | Mar_Status | Feb_Status | Jan_Status | Jun_Bill | May_Bill | Apr_Bill | Mar_Bill | Feb_Bill | Jan_Bill | Jun_Payment | May_Payment | Apr_Payment | Mar_Payment | Feb_Payment | Jan_Payment | Jun_PayPercent | May_PayPercent | Apr_PayPercent | Mar_PayPercent | Feb_PayPercent | Jan_PayPercent
#   1230 | 0 | 1 | 1.0 | 3.0 | 2 | 46.0 | 1700 | 0 | 0 | 0 | 0 | 0 | 0 | 1714.48 | 962.8 | 709.51 | 944.74 | 547.71 | 572.71 | 47.47 | 68.78 | 34.41 | 34.2 | 34.2 | 34.2 | 0.0277 | 0.0714 | 0.0485 | 0.0362 | 0.0624 | 0.0597
#   26676 | 0 | 1 | 1.0 | 2.0 | 2 | 24.0 | 1700 | 0 | 0 | 0 | 0 | 0 | 0 | 1730.62 | 1713.97 | 1643.75 | 1497.52 | 315.53 | 329.28 | 85.5 | 85.67 | 86.56 | 18.13 | 68.4 | 34.2 | 0.0494 | 0.05 | 0.0527 | 0.0121 | 0.2168 | 0.1039
#   21278 | 0 | 1 | 1.0 | 2.0 | 1 | 50.0 | 700 | 0 | 0 | 0 | 0 | 0 | 2 | 188.75 | 226.4 | 258.42 | 293.09 | 342.68 | 332.25 | 41.04 | 41.04 | 44.46 | 54.72 | 0.0 | 20.52 | 0.2174 | 0.1813 | 0.172 | 0.1867 | 0.0 | 0.0618
#   22051 | 0 | 2 |  |  | 2 |  | 4100 | -1 | -1 | -1 | -1 | -1 | -1 | 11.22 | 14.12 | 102.84 | 65.9 | 62.86 | 84.54 | 14.26 | 103.01 | 65.9 | 62.86 | 84.54 | 80.06 | 1.0 | 1.0 | 0.6408 | 0.9539 | 1.0 | 0.947
#   19021 | 0 | 2 | 2.0 | 1.0 | 2 | 28.0 | 8200 | 0 | 0 | 0 | 0 | 0 | 0 | 8423.8 | 8558.62 | 8555.78 | 8532.73 | 6578.99 | 6723.21 | 359.1 | 338.58 | 373.02 | 239.4 | 273.6 | 257.7 | 0.0426 | 0.0396 | 0.0436 | 0.0281 | 0.0416 | 0.0383
#   6137 | 0 | 3 | 1.0 | 2.0 | 2 | 30.0 | 10900 | 0 | 0 | 0 | 0 | 0 | 0 | 1014.65 | 1035.64 | 1051.21 | 919.13 | 900.08 | 885.1 | 46.27 | 41.86 | 25.27 | 25.38 | 25.68 | 24.04 | 0.0456 | 0.0404 | 0.024 | 0.0276 | 0.0285 | 0.0272
#   10780 | 0 | 2 | 2.0 | 3.0 | 1 | 48.0 | 4400 | 0 | -1 | -1 | -1 | -1 | -1 | 464.09 | 26.85 | 52.46 | 13.58 | 761.36 | 8.58 | 26.85 | 52.46 | 13.58 | 761.36 | 8.58 | 47.2 | 0.0579 | 1.0 | 0.2589 | 1.0 | 0.0113 | 1.0
#   8219 | 0 | 1 | 2.0 | 2.0 | 2 |  | 1000 | 0 | 0 | 0 | 0 | 0 | 0 | 1094.61 | 2054.7 | 919.36 | 942.89 | 960.17 | 994.02 | 60.02 | 51.3 | 38.82 | 37.62 | 49.76 | 26.51 | 0.0548 | 0.025 | 0.0422 | 0.0399 | 0.0518 | 0.0267

from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode

data_map = {
    'Customer':        [DT.Interval, (0.99, 29998.01)],
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
