# ANSI color codes - to print in color, the package colorama must be installed
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

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
print_boundary("Lasso Linear Regression")

import pandas as pd
import numpy  as np
from sklearn.linear_model                  import Lasso, LassoCV
from AdvancedAnalytics.Regression          import linreg
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT

data_map = {
	'Log_Cum_Production': 	[ DT.Interval , (8.0, 15.0) ],
	'Log_Proppant_LB': 	    [ DT.Interval , (6.0, 18.0) ],
	'Log_Carbonate': 	    [ DT.Interval , (-4.0, 4.0) ],
	'Log_Frac_Fluid_GL': 	[ DT.Interval , (7.0, 18.0) ],
	'Log_GrossPerforatedInterval': [ DT.Interval , (4.0, 9.0) ],
	'Log_LowerPerforation_xy': 	   [ DT.Interval , (8.0, 10.0) ],
	'Log_UpperPerforation_xy': 	   [ DT.Interval , (8.0, 10.0) ],
	'Log_TotalDepth': 	[ DT.Interval , (8.0, 10.0) ],
	'N_Stages': 	[ DT.Interval , (2, 14) ],
	'X_Well': 	[ DT.Interval , (-100.0, -95.0) ],
	'Y_Well': 	[ DT.Interval , (30.0, 35.0) ],
	'Operator': 	[ DT.Nominal , tuple(range(1, 29))],
	'County': 	[ DT.Nominal , tuple(range(1, 15))]
}
target = "Log_Cum_Production" # Identify Target Attribute in Data File
df  = pd.read_csv("../data/OilProduction.csv")
rie = ReplaceImputeEncode(data_map=data_map, nominal_encoding='one-hot', 
                          display=True)

encoded_df = rie.fit_transform(df)
print("Encoded Data has", encoded_df.shape[1], "Columns\n")

# Define target and features
target = "Log_Cum_Production"
y = encoded_df[target]
X = encoded_df.drop(target, axis=1)
feature_names = X.columns.tolist()

n_fold = 5
print_boundary(str(n_fold)+"-Fold Cross-Validation used to Select Alpha")
# Find optimal alpha using LassoCV
lasso_cv = LassoCV(cv=n_fold, random_state=42, max_iter=10000)
lasso_cv.fit(X, y)
optimal_alpha = lasso_cv.alpha_

print(f"\n{GOLD}Optimal alpha from CV: {optimal_alpha:.6f}{RESET}")

# Fit final model with optimal alpha
final_lasso = Lasso(alpha=optimal_alpha, max_iter=10000)
final_lasso.fit(X, y)

# Get selected features
selected_indices  = np.where(final_lasso.coef_ != 0)[0]
selected_features = [feature_names[i] for i in selected_indices]
selected_coef     = final_lasso.coef_[selected_indices]

linreg.display_metrics(final_lasso, X, y)
n_features = X.shape[1]
txt = "Lasso Selected "+str(len(selected_features)) +\
      " out of "+str(n_features)+" Attributes"
print_boundary(txt, b_width=45)
print(f"{GREEN}", 38*" ", f"COEF")
print(46*"=", f"{RESET}")
for feature, coef in zip(selected_features, selected_coef):
    print("{:.<38s}{:8.4f}".format(feature, coef))


