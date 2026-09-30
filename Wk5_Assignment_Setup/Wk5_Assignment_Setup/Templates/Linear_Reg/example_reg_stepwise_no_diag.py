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
print_boundary("Stepwise Linear Regression Without Diagnostic Plots")

def print_ase_ratio(scores, n_folds, n):
    train_ase  = -scores["train_score"].mean()
    train_sase = 2.0*scores["train_score"].std()
    val_ase    = -scores["test_score"].mean()
    val_sase   = 2.0*scores["test_score"].std()
    ratios     = scores["test_score"]/scores["train_score"]
    ratio      = ratios.mean()
    s_ratio    = 2.0*ratios.std()

    print(f"\n{TEAL}====== {n_folds:.0f}-Fold Cross Validation ======")
    print(f"  {GREEN}Train Avg. ASE..... {train_ase:.4f} +/-{train_sase:.4f}")
    print(f"  Test  Avg. ASE..... {val_ase:.4f} +/-{val_sase:.4f}")
    print(f"  Mean ASE Ratio..... {ratio:.4f} +/-{s_ratio:.4f}")
    print(f"{TEAL}", 38*"=")
    n_v = n*(1.0/n_folds)
    n_t = n - n_v
    print(f"{GREEN}Equivalent to {n_folds:.0f} splits each with "+
          f"{n_t:.0f}/{n_v:.0f} Cases{RESET}")
    
import pandas as pd
from sklearn.model_selection import train_test_split   #Hold-Out Sets
from sklearn.linear_model    import LinearRegression   #Ordinary Regression
from sklearn.model_selection import cross_validate     #k-fold validation
from sklearn.metrics         import mean_squared_error #ASE
from AdvancedAnalytics.Regression          import linreg, stepwise
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
	'Operator': 	[ DT.Nominal , tuple(range(1, 29)) ],
	'County': 	[ DT.Nominal , tuple(range(1, 15))]
}
target = "Log_Cum_Production" # Identify Target Attribute in Data File
rie = ReplaceImputeEncode(data_map=data_map, nominal_encoding='one-hot', 
                          no_impute=[target], drop=False, display=True)
df  = pd.read_csv("../data/OilProduction.csv")
encoded_df = rie.fit_transform(df)
print("Encoded Data has", encoded_df.shape[1], "Columns\n")

# Define target and features
target = "Log_Cum_Production"
# Hyperparameter Optimization to Select Features
print_boundary("STEPWISE LINEAR REGRESSION")

selected = stepwise(encoded_df, target, reg="linear", method="stepwise",
                    crit_in=0.05, crit_out=0.05, verbose=True).fit_transform()
y = encoded_df[target]
X = encoded_df[selected]
print(f"{GREEN}", len(selected), "out of",  encoded_df.shape[1]-1,
      "Features were selected by Stepwise.", f"{RESET}\n")

print_boundary("70/30 HOLD-OUT VALIDATION")
X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                    test_size=0.3, random_state=12345)
# Fit Regression Model to Training Data
lr = LinearRegression()
lr = lr.fit(X_train, y_train)

# Display hold-out metrics using AdvancedAnalytics
print("\nTraining and Validation Metrics:")
linreg.display_split_metrics(lr, X_train, y_train, X_val, y_val)

# Examine Possible Overfitting
train_predict = lr.predict(X_train)
val_predict   = lr.predict(X_val)
ASE_train     = mean_squared_error(y_train, train_predict)
ASE_val       = mean_squared_error(y_val, val_predict)
overfit_ratio = ASE_val / ASE_train
print(f"\n{GREEN}ASE ratio (validation/train): {ASE_val/ASE_train:.2f}{RESET}")
# Check for overfitting
if overfit_ratio > 1.2:
    print("Warning: Potential Overfitting Detected")

print_boundary("N-FOLD CROSS-VALIDATION")

n  = X.shape[0]
lr = LinearRegression()
for n_folds in range(2, 6):
    lr = LinearRegression()
    scores  = cross_validate(lr, X[selected], y, 
                             scoring="neg_mean_squared_error", 
                             cv=n_folds, return_train_score=True, )
    print_ase_ratio(scores, n_folds, n)






