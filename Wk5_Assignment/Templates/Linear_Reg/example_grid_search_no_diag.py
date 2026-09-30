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
print_boundary("Grid Optimization in Linear Regression")

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
import numpy as np
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import cross_validate     #k-fold validation
from itertools               import combinations
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
	'Operator': 	[ DT.Ignore , ()], #Minor effect in stepwise
	'County': 	[ DT.Nominal , (1, 6, 9, 11, 13, 14)] #counties from stepwise
}
drop_counties  = [2, 3, 4, 5, 7, 8, 10, 12, 15]
target = "Log_Cum_Production" # Identify Target Attribute in Data File
df  = pd.read_csv("../data/OilProduction.csv")
df  = df[~df["County"].isin(drop_counties)]

rie = ReplaceImputeEncode(data_map=data_map, nominal_encoding='one-hot', 
                          display=True)

encoded_df = rie.fit_transform(df)
print("Encoded Data has", encoded_df.shape[1], "Columns\n")

# Define target and features
target   = "Log_Cum_Production"
y        = encoded_df[target]
features = [col for col in encoded_df.columns if col != target]

# Function to evaluate feature combination
def evaluate_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    n = len(y)
    k = X.shape[1] + 2 # Number of features + 2 for the intercept and sigma
    
    # Calculate metrics
    ase = np.sum(residuals**2) / n  # Sum of squared residuals
    bic = n * np.log(ase) + k * np.log(n) + n * (1.0+np.log(2*np.pi))
    
    return {"ase": ase, "bic": bic, "model": model}
    
# Store results
results = []

# Try combinations up to max_features
max_features = min(17, len(features))  # Limit for computational feasibility
print(f"\n{GOLD}Grid Search using Maximum Attributes of {max_features} {RESET}")
for k in range(1, max_features + 1):
    if k > 1:
        print(f"Evaluating combinations with {k} features...")
    else:
        print(f"Evaluating combinations with {k} feature...")
    for combo in combinations(features, k):
        X = encoded_df[list(combo)]
        result = evaluate_model(X, y)
        results.append({
            "features": combo,
            "n_features": k,
            "ase": result["ase"],
            "bic": result["bic"],
            "model": result["model"]
        })

# Sort by ASE and BIC
results_ase = sorted(results, key=lambda x: x["ase"])
results_bic = sorted(results, key=lambda x: x["bic"])

# Display top 3 for each criterion

print_boundary("Grid Optimization using ASE")
for i, result in enumerate(results_ase[:3]):
    print(f"\n{i+1}. Features: {result['features']}")
    print(f"ASE: {result['ase']:.4f}, BIC: {result['bic']:.4f}")

# Fit final models
best_ase_features = results_ase[0]["features"]
best_bic_features = results_bic[0]["features"]

X_ase = encoded_df[list(best_ase_features)]
X_bic = encoded_df[list(best_bic_features)]

lr_ase = LinearRegression().fit(X_ase, y)
lr_bic = LinearRegression().fit(X_bic, y)
    
# Display results
print_boundary("BEST ASE MODEL")
linreg.display_coef(lr_ase, X_ase, y)
linreg.display_metrics(lr_ase, X_ase, y)

print_boundary("N-fold Cross-Validation for Best ASE Model")
n = X_ase.shape[0]
for n_folds in range(2, 6):
    scores  = cross_validate(lr_ase, X_ase, y, 
                             scoring="neg_mean_squared_error", 
                             cv=n_folds, return_train_score=True, )
    print_ase_ratio(scores, n_folds, n)

print_boundary("Grid Optimization using BIC")
for i, result in enumerate(results_bic[:3]):
    print(f"\n{i+1}. Features: {result['features']}")
    print(f"ASE: {result['ase']:.4f}, BIC: {result['bic']:.4f}")
    
    
print_boundary("BEST ASE MODEL")
linreg.display_coef(lr_bic, X_bic, y)
linreg.display_metrics(lr_bic, X_bic, y)

print_boundary("N-fold Cross-Validation for Best BIC Model")
n = X_bic.shape[0]
for n_folds in range(2, 6):
    scores  = cross_validate(lr_bic, X_bic, y, 
                             scoring="neg_mean_squared_error", 
                             cv=n_folds, return_train_score=True, )
    print_ase_ratio(scores, n_folds, n)
