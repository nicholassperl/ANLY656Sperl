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
import numpy  as np
import matplotlib.pyplot as plt
from   matplotlib.lines import Line2D
from   scipy.stats      import norm
import statsmodels.api  as sm
from   statsmodels.stats.outliers_influence import OLSInfluence
from   sklearn.model_selection import train_test_split   #Hold-Out Sets
from   sklearn.linear_model    import Lasso, LassoCV
from   sklearn.linear_model    import LinearRegression   #Ordinary Regression
from   sklearn.model_selection import cross_validate     #k-fold validation
from   sklearn.metrics         import mean_squared_error #ASE
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
df  = pd.read_csv("../data/OilProduction.csv")
print(f"{GOLD}Data loaded: {RED}{df.shape[0]}{GOLD} observations", 
      f"{RED}{df.shape[1]} {GOLD}columns.{RESET}")

print(f"{GOLD}", 15*"=", "DATA MAP", 15*"=")
lk = len(max(data_map, key=len)) + 1
ignored = 0
for col, (dt_type, valid_values) in data_map.items():
    if dt_type.name == "ID" or dt_type.name=="Ignore":
        ignored += 1
    print(f"  {TEAL}{col:.<{lk}s} {GOLD}{dt_type.name:9s}{GREEN}{valid_values}")
print(f"{GOLD} === Data Map has{RED}", len(data_map)-ignored,
      f"{GOLD}attribute columns", 3*"=",f"{RESET}")

# Set target variable
target = "Log_Cum_Production" # Identify Target Attribute in Data File
print(f"{GOLD}")

rie    = ReplaceImputeEncode(data_map=data_map, nominal_encoding='one-hot', 
                             no_impute=[target], no_encode=[target], 
                             drop=False, display=True)

encoded_df = rie.fit_transform(df)
print("Encoded Data has", encoded_df.shape[1], "Columns\n")

# Define target and features
target = "Log_Cum_Production"
y = encoded_df[target]
X = encoded_df.drop(target, axis=1)
feature_names = X.columns.tolist()

print_boundary("Grid Search used to Optimize Lasso Regularization")
alpha_list = [0.0001, 0.0002, 0.0004, 0.0005, 0.0006, 0.001, 0.005, 0.01, 
              0.05, 0.1, 0.5, 1.0]
n_fold = 5
print_boundary(str(n_fold)+"-Fold Cross-Validation used to Select Alpha")
# Find optimal alpha using LassoCV
lasso_cv = LassoCV(cv=n_fold, alphas=alpha_list, random_state=12345,
                   max_iter=10000)
lasso_cv.fit(X, y)
optimal_alpha = lasso_cv.alpha_
print(f"\n{GOLD}Optimal alpha from CV: {optimal_alpha:.6f}")
print(f"Metrics are from Sklearn Lasso Fit")
# Fit final model with optimal alpha
final_lasso = Lasso(alpha=optimal_alpha, max_iter=10000)
final_lasso.fit(X, y)

# Get selected features
selected_indices  = np.where(np.abs(final_lasso.coef_)>1e-4)[0]
selected_features = [feature_names[i] for i in selected_indices]
selected_coef     = final_lasso.coef_[selected_indices]

linreg.display_metrics(final_lasso, X, y)
n_features = X.shape[1]
txt = "Lasso Selected "+str(len(selected_features)) +\
      " out of "+str(n_features)+" Attributes"
print_boundary(txt, b_width=45)
print(f"{GREEN}", 38*" ", "COEF")
print(f"{TEAL}",  46*"=", f"{RESET}")
for feature, coef in zip(selected_features, selected_coef):
    print(f"{TEAL}{feature:.<38s}{GREEN}{coef:8.4f}")

print_boundary("N-Fold Cross Validation", b_width=45)
n  = X.shape[0]
lr = LinearRegression()
for n_folds in range(2, 6):
    lr = LinearRegression()
    scores  = cross_validate(lr, X[selected_features], y, 
                             scoring="neg_mean_squared_error", 
                             cv=n_folds, return_train_score=True, )
    print_ase_ratio(scores, n_folds, n)

print(f"\n{GREEN}{len(selected_features)}{TEAL} out of",
      f"{GREEN}{encoded_df.shape[1]-1}",
      f"{TEAL}Features were selected by {GREEN}lasso")
print(f"{TEAL}Metrics are from Statsmodels GLM")
# Extract target and predictors from encoded_df
y = encoded_df[target]
X = encoded_df[selected_features]

# Add constant for intercept
X_sm = sm.add_constant(X)

# Fit regularized model with L1_wt=1.0 for pure Lasso
lasso_model   = sm.OLS(y, X_sm)
lasso_results = lasso_model.fit_regularized(alpha=optimal_alpha, L1_wt=1.0)
# Print estimated parameters
print(lasso_results.params)

refitted  = lasso_model.fit(params=lasso_results.params)
y_pred    = lasso_results.predict(X_sm)          #Predicted Values
influence = OLSInfluence(refitted)               #Statsmodels Influence Object
std_resid = influence.resid_studentized_internal #Studentized Residuals
cooks_d   = influence.cooks_distance[0]          #Cooks D
count_gt2 = sum(1 for resid in std_resid if abs(resid) > 2)
count_gt3 = sum(1 for resid in std_resid if abs(resid) > 3) 
count_gt6 = sum(1 for resid in std_resid if abs(resid) > 6)
print(f"Abs Std. Residuals >2 {count_gt2: 4d}")
print(f"Abs Std. Residuals >3 {count_gt3: 4d}")
if count_gt6 == 0:
    print(f"Abs Std. Residuals >6 {count_gt6: 4d}")
else:
    print(f"Abs Std. Residuals >6 {RED}{count_gt6: 4d}{TEAL}")

# Determine threshold based on sample size
threshold = 3 
if len(y) > 500:
    threshold = 6
# Create mask for outliers (points beyond threshold)
outlier_mask = np.abs(std_resid) > threshold
yellow_mask  = np.abs(std_resid) > 3

gold = '#D4AF37'
plt.style.use('dark_background')
    
y = encoded_df[target]
X = encoded_df[selected_features]

# Create a table of expected vs actual cases beyond different sigma thresholds
# Calculate total number of observations
n_observations = len(std_resid)

# Calculate expected and actual cases beyond different sigma thresholds
sigma_levels = [2, 3, 4, 5, 6]
table_data = []

for sigma in sigma_levels:
    # Calculate expected cases beyond sigma (two-tailed)
    # For standard normal distribution:
    # P(|Z| > sigma) = 2 * (1 - Φ(sigma)) where Φ is the CDF of standard normal
    expected_prob = 2 * (1 - norm.cdf(sigma))
    expected_cases = expected_prob * n_observations  # Keep as float for comparison
    expected_cases_rounded = round(expected_cases)  # Round for display

    # Calculate actual cases beyond sigma
    actual_cases = np.sum(np.abs(std_resid) > sigma)

    # Store in table
    table_data.append([sigma, expected_cases,
                       expected_cases_rounded, actual_cases])

# Print the table
print(f"\n{TEAL}Expected vs Observed Standardized Residuals")
print("=" * 43)
print(f"{GREEN}{'Sigma':^10}{'Expected Cases':>10}{'Observed Cases':>18}{TEAL}")
print("-" * 43, f"{GREEN}")

for row in table_data:
    sigma, expected_float, expected_rounded, actual = row

    # Format actual count in red if expected is very small (less than 1)
    if expected_float < 1 and actual > 0:
        # Add padding to ensure right alignment with ANSI codes
        spaces = 17 - len(str(actual))
        actual_str = " " * spaces + f"{RED}{actual}"
    else:
        # Add padding for normal values too
        spaces = 17 - len(str(actual))
        actual_str = " " * spaces + f"{GREEN}{actual}"

    # Don't use alignment specifier for actual_str since padding manually
    print(f"{GREEN}{sigma:^10.1f}{expected_rounded:>10d}{actual_str}{TEAL}")
print("=" * 43)

# Determine threshold based on sample size
threshold = 3
if len(y) > 500:
    threshold = 6
print(f"Using threshold of ±{threshold} for standardized residuals")

gold = '#D4AF37'
plt.style.use('dark_background')
# Create mask for outliers (points beyond threshold)
outlier_mask = np.abs(std_resid) > threshold
yellow_mask  = np.abs(std_resid) > 3
print(f"Found {np.sum(outlier_mask)} outliers beyond threshold")

print_boundary("HOLD-OUT VALIDATION")
X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                    test_size=0.3, random_state=12345)

# Fit Regression Model to Training Data
lr = LinearRegression()
lr = lr.fit(X_train, y_train)

# Display hold-out metrics using AdvancedAnalytics
print(f"{TEAL}\nTraining and Validation Metrics",
      f"from {GREEN}Lasso with OLS{GOLD}")
linreg.display_split_metrics(lr, X_train, y_train, X_val, y_val)

# Examine Possible Overfitting
train_predict = lr.predict(X_train)
val_predict   = lr.predict(X_val)
ASE_train     = mean_squared_error(y_train, train_predict)
ASE_val       = mean_squared_error(y_val, val_predict)
overfit_ratio = ASE_val / ASE_train
if overfit_ratio < 1.2:
    print(f"\n{TEAL}ASE ratio (validation/train) {GREEN}{overfit_ratio:.2f}")
# Check for overfitting
else:
    print(f"\n{TEAL}ASE ratio (validation/train) {RED}{overfit_ratio:.2f}")
    print(f"{RED}Warning: Potential Overfitting Detected{RESET}")

print_boundary("N-FOLD CROSS-VALIDATION")
n  = X.shape[0]
lr = LinearRegression()
for n_folds in range(2, 6):
    lr = LinearRegression()
    scores  = cross_validate(lr, X[selected_features], y, 
                             scoring="neg_mean_squared_error", 
                             cv=n_folds, return_train_score=True, )
    print_ase_ratio(scores, n_folds, n)

# Standardized Residuals vs Predicted Values
plt.figure(figsize=(12, 6))
# Plot regular points in cyan
plt.scatter(y_pred[~yellow_mask], std_resid[~yellow_mask], 
           alpha=0.9, color="cyan", edgecolors='k')
# Plot outlier points in gold or red
if np.any(yellow_mask):
    plt.scatter(y_pred[yellow_mask], std_resid[yellow_mask], 
               alpha=0.9, color=gold, edgecolors='k')
if np.any(outlier_mask):
    plt.scatter(y_pred[outlier_mask], std_resid[outlier_mask], 
               alpha=0.9, color='r', edgecolors='k')
plt.axhline(y= 0, color='r',  linestyle='-', linewidth=1.5)
plt.axhline(y= 3,  color=gold,linestyle='--',alpha=0.8, linewidth=2)
plt.axhline(y=-3, color=gold, linestyle='--',alpha=0.8, linewidth=2)
plt.axhline(y= 6, color='r',  linestyle='-', linewidth=1.5)
plt.axhline(y=-6, color='r',  linestyle='-', linewidth=1.5)
plt.xlabel("Predicted "+target, color=gold, 
                           fontweight="bold", fontsize=14)
plt.ylabel('Standardized Residuals', color=gold, 
                           fontweight="bold", fontsize=14)
plt.title('Standardized Residuals vs Predicted', 
          color=gold, fontweight="bold", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
# Create custom legend with explicit colors
legend_elements = [
    Line2D([0], [0], marker='o', color=gold, markerfacecolor=gold, 
           markersize=10, label='3 Sigma'),
    Line2D([0], [0], marker='o', color='r', markerfacecolor='r', 
           markersize=10, label='6 Sigma')
]
# Add legend
legend_properties = {'size': 14, 'weight': 'bold'}
plt.legend(handles=legend_elements, loc='lower center', framealpha=0.9, 
           prop=legend_properties)
plt.tight_layout()
plt.savefig('residuals_vs_predicted.png', dpi=300)
plt.show()

# Standardized Residuals vs Sequence Number
plt.figure(figsize=(12, 6))
plt.scatter(std_resid.index[~yellow_mask], std_resid[~yellow_mask], 
           alpha=0.9, color="cyan", edgecolors='k')
# Plot outlier points in gold or red
if np.any(yellow_mask):
    plt.scatter(std_resid.index[yellow_mask], std_resid[yellow_mask], 
               alpha=0.9, color=gold, edgecolors='k')
if np.any(outlier_mask):
    plt.scatter(std_resid.index[outlier_mask], std_resid[outlier_mask], 
               alpha=0.9, color='r', edgecolors='k')

plt.axhline(y= 0, color='r', linestyle='-', linewidth=1.5)
plt.axhline(y= 3,  color=gold, linestyle='--', alpha=0.8, linewidth=2)
plt.axhline(y=-3, color=gold, linestyle='--',  alpha=0.8, linewidth=2)
plt.axhline(y= 6, color='r', linestyle='-', linewidth=1.5)
plt.axhline(y=-6, color='r', linestyle='-',linewidth=1.5)
plt.xlabel('Observation Number', color=gold, 
                           fontweight="bold", fontsize=14)
plt.ylabel('Standardized Residuals', color=gold, 
                           fontweight="bold", fontsize=14)
plt.title('Time Series of Standardized Residuals', color=gold, 
                           fontweight="bold", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
# Create custom legend with explicit colors
legend_elements = [
    Line2D([0], [0], marker='o', color=gold, markerfacecolor=gold, 
           markersize=10, label='3 Sigma'),
    Line2D([0], [0], marker='o', color='r', markerfacecolor='r', 
           markersize=10, label='6 Sigma')
]
# Add legend
legend_properties = {'size': 14, 'weight': 'bold'}
plt.legend(handles=legend_elements, loc='lower center', framealpha=0.9, 
           prop=legend_properties)
plt.tight_layout()
plt.savefig('residuals_vs_sequence.png', dpi=300)
plt.show()

# Standardized Residuals vs Cook's Distance
plt.figure(figsize=(12, 6))
# Plot regular points in cyan
plt.scatter(cooks_d[~yellow_mask], std_resid[~yellow_mask], 
           alpha=0.9, color="cyan", edgecolors='k')
# Plot outlier points in gold or red
if np.any(yellow_mask):
    plt.scatter(cooks_d[yellow_mask], std_resid[yellow_mask], 
               alpha=0.9, color=gold, edgecolors='k')
if np.any(outlier_mask):
    plt.scatter(cooks_d[outlier_mask], std_resid[outlier_mask], 
               alpha=0.9, color='r', edgecolors='k')
plt.axhline(y= 0, color='r', linestyle='-', linewidth=1.5)
plt.axhline(y= 3, color=gold,linestyle='--', alpha=0.8, linewidth=2)
plt.axhline(y=-3, color=gold,linestyle='--', alpha=0.8, linewidth=2)
plt.axhline(y= 6, color='r', linestyle='-', linewidth=1.5)
plt.axhline(y=-6, color='r', linestyle='-',linewidth=1.5)
plt.axvline(x=4/len(X), color='r', linestyle='--', label="Cook's D Threshold")
plt.ylabel('Standardized Residuals', color=gold, 
                           fontweight="bold", fontsize=14)
plt.xlabel("Cook's Distance", color=gold, 
                           fontweight="bold", fontsize=14)
plt.title("Standardized Residuals vs Cook's Distance", color=gold, 
                           fontweight="bold", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
# Create custom legend with explicit colors
legend_elements = [
    Line2D([0], [0], marker='o', color=gold, markerfacecolor=gold, 
           markersize=10, label='3 Sigma'),
    Line2D([0], [0], marker='o', color='r', markerfacecolor='r', 
           markersize=10, label='6 Sigma')
]
# Add legend
legend_properties = {'size': 14, 'weight': 'bold'}
plt.legend(handles=legend_elements, loc='lower center', framealpha=0.9, 
           prop=legend_properties)
plt.tight_layout()
plt.savefig('residuals_vs_cooks_d.png', dpi=300)
plt.show()
plt.close()

