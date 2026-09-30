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
print_boundary("Stepwise Linear Regression with Diagnostic Plots")

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
from sklearn.linear_model               import Lasso, LassoCV, LinearRegression
from sklearn.metrics                    import mean_squared_error
from sklearn.model_selection            import cross_validate, train_test_split
from AdvancedAnalytics.Regression          import linreg, stepwise
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT

data_map = {
    'fixed_acidity':    [DT.Interval, (5.0, 10.0)],
    'volatile_acidity': [DT.Interval, (0.0, 1.0)],
    'citric_acid':      [DT.Interval, (0.0, 1.0)],
    'residual_sugar':   [DT.Interval, (0.0, 100.0)],
    'chlorides':        [DT.Interval, (0.0, 1.0)],
    'free_sulfur_dioxide':  [DT.Interval, (0.0, 100.0)],
    'total_sulfur_dioxide': [DT.Interval, (0.0, 200.0)],
    'density':   [DT.Interval, (0.9, 1.0)],
    'pH':        [DT.Interval, (2.0, 5.0)],
    'sulphates': [DT.Interval, (0.0, 1.0)],
    'alcohol':   [DT.Interval, (8.0, 14.0)],
    'quality':   [DT.Interval, (0.0, 10.0)]  # Target - do not impute
}

df = pd.read_csv("../data/WhiteWineData.csv")
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
target = "quality"  # Identify Target Attribute in Data File
print(f"{GOLD}")

rie = ReplaceImputeEncode(data_map=data_map, nominal_encoding='one-hot',
                          no_impute=[target], no_encode=[target],
                          drop=False, display=True)

encoded_df = rie.fit_transform(df)

# Define target and features
y = encoded_df[target]
X = encoded_df.drop(target, axis=1)
feature_names = X.columns.tolist()

alpha_list = [0.0001, 0.0002, 0.0004, 0.0005, 0.0006, 0.001, 0.005, 0.01,
              0.05, 0.1, 0.5, 1.0]
# Find optimal alpha using LassoCV
n_fold = 5
print_boundary(str(n_fold)+"-Fold Cross-Validation used to Select Alpha")
# Find optimal alpha using LassoCV
lasso_cv = LassoCV(cv=n_fold, alphas=alpha_list, random_state=12345, 
                   max_iter=10000)
lasso_cv.fit(X, y)
optimal_alpha = lasso_cv.alpha_
print(f"\n{GOLD}Optimal alpha from CV: {optimal_alpha:.6f}")
print(f"Metrics are from Sklearn Lasso Fit{RESET}")

# Fit final model with optimal alpha
final_lasso = Lasso(alpha=optimal_alpha, max_iter=10000)
final_lasso.fit(X, y)

# Get selected features
selected_indices = np.where(np.abs(final_lasso.coef_) > 1e-4)[0]
selected_features = [feature_names[i] for i in selected_indices]
selected_coef = final_lasso.coef_[selected_indices]

print(f"{GOLD}")
linreg.display_metrics(final_lasso, X, y)
n_features = X.shape[1]
txt = "Lasso Selected "+str(len(selected_features)) +\
      " out of "+str(n_features)+" Attributes"
print_boundary(txt, b_width=45)
print(f"{GREEN}", 39*" ", "COEF")
print(f"{TEAL}",  46*"=", f"{RESET}")

for feature, coef in zip(selected_features, selected_coef):
    print(f"{TEAL}{feature:.<38s}{GREEN}{coef:8.4f}{RESET}")

# Stepwise Feature Selection
print_boundary("Stepwise Feature Selection")
print(f"{TEAL}")
stepwise_selected = stepwise(encoded_df, target, reg="linear", 
                             method="stepwise", crit_in=0.05, crit_out=0.05, 
                             verbose=True).fit_transform()
print(f"{GREEN}", len(stepwise_selected), "out of",  encoded_df.shape[1]-1,
      "Features were selected by Stepwise.", f"{RESET}\n")

# Extract target and predictors from encoded_df using stepwise selected features
y_step = encoded_df[target]
X_step = encoded_df[stepwise_selected]

# Add constant for intercept
X_step_sm = sm.add_constant(X_step)
# Fit the model
stepwise_model = sm.OLS(y_step, X_step_sm).fit()

# Calculate ASE for stepwise model BEFORE displaying summary
y_pred = stepwise_model.predict(X_step_sm)  # Predicted Values
ASE_stepwise = mean_squared_error(y_step, y_pred)
print_boundary(f"STEPWISE MODEL ASE: {ASE_stepwise:.4f}", b_width=77)

# Display summary
print(f"{GOLD}")
print(stepwise_model.summary())
influence = OLSInfluence(stepwise_model)               #Statsmodels Influence Object
std_resid = influence.resid_studentized_internal #Studentized Residuals
cooks_d   = influence.cooks_distance[0]          #Cooks D
count_gt2 = sum(1 for resid in std_resid if abs(resid) > 2)
count_gt3 = sum(1 for resid in std_resid if abs(resid) > 3)
count_gt6 = sum(1 for resid in std_resid if abs(resid) > 6)
print(f"Abs Std. Residuals >2 {count_gt2: 4d}")
print(f"Abs Std. Residuals >3 {count_gt3: 4d}")
print(f"Abs Std. Residuals >6 {count_gt6: 4d}")

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
        spaces = 15 - len(str(actual))
        actual_str = " " * spaces + f"{RED}{actual}"
    else:
        # Add padding for normal values too
        spaces = 15 - len(str(actual))
        actual_str = " " * spaces + f"{GREEN}{actual}"

    # Don't use alignment specifier for actual_str since padding manually
    print(f"{GREEN}{sigma:^10.1f}{expected_rounded:>10d}{actual_str}{TEAL}")
print("=" * 43)

# Determine threshold based on sample size
threshold = 3
if len(y_step) > 500:
    threshold = 6
print(f"Using threshold of ±{threshold} for standardized residuals")

gold = '#D4AF37'
plt.style.use('dark_background')
# Create mask for outliers (points beyond threshold)
outlier_mask = np.abs(std_resid) > threshold
yellow_mask  = np.abs(std_resid) > 3
print(f"Found {np.sum(outlier_mask)} outliers beyond threshold")

# K-fold Cross Validation for Stepwise Model
print_boundary("N-Fold Cross Validation", b_width=45)
n  = X_step.shape[0]
lr = LinearRegression()
for n_folds in range(2, 6):
    lr = LinearRegression()
    scores  = cross_validate(lr, X_step, y_step,
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
plt.axvline(x=4/len(X_step), color='r', linestyle='--', label="Cook's D Threshold")
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
