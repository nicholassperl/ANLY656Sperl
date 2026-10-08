# AI Development Productivity Analysis - decision Tree
# Step-by-step optimization and validation of decision Tree model

# ANSI color codes - to print in color, the package colorama must be installed
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

# Import required packages
import warnings, time, math
import pandas as pd
import numpy  as np
import graphviz
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Tree                import tree_regressor
from matplotlib              import pyplot as plt
from sklearn.tree            import DecisionTreeRegressor, plot_tree
from sklearn.tree            import export_graphviz
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics         import mean_squared_error 
from sklearn.model_selection import GridSearchCV
from pydotplus.graphviz      import graph_from_dot_data

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

def print_ase_ratio(scores, n):
    n_folds    = len(scores["train_score"])
    train_ase  = -scores["train_score"]
    train_sase = 2.0*(train_ase).std()
    val_ase    = -scores["test_score"]
    val_sase   = 2.0*(val_ase).std()
    ratio_ase  = np.zeros(n_folds)
    for i in range(0, n_folds):
        if train_ase[i]>0:
            ratio_ase[i] = val_ase[i]  / train_ase[i]
        elif val_ase[i]>0:
            ratio_ase[i] = np.inf
        else:
            ratio_ase[i] = 1.0
    try:
        s_ratio = 2.0*ratio_ase.std()
    except:
        s_ratio = np.nan
    train_ase  = train_ase.mean()
    val_ase    = val_ase.mean()
    ratio      = val_ase/train_ase if train_ase>0 else np.inf
    print(f"{TEAL}\n")
    print(f" ====== {n_folds:.0f}-Fold Cross Validation =======")
    print(f"  Train Avg. ASE..... {train_ase:.4f} +/-{train_sase:.4f}")
    print(f"  Test  Avg. ASE..... {val_ase:.4f} +/-{val_sase:.4f}")
    if s_ratio == np.nan or s_ratio == np.inf:
        print(f"  Mean ASE Ratio..... {ratio:.4f}")
    else:
        print(f"  Mean ASE Ratio..... {ratio:.4f} +/-{s_ratio:.4f}")
    print(" ", 39*"=", f"{RESET}")
    n_v = n*(1.0/n_folds)
    n_t = n - n_v
    print(f"Equivalent to {n_folds:.0f} splits each with "+
          f"{n_t:.0f}/{n_v:.0f} Cases")
		   
def print_summary(train_ase, val_ase):
    if train_ase>0:
        ratio_ase  = val_ase/train_ase
    else:
        ratio_ase  = np.inf
    #print accuracy and aselassification summary for train/validation
    print(f"{GREEN}{'TRAIN':>28s} {'VALIDATION':>11s} {'RATIO':>7s}")
    if ratio_ase < 1.2:
        color = GREEN
    else:
        color = RED
    print(f"{GREEN} {'ASE':.<20s}{GOLD}{train_ase:>7.4f}",
      f"  {val_ase:>7.4f}   {color}{ratio_ase:>7.4f}{RESET}")

# Step 1: Read the data
lbl = "Step 1: Reading Data"
print_boundary(lbl)
df = pd.read_excel("../../data/diamonds_below_5.xlsx")
print(f"{GOLD}Data loaded: {df.shape[0]} observations and {df.shape[1]} columns.{RESET}")

# Step 2: Create data map and apply ReplaceImputeEncode
lbl = "Step 2: ReplaceImputeEncode (RIE) Processing"
print_boundary(lbl)
# Create data map based on data dictionary
data_map = {
    'obs':      [DT.Ignore, ("")],
    'carat':    [DT.Interval, (0, 1.0)],
    'cut':      [DT.Nominal, ('Fair', 'Good', 'Ideal', 'Premium', 'Very Good')],
    'color':    [DT.Nominal, ('D', 'E', 'F', 'G', 'H', 'I', 'J')],
    'clarity':  [DT.Nominal, ('I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 
							  'VVS1', 'VVS2')],
    'depth':    [DT.Interval, (43.0, 79.0)],
    'table':    [DT.Interval, (43.0, 95.0)],
    'price':    [DT.Interval, (326.0, 9800.0)],
    'x':        [DT.Interval, (0, 11.0)],
    'y':        [DT.Interval, (0, 59.0)],
    'z':        [DT.Interval, (0, 32.0)],
}
print(f"{GOLD}")
print(15*"=", "DATA MAP", 15*"=")
lk = len(max(data_map, key=len)) + 1
ignored = 0
for col, (dt_type, valid_values) in data_map.items():
    if dt_type.name == "ID" or dt_type.name=="Ignore":
        ignored += 1
    print(f"  {TEAL}{col:.<{lk}s} {GOLD}{dt_type.name:9s}{GREEN}{valid_values}")
print(f"{GOLD} === Data Map has{RED}", len(data_map)-ignored,
      f"{GOLD}attribute columns", 3*"=",f"{RESET}")

# Set target variable
target = "price"
print(f"{GOLD}")
# Apply ReplaceImputeEncode preprocessing
rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale=None,  # No standardization of interval features
                          no_impute=[target],    # Do not impute target variable
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          drop=False,             # Keep all columns
                          display=True)

# Transform the data
encoded_df = rie.fit_transform(df)
print(f"\n{RED}encoded_df    {RESET}:",
      f"{encoded_df.shape[0]} cases and",
      f"{encoded_df.shape[1]} columns,\n",
      "               including targets.")
print(f"{RESET}")

# Step 3: Kitchen Sink decision Tree Evaluation
lbl = "Step 3: Kitchen Sink Decision Tree (Default Parameters)"
print_boundary(lbl)

y = encoded_df[target]
X = encoded_df.drop(target, axis=1)

# First show the overfitting on full data
print(f"{GOLD}Fitting kitchen sink decision Tree using entire dataset")
kitchen_sink_Tree =DecisionTreeRegressor(random_state=42)  # Default parameters
kitchen_sink_Tree = kitchen_sink_Tree.fit(X, y)
tree_regressor.display_metrics(kitchen_sink_Tree, X, y)

# Now evaluate with proper holdout validation
lbl = "70/30 Holdout Validation of Kitchen Sink Tree"
print_boundary(lbl)

# Split the data 70/30
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, 
								  random_state=31415)

# Fit kitchen sink Tree on training data only
kitchen_sink_Tree_cv =DecisionTreeRegressor(random_state=42)  # Defaults
kitchen_sink_Tree_cv = kitchen_sink_Tree_cv.fit(X_train, y_train)

# Evaluate using AdvancedAnalytics display_split_metrics
print(f"{GOLD}")
tree_regressor.display_split_metrics(kitchen_sink_Tree_cv,
                                      X_train, y_train, X_val, y_val)

# Calculate and display accuracy ratio and misclassification ratio
train_pred = kitchen_sink_Tree_cv.predict(X_train)
val_pred   = kitchen_sink_Tree_cv.predict(X_val)
train_ase  = mean_squared_error(y_train, train_pred)
val_ase    = mean_squared_error(y_val, val_pred)

lbl = "Kitchen Sink 70/30 Validation"
print_boundary(lbl, 47)
print_summary(train_ase, val_ase) #train/validation summary

# Show feature importance from the properly trained model
print(f"{GOLD}\nTop 15 Feature Importance (from training data):")
tree_regressor.display_importance(kitchen_sink_Tree_cv, X.columns, 
                                     top=15, plot=True)

# Step 4: decision Tree Hyperparameter Optimization
lbl = "Step 4: decision Tree Hyperparameter Optimization"
print_boundary(lbl)
param_grid = {
    'criterion':         ['squared_error'],
    'max_depth':         [5, 6, 7, 8, 9],
    'min_samples_split': [40, 43, 46, 49, 52, 55, 58],
    'min_samples_leaf':  [40, 42, 44, 46, 48]
}
k  = X.shape[1]
m  = math.floor(0.005*X.shape[0]*0.7)
m  = min(m, 20)
ks = min(k, m, 5)
param_grid['max_depth']         = list(range(ks, min(ks+3, 20), 1))
#param_grid['max_depth'].append(None)
param_grid['min_samples_leaf']  = list(range(m, m+10, 2))
param_grid['min_samples_split'] = list(range(m, m+16, 2))

lbl = "Hyperparameters"
print_boundary(lbl, 47)
for parm in param_grid:
    print(f"{GREEN} {parm:.<20s}{GOLD}{param_grid[parm][0:]}{RESET}")
"""
 =============================================== 
 *************** Hyperparameters *************** 
 =============================================== 
 criterion...........['squared_error']
 max_depth...........[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, None]
 min_samples_split...[40, 43, 46, 49, 52, 55, 58]
 min_samples_leaf....[40, 42, 44, 46, 48]

Grid Search: 385 parameter combinations
Using a 4-fold CV requires 1540 total fits
Parallel processing with n_jobs=-1 for maximum speed

Starting grid search...
Grid search completed in 28.9 seconds
Average time per parameter combination: 0.08 seconds

 =============================================== 
 **** Optimum Decision Tree Hyperparameters **** 
 =============================================== 
            criterion...........   squared_error
            max_depth...........            None
            min_samples_leaf....              40
            min_samples_split...              40

 =============================================== 
 ******* Optimum Tree Performance Metrics ****** 
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ASE.................431820.2623   3624609.5405    8.3938
 """
# Calculate and display grid search information
total_combinations = 1
for param_list in param_grid.values():
    total_combinations *= len(param_list)
total_fits = total_combinations * 4  # cv=4

njobs = -1
print(f"\n{GOLD}Grid Search: {total_combinations} parameter combinations") 
print(f"Using a 4-fold CV requires {total_fits} total fits")
print(f"Parallel processing with {GOLD}n_jobs={njobs}", 
      f"{RED}for maximum speed")

# Start timing and run grid search
start_time = time.time()
print(f"\n{GOLD}Starting grid search...{RESET}")
rf          =DecisionTreeRegressor(random_state=31415)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=4, scoring='neg_mean_squared_error',
                           return_train_score=True, n_jobs=njobs).fit(X,y)
end_time = time.time()
elapsed_time = end_time - start_time
minutes = math.floor(elapsed_time/60)
seconds = elapsed_time - 60*minutes
print(f"{GOLD}Grid search completed in {minutes:.0f} min. {seconds:.0f} sec.")
print(f"Average time per parameter combination ", 
      f"{elapsed_time/total_combinations:.2f} seconds{RESET}")

lbl = "Optimum Decision Tree Hyperparameters"
print_boundary(lbl, 47)
# Find the longest parameter value for consistent formatting
max_param_len = max(len(str(val)) for val in grid_search.best_params_.values())
sze = max_param_len + 3
for parm in grid_search.best_params_:
    parameter = str(grid_search.best_params_[parm])
    print(f"{GREEN}            {parm:.<20s}{GOLD}{parameter:>{sze}s}{RESET}")

best_idx   = np.argmin(grid_search.cv_results_['rank_test_score'])
val_ase    = -grid_search.cv_results_['mean_test_score'][best_idx]
train_ase  = -grid_search.cv_results_['mean_train_score'][best_idx]

lbl = "Optimum Tree Performance Metrics"
print_boundary(lbl, 47)
print_summary(train_ase, val_ase)

best_tree = grid_search.best_estimator_

# Step 5: Holdout Validation (70/30 split)
lbl = "Step 5: Holdout Validation (70/30 split)"
print_boundary(lbl)

Xt, Xv, yt, yv  = train_test_split(X, y, test_size=0.3, random_state=42)
# Train optimized model on final training set
hold_out_Tree = best_tree
hold_out_Tree = hold_out_Tree.fit(Xt, yt)
print(f"{GOLD}")
tree_regressor.display_split_metrics(hold_out_Tree, Xt, yt, Xv, yv)
print(f"{RESET}")
# Calculate final performance metrics
train_pred = hold_out_Tree.predict(Xt)
val_pred   = hold_out_Tree.predict(Xv)
train_ase  = mean_squared_error(yt, train_pred)
val_ase    = mean_squared_error(yv, val_pred)
ratio_ase  = train_ase / val_ase if val_ase > 0 else np.inf

lbl = "Holdout Validation Performance Summary"
print_boundary(lbl, 47)
print_summary(train_ase, val_ase)

# Show feature importance from the optimized model
print(f"{GOLD}\nTop 15 Feature Importance (optimized model):")
tree_regressor.display_importance(hold_out_Tree, X.columns, 
                                     top=15, plot=True)
"""
Model Metrics.........  Training     Validation
Observations...........    37758          16182
Split Criterion........squared_error  squared_error
Max Depth..............     None           None
Minimum Split Size.....       40             40
Minimum Leaf  Size.....       40             40
R-Squared..............   0.9733         0.9704
Mean Absolute Error.... 339.4516       356.6202
Median Absolute Error.. 134.4856       144.1561
Avg Squared Error......429124.8142    461843.1340
Square Root ASE........ 655.0762       679.5904


 =============================================== 
 **** Holdout Validation Performance Summary *** 
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ASE.................429124.8142   461843.1340    1.0762
 """
# Step 6: K-Fold Cross Validation
lbl = "Step 6: K-Fold Cross-Validation"
print_boundary(lbl)

warnings.filterwarnings('ignore', category=RuntimeWarning)
n            = X.shape[0]
best_val_ase = np.inf
for k in range(2, 11):  # Test 2-fold through 10-fold CV
    scores  = cross_validate(best_tree, X, y, 
                             scoring='neg_mean_squared_error',
                             cv=k, return_train_score=True)
    # Calculate metrics
    train_ase = -scores["train_score"].mean()
    val_ase   = -scores["test_score"].mean()

    print_ase_ratio(scores, n)
    if val_ase < best_val_ase:
        best_k = k
        best_train_ase = train_ase
        best_val_ase   = val_ase

print(f"\n{GOLD} Best K :",
      f"{RED}{best_k}-Fold{GOLD}")
lbl = "K-Fold Cross-Validation Performance Summary"
print_boundary(lbl, 47)
print_summary(best_train_ase, best_val_ase)
"""
Best K : 10-Fold

 =============================================== 
 * K-Fold Cross-Validation Performance Summary * 
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ASE.................410504.9784   1599960.9210    3.8975
 """
lbl = "Step 7: Display and Save Decision Tree"
print_boundary(lbl)

features = X.columns
plt.style.use({'text.color': "black", 'figure.facecolor': "antiquewhite"})
fig      = plt.figure(figsize=(16, 8))
myplot   = plot_tree(best_tree, feature_names=features,
                     rounded=True, fontsize=12, filled=True, impurity=True, 
                     proportion=True, precision=2, max_depth=3)
plt.show()

features = X.columns
dot_data = export_graphviz(best_tree, feature_names=features, 
                           filled=True, 
                           impurity=True, proportion=True, 
                           precision=2, max_depth=3)
graph    = graph_from_dot_data(dot_data)
graph.write_png("decision_tree.png")
graph_pdf = graphviz.Source(dot_data)
graph_pdf.view("tree")

lbl = "AI Development Productivity Decision Tree Analysis Complete"
print_boundary(lbl)
