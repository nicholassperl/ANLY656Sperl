"""
@purpose: Step-by-step fitting and validation of a random forest for
          a nominal target using all attributes.  
@data:    CellphoneActivity_StratifiedRS.csv
@Date:    Oct 2026
@Author:  eJones
@Email:   ejones@tamu.edu
"""
# ANSI color codes for output formatting
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

RANDOM_STATE        = 31415
FEATURE_THRESHOLD   = 0.99  #Proportion of features to select for optimization
OPTIMIZATION_SAMPLE = 1.00  #Proportion of samples  to select for optimization

# Import required packages
import pandas as pd
import numpy  as np
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Forest              import forest_classifier

from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics         import accuracy_score
import warnings, time, math

def print_boundary(lbl, b_width=60, boundary=True):
    """Print formatted section boundary with label"""
    print("")
    margin = b_width - len(lbl) - 2
    lmargin = int(margin/2)
    rmargin = lmargin
    if lmargin+rmargin < margin:
        lmargin += 1
    if boundary:
        print(f"{TEAL}", "="*b_width, f"{RESET}")
    print(f"{GREEN}", lmargin*"*", lbl, rmargin*"*", f"{RESET}")
    if boundary:
        print(f"{TEAL}", "="*b_width, f"{RESET}")

def print_acc_ratio(scores, n):
    n_folds     = len(scores["train_score"])
    train_misc  = (1.0 - scores["train_score"])
    train_smisc = 2.0*(1.0 - scores["train_score"]).std()
    val_misc    = (1.0 - scores["test_score"])
    val_smisc   = 2.0*(1.0 - scores["test_score"]).std()
    ratio_misc  = np.zeros(n_folds)
    for i in range(0, n_folds):
        if train_misc[i]>0:
            ratio_misc[i] = val_misc[i]  / train_misc[i]
        elif val_misc[i]>0:
            ratio_misc[i] = np.inf
        else:
            ratio_misc[i] = 1.0
    try:
        s_ratio = 2.0*ratio_misc.std()
    except:
        s_ratio = np.nan
    train_misc  = train_misc.mean()
    val_misc    = val_misc.mean()
    ratio       = val_misc/train_misc if train_misc>0 else np.inf
    print(f"{TEAL}\n")
    print(f" ====== {n_folds:.0f}-Fold Cross Validation =======")
    print(f"  Train Avg. MISC..... {train_misc:.4f} +/-{train_smisc:.4f}")
    print(f"  Test  Avg. MISC..... {val_misc:.4f} +/-{val_smisc:.4f}")
    if s_ratio == np.nan or s_ratio == np.inf:
        print(f"  Mean Misc Ratio..... {ratio:.4f}")
    else:
        print(f"  Mean Misc Ratio..... {ratio:.4f} +/-{s_ratio:.4f}")
    print(" ", 39*"=", f"{RESET}")
    n_v = n*(1.0/n_folds)
    n_t = n - n_v
    print(f"Equivalent to {n_folds:.0f} splits each with "+
          f"{n_t:.0f}/{n_v:.0f} Cases")

def print_summary(train_acc, val_acc):
    train_misc = 1.0 - train_acc
    val_misc   = 1.0 - val_acc
    ratio_acc  = train_acc / val_acc    if val_acc>0    else np.inf
    if train_misc>0:
        ratio_misc = val_misc  / train_misc
    elif val_misc>0:
        ratio_misc = np.inf
    else:
        ratio_misc = 1.0
    #print accuracy and misclassification summary for train/validation
    print(f"{GREEN}{'TRAIN':>28s} {'VALIDATION':>11s} {'RATIO':>7s}")
    if ratio_acc < 1.2:
        color = GREEN
    else:
        color = RED
    print(f"{GREEN} {'ACCURACY':.<20s}{GOLD}{train_acc:>7.4f}",
      f"  {val_acc:>7.4f}   {color}{ratio_acc:>7.4f}{RESET}")

    if ratio_misc < 1.2:
        color = GREEN
    else:
        color = RED
    print(f"{GREEN} {'MISCLASSIFICATION':.<20s}{GOLD}{train_misc:>7.4f}",
      f"  {val_misc:>7.4f}   {color}{ratio_misc:>7.4f}{RESET}")
    print(f"{TEAL}","-"*47, f"{RESET}")

def tree_selection(X, y, threshold=0.9):
    if threshold >= 1.0:
        return X
    dt_selector = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X, y)
    # Get feature importances
    feature_importance = dt_selector.feature_importances_
    feature_name       = X.columns

    # Create a DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'feature': feature_name,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()

    print(f"\n{GREEN}            Feature Importance Analysis{GOLD}")
    print(f"{'='*51}")
    print(f"{'Feature':<27} {'Importance':<12} {'Cumulative':<12}")
    print(f"{'-'*51}")
    lne = False
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:.<29} {row['importance']:<12.4f} ", 
              f"{row['cumulative_importance']:<12.4f}")
        if row['cumulative_importance'] > threshold and not lne:
            print(f"{RED}{25*'- '}{GOLD}"); lne = True
    print(f"{'='*51}")
    # Select features that account for at least threshold of total importance
    cumulative_threshold = threshold
    selected_mask = importance_df['cumulative_importance'] >= cumulative_threshold

    if selected_mask.any():
        # Find the first feature that makes cumulative importance >= 90%
        first_idx = selected_mask.idxmax()
        selected_features = importance_df.loc[:first_idx, 'feature'].tolist()
        threshold_reached = importance_df.loc[first_idx, 'cumulative_importance']
    else:
        # If no combination reaches 90%, take all features
        selected_features = importance_df['feature'].tolist()
        threshold_reached = importance_df['cumulative_importance'].max()
        print(f"\n{RED}No feature combination reaches 90% importance. ", 
              f"Using all features (cumulative: {threshold_reached:.1%})")

    print(f"\n{GREEN}Selected {len(selected_features)} features accounting for ", 
          f"{threshold_reached:.0%} of importance")
    # Reduce feature set to selected features
    X_selected = X[selected_features]
    print(f"{GREEN}Feature selection complete - reduction: ", 
          f"{RED}{X.shape[1]} -→ {X_selected.shape[1]} {GREEN}features{RESET}")
        
    return X_selected
""" ======================================================================= """
lbl = "Step 1: Reading Data"
print_boundary(lbl)
df = pd.read_csv("../../data/CellphoneActivity_StratifiedRS.csv")

print(f"{GOLD}Data loaded: {df.shape[0]} ", 
      f"observations and {df.shape[1]} columns.{RESET}")

# Display first few rows to verify data structure
print(f"\n{GOLD}First 5 rows of the data:{RESET}")
print(df.head())

# Step 2: Create Data Map and Apply ReplaceImputeEncode
lbl = "Step 2: Data Map and ReplaceImputeEncode (RIE) Processing"
print_boundary(lbl)

data_map = {
    'activity':[DT.String, ('sitting',  'sittingdown',
                             'standing', 'standingup', 'walking')],
    'user':    [DT.Nominal, ('wallace', 'debora', 'katia', 'jose_carlos')],
    'gender':  [DT.Binary, ('Man', 'Woman')],
    'age':     [DT.Interval,(20,     80)],
    'height':  [DT.Interval,(1.5,  1.75)],
    'weight':  [DT.Interval,(50,     85)],
    'BMI':     [DT.Interval,(20,     30)],
    'x1':      [DT.Interval,(-750, +750)],
    'y1':      [DT.Interval,(-750, +750)],
    'z1':      [DT.Interval,(-750, +750)],
    'x2':      [DT.Interval,(-750, +750)],
    'y2':      [DT.Interval,(-750, +750)],
    'z2':      [DT.Interval,(-750, +750)],
    'x3':      [DT.Interval,(-750, +750)],
    'y3':      [DT.Interval,(-750, +750)],
    'z3':      [DT.Interval,(-750, +750)],
    'x4':      [DT.Interval,(-750, +750)],
    'y4':      [DT.Interval,(-750, +750)],
    'z4':      [DT.Interval,(-750, +750)]
}

print(f"{GOLD}")
print(15*"=", "DATA MAP", 15*"=")
lk = len(max(data_map, key=len)) + 1
ignored = 0
for col, (dt_type, valid_values) in data_map.items():
    if dt_type.name == "ID" or dt_type.name == "Ignore":
        ignored += 1
    print(f"  {TEAL}{col:.<{lk}s} {GOLD}{dt_type.name:9s}{GREEN}{valid_values}")
print(f"{GOLD} === Data Map has{RED}", len(data_map)-ignored,
      f"{GOLD}attribute columns", 3*"=",f"{RESET}")

# Step 3: Apply RIE preprocessing
# Set target variable
target = "activity"
print(f"{GOLD}Target variable: {target}")
if data_map[target][0] == DT.String:
	target_ = None
else:
	target_ = [target]
rie   = ReplaceImputeEncode(data_map=data_map,
                          interval_scale= None, # No standardization
                          no_impute=target_,    # Don't impute target
                          binary_encoding ="one-hot",
                          nominal_encoding="one-hot",
                          drop=False,  # Keep all encoded columns
                          display=True)

# Transform the data
encoded_df = rie.fit_transform(df)
print(f"\n{RED}encoded_df:{GOLD}:",
      f"{encoded_df.shape[0]} cases and",
      f"{encoded_df.shape[1]} columns,\n",
      "               including targets.")

print(f"\n{GOLD}Preprocessing complete. Ready for next step.{RESET}")

# Step 3: Kitchen Sink Random Forest Evaluation
lbl = "Step 3: Kitchen Sink Random Forest (Default Parameters)"
print_boundary(lbl)

y = df[target]
X = encoded_df
#X = encoded_df.drop(target, axis=1)

# First show the overfitting on full data
print(f"{GOLD}Fitting kitchen sink random forest using entire dataset")
kitchen_sink_forest = RandomForestClassifier(random_state=RANDOM_STATE)
kitchen_sink_forest = kitchen_sink_forest.fit(X, y)
forest_classifier.display_metrics(kitchen_sink_forest, X, y)

# Now evaluate with proper holdout validation
lbl = "70/30 Holdout Validation of Kitchen Sink Forest"
print_boundary(lbl)

# Split the data 70/30
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                  stratify=y, 
												  random_state=RANDOM_STATE)

# Fit kitchen sink forest on training data only
kitchen_sink_forest_cv = RandomForestClassifier(random_state=RANDOM_STATE)
kitchen_sink_forest_cv = kitchen_sink_forest_cv.fit(X_train, y_train)

# Evaluate using AdvancedAnalytics display_split_metrics
print(f"{GOLD}")
forest_classifier.display_split_metrics(kitchen_sink_forest_cv,
                                      X_train, y_train, X_val, y_val)

# Calculate and display accuracy ratio and misclassification ratio
train_pred = kitchen_sink_forest_cv.predict(X_train)
val_pred   = kitchen_sink_forest_cv.predict(X_val)
train_acc  = accuracy_score(y_train, train_pred)
val_acc    = accuracy_score(y_val, val_pred)

lbl = "Kitchen Sink 70/30 Validation"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc) #train/validation summary

# Show feature importance from the properly trained model
print(f"{GOLD}\nTop 10 Feature Importance (from training data):")
forest_classifier.display_importance(kitchen_sink_forest_cv, X.columns,
                                     top=10, plot=True)
print(f"{RESET}")

# Step 4.1: Feature Selection using Decision Tree Importance
lbl = "Step 4.1: Feature Selection using Decision Tree Importance"
print_boundary(lbl)

#Select top features for hyperparameter optimization
if FEATURE_THRESHOLD < 1.0:
	X_selected = tree_selection(X, y, FEATURE_THRESHOLD)
else:
	X_selected = X
	print(f"{GREEN}")
	print(f"Feature Threshold = 100%, using all features.{RESET}")
selected_features = X_selected.columns

# Step 4,2: Case Reduction - Construct Stratified Random Sample
lbl = "Step 4.2: Case Reduction using Stratified Random Sample"
print_boundary(lbl)
#Using different random_state (123) than Step 3 (42) to get different split
train_size = OPTIMIZATION_SAMPLE #Sample proportion for hyperparameter opt.
print(f"{GREEN}")
if train_size >= 1:
	print(f"Optimization Sample = 100%, using all cases.{RESET}")
else:
	c = round(train_size*X.shape[0])
	print(f"Optimization Sample = {train_size:0.1%}, using {c} cases.{RESET}" )

if train_size < 1.0:
    X_train, X_val, y_train, y_val = \
                   train_test_split(X_selected, y, train_size=train_size, 
									stratify=y, random_state=RANDOM_STATE)
else:
    X_train = X
    y_train = y
                           
# Step 4.3: Random Forest Hyperparameter Optimization using selected features
lbl = "Step 4.3: Random Forest Hyperparameter Optimization"
print_boundary(lbl)
#n=8281; 50% N = 4141; 0.5%xN = 41
param_grid = {
    'n_estimators':      [50, 100],
    'criterion':         ['gini', 'entropy'],
    'max_depth':         [5, 7, 9, 11, 13],
    'min_samples_split': [74, 78, 82, 86, 90],
    'min_samples_leaf':  [37, 39, 41, 43, 45],
    'max_features':      [3, 5, 7, 9, 11, 13]
}
k     = X_selected.shape[1]
m     = math.floor(np.sqrt(k)) #equivalent to 'sqrt(k)'
steps = math.ceil((k-m)/6)
maxfeatures = list(range(m, k-1, steps))
#if max(maxfeatures) < k:
#	maxfeatures.append(k)
param_grid['max_features'] = maxfeatures

m  = math.floor(0.005*y.shape[0]) - (y.nunique() - 1)
ks = min(k, m, 5)
param_grid['max_depth']         = list(range(ks, min(ks + 10, 20), 2))
param_grid['min_samples_leaf']  = list(range(m, m+10, 2))
param_grid['min_samples_split'] = list(range(2*m, 2*(m+10), 4))

"""
Grid Search: 3000 parameter combinations with
4-fold CV requires 12000 total fits.

Hyperparameter optimization uses only the top 99.0% of 
features and 100.0% of the dataset.

Setting n_jobs=-1 for faster calculations by using all available computer cores.

Starting grid search...
Grid search completed in 5 min. 49 sec.
Average time per parameter combination  0.12 seconds

 =============================================== 
 *********** Optimum Hyperparameters *********** 
 =============================================== 
            criterion...........   entropy
            max_depth...........        13
            max_features........         5
            min_samples_leaf....        37
            min_samples_split...        74
            n_estimators........       100

 =============================================== 
 ****** Optimum Forest Performance Metrics ***** 
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ACCURACY............ 0.9518    0.9431    1.0092
 MISCLASSIFICATION... 0.0482    0.0569    1.1795
 ----------------------------------------------- 
"""
lbl = "Hyperparameters"
print_boundary(lbl, 47)
for parm in param_grid:
    print(f"{GREEN} {parm:.<20s}{GOLD}{param_grid[parm][0:]}{RESET}")

# Calculate and display grid search information
total_combinations = 1
for param_list in param_grid.values():
    total_combinations *= len(param_list)
total_fits = total_combinations * 4  # cv=4

njobs = -1
print(f"\n{GOLD}Grid Search: {total_combinations} parameter combinations with")
print(f"4-fold CV requires {total_fits} total fits.\n")
t_size = 100*train_size; t = 100*FEATURE_THRESHOLD
print(f"Hyperparameter optimization uses only the top {t}% of", 
      f"\nfeatures and {t_size}% of the dataset.\n")
print(f"{GOLD}Setting {RED}n_jobs={njobs}{GOLD} for faster calculations",
	  f"by using all available computer cores.{RESET}")

# Start timing and run grid search
start_time  = time.time()
print(f"\n{GREEN}Starting grid search...{RESET}")
rf          = RandomForestClassifier(random_state=RANDOM_STATE)
#Grid Search using only selected features
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=4, scoring='accuracy', return_train_score=True, 
                           n_jobs=njobs).fit(X_train, y_train)
end_time = time.time()
elapsed_time = end_time - start_time
minutes = math.floor(elapsed_time/60)
seconds = elapsed_time - 60*minutes
print(f"{GOLD}Grid search completed in {minutes:.0f} min. {seconds:.0f} sec.")
print(f"Average time per parameter combination ", 
      f"{elapsed_time/total_combinations:.2f} seconds{RESET}")

lbl = "Optimum Hyperparameters"
print_boundary(lbl, 47)
# Find the longest parameter value for consistent formatting
max_param_len = max(len(str(val)) for val in grid_search.best_params_.values())
sze = max_param_len + 3
for parm in grid_search.best_params_:
    parameter = str(grid_search.best_params_[parm])
    print(f"{GREEN}            {parm:.<20s}{GOLD}{parameter:>{sze}s}{RESET}")

best_idx   = np.argmin(grid_search.cv_results_['rank_test_score'])
val_acc    = grid_search.cv_results_['mean_test_score'][best_idx]
train_acc  = grid_search.cv_results_['mean_train_score'][best_idx]

lbl = "Optimum Forest Performance Metrics"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

lbl = "Optimum Random Forest Importance"
print_boundary(lbl, 47)

opt_forest       = grid_search.best_estimator_
importance       = opt_forest.feature_importances_
feature          = X_train.columns
data             = {'feature': feature, 'importance':importance}
df               = pd.DataFrame(data)
df               = df.sort_values(by='importance', ascending=False)
df['cumulative'] = df['importance'].cumsum()
print(f"{GOLD}")
print(df.to_string(index=False))

# Step 5: Holdout Validation (70/30 split) - Selected Features
lbl = "Step 5: Holdout Validation (Selected Features)"
print_boundary(lbl)

Xt, Xv, yt, yv = train_test_split(X_selected, y, test_size=0.3,
                                  stratify=y, random_state=RANDOM_STATE)
# Train optimized model on final training set
hold_out_forest = opt_forest
hold_out_forest = hold_out_forest.fit(Xt, yt)
print(f"{GOLD}")
forest_classifier.display_split_metrics(hold_out_forest, Xt, yt, Xv, yv)
print(f"{RESET}")

# Calculate final performance metrics
train_pred = hold_out_forest.predict(Xt)
val_pred   = hold_out_forest.predict(Xv)
train_acc  = accuracy_score(yt, train_pred)
val_acc    = accuracy_score(yv, val_pred)
train_misc = 1.0 - train_acc
val_misc   = 1.0 - val_acc
ratio_acc  = train_acc / val_acc if val_acc > 0 else np.inf
ratio_misc = val_misc  / train_misc if train_misc > 0 else np.inf
"""
============================================================ 
 ****** Step 5: Holdout Validation (Selected Features) ****** 
 ============================================================ 
Model Metrics..........       Training     Validation
Observations...........           5796           2485
Features...............             15             15
Trees in Forest........            100            100
Maximum Tree Depth.....             13             13
Minimum Leaf Size......             37             37
Minimum split Size.....             74             74
Avg Squared Error......         0.0231         0.0255
Root ASE...............         0.1519         0.1597
Accuracy...............         0.9513         0.9284
Precision..............         0.9366         0.9019
Recall (Sensitivity)...         0.9003         0.8500
F1-score...............         0.9134         0.8651
Total Misclassifications...        282            178
MISC (Misclassification)...       4.9%           7.2%
     class sitting.........       0.4%           1.2%
     class sittingdown.....      10.9%          18.1%
     class standing........       2.0%           2.8%
     class standingup......      33.1%          48.9%
     class walking.........       3.5%           4.0%

Training
Confusion Matrix  Class 0   Class 1   Class 2   Class 3   Class 4  
0 sitting......      1764         4         0         3         0
1 sittingdown..        22       369         7         8         8
2 standing.....         0         5      1624         3        25
3 standingup...        17        50        44       291        33
4 walking......         0         4        48         1      1466


Training 
Metrics:
               precision    recall  f1-score   support

     sitting       0.98      1.00      0.99      1771
 sittingdown       0.85      0.89      0.87       414
    standing       0.94      0.98      0.96      1657
  standingup       0.95      0.67      0.79       435
     walking       0.96      0.97      0.96      1519

    accuracy                           0.95      5796
   macro avg       0.94      0.90      0.91      5796
weighted avg       0.95      0.95      0.95      5796

Validation
Confusion Matrix  Class 0   Class 1   Class 2   Class 3   Class 4  
0 sitting......       751         5         0         4         0
1 sittingdown..        11       145         4         7        10
2 standing.....         0         2       691         0        18
3 standingup...         3        30        30        95        28
4 walking......         0         1        25         0       625


Validation 
Metrics:
               precision    recall  f1-score   support

     sitting       0.98      0.99      0.98       760
 sittingdown       0.79      0.82      0.81       177
    standing       0.92      0.97      0.95       711
  standingup       0.90      0.51      0.65       186
     walking       0.92      0.96      0.94       651

    accuracy                           0.93      2485
   macro avg       0.90      0.85      0.87      2485
weighted avg       0.93      0.93      0.92      2485

 =============================================== 
  Holdout Validation Performance Summary (Selected Features)  
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ACCURACY............ 0.9513    0.9284    1.0247
 MISCLASSIFICATION... 0.0487    0.0716    1.4722
 ----------------------------------------------- 
"""
lbl = "Holdout Validation Performance Summary (Selected Features)"
print_boundary(lbl, 47)
print_summary(train_acc, val_acc)

# Show feature importance from the optimized model (selected features only)
print(f"{GOLD}\nFeature Importance (optimized model - selected features):")
forest_classifier.display_importance(hold_out_forest, selected_features,
                                     top='all', plot=True)

# Step 6: K-Fold Cross Validation - Selected Features
lbl = "Step 6: K-Fold Cross-Validation (Selected Features)"
print_boundary(lbl)

warnings.filterwarnings('ignore', category=RuntimeWarning)
n            = X_selected.shape[0]
best_val_acc = 0
for k in range(2, 11):  # Test 2-fold through 10-fold CV
    scores  = cross_validate(opt_forest, X_selected, y, 
							 scoring='accuracy', cv=k, return_train_score=True)
    # Calculate metrics
    train_acc = scores["train_score"].mean()
    val_acc   = scores["test_score"].mean()

    print_acc_ratio(scores, n)
    if val_acc > best_val_acc:
        best_k = k
        best_train_acc = train_acc
        best_val_acc   = val_acc

print(f"\n{GOLD} Best K :",
      f"{RED}{best_k}-Fold{GOLD}")
lbl = "K-Fold Cross-Validation Performance Summary (Selected Features)"
print_boundary(lbl, 47)
print_summary(best_train_acc, best_val_acc)
lbl = "Random Forest Analysis Complete"
print_boundary(lbl)
"""
 ====== 10-Fold Cross Validation =======
  Train Avg. MISC..... 0.0447 +/-0.0037
  Test  Avg. MISC..... 0.0533 +/-0.0148
  Mean Misc Ratio..... 1.1919 +/-0.3834
  ======================================= 
Equivalent to 10 splits each with 7453/828 Cases

 Best K : 10-Fold

 =============================================== 
  K-Fold Cross-Validation Performance Summary (Selected Features)  
 =============================================== 
                       TRAIN  VALIDATION   RATIO
 ACCURACY............ 0.9553    0.9467    1.0091
 MISCLASSIFICATION... 0.0447    0.0533    1.1919
 ----------------------------------------------- 
"""