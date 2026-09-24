#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update on Sep 20, 2026
@purpose: Step-by-step optimization and validation of binary logistic reg.
@data:    CreditDefaultData.csv
@author:  EJones
@email:   ejones@tamu.edu
"""
# ANSI color codes - to print in color, the package colorama must be installed
RED   = "\033[38;5;197m"  
GOLD  = "\033[38;5;185m"
TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m"
RESET = "\033[0m"

# Import required packages
import pandas as pd
import numpy  as np
import statsmodels.api   as sm
import matplotlib.pyplot as plt
from sklearn.linear_model    import LogisticRegression, LogisticRegressionCV
from sklearn.metrics         import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Regression          import logreg, stepwise

def print_boundary(lbl, b_width=60):
    print("")
    margin = b_width - len(lbl) - 2
    lmargin = int(margin / 2)
    rmargin = lmargin
    if lmargin + rmargin < margin:
        lmargin += 1
    print(f"{TEAL}", "=" * b_width, f"{RESET}")
    print(f"{GREEN}", lmargin * "*", lbl, rmargin * "*", f"{RESET}")
    print(f"{TEAL}", "=" * b_width, f"{RESET}")

def print_acc_ratio(scores, n):
    n_folds     = len(scores["train_score"])
    train_misc  = (1.0 - scores["train_score"])
    train_smisc = 2.0*(1.0 - scores["train_score"]).std()
    val_misc    = (1.0 - scores["test_score"])
    val_smisc   = 2.0*(1.0 - scores["test_score"]).std()
    ratios      = val_misc/train_misc
    ratio       = ratios.mean()
    s_ratio     = 2.0*ratios.std()
    train_misc  = train_misc.mean()
    val_misc    = val_misc.mean()
    print(f"{TEAL}\n")
    print(f" ====== {n_folds:.0f}-Fold Cross Validation =======")
    print(f"  Train Avg. MISC..... {train_misc:.4f} +/-{train_smisc:.4f}")
    print(f"  Test  Avg. MISC..... {val_misc:.4f} +/-{val_smisc:.4f}")
    print(f"  Mean Misc Ratio..... {ratio:.4f} +/-{s_ratio:.4f}")
    print(" ", 39*"=", f"{RESET}")
    n_v = n*(1.0/n_folds)
    n_t = n - n_v
    print(f"Equivalent to {n_folds:.0f} splits each with "+
          f"{n_t:.0f}/{n_v:.0f} Cases")
    
def lr_plot(lr, X, y, bestc):
    shrinkage = np.logspace(-3, 2, 20)
    coefs  = []
    acc    = []
    for a in shrinkage:
        lr.set_params(C=a)
        lr.fit(X, y)
        coefs.append(lr.coef_[0])
        predictions = lr.predict(X)
        accuracy = accuracy_score(y, predictions)
        acc.append(accuracy)

    gold = '#D4AF37'
    plt.style.use('dark_background')
    fs = 12
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.grid(True, linestyle='--', alpha=0.3)
    ax = plt.gca()
    clabel = "C="+str(bestc)
    ax.axvline(x=bestc, color='r', linestyle=":", label=clabel)
    ax.plot(shrinkage, coefs)
    ax.set_xscale('log')
    ax.legend(loc="lower left")

    parms = lr.get_params()
    plt.xlabel('Shrinkage', color=gold, fontsize=fs, fontweight='bold')
    if parms['penalty']=='l1':
        plt.ylabel('L1 Coefficients',color=gold, fontsize=fs, fontweight='bold')
        pltFile = 'l1.png'
    elif parms['penalty']=='l2':
        plt.ylabel('L2 Coefficients',color=gold, fontsize=fs, fontweight='bold')
        pltFile = 'l2.png'
    else:
        return

    plt.axis('tight')

    plt.subplot(122)
    plt.grid(True, linestyle='--', alpha=0.3)
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    clabel = "C="+str(bestc)
    ax.axvline(x=bestc, color='r', linestyle=":", label=clabel)
    ax.plot(shrinkage, acc, linewidth=3, )
    ax.set_xscale('log')
    ax.legend(loc="lower right")

    plt.xlabel('Shrinkage', color=gold, fontsize=fs, fontweight='bold')
    if parms['penalty']=='l1':
        plt.ylabel('L1 Accuracy',color=gold, fontsize=fs, fontweight='bold')
        pltFile = 'l1.png'
    elif parms['penalty']=='l2':
        plt.ylabel('L2 Accuracy',color=gold, fontsize=fs, fontweight='bold')
        pltFile = 'l2.png'
    else:
        return
    plt.axis('tight')
    plt.savefig(pltFile, pad_inches=0.1, dpi=256, bbox_inches='tight')
    plt.show()

lbl = "Step 1: Reading Data and Preparing Data Map"
print_boundary(lbl)

data_file = "CreditDefaultData.csv"
df        = pd.read_csv("../data/"+data_file)
print(f"{GOLD}Data loaded: {RED}{df.shape[0]}{GOLD} observations", 
      f"{RED}{df.shape[1]} {GOLD}columns.{RESET}")

# Create data map based on data dictionary
data_map = {
    "Customer":      [DT.ID, ("")],
    "Default":       [DT.Binary, (0, 1)],
    "card_class":    [DT.Nominal, (1, 2, 3)],
    "Gender":        [DT.Binary, (1, 2)],  # 1=female, 2=male
    "Education":     [DT.Ignore, ("")],    # Listed as "Ignore" in data dict
    "Marital_Status":[DT.Nominal, (0, 1, 2, 3)],
    "Age":           [DT.Interval, (20, 80)],
    "Credit_Limit":  [DT.Interval, (100, 800000)],
    # Payment status for 6 months (-2 to 8)
    "Jun_Status":    [DT.Interval, (-2, 8)],
    "May_Status":    [DT.Interval, (-2, 8)],
    "Apr_Status":    [DT.Interval, (-2, 8)],
    "Mar_Status":    [DT.Interval, (-2, 8)],
    "Feb_Status":    [DT.Interval, (-2, 8)],
    "Jan_Status":    [DT.Interval, (-2, 8)],
    # Bill amounts for 6 months (-12000 to +32000)
    "Jun_Bill":      [DT.Interval, (-12000, 32000)],
    "May_Bill":      [DT.Interval, (-12000, 32000)],
    "Apr_Bill":      [DT.Interval, (-12000, 32000)],
    "Mar_Bill":      [DT.Interval, (-12000, 32000)],
    "Feb_Bill":      [DT.Interval, (-12000, 32000)],
    "Jan_Bill":      [DT.Interval, (-12000, 32000)],
    # Payment amounts for 6 months (0 to 60000)
    "Jun_Payment":   [DT.Interval, (0, 60000)],
    "May_Payment":   [DT.Interval, (0, 60000)],
    "Apr_Payment":   [DT.Interval, (0, 60000)],
    "Mar_Payment":   [DT.Interval, (0, 60000)],
    "Feb_Payment":   [DT.Interval, (0, 60000)],
    "Jan_Payment":   [DT.Interval, (0, 60000)],
    # Payment percentages for 6 months (0 to 1)
    "Jun_PayPercent":[DT.Interval, (0, 1)],
    "May_PayPercent":[DT.Interval, (0, 1)],
    "Apr_PayPercent":[DT.Interval, (0, 1)],
    "Mar_PayPercent":[DT.Interval, (0, 1)],
    "Feb_PayPercent":[DT.Interval, (0, 1)],
    "Jan_PayPercent":[DT.Interval, (0, 1)]
}
print(f"{GOLD}", 15*"=", "DATA MAP", 15*"=")
lk = len(max(data_map, key=len)) + 1
ignored = 0
for col, (dt_type, valid_values) in data_map.items():
    if dt_type.name == "ID" or dt_type.name=="Ignore":
        ignored += 1
    print(f"  {TEAL}{col:.<{lk}s} {GOLD}{dt_type.name:9s}{GREEN}{valid_values}")
print(f"{GOLD} === Data Map has{RED}", len(data_map)-ignored,
      f"{GOLD}attribute columns", 3*"=",f"{RESET}")

lbl = "Step 2: ReplaceImputeEncode (RIE) Processing"
print_boundary(lbl)

# Set target variable
target = "Default"
print(f"{GOLD}")
# Apply ReplaceImputeEncode preprocessing
rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale=None,  # No standardization of interval features
                          no_impute=[target],    # Do not impute target variable
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          drop=False,             # Drop one column from each encoded nominal set
                          display=True)

# Transform the data
encoded_df = rie.fit_transform(df)
print(f"{RESET}")
print(f"{RED}encoded_df{GOLD} created from {TEAL}{data_file}{GOLD} containing",
      f"{TEAL}{encoded_df.shape[0]} {GOLD}cases & {TEAL}{encoded_df.shape[1]}",
      f" {GOLD}columns, including the target {RED}'{target}'{RESET}")

# Create version without dropped columns for analysis using ALL attributes
rie = ReplaceImputeEncode(data_map=data_map,
                                  interval_scale=None,
                                  no_impute=[target],
                                  binary_encoding="one-hot",
                                  nominal_encoding="one-hot",
                                  drop=True,  # Keep all columns for stepwise
                                  display=False)
encoded_drp_df = rie.fit_transform(df)
print(f"{RESET}")
print(f"{RED}encoded_drp_df{GOLD} created from {TEAL}{data_file}{GOLD}", 
      f"containing {TEAL}{encoded_drp_df.shape[0]} {GOLD}cases &", 
      f"{TEAL}{encoded_drp_df.shape[1]}{GOLD} columns, including the", 
      f"target {RED}'{target}'{RESET}")

#***************************************************************************
#**************** All Features Logistic Regression *************************
lbl = " STEP 3: Logistic Regression using All Attributes"
print_boundary(lbl)
print(f"\n{GOLD}Predicting {RED}'{target}'{GOLD} using", 
      f"{RED}'All'{GOLD} Attributes in {RED}encoded_drp_df{RESET}")

y = encoded_drp_df[target]
X = encoded_drp_df.drop(target, axis=1)
Xc = sm.add_constant(X)

glm_model = sm.GLM(y, Xc, family=sm.families.Binomial()).fit()
print(f"{GOLD}")
print(glm_model.summary())

# Display the confusion Matrix
pred_prob = glm_model.predict(Xc)  # Predicted prob(target=1) target=0 or 1
pred_class = list(map(round, pred_prob))  # Predicted class 0 or 1 using pred
accuracy_all = accuracy_score(y, pred_class)
conf_mat = confusion_matrix(y, pred_class)
misc_all = conf_mat[0, 1] + conf_mat[1, 0]
logreg.display_confusion(conf_mat)
n = df.shape[0]
misc_p = misc_all/n
print(f"{RESET}")
print(f"{TEAL}Logistic Regression using {GREEN}'ALL' {TEAL}attributes ****")
print(f"{TEAL}Accuracy for All Features Logistic Reg: {GREEN}{accuracy_all: 5.4f}")
print(f"{TEAL}Total Misclassifications: {misc_all}/{n}:    {GREEN}{misc_p: 5.4f}")
print(f"{RESET}")
feature_all = list(X.columns)

#***************************************************************************
#**************** Stepwise Feature Selection *******************************
lbl = "STEP 4: STEPWISE SELECTION"
print_boundary(lbl)
print(f"{TEAL}")
# Hyperparameter Optimization to Select Features
selected = stepwise(encoded_df, target, reg="logistic", method="stepwise",
                    crit_in=0.05, crit_out=0.05, verbose=True).fit_transform()
print(F"{TEAL}", "-"*80, f"{RESET}")
print(f" {GOLD}Stepwise selected {RED}{len(selected)}{GOLD} out of", 
      f"{RED}{encoded_df.shape[1]-1} {GOLD}features.{RESET}")
print(F"{TEAL}", "="*80, f"{RESET}")

# Extract target and predictors from encoded_df
y = encoded_df[target]
X = encoded_df[selected]
print(f"{RESET}")
print(f"\n{RED}Stepwise{GOLD} Logistic Regression {RESET}")
# Add constant for intercept
X_sm = sm.add_constant(X)
glm_model = sm.GLM(y, X_sm, family=sm.families.Binomial()).fit()
print(f"{GOLD}")
print(glm_model.summary())

# Display the confusion Matrix
pred_prob = glm_model.predict(X_sm)  # Predicted prob(target=1) target=0 or 1
pred_class = list(map(round, pred_prob))  # Predicted class 0 or 1 using pred
accuracy_step = accuracy_score(y, pred_class)
conf_mat = confusion_matrix(y, pred_class)
logreg.display_confusion(conf_mat)

print(f"{RESET}")
misc_step = conf_mat[1, 0] + conf_mat[0, 1]
n = df.shape[0]
misc_p = misc_step/n
n_selected = len(selected)
print(f"{RED}Stepwise Logistic Regression {TEAL}selected",
      f"{GREEN}{n_selected}{TEAL} attributes,")
print(f"{TEAL}Accuracy..............................{GREEN}{accuracy_step: 5.1%}")
print(f"{TEAL}Total Misclassifications: {misc_step}/{n}...{GREEN}{misc_p: 5.1%}")
print(f"{RESET}")
feature_step = list(selected)

#***************************************************************************
#**************** L1 Regularization (Lasso) *********************************
lbl = "STEP 5: L1 REGULARIZATION SELECTION"
print_boundary(lbl)

y = encoded_df[target]  
X = encoded_df.drop(target, axis=1)  # Features without target
nf = X.shape[1]

# Test different regularization strengths
C = [0.001, 0.01, 0.08, 0.09, 0.1, 0.2, 1.0, 5.0, 10.0, 50.0, np.inf]
accuracy  = []
best_l1_c = 0.0
best_acc  = 0.0

for c in C:
    lgr = LogisticRegression('l1', C=c, tol=1e-4, max_iter=500, l1_ratio=1,
                             solver="liblinear", random_state=31415).fit(X, y)
    acc = lgr.score(X, y)
    if acc > best_acc:
        best_acc = acc
        best_l1_c = c
    #accuracy.append(lgr.score(X, y))

# Fit final L1 model with best C
lgr = LogisticRegression('l1', C=best_l1_c, tol=1e-8, max_iter=10000, l1_ratio=1,
                        solver="liblinear", random_state=31415).fit(X, y)

# Get selected features (non-zero coefficients)
ncoef = len(lgr.coef_[0])
feature = X.columns
selected = []
i = 0
for coef in lgr.coef_[0]:
    if np.abs(coef) > 0.0001:  # Threshold for "selected"
        selected.append(feature[i])
    i += 1

Xs = X[selected]
print("Number of Selected Attributes: ", len(selected))
lgr_l1 = LogisticRegression('l1', C=best_l1_c, tol=1e-8, max_iter=10000, l1_ratio=1,
                        solver="liblinear", random_state=31415).fit(Xs, y)
print(f"{GOLD}")
logreg.display_coef(lgr_l1, Xs, y)
logreg.display_metrics(lgr_l1, Xs, y)

accuracy_l1 = lgr_l1.score(Xs, y)
n_selected  = len(selected)
pred_class  = lgr_l1.predict(Xs)
conf_mat    = confusion_matrix(y, pred_class)
misc_l1     = conf_mat[1, 0] + conf_mat[0, 1]
n           = df.shape[0]
misc_p      = misc_l1/n
print(f"{RED}L1 Logistic Regression {TEAL}using {GREEN}c={best_l1_c}",
      f"{TEAL}selected {GREEN}{n_selected}{TEAL} attributes")
print(f"{TEAL}Accuracy.............................{GREEN}{accuracy_l1:6.1%}")
print(f"{TEAL}Total Misclassifications: {misc_l1}/{n}..{GREEN}{misc_p: 6.1%}")
print(f"{RESET}")
feature_l1 = list(selected)
lr_plot(lgr_l1, Xs, y, best_l1_c)

#***************************************************************************
#**************** L2 Regularization ***************************************
lbl = "STEP 6: L2 REGULARIZATION"
print_boundary(lbl)

y = encoded_df[target]  # Original target
X = encoded_df.drop(target, axis=1)

# Test different regularization strengths
C = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 1.0, 5.0, 10.0, 50.0, 
     100.0, 500.0, np.inf]
accuracy  = []
best_l2_c = 0.0
best_acc  = 0.0

for c in C:
    lgr = LogisticRegression("l2", C=c, tol=1e-4, max_iter=500, l1_ratio=0,
                             solver="newton-cg", random_state=31415).fit(X, y)
    acc = lgr.score(X, y)
    if acc > best_acc:
        best_acc = acc
        best_l2_c = c
    #accuracy.append(lgr.score(X, y))

# Fit final L2 model with best C
lgr = LogisticRegression("l2",C=best_l2_c,tol=1e-8, max_iter=10000, l1_ratio=0,
                        solver="newton-cg", random_state=31415).fit(X, y)

# Get features with significant coefficients
feature_l2 = []
coef_l2    = lgr.coef_[0]
i = 0
for coef in coef_l2:
    if np.abs(coef) >= 0.0001:
        feature_l2.append(X.columns[i])
    i += 1

Xs = X[feature_l2]
lgr_l2 = LogisticRegression('l2',C=best_l2_c,tol=1e-8,max_iter=10000,
                            l1_ratio=0, solver="newton-cg", 
                            random_state=31415).fit(Xs, y)
print(f"{GOLD}")
logreg.display_coef(lgr_l2, Xs, y)
logreg.display_metrics(lgr_l2, Xs, y)

accuracy_l2 = lgr_l2.score(Xs, y)
pred_class  = lgr_l2.predict(Xs)
conf_mat    = confusion_matrix(y, pred_class)
misc_l2     = conf_mat[1, 0] + conf_mat[0, 1]
n           = df.shape[0]
misc_p      = misc_l2/n
n_selected  = len(feature_l2)
print(f"{RED}L2 Logistic Regression {TEAL}using {GREEN}c={best_l2_c}",
      f"{TEAL}selected {GREEN}{n_selected}{TEAL} attributes")
print(f"{TEAL}Accuracy.............................{GREEN}{accuracy_l2:6.1%}")
print(f"{TEAL}Total Misclassifications: {misc_l2}/{n}..{GREEN}{misc_p: 6.1%}")
print(f"{RESET}")
lr_plot(lgr_l2, X, y, best_l2_c)

#*************** OPTIMIZATION SUMMARY ***************************************
lbl = "STEP 7: HYPERPARAMETER OPTIMIZATION SUMMARY"
print_boundary(lbl)

print("")
print(f"{TEAL}", 44*"="+f"{RESET}")
models   = ["ALL", "Stepwise", "L1 Reg.", "L2 Reg."]
fitted   = [lgr, glm_model, lgr_l1, lgr_l2]
m_acc    = [accuracy_all, accuracy_step, accuracy_l1, accuracy_l2]
misc     = [misc_all, misc_step, misc_l1, misc_l2]
features = [feature_all, feature_step, feature_l1, feature_l2]
best_i   = 0
best_acc = 0
for i in range(1,4):
    if m_acc[i] > best_acc:
        best_acc = m_acc[i]
        best_i     = i
        
print(f"{GOLD} MODEL               ACCURACY          MISC{RESET}")
n = len(y)
for i in range(4):
    if i == best_i:
        print(f"{RED} {models[i]:.<19s}{RESET} {GREEN}{m_acc[i]: 7.2%} ",
              f"    {misc[i]: 5d}/{n:5d}{RESET}")
    else:
        print(f"{TEAL} {models[i]:.<19s}{RESET} {GREEN}{m_acc[i]: 7.2%} ",
              f"    {misc[i]: 5d}/{n:5d}{RESET}")
print(f"{TEAL}", 44*"="+f"{RESET}")

print("")
print(f"{TEAL}", 44*"="+f"{RESET}")
print(f"{GOLD} ATTRIBUTE............... STEPWISE  L1    L2{RESET}")
print(f"{TEAL}", 44*"-"+f"{GREEN}")

X = encoded_df.drop(target, axis=1)
features_all = X.columns
for i in range(X.shape[1]):
    attribute = features_all[i]
    if attribute == target:
        continue
    flag = 0
    if attribute in feature_step:
        select = "    X      "
        flag = 1
    else:
        select = "           "
    if attribute in feature_l1:
        select = select + "X      "
        flag = 1
    else:
        select = select + "       "
    if attribute in feature_l2:
        select = select + "X      "
        flag = 1
    else:
        select = select + "       "
    
    feature = attribute[0:25] #Truncating attributes labels to 25 characters
    if flag == 1:
        print(f"{GOLD}{feature:.<25s}{RED}{select:25s}{RESET}")
    else:
        print(f"{GOLD}{feature:.<25s}{RESET}")
print(f"{TEAL}", 44*"-"+f"{RESET}")
ns = len(feature_step)
n1 = len(feature_l1)
n2 = len(feature_l2)
print(f"{GOLD} Selected................ {RED}{ns:>4d}{n1:>7d}{n2:>7d}{RESET}")
print(f"{TEAL}", 44*"="+f"{RESET}")

#****************************** HOLD-OUT VALIDATION *************************
lbl = "STEP 8: HOLD-OUT VALIDATION"
print_boundary(lbl)

best_features = features[best_i]
best_model    = models[best_i]
y = df[target]  # Use original target values (not encoded)
X = encoded_df[best_features]  # Use features from best model
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                    test_size=0.3, random_state=12345)
if best_model == "Stepwise":
    lgr = LogisticRegression(None, tol=1e-8, max_iter=10000,
                         solver="newton-cg", random_state=31415)
elif best_model == "L1 Reg.":
    lgr = LogisticRegression('l1', C=c, tol=1e-4, max_iter=500, l1_ratio=1,
                             solver="liblinear", random_state=31415).fit(X, y)
elif best_model == "L2 Reg.":
    lgr = LogisticRegression("l2", C=best_l2_c, tol=1e-8, max_iter=10000,
                             solver="newton-cg", random_state=31415)
lgr = lgr.fit(X_train, y_train)
print(f"{GOLD}")
logreg.display_split_metrics(lgr, X_train, y_train, X_val, y_val)
print(f"{RESET}")
# Examine Possible Overfitting
misc_train     = 1.0 - lgr.score(X_train, y_train)
misc_val       = 1.0 - lgr.score(X_val,   y_val)
if misc_val > 0:
    overfit_ratio = misc_val/misc_train
else:
    overfit_ratio = np.inf
# Check for overfitting
if overfit_ratio > 1.2:
    print(f"{RED}Warning Overfit more than 20%{RESET}")
    print(f"{GREEN}Misclassification overfit_ratio ",
          f"(misc_val/misc_train):{RESET} {RED}{overfit_ratio:.2f}{RESET}")
else:
    print(f"{GREEN}Misclassification overfit_ratio ",
          f"(misc_val/misc_train):{RESET} {GREEN}{overfit_ratio:.2f}{RESET}")

#****************************** CROSS VALIDATION *************************
lbl = "STEP 9: K-FOLD CROSS VALIDATION"
print_boundary(lbl)

n  = X.shape[0]
if best_model == "Stepwise":
    lgr = LogisticRegression(None, tol=1e-8, max_iter=10000,
                         solver="newton-cg", random_state=31415)
elif best_model == "L1 Reg.":
    lgr = LogisticRegression('l1', C=c, tol=1e-4, max_iter=10000, l1_ratio=1,
                             solver="liblinear", random_state=31415).fit(X, y)
elif best_model == "L2 Reg.":
    lgr = LogisticRegression("l2", C=best_l2_c, tol=1e-8, max_iter=10000,
                             solver="newton-cg", random_state=31415)
    
for n_folds in range(2, 11):
    scores  = cross_validate(lgr, X, y,
                             scoring="accuracy",
                             cv=n_folds, return_train_score=True, )
    print_acc_ratio(scores, n)

lbl = "Analysis of Binary Logistic Reg. Data Complete"
print_boundary(lbl)