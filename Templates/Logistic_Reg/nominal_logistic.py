# 33,126 rows x 14 columns   (pandas 3.0.3, max_n=10, max_s=30)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update on Sep 20, 2026
@purpose: Step-by-step optimization and validation of binary logistic reg.
@data:    CellphoneGender_StratifiedRS.csv
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
import matplotlib.pyplot as plt
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import accuracy_score, confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.metrics         import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_validate
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Regression          import logreg
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
        pred     = lr.predict(X)
        accuracy = accuracy_score(y, pred)
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

data_file = "CellphoneActivity_StratifiedRS.csv" #Reduced Dataset
df        = pd.read_csv("../data/"+data_file)
print(f"{GOLD}Data loaded: {RED}{df.shape[0]}{GOLD} observations",
      f"{RED}{df.shape[1]} {GOLD}columns.{RESET}")

#In these data, the target is nominal.  After ReplaceImputeEncode, it
#must be added back into the encoded dataframe as a single attribute

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
target = "activity"
print(f"{GOLD}")
# Apply ReplaceImputeEncode preprocessing
rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale=None, # No scaling of interval
                          no_encode=[target],  # Do not encode target
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          drop=False,             # Keep all encoded nominal columns
                          display=True)

# Transform the data
encoded_df = rie.fit_transform(df)
encoded_df = pd.concat([df[target], encoded_df], axis=1) #Insert Target back
print(f"{RESET}")
print(f"{RED}encoded_df{GOLD} created from {TEAL}{data_file}{GOLD} containing",
      f"{TEAL}{encoded_df.shape[0]} {GOLD}cases & {TEAL}{encoded_df.shape[1]}",
      f" {GOLD}columns, including the target {RED}'{target}'{RESET}")

print(f"{GOLD}")
# Create version with last dummy column dropped (avoids dummy trap in full GLM)
rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale=None, # No scaling of interval
                          no_encode=[target],  # Do not encode target
                          binary_encoding ="one-hot",
                          nominal_encoding="one-hot",
                          drop=True, # Drop last column of each encoded nominal
                          display=False)
encoded_drp_df = rie.fit_transform(df)
encoded_drp_df = pd.concat([df[target], encoded_drp_df], axis=1) #Insert Traget
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

lgr = LogisticRegression(solver='newton-cg', tol=1e-4, max_iter=10000, 
                         random_state=31415)
X     = encoded_drp_df.drop([target], axis=1)
y     = encoded_drp_df[target]
model = lgr.fit(X, y)
prob  = model.predict_proba(X)
pred  = model.predict(X)
print(classification_report(y, pred))
conf_matrix = confusion_matrix(y, pred)
ConfusionMatrixDisplay.from_estimator(model, X, y, normalize='true', 
                                      xticks_rotation=45, cmap='Blues')
plt.title("Recall Matrix (Correct vs. Truth)")
plt.show()

n_correct = 0.0
for i in range(prob.shape[1]):
    n_correct += conf_matrix[i,i]
n = y.shape[0]
accuracy_all = n_correct/n
misc_all = n - n_correct
misc_p   = misc_all/n

print(f"{RESET}")
print(f"{TEAL}Logistic Regression using {GREEN}'ALL' {TEAL}attributes ****")
print(f"{TEAL}Accuracy for All Features Logistic Reg: {GREEN}{accuracy_all: 5.2%}")
print(f"{TEAL}Total Misclassifications: {misc_all}/{n}:  {GREEN}{misc_p: 5.2%}")
print(f"{RESET}")
feature_all = list(X.columns)



#**************************************************************************
#****************************** HOLD-OUT VALIDATION ***********************
lbl = "STEP 8: HOLD-OUT VALIDATION"
print_boundary(lbl)

y = encoded_df[target]  # Use original target values (not encoded)
X = encoded_df.drop([target], axis=1) # Use features from best model
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                    test_size=0.3, random_state=12345)
print(f"\n{GOLD}Predicting {RED}'{target}'{GOLD} using", 
      f"{RED}'All'{GOLD} Attributes in {RED}encoded_df{RESET}")

lgr = LogisticRegression(solver='newton-cg', tol=1e-4, max_iter=10000, 
                         random_state=31415)
model = lgr.fit(X_train, y_train)

ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, normalize='true', 
                                      xticks_rotation=45, cmap='Blues')
plt.title("Recall Matrix (Training Data)")
plt.show()

ConfusionMatrixDisplay.from_estimator(model, X_val, y_val, normalize='true', 
                                      xticks_rotation=45, cmap='Blues')
plt.title("Recall Matrix (Validation Data)")
plt.show()

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
    print(f"{TEAL}Misclassification overfit ratio",
          f"({misc_val:5.2%}/{misc_train:5.2%}):",
          f" {RED}{overfit_ratio:.2f}{RESET}")
else:
    print(f"{TEAL}Misclassification overfit ratio",
          f"({misc_val:5.1%}/{misc_train:5.1%}):",
          f" {GREEN}{overfit_ratio:.2f}{RESET}")
    
#*************************************************************************
#****************************** CROSS VALIDATION *************************
lbl = "STEP 9: K-FOLD CROSS VALIDATION"
print_boundary(lbl)
    
for n_folds in range(2, 6):
    lgr = LogisticRegression(solver='newton-cg', tol=1e-4, max_iter=10000, 
                             random_state=31415)
    scores  = cross_validate(lgr, X, y,
                             scoring="accuracy",
                             cv=n_folds, return_train_score=True, )
    print_acc_ratio(scores, n)

lbl = "Analysis of Binary Logistic Reg. Data Complete"
print_boundary(lbl)