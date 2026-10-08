"""
Update on Sep 26, 2026
@purpose: Step-by-step fitting and validation of logistic regression for
          a nominal target using all attributes: multinomial and
          one-vs-rest (one binary model per class).
@data:    CellphoneActivity_StratifiedRS.csv
@author:  EJones
@email:   ejones@tamu.edu
@version: One standardized encoding (interval_scale="std", no dropped
          one-hot columns) is used for fitting and both validation steps.
"""
# ANSI color codes - to print in color, the package colorama must be installed
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

# Import required packages
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.linear_model    import LogisticRegression
from sklearn.multiclass      import OneVsRestClassifier
from sklearn.metrics         import accuracy_score, confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.metrics         import ConfusionMatrixDisplay
from sklearn.model_selection import (train_test_split, cross_validate,
                                     StratifiedKFold)
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

def confusion_heatmap(model, X, y, title, ax=None, plt_file=None,
                      subtitle=None):
    """Plot normalized confusion matrix with a descriptive title."""
    fs = 12
    fs_sub = 9
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    acc = accuracy_score(y, model.predict(X))
    ConfusionMatrixDisplay.from_estimator(model, X, y, normalize='true',
                                          xticks_rotation=45, cmap='Blues',
                                          ax=ax, colorbar=own_fig)
    if subtitle == 'auto':
        subtitle = f"n={len(y):,} & Accuracy={acc:.1%}"
    ax.set_title(title, fontsize=fs, fontweight='bold',
                 pad=22 if subtitle else 14)
    if subtitle is not None:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=fs_sub,
                fontweight='bold')
    ax.set_xlabel(f"Predicted {target}", fontsize=fs - 1)
    ax.set_ylabel(f"True {target}", fontsize=fs - 1)
    if own_fig:
        plt.tight_layout()
        if plt_file is not None:
            plt.savefig(plt_file, pad_inches=0.3, dpi=256, bbox_inches='tight')
        plt.show()

def confusion_heatmap_pair(model, Xt, yt, Xv, yv, target, plt_file=None):
    """Side-by-side hold-out training and validation confusion heat maps."""
    if plt_file is None:
        plt_file = 'confusion_heatmap_holdout.png'
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Logistic Regression for '{target}' using All Features",
                 fontsize=12, fontweight='bold')
    confusion_heatmap(model, Xt, yt,
                      "Training (70%)",
                      subtitle='auto', ax=axes[0])
    confusion_heatmap(model, Xv, yv,
                      "Validation (30%)",
                      subtitle='auto', ax=axes[1])
    plt.tight_layout()
    plt.savefig(plt_file, pad_inches=0.3, dpi=256, bbox_inches='tight')
    plt.show()
	
def best_lgr():
    # Unfitted copy of the best model, for hold-out and k-fold validation
    lgr = LogisticRegression(C=np.inf, solver='newton-cg', tol=1e-4,
                             max_iter=10000, random_state=31415)
    if best_model == "One-vs-Rest":
        return OneVsRestClassifier(lgr)
    return lgr
""" ======================================================================= """
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

# Attributes that take a single value within every level of a nominal
# attribute carry no information beyond that nominal. All are kept, but
# their coefficients cannot be separated from the nominal's coefficients.
nominal = [col for col, (dt_type, _) in data_map.items()
           if dt_type.name == "Nominal"]
for nom in nominal:
    others   = [col for col, (dt_type, _) in data_map.items()
                if col != nom and dt_type.name in ("Interval", "Binary",
                                                   "Nominal")]
    constant = [col for col in others
                if df.groupby(nom)[col].nunique().max() == 1]
    if constant:
        print(f"{RED}NOTE: {', '.join(constant)} take one value per level",
              f"of '{nom}'")
        print(f"      ({df[nom].nunique()} levels). They add nothing",
              f"beyond '{nom}', and their")
        print(f"      coefficients (and '{nom}') should not be interpreted.",
              f"{RESET}")

lbl = "Step 2: ReplaceImputeEncode (RIE) Processing"
print_boundary(lbl)

# Set target variable
target = "activity"
print(f"{GOLD}")
# One encoding is used throughout.
#   interval_scale="std" - interval attributes on a common scale
#       (per standard deviation), as in the binary templates.
#   drop=False (default) - every one-hot column is kept.
rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale="std",
                          no_encode=[target],  # Do not encode target
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          display=True)

# Transform the data
encoded_df = rie.fit_transform(df)
encoded_df = pd.concat([df[target], encoded_df], axis=1) #Insert Target back
print(f"{RESET}")
print(f"{RED}encoded_df{GOLD} created from {TEAL}{data_file}{GOLD} containing",
      f"{TEAL}{encoded_df.shape[0]} {GOLD}cases & {TEAL}{encoded_df.shape[1]}",
      f" {GOLD}columns, including the target {RED}'{target}'{RESET}")

#***************************************************************************
#**************** All Features Logistic Regression *************************
lbl = "STEP 3: Multinomial Logistic Regression (All Attributes)"
print_boundary(lbl)
print(f"\n{GOLD}Predicting {RED}'{target}'{GOLD} using", 
      f"{RED}'All'{GOLD} Attributes in {RED}encoded_df{RESET}")

# C=np.inf means no penalty. Without it, sklearn's LogisticRegression
# silently applies an L2 (ridge) penalty with C=1.0 by default.
lgr = LogisticRegression(C=np.inf, solver='newton-cg', tol=1e-4,
                         max_iter=10000, random_state=31415)
X     = encoded_df.drop([target], axis=1)
y     = encoded_df[target]
model = lgr.fit(X, y)
prob  = model.predict_proba(X)
pred  = model.predict(X)
print(classification_report(y, pred))
conf_matrix = confusion_matrix(y, pred)
confusion_heatmap(model, X, y,
                  "All-Features Logistic Regression",
                  subtitle='auto',
                  plt_file='confusion_all_features.png')

n_correct = 0.0
for i in range(prob.shape[1]):
    n_correct += conf_matrix[i,i]
n = y.shape[0]
accuracy_all = n_correct/n
misc_all = int(n - n_correct)
misc_p   = misc_all/n

print(f"{RESET}")
print(f"{TEAL}Logistic Regression using {GREEN}'ALL' {TEAL}attributes ****")
print(f"{TEAL}Accuracy for All Features Logistic Reg: {GREEN}{accuracy_all: 5.2%}")
print(f"{TEAL}Total Misclassifications: {misc_all}/{n}:  {GREEN}{misc_p: 5.2%}")
print(f"{RESET}")
feature_all = list(X.columns)

#***************************************************************************
#**************** One-vs-Rest Logistic Regression **************************
lbl = "STEP 4: One-vs-Rest Logistic Regression (All Attributes)"
print_boundary(lbl)

# One binary logistic regression per class (that class vs. all others).
# Each case is assigned to the class whose model gives the highest
# probability, so a rare class only has to beat the other classes, not 0.5.
ovr = OneVsRestClassifier(
          LogisticRegression(C=np.inf, solver='newton-cg', tol=1e-4,
                             max_iter=10000, random_state=31415))
ovr = ovr.fit(X, y)
pred_ovr = ovr.predict(X)
print(f"{GOLD}")
print(classification_report(y, pred_ovr))
confusion_heatmap(ovr, X, y,
                  "One-vs-Rest Logistic Regression",
                  subtitle='auto',
                  plt_file='confusion_ovr.png')
accuracy_ovr = accuracy_score(y, pred_ovr)
misc_ovr     = int((pred_ovr != y).sum())

# Sensitivity (recall) = percent of each class's actual cases that are
# predicted as that class. The three columns differ only in decision rule:
#   BINARY CUTOFF 0.5 - that class's own binary model alone; a case is the
#                       class if its probability >= 0.5 (full data, no
#                       sampling). Rare classes are mostly missed.
#   ONE-VS-REST       - all binary models; the highest probability wins.
#   MULTINOMIAL       - one model for all classes; highest probability wins.
print(f"{GOLD}{24*' '}SENSITIVITY (RECALL) BY DECISION RULE{RESET}")
print(f"{GOLD} CLASS.......... EVENT RATE  BINARY CUTOFF 0.5  ONE-VS-REST",
      f" MULTINOMIAL{RESET}")
for k, cls in enumerate(ovr.classes_):
    event   = (y == cls)
    p_bin   = ovr.estimators_[k].predict_proba(X)[:, 1]
    sens_b  = (p_bin[event] >= 0.5).mean()
    sens_o  = (pred_ovr[event] == cls).mean()
    sens_m  = (pred[event] == cls).mean()
    print(f" {TEAL}{cls:.<15s}{GREEN}{event.mean():9.1%}  {sens_b:17.1%}",
          f" {sens_o:11.1%}  {sens_m:11.1%}{RESET}")

# Choose the model with the higher accuracy (ties keep Multinomial)
models   = ["Multinomial", "One-vs-Rest"]
m_acc    = [accuracy_all, accuracy_ovr]
misc     = [misc_all, misc_ovr]
best_i   = 1 if accuracy_ovr > accuracy_all else 0
best_model = models[best_i]
print(f"\n{GOLD} MODEL               ACCURACY          MISC{RESET}")
for i in range(2):
    color = RED if i == best_i else TEAL
    print(f"{color} {models[i]:.<19s}{RESET} {GREEN}{m_acc[i]: 7.2%} ",
          f"    {misc[i]: 5d}/{n:5d}{RESET}")
print(f"{GOLD}Best model: {RED}{best_model}{RESET}")

#**************************************************************************
#****************************** HOLD-OUT VALIDATION ***********************
lbl = "STEP 5: HOLD-OUT VALIDATION"
print_boundary(lbl)

# Same X and y as Step 3 (all attributes, target not encoded)
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                    test_size=0.3, random_state=12345,
                                    stratify=y)
print(f"\n{GOLD}Predicting {RED}'{target}'{GOLD} with the",
      f"{RED}{best_model}{GOLD} model using {RED}'All'{GOLD} Attributes{RESET}")

lgr = best_lgr().fit(X_train, y_train)

confusion_heatmap_pair(lgr, X_train, y_train, X_val, y_val, target)

print(f"{GOLD}")
if best_model == "One-vs-Rest":
    # logreg.display_split_metrics needs a single LogisticRegression
    print("Training (70%)")
    print(classification_report(y_train, lgr.predict(X_train)))
    print("Validation (30%)")
    print(classification_report(y_val, lgr.predict(X_val)))
else:
    logreg.display_split_metrics(lgr, X_train, y_train, X_val, y_val)
print(f"{RESET}")
"""
Predicting 'activity' with the Multinomial model using 'All' Attributes

Model Metrics..............   Training   Validation
Observations...............      5796       2485
Coefficients...............       110        110
DF Error...................      5686       2375
Iterations.................        10         10
ASE........................    0.0469     0.0461
Root ASE...................    0.2166     0.2148
Mean Absolute Error........    0.0969     0.0951
Accuracy...................    0.8302     0.8318
Precision..................    0.7994     0.7930
Recall (Sensitivity).......    0.7582     0.7490
F1-score...................    0.7755     0.7671
Total Misclassifications...       984        418
MISC (Misclassification)...     17.0%      16.8%
     class sitting.........      1.0%       0.4%
     class sittingdown.....     38.4%      39.5%
     class standing........     16.5%      17.7%
     class standingup......     41.8%      47.8%
     class walking.........     23.1%      20.0%

Misclassification overfit ratio (16.8%/17.0%):  0.99
"""
# Sensitivity by class on training and validation data. Rare classes are
# where a model is most likely to hold up in training but fail validation.
pred_train = lgr.predict(X_train)
pred_val   = lgr.predict(X_val)
print(f"{GOLD} CLASS.......... N (VAL)  TRAIN SENSITIVITY  VAL SENSITIVITY{RESET}")
for cls in lgr.classes_:
    sens_t = (pred_train[y_train == cls] == cls).mean()
    sens_v = (pred_val[y_val == cls] == cls).mean()
    print(f" {TEAL}{cls:.<15s}{GREEN}{(y_val == cls).sum():7d}  {sens_t:17.1%}",
          f" {sens_v:15.1%}{RESET}")
print("")

# Examine Possible Overfitting
misc_train     = 1.0 - lgr.score(X_train, y_train)
misc_val       = 1.0 - lgr.score(X_val,   y_val)
if misc_train > 0:
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
lbl = "STEP 6: K-FOLD CROSS VALIDATION"
print_boundary(lbl)
print(f"{GOLD}Best model: {RED}{best_model}{RESET}")

for n_folds in range(2, 11):
    cv_model = best_lgr()  # lgr keeps the hold-out fit for Step 7
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=12345)
    scores  = cross_validate(cv_model, X, y,
                             scoring="accuracy",
                             cv=cv, return_train_score=True)
    print_acc_ratio(scores, n)
	
"""
 ====== 3-Fold Cross Validation =======
  Train Avg. MISC..... 0.1662 +/-0.0048
  Test  Avg. MISC..... 0.1703 +/-0.0084
  Mean Misc Ratio..... 1.0253 +/-0.0783
  ======================================= 
Equivalent to 3 splits each with 5521/2760 Cases
"""
#****************************** INTERPRETATION *****************************
lbl = "STEP 7: FINAL MODEL INTERPRETATION"
print_boundary(lbl)

cv_misc  = (1.0 - scores["test_score"]).mean()
cv_ratio = ((1.0 - scores["test_score"]) / (1.0 - scores["train_score"])).mean()
print(f"{GOLD}Final model: {RED}{best_model}{GOLD} logistic regression for",
      f"{RED}'{target}'{GOLD} ({len(lgr.classes_)} classes) using all",
      f"{RED}{X.shape[1]}{GOLD} attributes.")
print(f"{GOLD}Hold-out validation misclassification: {GREEN}{misc_val:.1%}",
      f"{GOLD}(overfit ratio {GREEN}{overfit_ratio:.2f}{GOLD}).")
print(f"{GOLD}{n_folds}-fold CV misclassification: {GREEN}{cv_misc:.1%}",
      f"{GOLD}(mean overfit ratio {GREEN}{cv_ratio:.2f}{GOLD}).{RESET}")

# For each class: validation sensitivity and the class it is most often
# mistaken for (share of that class's validation cases)
print(f"\n{GOLD} CLASS.......... VAL SENSITIVITY  MOST OFTEN MISTAKEN FOR{RESET}")
for cls in lgr.classes_:
    wrong = pd.Series(pred_val[(y_val == cls) & (pred_val != cls)])
    sens  = (pred_val[y_val == cls] == cls).mean()
    if wrong.empty:
        mistaken = "(none)"
    else:
        top      = wrong.value_counts().index[0]
        share    = (wrong == top).sum() / (y_val == cls).sum()
        mistaken = f"{top} ({share:.1%})"
    print(f" {TEAL}{cls:.<15s}{GREEN}{sens:15.1%}  {GOLD}{mistaken}{RESET}")
"""
Final model: Multinomial logistic regression for 'activity' (5 classes) 
using all 21 attributes.
Hold-out validation misclassification: 16.8% (overfit ratio 0.99).
10-fold CV misclassification: 17.1% (mean overfit ratio 1.02).

 CLASS.......... VAL SENSITIVITY  MOST OFTEN MISTAKEN FOR
 sitting........          99.6%   standingup (0.4%)
 sittingdown....          60.5%   standing (18.6%)
 standing.......          82.3%   walking (16.7%)
 standingup.....          52.2%   standing (23.1%)
 walking........          80.0%   standing (15.7%)
"""
lbl = "Analysis of Nominal Logistic Reg. Data Complete"
print_boundary(lbl)