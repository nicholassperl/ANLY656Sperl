#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANLY 656 Midterm — Multinomial / One-vs-Rest Logistic Regression
Data: Data/sleep_rating.csv
Target: sleep_deficit (Nominal / String labels)
Data map drafted by Tools/Data_Map_Tool.py and reviewed for midterm.
Based on Templates/Logistic_Reg/nominal_logistic.py
"""
# ANSI color codes - to print in color, the package colorama must be installed
RED   = "\033[38;5;197m"; GOLD  = "\033[38;5;185m"; TEAL  = "\033[38;5;50m"
GREEN = "\033[38;5;82m";  RESET = "\033[0m"

import matplotlib
matplotlib.use("Agg")
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
    train_smisc = 2.0 * (1.0 - scores["train_score"]).std()
    val_misc    = (1.0 - scores["test_score"])
    val_smisc   = 2.0 * (1.0 - scores["test_score"]).std()
    ratios      = val_misc / train_misc
    ratio       = ratios.mean()
    s_ratio     = 2.0 * ratios.std()
    train_misc  = train_misc.mean()
    val_misc    = val_misc.mean()
    print(f"{TEAL}\n")
    print(f" ====== {n_folds:.0f}-Fold Cross Validation =======")
    print(f"  Train Avg. MISC..... {train_misc:.4f} +/-{train_smisc:.4f}")
    print(f"  Test  Avg. MISC..... {val_misc:.4f} +/-{val_smisc:.4f}")
    print(f"  Mean Misc Ratio..... {ratio:.4f} +/-{s_ratio:.4f}")
    print(" ", 39 * "=", f"{RESET}")
    n_v = n * (1.0 / n_folds)
    n_t = n - n_v
    print(f"Equivalent to {n_folds:.0f} splits each with " +
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
        plt.close()

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
    plt.close()

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

data_file = "sleep_rating.csv"
df        = pd.read_csv("Data/" + data_file)
print(f"{GOLD}Data loaded: {RED}{df.shape[0]}{GOLD} observations",
      f"{RED}{df.shape[1]} {GOLD}columns.{RESET}")

# Target is nominal. After RIE it is added back as class labels.
data_map = {
    'age':                       [DT.Interval, (17.99, 65.01)],
    'gender':                    [DT.Nominal, ('Female', 'Male', 'Non-Binary')],
    'bedtime_phone_minutes':     [DT.Interval, (0.99, 180.01)],
    'screen_brightness_pct':     [DT.Interval, (9.99, 100.01)],
    'blue_light_filter_active':  [DT.Binary, (0, 1)],
    'caffeine_post_5pm_mg':      [DT.Interval, (-0.01, 250.01)],
    'physical_activity_min':     [DT.Interval, (-0.01, 112.01)],
    'sleep_latency_min':         [DT.Interval, (5.99, 123.31)],
    'total_sleep_hours':         [DT.Interval, (3.19, 9.81)],
    'deep_sleep_pct':            [DT.Interval, (8.09, 28.01)],
    'rem_sleep_pct':             [DT.Interval, (9.59, 27.01)],
    'morning_alarm_snoozes':     [DT.Interval, (-0.01, 7.01)],
    'sleep_deficit':             [DT.String, ('0-none', 'mild', 'moderate', 'severe')],
    'occupation':                [DT.Nominal, (1, 2, 3, 4, 5)],
    'favorite_bedtime_app':      [DT.Nominal, (1, 2, 3, 4, 5, 6)],
    'clock_type':                [DT.Nominal, (1, 2, 3)],
}

print(f"{GOLD}", 15 * "=", "DATA MAP", 15 * "=")
lk = len(max(data_map, key=len)) + 1
ignored = 0
for col, (dt_type, valid_values) in data_map.items():
    if dt_type.name == "ID" or dt_type.name == "Ignore":
        ignored += 1
    print(f"  {TEAL}{col:.<{lk}s} {GOLD}{dt_type.name:9s}{GREEN}{valid_values}")
print(f"{GOLD} === Data Map has{RED}", len(data_map) - ignored,
      f"{GOLD}attribute columns", 3 * "=", f"{RESET}")

nominal = [col for col, (dt_type, _) in data_map.items()
           if dt_type.name == "Nominal"]
for nom in nominal:
    others = [col for col, (dt_type, _) in data_map.items()
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

target = "sleep_deficit"
print(f"{GOLD}")
rie = ReplaceImputeEncode(data_map=data_map,
                          interval_scale="std",
                          no_encode=[target],
                          binary_encoding="one-hot",
                          nominal_encoding="one-hot",
                          display=True)

encoded_df = rie.fit_transform(df)
encoded_df = pd.concat([df[target], encoded_df], axis=1)
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

# C=np.inf means no penalty.
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
    n_correct += conf_matrix[i, i]
n = y.shape[0]
accuracy_all = n_correct / n
misc_all = int(n - n_correct)
misc_p   = misc_all / n

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

print(f"{GOLD}{24 * ' '}SENSITIVITY (RECALL) BY DECISION RULE{RESET}")
print(f"{GOLD} CLASS.......... EVENT RATE  BINARY CUTOFF 0.5  ONE-VS-REST",
      f" MULTINOMIAL{RESET}")
for k, cls in enumerate(ovr.classes_):
    event  = (y == cls)
    p_bin  = ovr.estimators_[k].predict_proba(X)[:, 1]
    sens_b = (p_bin[event] >= 0.5).mean()
    sens_o = (pred_ovr[event] == cls).mean()
    sens_m = (pred[event] == cls).mean()
    print(f" {TEAL}{cls:.<15s}{GREEN}{event.mean():9.1%}  {sens_b:17.1%}",
          f" {sens_o:11.1%}  {sens_m:11.1%}{RESET}")

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

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                    test_size=0.3, random_state=12345,
                                    stratify=y)
print(f"\n{GOLD}Predicting {RED}'{target}'{GOLD} with the",
      f"{RED}{best_model}{GOLD} model using {RED}'All'{GOLD} Attributes{RESET}")

lgr = best_lgr().fit(X_train, y_train)

confusion_heatmap_pair(lgr, X_train, y_train, X_val, y_val, target,
                       plt_file='confusion_heatmap_holdout.png')

print(f"{GOLD}")
if best_model == "One-vs-Rest":
    print("Training (70%)")
    print(classification_report(y_train, lgr.predict(X_train)))
    print("Validation (30%)")
    print(classification_report(y_val, lgr.predict(X_val)))
else:
    logreg.display_split_metrics(lgr, X_train, y_train, X_val, y_val)
print(f"{RESET}")

pred_train = lgr.predict(X_train)
pred_val   = lgr.predict(X_val)
print(f"{GOLD} CLASS.......... N (VAL)  TRAIN SENSITIVITY  VAL SENSITIVITY{RESET}")
for cls in lgr.classes_:
    sens_t = (pred_train[y_train == cls] == cls).mean()
    sens_v = (pred_val[y_val == cls] == cls).mean()
    print(f" {TEAL}{cls:.<15s}{GREEN}{(y_val == cls).sum():7d}  {sens_t:17.1%}",
          f" {sens_v:15.1%}{RESET}")
print("")

misc_train = 1.0 - lgr.score(X_train, y_train)
misc_val   = 1.0 - lgr.score(X_val, y_val)
if misc_train > 0:
    overfit_ratio = misc_val / misc_train
else:
    overfit_ratio = np.inf
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
    cv_model = best_lgr()
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=12345)
    scores = cross_validate(cv_model, X, y,
                            scoring="accuracy",
                            cv=cv, return_train_score=True)
    print_acc_ratio(scores, n)

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

# Coefficients help identify attribute effects on sleep_deficit classes.
if best_model == "Multinomial":
    print(f"\n{GOLD}Multinomial coefficients (hold-out trained model):{RESET}")
    logreg.display_coef(lgr, X_train, y_train, col=list(X.columns))
else:
    print(f"\n{GOLD}One-vs-Rest binary coefficients by class:{RESET}")
    for k, cls in enumerate(lgr.classes_):
        print(f"\n{TEAL}=== Class: {cls} ==={RESET}")
        logreg.display_coef(lgr.estimators_[k], X_train, (y_train == cls).astype(int),
                            col=list(X.columns))

lbl = "Analysis of Sleep Deficit Logistic Reg. Complete"
print_boundary(lbl)
