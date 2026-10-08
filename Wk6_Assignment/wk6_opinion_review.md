# Independent Opinion Review

**ANLY656 Week 6 — Binary Decision Tree for Credit Default**

| | |
|---|---|
| Reviewer | Grok 4.7, independent opinion review |
| Scope | Evaluation of the analysis that was produced. The solution was not rewritten, and the model was not re-run. |
| Materials | `wk6_solution.py`, `wk6_output.txt`. Assignment brief `Wk6_Assignment.pdf` and `Notes/wk6_assignment_context.md` used only as context. |
| Date | 8 October 2026 |

---

## 1. Interpretation of the results

The program builds a classification tree to flag bank customers who default on card payments. The file has 8,000 accounts and 32 columns. The target `Default` is exactly balanced: 4,000 defaults and 4,000 non-defaults. `Customer` is removed as an identifier. After `ReplaceImputeEncode`, the modeling table has 8,000 rows and 41 predictors.

Three fields arrive with missing values and nothing else does: Gender 832 (10.4%), Education 1,182 (14.8%), and Age 1,592 (19.9%). The preprocessor reports zero outliers. The target is left unimputed, which is the right choice.

### The unrestricted tree memorizes the sample

Fit on all 8,000 rows with default scikit-learn settings, the tree reaches depth 33 and 1,264 leaves and records perfect training accuracy. That fit is a memorization check, and it behaves like one.

On a stratified 70/30 split (5,600 / 2,400, same seed later used for the tuned tree), training misclassification stays at 0% and validation misclassification rises to 31.8% (764 of 2,400). Validation accuracy is 68.2%, precision for the default class is 0.675, and recall is 0.701. Because training error is zero, the misclassification ratio is infinite. The validation errors are roughly even across classes (33.8% of non-defaults and 29.9% of defaults). An unrestricted tree on these 41 features is not a usable screen.

### The selected tree is shallow, stable, and only moderately accurate

The search uses 4-fold stratified cross-validation over Gini and entropy, depths from 4 through unlimited, and leaf sizes tied to twice the minimum split size. A coarse grid (126 combinations, 504 fits, about 7 seconds) is followed by a one-leaf-at-a-time scan from leaf size 15 to 60.

Both the lowest validation error in the grid and the tree that passes the course overfitting rule land on the same shape: entropy and maximum depth 4. The rule-selected leaf size is 26 (minimum split 52). Its cross-validated misclassification is 23.68% in the training folds and 24.16% in the test folds, ratio 1.02. Refit on all 8,000 rows, that specification grows a depth-4 tree with 16 leaves.

A binary tree cannot have more than 16 leaves once depth is capped at 4, and it has exactly 16 only when every leaf sits on the bottom level. The depth cap is the constraint that is actually shaping the model. The leaf minimum of 26 did not cut any branch short on the full sample.

The later 70/30 holdout agrees with the cross-validation. Training accuracy is 0.7682 (misclassification 23.2%, 1,298 of 5,600). Validation accuracy is 0.7512 (misclassification 24.9%, 597 of 2,400). The misclassification ratio is 1.07, inside the 1.2 limit used in the code. Against a 50% coin-flip baseline on this balanced file, the holdout cuts the error rate about in half: from 50% to 24.9%.

Reading the validation confusion matrix as a screening rule:

|  | Predicted no default | Predicted default |
|---|---:|---:|
| Actually no default (1,200) | 1,041 | 159 |
| Actually default (1,200) | 438 | 762 |

When the tree says “default,” it is right 762 / 921 times (precision 0.827). It finds 762 of 1,200 defaults (recall 0.635) and misses 438. It falsely flags 159 of 1,200 good accounts (13.2%). Specificity is 0.868. Class-wise validation misclassification is 13.2% for non-defaults and 36.5% for defaults, and both class ratios stay at or under 1.19.

Compared with the kitchen-sink tree on the same split, overall validation error falls from 31.8% to 24.9% (167 fewer mistakes on 2,400 accounts). The gain is uneven. Non-default error falls from 33.8% to 13.2%. Default error rises from 29.9% to 36.5%. The pruned tree is a more conservative screen: fewer false alarms, and also fewer caught defaults. Kitchen-sink recall was 0.701; the selected tree’s recall is 0.635. That trade is the main business result in the output, and the class-wise table makes it visible.

The printed mean absolute error (0.341 validation) and average squared error (0.175 validation) are larger and smaller, respectively, than the 0.249 misclassification rate in a way that lines up with probability scoring. On the kitchen-sink tree, where every leaf is pure, those two figures equal the misclassification rate (0.318). On the pruned tree, leaves are mixed, predicted probabilities sit away from 0 and 1, and squared error drops below the hard-label error. The validation squared error of 0.175 is the Brier score of P(Default). A constant forecast of 0.50 on a balanced sample has Brier score 0.25, so the leaf probabilities improve on a null forecast. They are still coarse: average absolute probability error remains about 0.34.

### Repayment status accounts for almost all of the splits

On the refit tree, impurity reduction is concentrated:

| Feature | Importance |
|---|---:|
| Jun_Status | 0.792 |
| Mar_Status | 0.071 |
| May_Status | 0.053 |
| May_Payment | 0.039 |
| Credit_Limit | 0.015 |
| Gender | 0.009 |
| Jan_Status | 0.008 |
| Mar_Payment | 0.004 |
| May_Bill | 0.004 |
| Feb_Payment | 0.003 |

The top ten features sum to about 0.997. Four month-status variables alone sum to 0.923. June’s status supplies 79% by itself. Card class, education, marital status, age, and most bill and payment-percent columns do not appear. For a credit book, that is a coherent story: how late the account already is dominates who the customer is and what the balance looks like. The kitchen-sink importances tell a softer version of the same story (June status 0.25, then credit limit, payments, and age), because a depth-33 tree can spend impurity reduction on many correlated substitutes. The 79% figure is the share of impurity reduction inside this one shallow tree. It is evidence that the first splits are repayment-status splits. It is weak evidence that the other months would be useless if June were removed.

### Cross-validation says the error is stable across fold counts

With the selected hyperparameters held fixed, stratified K-fold for K from 2 to 10 produces test misclassification between 0.2416 (4-fold) and 0.2482 (8-fold). Every mean ratio sits between 1.02 and 1.04. Training error stays near 0.236 to 0.239. The program then names 4-fold the winner because it has the lowest test error, and it reprints accuracy 0.7632 / 0.7584.

That 4-fold number is the same pair already reported by the hyperparameter search, which itself used 4-fold cross-validation and the same random seed. The sweep is still useful: it shows that the error barely moves when the fold count changes. The spread of 0.66 percentage points is smaller than the fold-to-fold standard deviation at K=4 (0.72 points) and much smaller than the standard deviations at K=8 to 10 (about 2 points). Declaring a best K overstates a difference the table itself shows is flat.

A reviewer standard error for the single holdout accuracy is about `sqrt(0.7512 × 0.2488 / 2400) ≈ 0.009`. The kitchen-sink gap of about 7 points is large relative to that noise. The fine-scan gap that picked leaf 26 over leaf 30 is 0.03 points (validation misclassification 0.2416 versus 0.2419), and leaves 26, 27, and 28 are identical through four decimals. The selected leaf size sits on a plateau, not on a peak.

---

## 2. What worked well

The solution follows the full binary-tree sequence the assignment asks for: a documented data map, imputation and encoding, an unrestricted reference tree, a two-pass hyperparameter search, a stratified holdout, a K-fold sweep from 2 to 10, and a saved picture of the tree. Seeds are fixed at 12345, splits are stratified, the target is excluded from imputation, and the customer identifier is excluded from the predictors. Those choices are visible in the code and are reflected in the output.

The overfitting comparison is done properly. The kitchen-sink tree and the selected tree are scored on the same 70/30 split, and the output prints overall and class-wise misclassification for both. A reader can see both the drop in overall error and the shift of error from good accounts onto defaulters. The search also reports the lowest validation error with the overfitting rule turned off. That tree (entropy, depth 4, leaf 30) is almost the same as the rule-selected tree (leaf 26). Showing both makes it clear that the rule is not hiding a deeper, more accurate model inside this grid.

The arithmetic in the output hangs together. Holdout counts, rates, precision, recall, and the 1.07 ratio all match the confusion matrix. The cross-validated ratio of 1.02 and the holdout ratio of 1.07 tell the same story: once depth and leaf size are constrained, training and validation error move together. That is the result the kitchen-sink section was built to contrast, and the contrast is large enough to trust (about 7 points, against a holdout standard error near 1 point).

The feature ranking is credible for this kind of file. A shallow tree that spends almost all of its splits on months-behind status, and that ignores card class and education, is what a prudent credit screen often looks like. Encoding leaves the interval predictors unscaled, which is appropriate for a tree. The grid is small enough to finish in seconds and large enough to cover criterion, depth, and a wide leaf-size range, including a second pass at unit resolution around the coarse winner.

Reproducibility of the run record is good. The output states the grid, the timings, the selected parameters, the refit size (depth 4, 16 leaves), and the paths of the saved tree files.

---

## 3. Shortcomings and limitations

**The depth grid cannot choose a simpler tree.** Every depth in the search is 4 or greater, and the winner is 4. Combined with a perfect 16-leaf tree, that means the lower bound of the grid is an active constraint. Depths 1, 2, and 3 were never scored. Given that one feature carries 79% of the impurity reduction, a two- or three-level tree, or a June-status stump, could sit close to 24% error and would be easier to explain to the bank. The analysis cannot say so, because those models are outside the grid.

**The leaf-size “optimum” is a tie.** Leaves 26, 27, and 28 share training misclassification 0.2368, validation misclassification 0.2416, and ratio 1.0204. The code keeps the first index. Leaf 30, the unconstrained grid winner, is 0.0003 worse, which is far inside the 4-fold test-error standard deviation of 0.0072. Reporting a single optimum overstates the precision of the search.

**The overfitting rule does little work once the tree is regularized.** The rule passes a model when validation misclassification is at most 1.2 times training misclassification, or at most 0.5 points higher. Around a training error of 0.24, the ratio clause allows validation error up to about 0.28. Essentially every leaf size in the fine scan passes, so selection collapses to “lowest validation error.” The rule does the job it was given on the kitchen-sink tree, where the ratio is infinite. It does not separate the serious candidates from each other.

**Hyperparameters were chosen on the same rows later used as the holdout.** Grid search sees all 8,000 accounts. The 70/30 split is drawn afterward, with the same seed as the kitchen-sink split. The holdout is a useful second measurement — 24.9% versus a cross-validated 24.2% — and it did not come back wildly optimistic. It is still a reuse of rows that already influenced the chosen depth, criterion, and leaf size. A locked 30% would have been the cleaner confirmation.

**Naming 4-fold the best K repeats the tuning criterion.** The hyperparameters minimize 4-fold error. The K sweep then discovers that 4-fold error is smallest, and the reprinted summary is identical to the search summary. The substantive finding is the narrow band from K=2 to K=10. The label “best K” is stronger than that band.

**The operating point is an unexamined 50/50 vote.** The assignment asks for a model that identifies customers likely to default. On the holdout the selected tree misses 36.5% of defaults in exchange for flagging only 13.2% of good accounts. No probability threshold, no cost ratio, and no expected-loss table is reported. If a missed default costs more than a false alarm, the leaf vote at one half is the wrong cutoff even if the tree splits are right. The sample is also exactly balanced, while customer identifiers are valid up to about 30,000 and only 8,000 rows are present. Precision of 0.83 is the precision at a 50% default rate. A different base rate in the full book would move precision and would move the value of this cutoff.

**Education is kept against the data dictionary, without a sensitivity check.** The dictionary types Education as Ignore. The code types it as nominal with levels 0 through 6 and says so in the header comment. It never enters the top ten importances, so the selected splits almost certainly ignore it. The output still does not show the with-and-without comparison, and 1,182 education values are missing, with several codes that are typically residual categories. Keeping the column is a real deviation from the dictionary and should have been measured.

**Interval limits look like padded sample ranges, so “zero outliers” is a weak audit.** The dictionary describes age on roughly 20–80, credit limit up to 800,000, and bills on a much wider scale than the map. The map uses tight bounds such as age (20.99, 75.01) and credit limit (299.99, 27,400.01). With bounds wrapped around the observed sample, the outlier count is expected to be zero. The screen does not test the dictionary domains. May status is capped at 7.01 while the other status fields allow 8, which is another sign the bounds follow this file rather than the stated scale.

**Missingness is filled in before the tree can see it.** About one in five ages is imputed, along with a tenth of gender and a seventh of education. Filling age with a typical value stacks those customers on a single point and removes any signal carried by “age was missing.” A tree can split on a missingness flag. This pipeline does not give it one.

**Importances are easy to over-read.** June, May, April, March, February, and January status measure the same behavior one month apart. Bill, payment, and payment-percent are algebraic relatives. A single tree assigns the split to whichever cousin reduces impurity first. June’s 79% can shrink if June is dropped and May or March takes the root. The output provides no correlation table and no drop-one check.

**The delivered picture hides the last level of the tree.** The chosen model is depth 4 with 16 leaves. The plot, the PDF figure, and the DOT export are all drawn with `max_depth=3`. The text log never lists the rules. A reader of `wk6_output.txt` can see which variables matter and cannot see the thresholds a credit officer would apply. The displayed tree is also the refit on all 8,000 rows, while the holdout metrics come from a tree fit on 5,600. Those thresholds can differ. The log does say the picture is the full-sample refit, which keeps the record honest, and it still leaves the rules unpublished.

**The metric block can be misread.** Mean absolute error of 0.34 sits in the same table as misclassification of 24.9%. Nothing in the output says that the absolute and squared errors are scored on predicted probabilities. A reader can think the model has two conflicting error rates.

**Redundant encoding is unused insurance.** Nominal one-hot encoding is requested with `drop=False`, which duplicates a full set of category columns. Trees tolerate that. It still widens a 30-predictor file to 41 columns and can split importance across dummies. In this run the nominal fields stayed out of the top ten, so the extra columns did not drive the result.

---

## 4. Recommendations

1. **Extend the depth grid downward and publish the comparison.** Score maximum depths 1, 2, and 3, and score a tree that is allowed to use only `Jun_Status`. Put those validation errors next to the depth-4 result. If a two-level tree is within about one holdout standard error (about one accuracy point) of 75.1%, prefer the smaller tree and explain it in the write-up.

2. **Report the leaf-size plateau.** State that leaf sizes 26, 27, and 28 are tied at validation misclassification 0.2416, and that this margin is smaller than the cross-validation standard deviation. Keep one of them as the working model and stop describing the third decimal as an optimum.

3. **Tune inside the training split.** Draw the stratified 70/30 split first. Run the grid only on the 5,600. Touch the 2,400 once, after the parameters are frozen. Keep the kitchen-sink comparison on that same split.

4. **Add the decision the bank actually needs.** From the holdout probabilities, tabulate recall, false-alarm rate, and precision at several cutoffs (for example 0.3, 0.4, 0.5, 0.6). If a missed default and a false alarm have different costs, mark the cutoff that minimizes expected cost. Lead the conclusion with default recall, not only with overall accuracy. Overall accuracy treats the 438 missed defaults and the 159 false alarms as the same kind of mistake.

5. **State the baseline and the base rate.** Accuracy of 75.1% should be placed next to the 50% majority-class baseline on this file. Note that the classes were constructed to be equal (4,000 and 4,000) inside an identifier range that runs to about 30,000, so precision will change if the live book has a different default rate.

6. **Reconcile Education with the dictionary.** Refit the selected specification with Education excluded, and report the change in validation misclassification and in the top importances. If the change is negligible, say so and either drop the column or document why it stays. Do the same check for a missingness indicator on Age, Gender, and Education.

7. **Rebuild the interval bounds from the dictionary domains, then rerun the outlier report.** Age, credit limit, bills, and payments should be screened on the ranges in the data dictionary. Sample-padded bounds can remain as a descriptive note. They should not be the only screen that supports the claim of zero outliers.

8. **Print the rules for all four levels.** Export the plot and the DOT file without a depth cutoff, and write the 16 leaf rules into the text output with their sample shares and predicted class. Qualify the importance list in one sentence: the other month-status fields are substitutes, and a drop-June refit is the check on whether 79% means “June specifically” or “recent delinquency.”

9. **Label the probability errors.** In the summary that accompanies the metrics, identify validation average squared error 0.175 as a Brier score and the 0.34 absolute error as probability error, and place the 0.25 Brier score of a constant 50% forecast beside them.

10. **Describe the K sweep as a stability check.** Report the band 24.2% to 24.8% test misclassification for K from 2 to 10. There is no need to crown a best fold count after the model was already chosen by 4-fold search.

---

## 5. Overall opinion

This is a complete and internally consistent Week 6 analysis. The unrestricted tree is shown to memorize the training rows (depth 33, 1,264 leaves, 0% training error, 31.8% validation error). The constrained tree is a depth-4 entropy tree with 16 leaves, about 75% holdout accuracy, a misclassification ratio of 1.07, and the same error, within a point, under every fold count from 2 to 10. The gains over the unrestricted tree are real. The structure of those gains is also real: the model becomes much better at clearing good accounts and somewhat worse at catching defaults, and almost every split it uses is a repayment-status split, led by June.

The claims that should be softened are the ones the search design cannot support. Leaf size 26 is a tie, depth 4 is the smallest depth that was tried, 4-fold is the fold count already used to tune the model, and 75% accuracy is accuracy on a deliberately balanced extract at a default one-half voting cutoff. The picture of the tree omits its last level, and the output never writes the rule a lender would use. The next improvement is interpretive and procedural: a shallower comparison model, a threshold table aimed at missed defaults, a true holdout, and the four levels of rules printed in the log. The fitted model itself is a reasonable, stable screen for this assignment.
