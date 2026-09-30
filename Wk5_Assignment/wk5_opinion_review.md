# Week 5 Assignment — Independent Opinion Review

| | |
|---|---|
| **Project** | Identifying speaker gender from voice-recording statistics (binary logistic regression) |
| **Reviewer** | **Claude Opus 5**, acting as the independent Frontier reasoning model reviewer required by Checklist Step 8. The reviewer did not write the code under review. |
| **Review date** | September 30, 2026 |
| **Files reviewed** | `Wk5_Assignment.pdf`, `wk5_solution.py`, `wk5_output.txt`, `Data/Gender_Voice_Data_Dictionary.pdf`, `l1.png`, `l2.png`, `confusion_heatmap.png`, `Templates/Logistic_Reg/binary_logistic.py`, `Notes/Cursor_Project_Checklist.pdf` |

All numbers quoted below were taken from `wk5_output.txt` and, where noted, independently re-computed from `Data/gender_voice_data.csv` by the reviewer. A list of the verification checks and their results appears in the Appendix.

---

## 1. Summary of Results

The data are 3,168 voice recordings described by 20 interval audio statistics and a binary `gender` label, encoded `female = 0`, `male = 1`. The classes are exactly balanced (1,584 / 1,584), so the no-information baseline is 50.0 % accuracy. `ReplaceImputeEncode` reported 0 missing values and 0 outliers, and the single standardized encoding (`interval_scale="std"`) was reused by every model and by both validation paths.

Four candidate models were fit and compared on the full 3,168 cases:

| Model | Features | Accuracy | Misclassified | Notes |
|---|---|---|---|---|
| ALL | 20 | 97.38 % | 83 / 3168 | Rank-deficient design (rank 20 of 21 columns) |
| **Stepwise** | **7** | **97.54 %** | **78 / 3168** | Selected as best; all p-values ≤ 0.008 |
| L1 (lasso), C = 5.0 | 15 | 97.44 % | 81 / 3168 | Zeroed `meanfreq`, `Q25`, `centroid`, `maxdom`, `dfrange` |
| L2 (ridge), C = 1.0 | 20 | 97.47 % | 80 / 3168 | Ridge never zeroes a coefficient, so "20 selected" is automatic |

The stepwise model retained `IQR`, `meanfun`, `minfun`, `kurt`, `sfm`, `spent`, and `modindx` (deviance 560.46, Cox–Snell pseudo R² 0.7016, versus 554.85 and 0.7021 for the 20-feature model).

Validation of the chosen 7-feature model:

| Check | Training | Validation / Test | Ratio |
|---|---|---|---|
| 70/30 stratified hold-out (MISC) | 2.5 % (55 / 2217) | 2.9 % (28 / 951) | 1.19 |
| Stratified *k*-fold, *k* = 2 … 10 (MISC) | 2.46 – 2.53 % | 2.49 – 2.65 % | 0.99 – 1.11 |
| Stratified 10-fold (MISC) | 2.49 % | 2.49 % | 1.01 |

Hold-out errors by class were 3.6 % for female and 2.3 % for male speakers. The five largest standardized coefficients were `meanfun` −5.418 (odds ratio 0.004 per +1 SD = 0.0323 kHz), `IQR` +2.495 (12.13), `sfm` −1.737 (0.176), `spent` +1.534 (4.636), and `minfun` +0.701 (2.015).

**Headline:** a 7-predictor logistic regression classifies speaker gender at roughly 97.5 % accuracy — a 47.5-point lift over the 50 % baseline — with no evidence of overfitting. I reproduced both the 97.5379 % stepwise and 97.3801 % ALL accuracies exactly from the raw CSV, so the reported results are sound and repeatable.

## 2. Interpretation

**The model is essentially a pitch detector with refinements.** `meanfun`, the mean fundamental frequency, carries almost all of the signal: its coefficient of −5.418 per standard deviation means that lowering mean pitch by one standard deviation (0.0323 kHz, about 32 Hz) multiplies the odds that the speaker is male by exp(5.418) ≈ 225. This matches the physiology — adult male vocal folds are longer and heavier, producing fundamental frequencies near 100–120 Hz against 180–220 Hz for adult females. The lasso path in `l1.png` makes the same point graphically: the first coefficient to leave zero as the penalty relaxes is the large negative one, and a model with a single non-zero coefficient already reaches about 95.4 % accuracy (reviewer check). The remaining six predictors are refinements worth roughly two further percentage points.

**The secondary predictors describe spectral shape, not pitch.** `IQR` (+2.495) says that a wider interquartile spread of spectral energy favors "male"; `sfm` (spectral flatness, −1.737) and `spent` (spectral entropy, +1.534) act in opposite directions and are the two most mutually entangled survivors (variance inflation factors 5.7 and 6.9, the highest in the final model — reviewer check); `minfun` (+0.701) and `modindx` (−0.404) contribute small adjustments. Every VIF in the final seven is below 7, so the chosen model — unlike the ALL model — is not collinearity-compromised and its coefficients can be read.

**The overfitting verdict is correct, but for the right reason.** The k-fold table shows training and test misclassification essentially equal (2.49 % versus 2.49 % at k = 10). That equality of *level*, not the ratio column, is what establishes the absence of overfitting. The single hold-out ratio of 1.19 sitting just under the 1.2 trigger should not be read as a near-miss: re-running the same 70/30 split across 200 random seeds gives ratios from 0.46 to 1.85, with 32 % of splits exceeding 1.2 (reviewer check). The solution's own remark that "a single 70/30 split is noisy; the 10-fold ratio is the more reliable overfitting check" is exactly right, and the variability above quantifies it.

**The four models are a statistical tie, not a ranking.** The spread from best to worst is 5 misclassifications out of 3,168 (0.16 percentage points). A McNemar test comparing the stepwise and ALL predictions gives 8 versus 3 discordant cases, p = 0.23, and the likelihood-ratio test for the 13 features stepwise discarded gives a deviance drop of 5.61 on 12 effective degrees of freedom, p = 0.93 (reviewer checks). Threshold-free performance is identical: AUC 0.9941 for stepwise, 0.9942 for ALL. The honest conclusion is that all four models perform the same and the 7-feature model is preferable because it is parsimonious and identified — not because it is more accurate.

## 3. What Worked Well

**The collinearity handling is the strongest part of this submission.** The solution proactively scans for interval pairs with |r| ≥ 0.90 in Step 1, reports all seven of them, and then prints an explicit warning in Step 3 that the ALL design matrix has rank 20 of 21 columns, naming the culprits: `centroid` equals `meanfreq`, and `IQR = Q75 − Q25` and `dfrange = maxdom − mindom` up to rounding in the file. I verified all three claims — `centroid` is bit-identical to `meanfreq`, and the other two identities hold to within 1×10⁻⁹ and 4×10⁻⁹. Many submissions would have reported the resulting coefficients (`maxdom` = 1.85×10⁹, `Q25` = 9.04×10⁶) as findings; this one correctly states that they are not identified while the predictions and accuracy remain valid. That is the right diagnosis, communicated clearly in the output where a reader will see it.

**One encoding, used consistently.** Standardizing every interval predictor once and reusing that frame for all four models and both validation steps is the correct choice, and the in-code justification is accurate: standardization is *necessary* for L1 and L2 (the penalty only treats predictors alike when they share a scale) and *harmless* for ALL and stepwise, where it changes coefficient units but not selection, p-values, or predictions.

**Validation was set up carefully.** Both the 70/30 split and the k-fold splits are stratified on the balanced binary target, and the `best_lgr()` helper reuses the exact estimator settings from the model-fitting step, so Steps 8 and 9 validate the chosen model rather than an unpenalized look-alike. This is a real trap in the template workflow and the solution avoided it deliberately.

**The target encoding is documented and correct.** The data dictionary lists `gender` as `('male', 'female')`; the solution deliberately maps `('female', 'male')` so that `male = 1` is the modeled event, and says so in the docstring, the data map, and the Step 10 header. The sign of the `meanfun` coefficient independently confirms the mapping is what the output claims.

**Stepwise selection turned out to be remarkably stable.** Re-running the stepwise procedure inside each of 10 cross-validation folds produced only two distinct feature sets: six of the seven features were selected in 10 of 10 folds, and `kurt` in 8 of 10 with its near-duplicate `skew` (r = 0.977) substituting in the other two (reviewer check). The chosen feature set is not an artifact of one pass over the data.

**Reproducibility and presentation.** Random seeds are fixed for the splits, the folds, and the solvers; the output is saved with ANSI codes stripped; plots are written to file; and `Notes/wk5_assignment_context.md` records the analysis context as Checklist Step 7 requires. Every number I attempted to reproduce matched.

## 4. What Was Missed, Incorrect, or Limiting

**Model and hyperparameter selection were done on resubstitution accuracy.** This is the most consequential methodological problem. Steps 3–7 score all four candidates on the same 3,168 rows used to fit them, and Step 7 then crowns a winner by a 5-case margin. The C grids in Steps 5 and 6 have the same defect, and here it has a visible consequence: in-sample accuracy for lasso is flat (0.9729 – 0.9744) for every C ≥ 0.08, so the `if acc > best_acc` loop lands on C = 5.0 — nearly unpenalized — essentially by chance. Ten-fold cross-validation over the same grid is equally flat (0.9713 – 0.9722), and C = 0.08 delivers the same cross-validated accuracy with 8 non-zero coefficients instead of 15 (reviewer check). Selecting on validation performance would have produced a materially more parsimonious lasso model at no cost in accuracy.

**There is a latent bug in the best-model loop.** In Step 7, `best_i = 0` and `best_acc = 0`, and the loop runs `for i in range(1, 4)`. Because `m_acc[1]` is always greater than 0, index 0 — the ALL model — can never be selected as best even if it has the highest accuracy. The bug is inherited from the template and did not bite here, but it means the comparison is structurally rigged against one of the four candidates.

**Checklist Step 5, diagnostics, was effectively skipped.** The checklist explicitly asks for an analysis of standardized residuals: counts beyond ±2, ±3, and ±6 standard deviations, and plots of residuals against data order, against fitted values, and against Cook's distance. None of this appears in `wk5_solution.py` or `wk5_output.txt`. For a GLM fit this is nearly free — `glm_model.get_influence()` yields standardized deviance and Pearson residuals, leverage, and Cook's distance — and it would answer a question the current output leaves open: are the 78 misclassified recordings scattered noise, or a structured subgroup (for example young male or low-pitched female speakers) that a different model could capture? This is the clearest gap against the project's own checklist.

**Evaluation rests entirely on accuracy at a fixed 0.5 cutoff.** No ROC curve, no AUC, no precision–recall curve, no threshold sweep, and no calibration check appear anywhere. AUC is 0.994 and is threshold-free, so it would have strengthened the "models are equivalent" argument considerably. The hold-out also shows asymmetric errors — 3.6 % of female speakers misclassified versus 2.3 % of male — and a small threshold shift would trade one against the other; with no stated cost assumption, the choice of 0.5 is left implicit rather than justified.

**Feature selection sits outside the validation loop.** Stepwise ran on all 3,168 cases, and Steps 8 and 9 then validate that already-chosen feature set, so the reported 2.5 % cross-validated misclassification is conditional on a selection that saw the test folds. I quantified the optimism by re-running stepwise inside each fold: nested cross-validation gives 2.56 % against the reported 2.49 %. The bias is therefore real but small — about 0.07 percentage points — and the conclusion survives. Still, the output should say that the CV figure evaluates the *fitted coefficients* of a fixed 7-feature model, not the *selection procedure*.

**The "0 missing, 0 outliers" result is vacuous as a data-quality check.** The dictionary bounds are the data's own maxima rounded up: `skew` reaches 34.73 against a bound of 35.0, `kurt` reaches 1309.61 against 1310.0, `maxdom` reaches 21.87 against 22.0, `meandom` reaches 2.96 against 3.0. Because the screen was drawn around the observed range, it could not flag anything, so the clean report is guaranteed by construction rather than evidence of clean data. The output presents it as the latter.

**A sentinel value in `mode` went unexamined.** `mode` is exactly 0 in 236 recordings (7.4 %), which almost certainly encodes "no modal frequency detected" rather than a 0 kHz mode. Because 0 lies inside the declared (0, 1.0) bound, `ReplaceImputeEncode` accepted it as a real measurement and reported no missing data. It is not innocuous: the zero occurs in 10.1 % of male recordings against 4.8 % of female ones (reviewer check), so the model can use it as a quasi-indicator of a measurement failure that happens to correlate with the target. It should have been recoded as missing and imputed, or paired with an explicit indicator column.

**The rank deficiency was diagnosed but not fixed.** Flagging the problem was good; leaving it in the submitted output was not necessary. Dropping `centroid` (an exact duplicate of `meanfreq`), `dfrange`, and one of the `Q25`/`Q75`/`IQR` triple produces a full-rank ALL model whose largest coefficient is 5.4 rather than 1.85×10⁹, whose accuracy is slightly *better* at 97.44 %, and in which `Q25` (−2.63, p < 0.001) and `Q75` (+1.23, p = 0.010) become genuinely interpretable (reviewer check). One edit to the data map would have removed a table of uninterpretable numbers from the submission and added two real findings.

**Uncertainty is reported but not interpreted.** The ± figures in Step 9 are two standard deviations *across folds*, not standard errors of the mean, and nothing in the output says so. For the 10-fold run the test-MISC band is ±0.0170 around 0.0249 and the ratio band is ±0.7158 around 1.0116 — far wider than the distance from 1.0 to the 1.2 overfitting trigger. The ratio column therefore cannot distinguish overfitting from its absence at any k, which is worth stating plainly. Relatedly, printing nine separate k-fold runs for k = 2 … 10 adds length without adding information, since only k = 10 feeds Step 10.

**Labeling ridge as a selection method is misleading.** The Step 7 attribute table shows L2 "selecting" 20 of 20 features. L2 shrinks but never zeroes coefficients, so the |coefficient| > 0.0001 threshold necessarily retains everything; the column is uninformative by construction. The same threshold flatters the lasso: it counts `meandom` (0.0111), `mindom` (−0.0242), and `maxfun` (−0.0239) as "selected," so the honest count is closer to 12 features with |coefficient| ≥ 0.05 than the reported 15.

**Interpretation and limitations are thinner than Checklist Step 8 asks for.** Step 10 lists the top five coefficients by magnitude but stops short of a substantive reading: no confidence intervals on the odds ratios, no acoustic explanation of why `sfm` and `spent` matter, no statement of which measurements an operator could stop collecting, and no comparison against the 50 % baseline. The code comment at the top of Step 10 says that dividing by the attribute's standard deviation converts coefficients to original units, but the code does not do that — it prints the standard deviation alongside instead, which is defensible but does not match the comment. There is also no discussion of external validity, which matters here: the data come from one public corpus with the spectral analysis restricted to roughly 0–280 Hz (visible in `maxfun` ≤ 0.279 and `mode` ≤ 0.280), the classes are balanced 1,584 / 1,584 by construction rather than by natural prevalence — so precision and recall would both shift in deployment — and the label is a binary construct that does not describe every speaker. Given that the stated motivation is auditing recordings to identify speakers, a paragraph on appropriate use and on the 2.5 % error rate applied at scale belongs in the report.

**Minor code hazards.** `lr_plot` calls `set_params` and refits the estimator it is handed, so `lgr_l1` and `lgr_l2` are left fitted at C = 100 after the call; this is harmless only because every metric is computed before the plotting call, and `best_lgr()` constructs fresh estimators. In Step 6 the plot is called with the full `X` although the model was fit on `Xs` (identical here only because ridge kept all 20 columns). Neither `l1.png` nor `l2.png` has a legend identifying the 20 coefficient traces, which limits their usefulness. And `pred_class = list(map(round, pred_prob))` relies on Python's banker's rounding at exactly 0.5 — immaterial in practice, but `(p >= 0.5)` states the intent.

## 5. Recommendations for Improvement

These are ordered by how much they would change the analysis, not by effort.

1. **Select models and penalties on held-out performance.** Wrap the four candidates and both C grids in `GridSearchCV` or `cross_val_score` with `StratifiedKFold`, report cross-validated accuracy with its standard error, and break ties with a one-standard-error parsimony rule. On this data that rule selects lasso at C ≈ 0.08 with 8 features instead of C = 5.0 with 15, at identical accuracy.
2. **Present the four models as a tie and justify the choice on parsimony.** Add a McNemar test between each candidate and the ALL model (p = 0.23 for stepwise) and the likelihood-ratio test for the discarded features (p = 0.93), then state that the 7-feature model is preferred because it is smaller, identified, and stable — not because it is more accurate.
3. **Fix the rank deficiency in the data map.** Mark `centroid` as `DT.Ignore` since it duplicates `meanfreq` exactly, and drop one member of each linear triple (`Q25`/`Q75`/`IQR` and `maxdom`/`mindom`/`dfrange`). The ALL model then reports finite, interpretable coefficients at no accuracy cost.
4. **Add the Checklist Step 5 diagnostics.** Report counts of standardized deviance residuals beyond ±2, ±3, and ±6, plus Cook's distance and leverage from `glm_model.get_influence()`, with a residual histogram and residual-versus-fitted plot. Then profile the 78 misclassified recordings against the seven predictors to see whether the errors are structured.
5. **Report threshold-free and cost-aware metrics.** Add ROC/AUC (0.994) and a precision–recall curve, sweep the decision threshold with an explicit statement of the relative cost of a false "male" versus a false "female," and check calibration with a reliability plot. Give the class-specific error rates with binomial confidence intervals — 17/476 and 11/475 are small counts.
6. **Move feature selection inside the validation loop.** Either use a `Pipeline` whose selection step is refit on each training fold, or report the nested figure alongside the conditional one (2.56 % versus 2.49 % here) so the reader knows which quantity is being estimated.
7. **Treat `mode == 0` as missing.** Recode the 236 sentinel zeros to `NaN` and let `ReplaceImputeEncode` impute them, or add a `mode_missing` indicator. Then re-check whether `mode` still contributes.
8. **State honestly that the outlier screen did no work.** Note that the dictionary bounds equal the observed maxima, and if possible substitute independent physiological limits so the screen can actually reject a value.
9. **Replace the k = 2 … 10 sweep with repeated stratified 10-fold** (for example five repeats), reporting mean ± standard error of the mean. Drop the misclassification-ratio column or annotate it as too noisy to interpret at this error rate.
10. **Deepen the interpretation section.** Convert the leading odds ratios into a plain-language sentence (a 32 Hz drop in mean pitch multiplies the odds of "male" by about 225), attach confidence intervals, and report the accuracy of a deliberately minimal model — `meanfun` alone reaches roughly 95.4 % — as the deployment baseline against which the other six features must justify their collection cost.
11. **Add two reference points.** State the 50 % majority-class baseline explicitly, and fit one nonlinear benchmark to bound what the linear logit gives up. A random forest reaches 97.95 % and gradient boosting 97.70 % under the same 10-fold protocol, against about 97.5 % for logistic regression, which shows that the logistic model is close to this data's ceiling and that the residual errors are mostly irreducible rather than a modeling failure. For the record, I also tested log transforms of the heavily skewed `skew` and `kurt`; cross-validated accuracy fell slightly to 97.35 %, so that particular refinement is not worth pursuing.
12. **Add a limitations and appropriate-use paragraph** covering corpus provenance, the 0–280 Hz analysis range, the artificially balanced prevalence, the binary label construct, and the consequences of a 2.5 % error rate in the stated auditing application.

## 6. Overall Opinion

This is a strong, careful submission that is above average for the assignment. The headline result is correct and I reproduced it exactly: a 7-predictor logistic regression identifies speaker gender at about 97.5 % accuracy (AUC 0.994) against a 50 % baseline, with training and test misclassification both at 2.5 % under 10-fold cross-validation and therefore no evidence of overfitting. The dominant predictor, mean fundamental frequency, is the physiologically right answer, and the chosen feature set proved stable when I re-derived it independently inside each cross-validation fold.

What distinguishes the work is its intellectual honesty about collinearity. The solution finds the rank deficiency, names the three exact linear dependencies, and tells the reader in the output not to interpret the affected coefficients. That is a more sophisticated response than the assignment required, and it is the sort of thing that separates a result from a report.

The weaknesses are real but correctable and none of them overturn the conclusion. Ranked by importance: model and penalty selection were decided on resubstitution accuracy over a 5-case margin that is not statistically significant; the Checklist Step 5 residual diagnostics are absent; evaluation never leaves accuracy at a fixed 0.5 threshold; and the interpretation and limitations discussion is thinner than Checklist Step 8 asks for. The diagnosed-but-unfixed rank deficiency and the unexamined `mode = 0` sentinel are smaller items that would each take a few lines to resolve. If I were to prioritize a single revision, it would be recommendation 1 — selecting on cross-validated rather than in-sample accuracy — because it changes how the winning model is chosen, and with it the defensibility of every comparison in Step 7.

---

## Appendix — Reviewer Verification Checks

Run read-only against `Data/gender_voice_data.csv` with pandas, statsmodels, and scikit-learn; no project file was modified.

| Check | Result |
|---|---|
| Reproduce reported accuracies | Stepwise 0.9753788, ALL 0.9738005 — exact match to `wk5_output.txt` |
| Class balance | 1,584 female / 1,584 male; baseline accuracy 50.0 % |
| `centroid` = `meanfreq` | Identical, maximum absolute difference 0.0 |
| `IQR` = `Q75 − Q25` | Holds to 1.0×10⁻⁹ |
| `dfrange` = `maxdom − mindom` | Holds to 4.0×10⁻⁹ |
| AUC (threshold-free) | Stepwise 0.9941, ALL 0.9942 |
| McNemar, stepwise vs. ALL | 8 vs. 3 discordant, exact p = 0.227 — not significant |
| Likelihood-ratio test, 13 dropped features | Deviance drop 5.61 on 12 df, p = 0.93 |
| Nested 10-fold CV (stepwise re-run per fold) | 2.56 % MISC vs. 2.49 % reported — optimism ≈ 0.07 pp |
| Stepwise selection stability across 10 folds | Only 2 distinct sets; 6 features in 10/10 folds, `kurt` 8/10, `skew` 2/10 |
| Lasso C grid, in-sample vs. 10-fold CV accuracy | In-sample flat 0.9729–0.9744 for C ≥ 0.08; CV flat 0.9713–0.9722; C = 0.08 gives 8 features at CV 0.9719 vs. C = 5.0 giving 15 at 0.9719 |
| ALL model after dropping `centroid`, `dfrange`, `IQR` | Full rank; maximum \|coefficient\| 5.37 (was 1.85×10⁹); accuracy 97.44 %; `Q25` −2.63 (p < 0.001), `Q75` +1.23 (p = 0.010) |
| VIF within the final 7 features | `spent` 6.89, `sfm` 5.72, `IQR` 2.53, `kurt` 1.85, `meanfun` 1.74, `minfun` 1.28, `modindx` 1.12 |
| Hold-out overfit ratio across 200 random seeds | Mean 1.10, SD 0.26, range 0.46–1.85; exceeds 1.2 in 32 % of splits |
| `mode` sentinel zeros | 236 cases (7.4 %); 10.1 % of male vs. 4.8 % of female recordings |
| Dictionary bounds vs. observed maxima | `skew` 34.73 / 35.0, `kurt` 1309.61 / 1310.0, `maxdom` 21.87 / 22.0, `meandom` 2.96 / 3.0 — screen cannot reject |
| Nonlinear benchmarks, 10-fold CV accuracy | Random forest 97.95 %, gradient boosting 97.70 % vs. logistic ≈ 97.5 % |
| Log transform of `skew` and `kurt`, 10-fold CV | 97.35 % — no improvement, not recommended |

*Independent review produced by Claude Opus 5 for Checklist Step 8. Opinions are the reviewer's own and are based solely on the files listed above.*
