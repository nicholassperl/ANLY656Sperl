# ANLY 656 Midterm Report  
## Predicting Sleep Deficit Among Software Engineers

**Data:** `sleep_rating.csv` (8,500 employees, 16 attributes)  
**Target:** `sleep_deficit` — nominal classes `0-none`, `mild`, `moderate`, `severe`  
**Objective:** Predict sleep-deficit level and identify attribute effects that help address its causes.  
**Programs:** `log_reg.py`, `tree.py`, `forest.py` (with matching `*_output.txt` files)

---

## 1. Problem and Approach

Sleep deficit among software engineers affects health and productivity. This analysis builds three classification models studied in ANLY 656 — multinomial logistic regression, a decision tree, and a random forest — using the same cleaned and encoded data.

**Data preparation (shared across models)**  
- Data map drafted with `Data_Map_Tool.py` and reviewed  
- `morning_alarm_snoozes` treated as **Interval** (ordered count)  
- Nominal one-hot encoding for gender, occupation, favorite bedtime app, and clock type  
- Target kept as class labels (`DT.String`)  
- ReplaceImputeEncode (RIE): 0 missing values and 0 outliers within map bounds  
- Encoded design matrix: **28 predictors** (forest later reduced features for tuning)

**Class imbalance**  
moderate 52.5% · mild 23.6% · 0-none 16.3% · severe 7.6%  
All hold-out and CV splits used stratification. The rare **severe** class is the hardest to recover.

Validation rules follow the course checklist: misclassification (MISC) and the ratio  
`MISC_validation / MISC_training`, with ratios above **1.2** flagged as possible overfitting.

---

## 2. Model Results

### 2.1 Logistic Regression (`log_reg.py`)

- RIE with `interval_scale="std"`  
- Compared **Multinomial** vs **One-vs-Rest**; selected **Multinomial** (97.89% vs 87.88% full-data accuracy)  
- Multinomial recovered `severe` far better than OvR on full data (≈86.6% vs ≈62.8% sensitivity)

| Metric | Training (70%) | Validation (30%) |
|--------|---------------:|-----------------:|
| Accuracy | 98.02% | **97.53%** |
| MISC | 2.0% | **2.5%** |
| Overfit ratio | — | **1.25** |

10-fold CV test MISC ≈ **2.5%** (mean overfit ratio ≈ 1.20).

**Validation sensitivity by class:** 0-none 99.3% · mild 99.0% · moderate 98.4% · **severe 83.5%**  
Severe cases were most often mistaken for moderate.

**Cause signals (coefficients):** Lower `total_sleep_hours`, higher `sleep_latency_min`, and more `morning_alarm_snoozes` associate with worse deficit classes (especially severe).

### 2.2 Decision Tree (`tree.py`)

- RIE without interval standardization  
- Kitchen-sink tree overfit (perfect train fit; 3.8% val MISC)  
- Grid search selected **gini, max_depth=7, min_samples_leaf=20, min_samples_split=20** (37 leaves)

| Metric | Training (70%) | Validation (30%) |
|--------|---------------:|-----------------:|
| Accuracy | 97.70% | **97.14%** |
| MISC | 2.3% | **2.9%** |
| Overfit ratio | — | **1.24** |

Best CV: **4-fold**, val MISC ≈ **2.71%** (ratio ≈ 1.21).

**Importance (causes):**  
1. `total_sleep_hours` (~87%)  
2. `morning_alarm_snoozes` (~10%)  
3. `sleep_latency_min` (~2%)  
Smaller roles for deep-sleep %, caffeine, brightness, and blue-light filter.

### 2.3 Random Forest (`forest.py`)

- Feature selection allowed by exam: kept features accounting for **99%** of DT importance → **9 of 28** predictors  
- Did **not** reduce the case count (all 8,500 observations used)  
- Grid search selected **entropy, n_estimators=50, max_depth=7, max_features=6, min_samples_leaf=39, min_samples_split=78**

| Metric | Training (70%) | Validation (30%) |
|--------|---------------:|-----------------:|
| Accuracy | 97.71% | **97.80%** |
| MISC | 2.3% | **2.2%** |
| Overfit ratio | — | **0.96** |

Best CV: **9-fold**, val MISC ≈ **2.51%** (ratio ≈ 1.19).

**Importance (causes):**  
1. `total_sleep_hours` (~82%)  
2. `morning_alarm_snoozes` (~14%)  
3. `sleep_latency_min` (~3%)  
Smaller: bedtime phone minutes, deep-sleep %.

---

## 3. Comparison

| Model | Val Accuracy | Val MISC | Overfit Ratio | Severe (notes) | Interpretability |
|-------|-------------:|---------:|--------------:|----------------|------------------|
| Multinomial logistic | 97.53% | 2.5% | 1.25 | Val sens. 83.5% | Coefficients (signed effects) |
| Decision tree | 97.14% | 2.9% | 1.24 | Val MISC ≈ 20% on severe | Rules / plot (depth 7) |
| Random forest | **97.80%** | **2.2%** | **0.96** | Val MISC ≈ 20% on severe | Importance ranks |

All three models agree that sleep quantity and bedtime/wake behaviors drive deficit class: **total sleep hours**, **morning alarm snoozes**, and **sleep latency**.

---

## 4. Strengths and Shortcomings

### Logistic regression
**Strengths:** Clear signed coefficients for interventions; multinomial beats OvR here; strong overall accuracy; useful per-class sensitivity reporting.  
**Shortcomings:** Hold-out overfit ratio slightly above 1.2; one-hot / collinear nominal blocks complicate coefficient interpretation; assumes linear log-odds (after scaling).

### Decision tree
**Strengths:** Transparent split rules and plot; importance ranking aligns with other models; kitchen-sink vs tuned contrast shows value of regularization.  
**Shortcomings:** Highest validation MISC of the three; overfit ratio still above 1.2; single tree is less stable than an ensemble; weak on rare severe class.

### Random forest
**Strengths:** Best hold-out accuracy and lowest MISC; overfit ratio under 1.2; feature selection shortened the model without dropping cases; stable CV performance.  
**Shortcomings:** Less directly interpretable than logistic coefficients or a single tree; grid search was computationally heavier; still struggles with severe vs moderate.

---

## 5. Preferred Model

**Preferred model: Random Forest.**

Reasons:
1. **Best validated accuracy / MISC** on the 70/30 hold-out (97.8% / 2.2%)  
2. **No overfitting** by the course ratio rule (0.96 vs 1.25 logistic and 1.24 tree)  
3. **Cause story is consistent** with the other models, with a short actionable feature list after selection  
4. Exam guidance explicitly allows forest feature selection, which we used without shrinking the sample

Logistic remains valuable for **signed effect interpretation** in a follow-up briefing. The tree is useful for **communicating simple rules**. For deployment of a single predictive model, the forest is the best balance of accuracy and generalization on these data.

---

## 6. Suggestions for Expanding This Work

1. **Cost-sensitive or balanced learning** for the rare severe class (class weights, threshold tuning, or focused sampling)  
2. **Ordered / ordinal models** treating deficit severity as ordered rather than unordered nominal  
3. **Intervention simulation** — estimate how changing sleep hours, snoozes, or latency shifts predicted class probabilities  
4. **External validation** on a later employee cohort or another company site  
5. **Policy packaging** — combine forest scores with logistic coefficients into an employee wellness checklist

---

## 7. Deliverables Checklist

| File | Role |
|------|------|
| `log_reg.py` / `log_reg_output.txt` | Logistic solution + output |
| `tree.py` / `tree_output.txt` | Decision tree solution + output |
| `forest.py` / `forest_output.txt` | Random forest solution + output |
| `midterm_report.pdf` | This summary report |

Supporting plots (confusion matrices, tree diagram, forest importance) are saved alongside the programs in the Midterm project folder.
