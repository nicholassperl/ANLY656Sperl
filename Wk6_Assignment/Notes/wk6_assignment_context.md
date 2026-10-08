# Wk6 Assignment Context — Credit Default Decision Tree

## Project
- **Folder:** `Wk6_Assignment/`
- **Solution:** `wk6_solution.py`
- **Output:** `wk6_output.txt`
- **Data:** `Data/CreditDefaultData.csv` (8,000 rows × 32 columns)
- **Target:** `Default` (Binary 0=no, 1=yes), balanced 4000/4000
- **Objective:** Identify customers likely to default on credit payments
- **Template followed:** `Templates/Decision_Tree/Binary_Target/DecisionTree_Forgery.py` (longer binary tree template)

## Data Map Decisions
- Drafted with `Tools/Data_Map_Tool.py`
- **Customer:** `DT.Ignore` (unique ID — not a predictor)
- **Education:** `DT.Nominal` (kept; course dictionary listed Ignore, but we retained it)
- Gender Binary; card_class & Marital_Status Nominal; remaining predictors Interval
- Missing before RIE: Gender 832, Education 1182, Age 1592; 0 outliers
- After RIE one-hot: `encoded_df` 8000 × 42 (41 predictors + target)

## How We Worked
- Step-by-step per Cursor Project Checklist
- Agent may run code in its environment; wait for instructions between steps (user later authorized Steps 4–8 together)

## Key Results
### Kitchen sink (Step 3)
- Depth 33, 1264 leaves; train accuracy 100%
- 70/30 holdout: train MISC 0%, val MISC 31.8% — severe overfitting

### Optimum tree (Steps 4–7)
- **criterion:** entropy  
- **max_depth:** 4  
- **min_samples_leaf:** 26  
- **min_samples_split:** 52  
- Refit depth 4, 16 leaves

### Holdout 70/30 (Step 5)
- Train acc 0.7682 / MISC 23.2%
- Val acc 0.7512 / MISC 24.9%
- MISC ratio **1.07** (passes ≤1.2 rule)
- Much better than kitchen sink (val MISC 31.8%)
- Class 1 (Default) harder: val MISC ~36.5% vs class 0 ~13.2%

### Cross-validation (Step 6)
- Best K: **4-Fold**
- Train MISC 0.2368, Test MISC 0.2416, ratio **1.02**

### Artifacts (Step 7)
- `CreditDefault_Tree.png`, `.pdf`, and DOT file `CreditDefault_Tree`

## Deliverables
- `wk6_solution.py`
- `wk6_output.txt`
- `wk6_opinion_review.pdf` (independent review)
