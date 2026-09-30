# Week 5 Assignment - Analysis Context (Checklist Step 7: Memory)

## Statement of Intent
- **What:** Binary logistic regression identifying a speaker's gender from 20
  statistics of their voice recording (3,168 recordings, `Data/gender_voice_data.csv`).
- **Why:** Analysts auditing recordings need to know who the speaker is; gender
  narrows that identification. Importance ranges from low to high with the
  value of the recording.
- **How:** Step by step, following `Templates/Logistic_Reg/binary_logistic.py`
  (Sep 26, 2026 version) and `AdvancedAnalytics` (ReplaceImputeEncode, Regression).
  Run in Spyder, environment `anly656`.

## Files
| File | Purpose |
|---|---|
| `wk5_solution.py` | Solution program (run from the `Wk5_Assignment` folder) |
| `wk5_output.txt` | Program output (ANSI color codes removed) |
| `Data/gender_voice_data_map.py` | Data map drafted by `Tools/Data_Map_Tool.py`, bounds taken from the data dictionary |
| `l1.png`, `l2.png` | L1 / L2 coefficient and accuracy paths |
| `confusion_heatmap.png` | Training vs. validation confusion matrices of the best model |
| `wk5_opinion_review.pdf` | Independent review (checklist Step 8) |

## Data
- Target `gender`: Binary ('female', 'male'); RIE encodes female=0, male=1. Balanced 1,584 / 1,584.
- 20 Interval predictors; no missing values and no outliers against the dictionary bounds.
- Redundancies: `centroid` equals `meanfreq`; `IQR` = `Q75` - `Q25` and
  `dfrange` = `maxdom` - `mindom` (up to rounding). `skew`/`kurt` r = 0.98.
  The ALL model is therefore rank-deficient; its coefficients for those attributes are not interpretable.

## Results
| Model | Features | Accuracy (all data) | MISC |
|---|---|---|---|
| ALL | 20 | 97.38% | 83 / 3168 |
| **Stepwise** (best) | 7 | **97.54%** | 78 / 3168 |
| L1 (C=5.0) | 15 | 97.44% | 81 / 3168 |
| L2 (C=1.0) | 20 | 97.47% | 80 / 3168 |

- Stepwise features: `IQR`, `meanfun`, `minfun`, `kurt`, `sfm`, `spent`, `modindx`.
- 70/30 stratified hold-out: validation MISC 2.9% (train 2.5%), overfit ratio 1.19 (just under 1.2).
- Stratified k-fold (2-10): test MISC 2.5-2.7%, mean ratio 0.99-1.11; 10-fold MISC 2.5%, ratio 1.01. No overfitting.
- Strongest predictor: `meanfun` (mean fundamental frequency, i.e. pitch); coef -5.42 per std. dev.
  (odds ratio 0.004), so higher pitch strongly lowers the odds of 'male'. Next: `IQR` (+2.50), `sfm` (-1.74), `spent` (+1.53), `minfun` (+0.70).
- Hold-out error by class: female 3.6%, male 2.3%.

## Resume a session
Point the new agent to `wk5_solution.py`, `wk5_output.txt` and this file.
