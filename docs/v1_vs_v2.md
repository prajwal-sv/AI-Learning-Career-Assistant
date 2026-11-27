# V1 vs V2

Both versions use the same 316 training and 79 test students.

**The final models and their test scores are identical to V1.** XGBoost and tuning did not beat the V1 models in cross-validation, so the V1 models were kept.

## Predicting the final mark (test students)

| Measure | V1 | V2 | Change |
|---|---|---|---|
| Model | Random Forest | Random Forest |  |
| MAE | 1.026 | 1.026 | +0.000 |
| RMSE | 1.571 | 1.571 | +0.000 |
| R2 | 0.877 | 0.877 | +0.000 |

## Finding students who may need support (test students)

| Measure | V1 (threshold 0.5) | V2 (threshold 0.5) | V2 (threshold 0.5) |
|---|---|---|---|
| Model | Logistic Regression | Logistic Regression | Logistic Regression |
| Failing students found (recall) | 0.923 | 0.923 | 0.923 |
| Flags that were right (precision) | 0.774 | 0.774 | 0.774 |
| F1 (fail group) | 0.842 | 0.842 | 0.842 |
| Failing students caught | 24 | 24 | 24 |
| Failing students missed | 2 | 2 | 2 |
| False alarms | 7 | 7 | 7 |

The V2 threshold was chosen using the training students only: the highest threshold that still found at least 0.9 of the failing students.

## How much can we trust small differences?

The test set has 79 students, 26 of whom failed. One failing student moves recall by about 3.8 percentage points, and one student moves accuracy by about 1.3. Differences of one or two students are noise, not proof of a better model.

## Model comparison on the fail group (training students, cross-validated)

| Model | Recall | Precision | F1 | ROC AUC |
|---|---|---|---|---|
| Logistic Regression | 0.913 | 0.905 | 0.909 | 0.982 |
| Decision Tree | 0.904 | 0.87 | 0.887 | 0.958 |
| Random Forest | 0.885 | 0.885 | 0.885 | 0.975 |
| XGBoost | 0.923 | 0.897 | 0.91 | 0.978 |

## Threshold analysis (training students, cross-validated)

A student is flagged when the estimated fail probability is at or above the threshold.

| Threshold | Flagged | Caught | Missed | False alarms | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| 0.1 | 134 | 102 | 2 | 32 | 0.981 | 0.761 | 0.857 |
| 0.15 | 130 | 101 | 3 | 29 | 0.971 | 0.777 | 0.863 |
| 0.2 | 121 | 100 | 4 | 21 | 0.962 | 0.826 | 0.889 |
| 0.25 | 118 | 99 | 5 | 19 | 0.952 | 0.839 | 0.892 |
| 0.3 | 115 | 98 | 6 | 17 | 0.942 | 0.852 | 0.895 |
| 0.35 | 113 | 97 | 7 | 16 | 0.933 | 0.858 | 0.894 |
| 0.4 | 109 | 96 | 8 | 13 | 0.923 | 0.881 | 0.901 |
| 0.45 | 108 | 96 | 8 | 12 | 0.923 | 0.889 | 0.906 |
| 0.5 | 105 | 95 | 9 | 10 | 0.913 | 0.905 | 0.909 |
| 0.55 | 101 | 92 | 12 | 9 | 0.885 | 0.911 | 0.898 |
| 0.6 | 100 | 91 | 13 | 9 | 0.875 | 0.91 | 0.892 |
| 0.65 | 96 | 89 | 15 | 7 | 0.856 | 0.927 | 0.89 |
| 0.7 | 90 | 84 | 20 | 6 | 0.808 | 0.933 | 0.866 |
| 0.75 | 89 | 83 | 21 | 6 | 0.798 | 0.933 | 0.86 |
| 0.8 | 85 | 80 | 24 | 5 | 0.769 | 0.941 | 0.847 |

Predictions are estimates, not guarantees. They should support teachers and counselors, not replace them.
