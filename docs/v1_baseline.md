# V1 baseline: the numbers V2 has to beat

Source: models/registry/v1_metrics.json (frozen copy of the V1 metrics).
Data: UCI Student Performance (Math). 316 training students, 79 test students.
Both V1 models use feature set A (with G1 and G2).

## Predicting the final mark (0 to 20): Random Forest

| Measure | V1 test score | Baseline (always the mean) |
|---|---|---|
| MAE (average error, in marks) | 1.026 | 3.322 |
| RMSE (error, big misses count more) | 1.571 | 4.482 |
| R-squared | 0.877 | 0.0 |

Cross-validation RMSE (training students only): 1.558, spread 0.294.

## Predicting pass or fail: Logistic Regression

| Measure | V1 test score | Baseline (always "pass") |
|---|---|---|
| Accuracy | 0.886 | 0.671 |
| Fail group: recall | 0.923 | 0.0 |
| Fail group: precision | 0.774 | 0.0 |
| Fail group: F1 | 0.842 | 0.0 |
| Pass group: F1 | 0.911 | 0.803 |

Cross-validation F1 (pass group): 0.955, spread 0.008.

## Confusion matrix on the 79 test students

|  | Predicted fail | Predicted pass |
|---|---|---|
| Actually failed (26) | 24 | 2 |
| Actually passed (53) | 7 | 46 |

In plain words:
- V1 found 24 of the 26 students who failed (2 were missed).
- V1 also flagged 7 students who actually passed.

## How big a change is real?

The test set is small. One student is about 1.3 percentage points of accuracy
(1 of 79) and about 3.8 points of fail-group recall (1 of 26).
Differences smaller than this are noise. The mark model's cross-validation
spread (0.294 RMSE) also means a tiny RMSE gain is not proof of a better model.