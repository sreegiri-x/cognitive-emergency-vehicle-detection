# Cognitive EV Detection — Research Handoff

## Current status

The Kaggle run of the Adaptive Cognitive Pressure Field (A-CPF) pipeline completed. It ran five-fold subject-level cross-validation, trained the final model, generated six figures, and saved benchmark tables and checkpoints.

The log says the dataset contains 20 subjects and 80 global numeric features. Each cross-validation fold trains on 16 subjects and validates on 4 held-out subjects.

## Reported benchmark results

Mean ± standard deviation across five folds:

| Model | RMSE mean | RMSE std | MAE mean | MAE std | R² mean | R² std |
|---|---:|---:|---:|---:|---:|---:|
| A-CPF + Koopman + DFCG | 0.1976 | 0.0536 | 0.1584 | 0.0455 | -1.0774 | 0.7852 |
| Linear Ridge Regression | 0.1884 | 0.0580 | 0.1574 | 0.0481 | -0.8687 | 0.7393 |
| Mean Predictor | 0.1859 | 0.0585 | 0.1561 | 0.0483 | 0.0000 | 0.0000 |
| Vanilla GRU | 0.1980 | 0.0579 | 0.1623 | 0.0472 | -1.0790 | 0.8369 |

Interpretation: lower RMSE/MAE and higher R² are better. The proposed model currently does not beat the mean predictor or Ridge on RMSE/MAE. Its average R² is negative, so performance is worse than a mean-based prediction under the standard R² definition. Do not claim superiority yet.

## Training observations

- Training losses decreased, but validation performance often worsened.
- Early stopping occurred in each cross-validation fold around epochs 7–9, and during final-model training at epoch 10.
- The model restored each fold's best checkpoint, but generalization remains weak.
- A large number of overlapping sequences does not mean the same number of independent samples; the dataset has only 20 subjects. Subject-level evaluation is therefore important.

## Outputs reported by the run

- `/kaggle/working/outputs/results/benchmark_summary.csv`
- `/kaggle/working/outputs/results/model_comparison_by_fold.csv`
- `/kaggle/working/outputs/figures/` (six publication figures)
- `/kaggle/working/outputs/checkpoints/final/`

## Code-review plan (source code still needs inspection)

1. Inspect how the cognitive-pressure target is defined and aligned with each physiological/time-series window. Confirm which of NASA-TLX, Stress, DES and SAM contribute to the target and how they are scaled.
2. Audit the subject split and preprocessing. Fit scalers, imputation, feature selection and any learned transformations on training subjects only, then apply them unchanged to validation subjects.
3. Verify sequence-window generation and label alignment; confirm overlapping windows cannot cross subject or isolated-segment boundaries.
4. Verify RMSE, MAE and R² calculation and target inverse-scaling. Report metrics consistently in the original target scale where appropriate.
5. Examine per-fold and per-subject errors before changing the architecture.
6. Add controlled ablations: (a) Ridge, (b) vanilla GRU, (c) GRU plus A-CPF, (d) add Koopman, (e) add DFCG. Use identical splits and metrics.
7. Add or update tests for subject isolation, target alignment, and metric calculations. Save output tables and model/config metadata for reproducibility.

## Simple technical explanation

The model is learning the training subjects, because training loss goes down. However, it does not generalize well to unseen subjects, because validation errors remain high and its scores are worse than simple baselines. This is consistent with overfitting or a problem in target construction, preprocessing, or evaluation; the logs alone cannot tell which cause is responsible. We need to inspect the code before making a safe fix.

## Important limitation

This handoff is based on the Kaggle log only. The repository files and source code have not yet been inspected or changed. No GitHub commit has been made.
