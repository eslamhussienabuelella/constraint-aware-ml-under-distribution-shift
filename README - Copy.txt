# Explainable and Constraint-Aware ML Under Distribution Shift

This repository contains the parsed and organised implementation for the MSc thesis **Explainable and Constraint-Aware Machine Learning for Robust Vehicle Fuel Consumption Prediction**.

## What is included

- Original notebook: `notebooks/Thesis_Submission.ipynb`
- Extracted reusable Python functions: `src/constraint_aware_ml/`
- Exported notebook figures: `results/figures/`
- Key result tables: `results/tables/`
- Methodology and findings summaries: `docs/`

## Experimental phases

| Phase | Description |
|---|---|
| A | Unconstrained baseline models |
| B | Physics-informed feature-level constraints |
| C | Constraint-aware model-level constraints |

## Key city-cycle results

| Phase | Model | RMSE | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| A | LR | 1.304 | 8.14 | 0.865 |
| A | RF | 0.575 | 2.92 | 0.974 |
| B | LR | 1.300 | 8.07 | 0.866 |
| B | RF | 0.574 | 2.95 | 0.974 |
| C | LR | 1.339 | 8.25 | 0.858 |
| C | RF | 1.083 | 5.91 | 0.907 |

## Main finding

Constraint-aware modelling reduces peak Random Forest accuracy but improves physical plausibility and reduces robustness gaps under manufacturer-level distribution shift.

## Repository structure

```text
data/
notebooks/
src/constraint_aware_ml/
results/tables/
results/figures/
docs/
tests/
```

## Installation

```bash
pip install -r requirements.txt
```

## Reproducibility note

The original notebook uses local Windows file paths. Add the raw CSV files to `data/raw/` and update the paths before re-running the full workflow.
