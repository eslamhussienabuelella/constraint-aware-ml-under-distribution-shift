![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

# Constraint-Aware Machine Learning Under Distribution Shift

A reproducible machine learning framework for vehicle fuel-consumption prediction using physically informed constraints, interpretable modelling, and robustness testing under distribution shift.

This repository contains the full implementation of the MSc thesis:

> **Explainable and Constraint-Aware Machine Learning for Robust Vehicle Fuel Consumption Prediction**

---

## Project overview

This project investigates how physical constraints affect predictive performance, interpretability, and robustness under out-of-distribution scenarios.

The framework is structured into three modelling phases:

### Phase A — Baseline Modelling

Unconstrained machine learning models:

- Linear Regression
- Lasso Regression
- Random Forest

---

### Phase B — Physics-Informed Feature Engineering

Adds physically meaningful engineered features:

- power-to-weight proxy
- mass-aerodynamic interaction

---

### Phase C — Constraint-Aware Learning

Imposes explicit physical constraints:

**Linear models**
- non-negative coefficients

**Random Forest**
- monotonic constraints

---

## Research objectives

This work addresses five practical machine learning challenges:

- target leakage prevention
- model interpretability
- physical consistency
- robustness under distribution shift
- early-design prediction feasibility

---

## Repository structure

```text
constraint-aware-ml-under-distribution-shift/

├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── notebooks/
│   └── full_experiment_pipeline.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── modelling/
│   ├── evaluation/
│   └── visualisation/
│
├── tests/
│   └── test_pipeline.py
│
├── data/
│   ├── raw/
│   │   ├── fuel_consumption_ratings/
│   │   │   ├── my1995-2014-fuel-consumption-ratings-5-cycle.csv
│   │   │   └── my2015-2024-fuel-consumption-ratings.csv
│   │   │
│   │   └── vehicle_specifications/
│   │       ├── 2011_en.csv
│   │       ├── ...
│   │       └── 2023_en.csv
│   │
│   └── processed/
│       ├── fc_veh_spec_all.csv
│       └── missing_vehicles.csv
│
├── results/
│   ├── figures/
│   │   ├── EDA/
│   │   ├── PhaseA/
│   │   ├── PhaseB/
│   │   ├── PhaseC/
│   │   ├── in_domain_city_cycle_performance_comparison.png
│   │   ├── out_of_distribution_ford_performance_comparison.png
│   │   └── highway_cycle_transfer_performance_comparison.png
│   │
│   └── tables/
│       ├── in_domain_city_results.csv
│       ├── manufacturer_holdout_ford_results.csv
│       └── highway_transfer_results.csv
```

---

## Dataset sources

The framework integrates two Canadian open datasets:

### 1. Fuel Consumption Ratings Dataset

Contains:

- city fuel consumption
- highway fuel consumption
- combined fuel consumption
- CO₂ emissions

Coverage:

```text
1995–2024
```

---

### 2. Vehicle Specifications Dataset

Contains:

- vehicle dimensions
- engine specifications
- curb weight
- transmission
- fuel type

Coverage:

```text
2011–2023
```

---

## Data integration pipeline

The processed dataset was created through a fuzzy data integration workflow using:

- text normalisation
- tokenisation
- Jaccard similarity matching
- tie-breaking logic
- quality-control validation

Workflow:

```text
Raw datasets
    ↓
Text cleaning
    ↓
Tokenisation
    ↓
Jaccard similarity matching
    ↓
Validation
    ↓
Integrated dataset
```

---

## Final dataset summary

| Metric | Value |
|---|---:|
| Observations | 11,086 |
| Variables | 32 |
| Model years | 2011–2023 |
| Hold-out manufacturer | Ford |

---

## Experimental design

Three evaluation settings:

### 1. In-domain evaluation

Training/testing on city-cycle data.

Results:

```text
results/tables/in_domain_city_results.csv
```

---

### 2. Manufacturer hold-out (OOD)

Ford completely excluded during training.

Tests:

- manufacturer generalisation
- robustness under brand shift

Results:

```text
results/tables/manufacturer_holdout_ford_results.csv
```

---

### 3. Cross-cycle transfer

Train on city-cycle → test on highway-cycle.

Tests:

- cross-regulatory transfer
- operational robustness

Results:

```text
results/tables/highway_transfer_results.csv
```

---

## Exploratory Data Analysis (EDA)

Stored in:

```text
results/figures/EDA/
```

Includes:

- model year distribution
- missing value analysis
- outlier detection
- weight analysis
- make distribution
- vehicle class distribution
- transmission distribution
- fuel type distribution
- feature correlation matrix
- categorical relationship heatmaps

---

## Model interpretability outputs

Stored by phase:

### Phase A

```text
results/figures/PhaseA/
```

Includes:

- LR/Lasso coefficients
- RF feature importance
- SHAP beeswarm

---

### Phase B

```text
results/figures/PhaseB/
```

Includes:

- physics-informed LR/Lasso coefficients
- RF feature importance
- SHAP beeswarm

---

### Phase C

```text
results/figures/PhaseC/
```

Includes:

- constraint-aware LR/Lasso coefficients
- RF feature importance
- SHAP beeswarm

---

## Main modelling pipeline

```text
Integrated dataset
        ↓
Train / Validation / Test split
        ↓
Feature preprocessing
        ↓
Model training
        ↓
Hyperparameter tuning
        ↓
Constraint application
        ↓
Evaluation
        ↓
Interpretability analysis
```

---

## Core methods

### Preprocessing

- robust scaling
- binary encoding
- feature engineering

---

### Models

- Linear Regression
- Lasso Regression
- Random Forest

---

### Constraints

Linear:

```text
positive=True
```

Random Forest:

```text
monotonic_cst
```

---

### Explainability

- model coefficients
- feature importance
- SHAP

---

## Installation

Clone repository:

```bash
git clone https://github.com/eslamhussienabuelella/constraint-aware-ml-under-distribution-shift.git
cd constraint-aware-ml-under-distribution-shift
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or environment:

```bash
conda env create -f environment.yml
conda activate constraint-aware-ml
```

---

## Run the notebook

```bash
jupyter notebook notebooks/
```

---

## Skills demonstrated

- machine learning
- constraint-aware learning
- robust modelling
- distribution shift evaluation
- explainable AI
- SHAP analysis
- feature engineering
- data integration
- reproducible pipelines

---

## Recommended GitHub topics

```text
machine-learning
constraint-aware-learning
physics-informed-ml
distribution-shift
robustness
explainable-ai
shap
random-forest
lasso-regression
feature-engineering
data-integration
python
```

---

## Repository description

Constraint-aware machine learning framework for robust vehicle fuel-consumption prediction under distribution shift using interpretable models, physics-informed features, and monotonic constraints.

---

## Author

**Eslam H. M. Abuelella**  
MSc Data Science — Coventry University  
MSc Geology — Cairo University

Machine Learning | Data Science | Geospatial Analytics | Earth Systems Modelling