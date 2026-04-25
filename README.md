![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

# Constraint-Aware, Robust and Explainable Vehicle Fuel Consumption Prediction

A reproducible machine learning framework for vehicle fuel-consumption prediction using physically informed constraints, interpretable modelling, and robustness testing under distribution shift.

This repository contains the full implementation of the MSc thesis:

> **Constraint-Aware, Robust and Explainable Vehicle Fuel Consumption Prediction**

---

## Project overview

This project investigates how physically informed constraints affect:

- predictive performance  
- interpretability  
- robustness under distribution shift  
- physical consistency of learned relationships  

The framework evaluates three machine learning paradigms across progressively constrained modelling stages.

---

## Research questions

This work addresses four core questions:

1. Do physical constraints improve robustness under distribution shift?  
2. How do interpretable linear models compare with ensemble models?  
3. Can static vehicle design features predict fuel consumption reliably?  
4. How stable are explanations under constrained learning?

---

## Modelling framework

The framework is structured into three experimental phases:

### Phase A — Baseline Modelling

Unconstrained models:

- Linear Regression  
- Lasso Regression  
- Random Forest  

---

### Phase B — Physics-Informed Feature Engineering

Introduces engineered physically meaningful features:

- power-to-weight proxy  
- mass-aerodynamic interaction  

---

### Phase C — Constraint-Aware Learning

Introduces explicit physical constraints:

**Linear models**
- non-negative coefficients  

**Random Forest**
- monotonic constraints  

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
│   └── constraint_aware_ml/
│
├── tests/
│   └── test_pipeline.py
│
├── data/
│   ├── raw/
│   │   ├── fuel_consumption_ratings/
│   │   └── vehicle_specifications/
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
│
└── docs/
    └── data_dictionary.md
```

---

## Dataset sources

This framework integrates two Canadian open datasets.

### Fuel Consumption Ratings Dataset

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

### Vehicle Specifications Dataset

Contains:

- engine size  
- cylinders  
- transmission  
- fuel type  
- curb weight  
- vehicle dimensions  

Coverage:

```text
2011–2023
```

---

## Data integration pipeline

The final analytical dataset was built through a fuzzy integration pipeline using:

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

Three robustness scenarios were tested.

---

### 1. In-domain evaluation

Train and test on city-cycle fuel consumption.

Stored in:

```text
results/tables/in_domain_city_results.csv
```

---

### 2. Manufacturer hold-out (OOD)

Ford excluded entirely during training.

Tests:

- out-of-distribution generalisation  
- manufacturer robustness  

Stored in:

```text
results/tables/manufacturer_holdout_ford_results.csv
```

---

### 3. Cross-cycle transfer

Train on city-cycle and test on highway-cycle.

Tests:

- transfer robustness  
- regulatory cycle shift  

Stored in:

```text
results/tables/highway_transfer_results.csv
```

---

## Key performance summary

### In-domain (best baseline)

| Model | Phase | R² | RMSE |
|---|---|---:|---:|
| Random Forest | A | 0.9738 | 0.5746 |
| Random Forest | B | 0.9739 | 0.5740 |
| Random Forest | C | 0.9069 | 1.0832 |

---

### Manufacturer hold-out (Ford)

Constraint-aware models reduce robustness degradation relative to unconstrained baselines.

---

### Cross-cycle transfer

Constraint-aware Random Forest shows improved transfer stability under cycle shift.

---

## Robustness metrics

Robustness is evaluated through:

```text
ΔRMSE = RMSE_shift − RMSE_in-domain
ΔR²   = R²_shift − R²_in-domain
```

Lower degradation indicates stronger robustness.

---

## Exploratory Data Analysis (EDA)

Stored in:

```text
results/figures/EDA/
```

Includes:

- model year distribution  
- missing values analysis  
- outlier inspection  
- vehicle make distribution  
- class distribution  
- transmission distribution  
- fuel type distribution  
- feature correlations  
- categorical relationship heatmaps  

---

## Interpretability analysis

Interpretability outputs are stored by phase.

---

### Phase A

```text
results/figures/PhaseA/
```

Includes:

- coefficient importance  
- feature importance  
- SHAP beeswarm  

---

### Phase B

```text
results/figures/PhaseB/
```

Includes:

- physics-informed coefficients  
- feature importance  
- SHAP beeswarm  

---

### Phase C

```text
results/figures/PhaseC/
```

Includes:

- constrained coefficients  
- feature importance  
- SHAP beeswarm  

---

## Main pipeline

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

Linear models:

```python
positive=True
```

Random Forest:

```python
monotonic_cst
```

---

### Explainability

- coefficients  
- feature importance  
- SHAP analysis  

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

Or use Conda:

```bash
conda env create -f environment.yml
conda activate constraint-aware-ml
```

---

## Run experiments

Launch notebook:

```bash
jupyter notebook notebooks/
```

Run tests:

```bash
pytest tests/
```

---

## Reproducibility

This repository includes:

- raw datasets  
- processed datasets  
- experiment outputs  
- figures  
- result tables  
- test scripts  
- environment specification  

to support full reproducibility.

---

## Skills demonstrated

- machine learning  
- constraint-aware learning  
- robustness testing  
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

## Citation

If you use this repository:

```text
Abuelella, E. H. M. (2026).
Constraint-Aware Machine Learning for Robust Vehicle Fuel Consumption Prediction.
MSc Thesis, Coventry University.
```

---

## Author

**Eslam H. M. Abuelella**  
MSc Data Science — Coventry University  
MSc Geology — Cairo University  

Machine Learning | Data Science | Geospatial Analytics | Earth Systems Modelling
