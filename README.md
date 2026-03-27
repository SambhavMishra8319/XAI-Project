## Stability of Feature Attributions under Class Imbalance
###  Project Overview

This project investigates the stability of feature attribution methods under varying levels of class imbalance in a credit card default dataset.

We replicate and analyze the methodology from a research paper that evaluates explanation stability using:

SHAP (SHapley Additive Explanations)
LIME (Local Interpretable Model-Agnostic Explanations)

The key metric used is:

SRA (Stability Rank Agreement)
### 🎯 Objective

To study how class imbalance affects explanation stability, and verify:

Whether stability improves as data becomes balanced
Whether SHAP is more stable than LIME
```
project/
│
├── data/
│   └── sampled/
│       ├── train_1.csv
│       ├── train_2.csv
│       ├── ...
│       └── train_50.csv
│
├── results/
│   └── fixed/
│       ├── shap_*.csv
│       └── lime_*.csv
│
├── generate_explanations_fixed.py
├── compute_sra_final.py
└── README.md
```
### ⚙️ Methodology
#### 1. Dataset Preparation
Pre-sampled datasets with different default rates:
```
1%, 2%, 5%, 10%, 15%, 20%, 30%, 40%, 45%, 50%
```
Target variable:
```
default (0 or 1)
```
#### 2. Model Training
Model used:
XG Boost
Trained separately for each imbalance level

### 3. Explanation Generation

For each dataset:

Select N target instances
Repeat multiple runs per instance
Apply input perturbation
Generate explanations using:
SHAP
### Perturbation Strategy

To ensure variability:
LIME
```
noise = np.random.normal(0, 0.2 * feature_std, x.shape)
x_perturbed = x + noise
```
This avoids unrealistic perfect stability.

### 4. Stability Computation (SRA)

We compute normalized SRA:

#### SRA = 1−observed deviation / maximum possible deviation

Range: 0 to 1
Higher = more stable

### 5. Evaluation Process

For each imbalance level:

Generate feature importance rankings

Compute rank variability across runs

Calculate SRA@K (K = 1 to 5)

Average across all target instances

## Results & Observations
#### 1. Effect of Class Imbalance

Stability is lower at high imbalance (1%)

Stability increases as data becomes balanced (50%)

✔ Matches research paper findings
#### 2. SHAP vs LIME
```
| Method | Observation                  |
| ------ | ---------------------------- |
| SHAP   | More stable, smoother curves |
| LIME   | Less stable, more variation  |
```
#### 3. Effect of K (Top Features)
SRA@1 > SRA@5

Top features are more stable

Lower-ranked features fluctuate more

## Output

### The project generates:

SRA vs Imbalance plots

Separate graphs for:

SHAP

LIME
## How to Run
#### Step 1: Generate Explanations
```
python generate_explanations_fixed.py
```
#### Step 2: Compute Stability
```
python compute_sra_final.py
```

Use sufficient runs for reliable stability:
```
N_TARGETS = 25
N_RUNS = 10
```
### Limitations
SHAP may appear overly stable due to:
deterministic nature of tree models
Results depend on perturbation strength
Not all experimental repetitions (as in paper) were implemented

### Conclusion
Explanation stability is strongly affected by class imbalance
Balanced datasets produce more reliable explanations
SHAP provides more consistent explanations than LIME

### Team Members
Bhoomika (23bcs030)

Sambhav Mishra (23bds050)

Sundaram (23bds060)
