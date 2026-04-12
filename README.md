# Stability of Feature Attributions under Class Imbalance

## Project Overview
This project studies how class imbalance affects the stability of feature attribution methods in credit risk models.

We replicate and extend the findings of Chen et al. (2024), focusing on:
- SHAP (SHapley Additive Explanations)
- LIME (Local Interpretable Model-Agnostic Explanations)

Stability is evaluated using:
- SRA (Stability Rank Agreement)
- CV (Coefficient of Variation)

---

## Objective
- Analyze how explanation stability changes with class imbalance
- Compare SHAP vs LIME stability
- Evaluate resampling techniques
- Identify root causes of instability

---

## Project Structure
project/
├── data/
├── results/
├── replicate_paper.py
├── phase3_extension.py
├── requirements.txt
└── README.md

---

## Methodology

### Dataset
- Taiwan Credit Card Dataset
- South German Credit Dataset

Default rates tested: 1% → 50%

### Model
- XGBoost classifier

### Explanation Methods
- SHAP
- LIME

### Stability Metrics
- SRA (ranking stability)
- CV (value stability)

Interpretation:
- Higher CV → less stable
- Higher SRA → less stable

---

## Phase 2: Replication Results
- Stability decreases at extreme imbalance (1–5%)
- Stability improves as data becomes balanced
- SHAP is more stable than LIME
- Top features are more stable

---

## Phase 3: Extensions

### Resampling Strategies
- Baseline
- SMOTE
- ADASYN
- Random Under-Sampling (RUS)
- Cost-sensitive XGBoost

Finding:
- Minimal improvement
- RUS worsens stability
- All converge at balanced data

---

### Experiment C: Class-Based Stability
- Compared defaulters vs non-defaulters

Finding:
- No meaningful difference
- Instability is system-wide

---

### Experiment D: SHAP Background Dataset
- Modified SHAP background only

Finding:
- Largest improvement (~8% CV reduction)
- Better than resampling
- No retraining needed

Insight:
Instability is driven by background dataset imbalance

---

### Experiment E: Feature-Level Stability
Finding:
- Payment history features = most unstable
- Financial amount features = more stable
- Critical features are least reliable

---

## How to Run

Install dependencies:
pip install -r requirements.txt

Run replication:
python replicate_paper.py --dataset both

Run extensions:
python phase3_extension.py --dataset both

---

## Key Conclusions
- Class imbalance reduces explanation stability
- SHAP > LIME in stability
- Resampling does not fix instability
- Background dataset is the root cause
- Important features are least stable

---

## Team
- Bhoomika (23BCS030)
- Sambhav Mishra (23BDS050)
- Sundaram (23BDS060)

---

## Reference
Chen, Y., Calabrese, R., & Martin-Barragan, B. (2024).
Interpretable machine learning for imbalanced credit scoring datasets.
European Journal of Operational Research.
