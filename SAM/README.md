# DS357 – Explainable AI | Course Project

## Replication of:
**"Interpretable machine learning for imbalanced credit scoring datasets"**
Chen, Calabrese, Martin-Barragan — EJOR 312 (2024) 357–372

---

## Repository Structure

```
├── replicate_paper.py      ← Phase 2: Full replication code
├── phase3_extension.py     ← Phase 3: Research gap + extension
├── requirements.txt
├── README.md
└── AI_USAGE.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets

**Taiwan Credit Card Dataset**
- URL: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- Download the XLS file → save as `taiwan_credit.csv`
- The file has two header rows; the code handles that automatically.

**South German Credit Dataset**
- URL: https://archive.ics.uci.edu/ml/datasets/South+German+Credit
- Download `SouthGermanCredit.asc` → place in the same directory.

Random seed used throughout: **42**

---

## Phase 2: Replication

Run the full experiment:
```bash
python replicate_paper.py --dataset both --model xgboost
```

Quick test (reduced iterations):
```bash
python replicate_paper.py --dataset both --model xgboost --quick
```

**What it reproduces:**
- Fig. 2 equivalent → SRA (feature ranking stability) vs default rate
- Fig. 3 equivalent → CV (feature importance value stability) boxplots
- Fig. 4 equivalent → VSI (LIME feature selection stability)

Results saved as:
- `results_Taiwan_xgboost.csv`
- `results_German_xgboost.csv`
- `stability_plots_Taiwan_xgboost.png`
- `stability_plots_German_xgboost.png`

**Expected finding (matching paper):**
Both LIME and SHAP become less stable as default rate decreases toward 1%.
CV and SRA values increase sharply below 5% default rate.

---

## Phase 3: Research Gap & Extension

### Research Gap

The paper demonstrates that class imbalance hurts XAI stability, but
**explicitly leaves open** (Section 7, Future Work) whether resampling
techniques can *restore* that stability:

> "the potential effects of these imbalanced learning techniques on the
> performance of interpretation methods could be investigated in future
> research." — Chen et al. (2024)

### Our Contribution

We test **five strategies** across the same default-rate grid:

| Strategy | Description |
|---|---|
| Baseline | No resampling (original paper) |
| SMOTE | Synthetic minority oversampling (Chawla et al., 2011) |
| ADASYN | Adaptive synthetic sampling (He et al., 2008) |
| RUS | Random under-sampling of majority class |
| Cost-sensitive | XGBoost `scale_pos_weight` (class-weight correction) |

**Hypothesis:** Cost-sensitive learning will improve XAI stability more
than resampling, because resampling adds randomness that increases variance
in the training distribution LIME/SHAP draw from.

### Run Phase 3
```bash
python phase3_extension.py --dataset both --iterations 30 --targets 50
```

Quick test:
```bash
python phase3_extension.py --dataset both --quick
```

**Outputs:**
- `phase3_results_Taiwan.csv` / `phase3_results_German.csv`
- `phase3_plot_Taiwan.png` — all 5 strategies × 5 metrics
- `phase3_heatmap_Taiwan.png` — % improvement vs baseline

---

## Evaluation Criteria Mapping

| Criterion | Where addressed |
|---|---|
| Implement the ML model | `train_xgboost()`, `train_random_forest()` in replicate_paper.py |
| Implement the XAI method | `get_lime_importances()`, `get_shap_importances()` |
| Use same dataset | Taiwan + German (open-source, same as paper Section 6.4) |
| Reproduce 2-3 main results | SRA plot, CV plot, VSI plot |
| Research gap | Section 7 of paper — resampling effect on stability |
| Improvement with references | SMOTE, ADASYN, cost-sensitive XGBoost with citations |
| Before vs after comparison | `plot_phase3()` and `plot_improvement_heatmap()` |

---

## Key Implementation Notes

1. **Sampling procedure** (Section 4.2): We fix sample size at 6258, vary
   default rate by under-sampling, NOT by oversampling. This matches Table 5.

2. **SRA** (Eq. 6-7): We implement the pooled-variance formulation. Lower
   SRA = more stable rankings.

3. **CV** (Eq. 8-9): Coefficient of Variation across 100 iterations per
   feature, then averaged. Higher CV = less stable values.

4. **VSI** (Eq. 10): Average pairwise feature overlap in LIME's selection
   step, normalized by n_features=10.

5. **Tree SHAP**: Uses `shap.TreeExplainer` with `feature_perturbation=
   "interventional"` and training set as background (matches paper Section 3.3).

---

## References

- Chen, Y., Calabrese, R., & Martin-Barragan, B. (2024). Interpretable machine
  learning for imbalanced credit scoring datasets. EJOR, 312, 357–372.
- Chawla, N.V. et al. (2002). SMOTE: Synthetic minority over-sampling technique.
  JAIR, 16, 321–357.
- He, H. et al. (2008). ADASYN: Adaptive synthetic sampling approach. IEEE IJCNN.
- Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model
  predictions. NeurIPS, 4768–4777.
- Ribeiro, M.T. et al. (2016). "Why should I trust you?" ACM SIGKDD, 1135–1144.
