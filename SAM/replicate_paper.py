"""
Replication of: "Interpretable machine learning for imbalanced credit scoring datasets"
Chen, Calabrese, Martin-Barragan (EJOR 2024)

Datasets used: Taiwan Credit Card Dataset + South German Credit Dataset
(The European Datawarehouse dataset is private)

Authors of replication: DS357 Project Team
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lime
import lime.lime_tabular
import shap
import warnings
warnings.filterwarnings("ignore")

# ─── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─── Config ─────────────────────────────────────────────────────────────────
DEFAULT_RATES = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
N_ITERATIONS  = 100          # B = 100 in the paper
N_TARGETS     = 100          # 100 defaults + 100 non-defaults = 200 targets
N_FEATURES_LIME = 10         # LIME selects top-10 features
SAMPLE_SIZE   = 6258         # Fixed sample size (paper Table 5)


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_taiwan():
    """
    Taiwan Credit Card Default Dataset
    Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    Download the CSV and place it as 'taiwan_credit.csv' in the working directory.
    Column 'default payment next month' is the target (1 = default).
    """
    # df = pd.read_csv("Taiwan_credit_dataset.xls", header=1)  
    df = pd.read_excel("Taiwan_credit_dataset.xls", header=1)# row-0 is extra header
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Standard rename for the target
    target_col = [c for c in df.columns if "default" in c][0]
    df = df.rename(columns={target_col: "default"})

    # Drop ID column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    X = df.drop(columns=["default"])
    y = df["default"].astype(int)
    feature_names = X.columns.tolist()
    return X.values, y.values, feature_names

def load_german():
    # 🔥 AUTO-DETECT separator (MOST IMPORTANT FIX)
    df = pd.read_csv("german_credit.csv", sep=None, engine="python")

    df.columns = df.columns.str.strip()

    # 🔍 Debug check
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    target_col = df.columns[-1]

    # ✅ Fix target
    y = (df[target_col].astype(str).str.strip() == "bad").astype(int)

    X = df.drop(columns=[target_col])

    # ❗ Ensure X is not empty
    if X.shape[1] == 0:
        raise ValueError("❌ ERROR: No features found after dropping target column!")

    # 🔥 Encode categorical features
    from sklearn.preprocessing import LabelEncoder
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    feature_names = X.columns.tolist()

    print("Final feature count:", len(feature_names))  # debug

    return X.values, y.values, feature_names
# def load_german():
#     """
#     South German Credit Dataset
#     Source: https://archive.ics.uci.edu/ml/datasets/South+German+Credit
#     Download 'SouthGermanCredit.asc' and place it in the working directory.
#     Last column is target: 1 = good credit, 2 = bad credit → recode to 0/1.
#     """
#     # df = pd.read_csv("german_credit.csv", sep=" ")
#     df = pd.read_csv("german_credit.csv", sep=None, engine='python')
#     df.columns = df.columns.str.strip()

#     target_col = df.columns[-1]                 # 'kredit' or similar
#     y = (df[target_col] == 2).astype(int)       # 2 = bad credit = default
#     X = df.drop(columns=[target_col])
#     feature_names = X.columns.tolist()
#     return X.values, y.values, feature_names


# ════════════════════════════════════════════════════════════════════════════
# 2. SAMPLING PROCEDURE  (Section 4.2 of paper)
# ════════════════════════════════════════════════════════════════════════════

def sample_targets(X, y, n_targets=N_TARGETS, seed=RANDOM_SEED):
    """
    Randomly select n_targets defaults and n_targets non-defaults as the
    fixed interpretation targets (these are held out from training).
    Returns indices into the full dataset.
    """
    rng = np.random.RandomState(seed)
    idx_def     = np.where(y == 1)[0]
    idx_nondef  = np.where(y == 0)[0]

    sel_def    = rng.choice(idx_def,    size=min(n_targets, len(idx_def)),    replace=False)
    sel_nondef = rng.choice(idx_nondef, size=min(n_targets, len(idx_nondef)), replace=False)
    return np.concatenate([sel_def, sel_nondef])


def build_training_set(X, y, target_idx, default_rate, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    Build ONE training set with a specific default_rate.
    - Excludes target instances.
    - Under-samples defaults and non-defaults to hit the requested rate.
    - Fixed total sample_size.
    """
    rng = np.random.RandomState(seed)

    pool_mask  = np.ones(len(y), dtype=bool)
    pool_mask[target_idx] = False
    pool_X = X[pool_mask]
    pool_y = y[pool_mask]

    pool_def    = np.where(pool_y == 1)[0]
    pool_nondef = np.where(pool_y == 0)[0]

    n_def    = max(1, int(np.round(sample_size * default_rate)))
    n_nondef = sample_size - n_def

    # Clip to available
    n_def    = min(n_def,    len(pool_def))
    n_nondef = min(n_nondef, len(pool_nondef))

    sel_def    = rng.choice(pool_def,    size=n_def,    replace=False)
    sel_nondef = rng.choice(pool_nondef, size=n_nondef, replace=False)

    idx = np.concatenate([sel_def, sel_nondef])
    rng.shuffle(idx)
    return pool_X[idx], pool_y[idx]


# ════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING  (Section 3.1)
# ════════════════════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, seed=RANDOM_SEED):
    """Train XGBoost with default hyper-parameters (simplified for speed)."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, seed=RANDOM_SEED):
    """Train Random Forest (for robustness check)."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ════════════════════════════════════════════════════════════════════════════
# 4. LIME EXPLANATIONS  (Section 3.2)
# ════════════════════════════════════════════════════════════════════════════

def get_lime_importances(model, X_train, target_instance, feature_names, n_features=N_FEATURES_LIME, seed=RANDOM_SEED):
    """
    Run LIME on a single target instance.
    Returns a dict {feature_name: |coefficient|}.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["non-default", "default"],
        mode="classification",
        random_state=seed,
    )
    explanation = explainer.explain_instance(
        data_row=target_instance,
        predict_fn=model.predict_proba,
        num_features=n_features,
    )
    coef_map = {feat: abs(coef) for feat, coef in explanation.as_list()}
    # Fill zeros for features not selected
    full_map = {f: coef_map.get(f, 0.0) for f in feature_names}
    return full_map


# ════════════════════════════════════════════════════════════════════════════
# 5. SHAP EXPLANATIONS  (Section 3.3)
# ════════════════════════════════════════════════════════════════════════════

def get_shap_importances(model, X_train, target_instance, feature_names):
    """
    Run Tree SHAP on a single target instance.
    Returns a dict {feature_name: |shap_value|}.
    """
    explainer = shap.TreeExplainer(model, data=X_train, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(target_instance.reshape(1, -1))
    # For binary classifiers shap may return list [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    abs_shap = np.abs(shap_vals[0])
    return dict(zip(feature_names, abs_shap))


# ════════════════════════════════════════════════════════════════════════════
# 6. STABILITY METRICS  (Section 5)
# ════════════════════════════════════════════════════════════════════════════

# ── 6a. Coefficient of Variation (CV) ─────────────────────────────────────

def compute_cv(importance_matrix):
    """
    importance_matrix: shape (B, P) — B iterations × P features.
    Returns mean CV across features (features appearing ≥ 2 times with nonzero mean).
    Eq. (8) and (9) from paper.
    """
    B, P = importance_matrix.shape
    cvs = []
    for p in range(P):
        vals = importance_matrix[:, p]
        nonzero = vals[vals > 0]
        if len(nonzero) < 2:
            continue
        mean_val = nonzero.mean()
        if mean_val == 0:
            continue
        std_val  = nonzero.std(ddof=1)
        cvs.append(std_val / mean_val)
    return np.mean(cvs) if cvs else np.nan


# ── 6b. Sequential Rank Agreement (SRA) ───────────────────────────────────

def importance_to_ranking(importance_dict, feature_names):
    """Convert importance dict to rank array (1 = most important)."""
    vals = np.array([importance_dict[f] for f in feature_names])
    # Descending rank: highest importance = rank 1
    order = np.argsort(-vals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(feature_names) + 1)
    return ranks


def compute_sra(ranking_matrix, depth=None):
    """
    ranking_matrix: shape (B, P) — each row is a ranking list.
    Returns SRA value at list depth `depth` (default = all features).
    Eq. (6) and (7) from paper.
    Lower SRA → more stable.
    """
    B, P = ranking_matrix.shape
    if depth is None:
        depth = P

    # Expected rank per feature
    mean_ranks = ranking_matrix.mean(axis=0)          # shape (P,)
    rank_stab  = ((ranking_matrix - mean_ranks) ** 2).mean(axis=0)   # variance per feature

    # Set S(d): features appearing in top-d of any list
    in_top_d = np.any(ranking_matrix <= depth, axis=0)   # bool (P,)
    S_d_idx  = np.where(in_top_d)[0]

    if len(S_d_idx) == 0:
        return 0.0

    sra_d = rank_stab[S_d_idx].mean()
    return sra_d


def compute_sra_all_depths(ranking_matrix, max_depth=5):
    """Return list of SRA values for depths 1..max_depth."""
    return [compute_sra(ranking_matrix, d) for d in range(1, max_depth + 1)]


# ── 6c. VSI and CSI (LIME internal/external stability) ───────────────────

def compute_vsi(feature_lists, n_select=N_FEATURES_LIME):
    """
    feature_lists: list of lists — each inner list contains the feature names
                   selected by one LIME run.
    VSI = average pairwise Jaccard overlap / n_select.
    Eq. (10) from paper.
    """
    M   = len(feature_lists)
    cnt = 0
    total = 0
    for i in range(M):
        for j in range(i + 1, M):
            same = len(set(feature_lists[i]) & set(feature_lists[j]))
            total += same
            cnt   += 1
    if cnt == 0:
        return np.nan
    return (total / cnt) / n_select


def compute_csi(coef_dicts, feature_names, n_select=N_FEATURES_LIME):
    """
    coef_dicts: list of {feature: coefficient} dicts from LIME runs.
    For each feature that appears ≥ 2 times, check if 95% CI pairs overlap.
    CSI = mean pairwise overlap fraction.
    Simplified CI: mean ± 1.96 * std across runs for that feature.
    Eq. (11) from paper.
    """
    from scipy import stats

    partial_indices = []
    for feat in feature_names:
        vals = [d.get(feat, None) for d in coef_dicts]
        vals = [v for v in vals if v is not None and v > 0]
        if len(vals) < 2:
            continue
        # Build one CI per run (bootstrap of 1 = use the value itself ± t-based margin)
        # Simplified: treat each value as a point; pair-overlap = |a-b| < 1.96*std_pooled
        std_v  = np.std(vals, ddof=1)
        margin = 1.96 * std_v if std_v > 0 else 1e-9
        ci_list = [(v - margin, v + margin) for v in vals]
        pairs = [(i, j) for i in range(len(ci_list)) for j in range(i + 1, len(ci_list))]
        overlaps = [
            1 if max(ci_list[i][0], ci_list[j][0]) <= min(ci_list[i][1], ci_list[j][1]) else 0
            for i, j in pairs
        ]
        partial_indices.append(np.mean(overlaps))
    return np.mean(partial_indices) if partial_indices else np.nan


# ════════════════════════════════════════════════════════════════════════════
# 7. MAIN EXPERIMENT LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_experiment(X, y, feature_names, dataset_name="dataset",
                   model_type="xgboost",
                   default_rates=DEFAULT_RATES,
                   n_iterations=N_ITERATIONS,
                   n_targets=N_TARGETS,
                   seed=RANDOM_SEED):
    """
    Full experiment following Fig. 1 of the paper.
    Returns a DataFrame with stability metrics per default_rate.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  |  Model: {model_type}")
    print(f"{'='*60}")

    # Step 2: Sample targets
    target_idx = sample_targets(X, y, n_targets=n_targets, seed=seed)
    X_target   = X[target_idx]
    y_target   = y[target_idx]
    P          = len(feature_names)

    results = []

    for dr in default_rates:
        print(f"\n  Default rate: {dr*100:.1f}%", end="", flush=True)

        # Per-target accumulators
        lime_importance_all = {i: [] for i in range(len(target_idx))}   # target → list of dicts
        shap_importance_all = {i: [] for i in range(len(target_idx))}
        lime_feat_lists_all = {i: [] for i in range(len(target_idx))}   # for VSI

        for b in range(n_iterations):
            if b % 20 == 0:
                print(".", end="", flush=True)

            # Step 3: Build training set
            X_tr, y_tr = build_training_set(X, y, target_idx, dr, seed=seed + b)

            # Step 4: Train model
            if model_type == "xgboost":
                model = train_xgboost(X_tr, y_tr, seed=seed + b)
            else:
                model = train_random_forest(X_tr, y_tr, seed=seed + b)

            # Step 5: Get importances for each target
            for i, (xi, yi) in enumerate(zip(X_target, y_target)):
                # LIME
                lime_imp = get_lime_importances(model, X_tr, xi, feature_names, seed=seed + b)
                lime_importance_all[i].append(lime_imp)
                lime_feat_lists_all[i].append([f for f, v in lime_imp.items() if v > 0])

                # SHAP
                shap_imp = get_shap_importances(model, X_tr, xi, feature_names)
                shap_importance_all[i].append(shap_imp)

        # Step 6: Compute stability metrics across targets
        lime_cv_vals, shap_cv_vals = [], []
        lime_sra_vals, shap_sra_vals = [], []     # averaged across depths 1-5
        lime_vsi_vals = []

        for i in range(len(target_idx)):
            # Build matrices shape (B, P)
            lime_mat = np.array([[lime_importance_all[i][b][f] for f in feature_names]
                                 for b in range(n_iterations)])
            shap_mat = np.array([[shap_importance_all[i][b][f] for f in feature_names]
                                 for b in range(n_iterations)])

            # CV
            lime_cv_vals.append(compute_cv(lime_mat))
            shap_cv_vals.append(compute_cv(shap_mat))

            # Ranking matrices
            lime_rank = np.array([importance_to_ranking(lime_importance_all[i][b], feature_names)
                                  for b in range(n_iterations)])
            shap_rank = np.array([importance_to_ranking(shap_importance_all[i][b], feature_names)
                                  for b in range(n_iterations)])

            # SRA (average of depths 1-5)
            lime_sra = np.mean(compute_sra_all_depths(lime_rank))
            shap_sra = np.mean(compute_sra_all_depths(shap_rank))
            lime_sra_vals.append(lime_sra)
            shap_sra_vals.append(shap_sra)

            # VSI
            lime_vsi_vals.append(compute_vsi(lime_feat_lists_all[i]))

        results.append({
            "dataset":      dataset_name,
            "model":        model_type,
            "default_rate": dr,
            "lime_cv_mean":  np.nanmean(lime_cv_vals),
            "lime_cv_std":   np.nanstd(lime_cv_vals),
            "shap_cv_mean":  np.nanmean(shap_cv_vals),
            "shap_cv_std":   np.nanstd(shap_cv_vals),
            "lime_sra_mean": np.nanmean(lime_sra_vals),
            "shap_sra_mean": np.nanmean(shap_sra_vals),
            "lime_vsi_mean": np.nanmean(lime_vsi_vals),
        })

    df = pd.DataFrame(results)
    out_path = f"results_{dataset_name}_{model_type}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 8. PLOTTING  (reproduces Figs. 2, 3, 4)
# ════════════════════════════════════════════════════════════════════════════

def plot_results(df, dataset_name):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Stability Results — {dataset_name}", fontsize=13, fontweight="bold")

    x = df["default_rate"] * 100

    # ── Fig 2 equivalent: SRA ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, df["lime_sra_mean"], "o-", label="LIME", color="#1f77b4")
    ax.plot(x, df["shap_sra_mean"], "s--", label="SHAP", color="#ff7f0e")
    ax.set_xlabel("Default rate (%)")
    ax.set_ylabel("Mean SRA (lower = more stable)")
    ax.set_title("Feature Ranking Stability (SRA)")
    ax.legend()
    ax.invert_xaxis()   # paper plots from balanced → imbalanced left-to-right

    # ── Fig 3 equivalent: CV ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(x, df["lime_cv_mean"], "o-", label="LIME (CV)", color="#1f77b4")
    ax.fill_between(x,
                    df["lime_cv_mean"] - df["lime_cv_std"],
                    df["lime_cv_mean"] + df["lime_cv_std"],
                    alpha=0.2, color="#1f77b4")
    ax.plot(x, df["shap_cv_mean"], "s--", label="SHAP (CV)", color="#ff7f0e")
    ax.fill_between(x,
                    df["shap_cv_mean"] - df["shap_cv_std"],
                    df["shap_cv_mean"] + df["shap_cv_std"],
                    alpha=0.2, color="#ff7f0e")
    ax.set_xlabel("Default rate (%)")
    ax.set_ylabel("Mean CV (higher = less stable)")
    ax.set_title("Feature Importance Value Stability (CV)")
    ax.legend()
    ax.invert_xaxis()

    # ── Fig 4 equivalent: VSI ────────────────────────────────────────────
    ax = axes[2]
    ax.plot(x, df["lime_vsi_mean"], "o-", label="LIME (VSI)", color="#2ca02c")
    ax.set_xlabel("Default rate (%)")
    ax.set_ylabel("Mean VSI (higher = more stable)")
    ax.set_title("LIME Feature Selection Stability (VSI)")
    ax.legend()
    ax.invert_xaxis()

    plt.tight_layout()
    out_path = f"stability_plots_{dataset_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replicate Chen et al. (EJOR 2024)")
    parser.add_argument("--dataset", choices=["taiwan", "german", "both"], default="both")
    parser.add_argument("--model",   choices=["xgboost", "rf", "both"],   default="xgboost")
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--quick",   action="store_true",
                        help="Run with fewer iterations & default rates for testing")
    args = parser.parse_args()

    if args.quick:
        default_rates = [0.01, 0.05, 0.10, 0.25, 0.50]
        n_iter = 10
        n_targets = 20
        print("⚡ Quick mode: reduced iterations for testing")
    else:
        default_rates = DEFAULT_RATES
        n_iter = args.iterations
        n_targets = N_TARGETS

    datasets = []
    if args.dataset in ("taiwan", "both"):
        try:
            X_tw, y_tw, feat_tw = load_taiwan()
            datasets.append((X_tw, y_tw, feat_tw, "Taiwan"))
        except FileNotFoundError:
            print("⚠  taiwan_credit.csv not found — skipping Taiwan dataset")

    if args.dataset in ("german", "both"):
        try:
            X_ge, y_ge, feat_ge = load_german()
            datasets.append((X_ge, y_ge, feat_ge, "German"))
        except FileNotFoundError:
            print("⚠  SouthGermanCredit.asc not found — skipping German dataset")

    models = []
    if args.model in ("xgboost", "both"):
        models.append("xgboost")
    if args.model in ("rf", "both"):
        models.append("rf")

    all_results = []
    for (X, y, feat, dname) in datasets:
        for mname in models:
            df_res = run_experiment(X, y, feat,
                                    dataset_name=dname,
                                    model_type=mname,
                                    default_rates=default_rates,
                                    n_iterations=n_iter,
                                    n_targets=n_targets)
            plot_results(df_res, f"{dname}_{mname}")
            all_results.append(df_res)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv("all_results_combined.csv", index=False)
        print("\n✅ All done! Combined results saved to all_results_combined.csv")
