"""
Phase 3 Extension:
"Does resampling improve LIME/SHAP stability under class imbalance?"

Research Gap:
The original paper shows class imbalance hurts XAI stability, but it
explicitly does NOT test whether common imbalanced-learning fixes
(SMOTE, ADASYN, cost-sensitive learning) can RESTORE that stability.
This is called out in Section 7 (Future Work) as an open question.

Our contribution:
We compare LIME/SHAP stability BEFORE and AFTER four imbalance corrections
across the same default-rate grid, showing which technique (if any) best
restores stability without sacrificing accuracy.

Techniques compared:
  1. Baseline  — no resampling (paper's original approach)
  2. SMOTE     — synthetic minority oversampling (Chawla et al., 2011)
  3. ADASYN    — adaptive synthetic sampling (He et al., 2008)
  4. Cost-sensitive XGBoost  — scale_pos_weight parameter
  5. RUS       — Random Under-Sampling of the majority class

Reference for gap:
  "the potential effects of these imbalanced learning techniques on the
   performance of interpretation methods could be investigated in future
   research." — Chen et al. (2024), Section 7
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from imblearn.over_sampling  import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# import lime
# import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Re-use helpers from the replication script
# from replicate_paper import (
#     load_taiwan, load_german,
#     sample_targets, build_training_set,
#     # get_lime_importances,
#      get_shap_importances,
#     compute_cv, compute_sra_all_depths, importance_to_ranking, compute_vsi,
#     DEFAULT_RATES, N_ITERATIONS, N_TARGETS, N_FEATURES_LIME, RANDOM_SEED,
# )
from replicate_paper import (
    load_taiwan, load_german,
    sample_targets, build_training_set,
    get_shap_importances,
    compute_cv, compute_sra_all_depths, importance_to_ranking,
    DEFAULT_RATES, N_ITERATIONS, N_TARGETS, RANDOM_SEED,
)

# ════════════════════════════════════════════════════════════════════════════
# IMBALANCE CORRECTION WRAPPERS
# ════════════════════════════════════════════════════════════════════════════

def apply_resampling(X_train, y_train, strategy, seed=RANDOM_SEED):
    """
    Apply a resampling strategy and return (X_res, y_res, model).
    strategy: one of 'baseline', 'smote', 'adasyn', 'rus', 'cost_sensitive'
    """
    if strategy == "baseline":
        return X_train, y_train, None

    if strategy == "smote":
        n_min = y_train.sum()
        if n_min < 2:
            return X_train, y_train, None
        k = min(5, n_min - 1)
        resampler = SMOTE(k_neighbors=k, random_state=seed)
        X_r, y_r = resampler.fit_resample(X_train, y_train)
        return X_r, y_r, None

    if strategy == "adasyn":
        n_min = y_train.sum()
        if n_min < 2:
            return X_train, y_train, None
        k = min(5, n_min - 1)
        try:
            resampler = ADASYN(n_neighbors=k, random_state=seed)
            X_r, y_r = resampler.fit_resample(X_train, y_train)
        except Exception:
            X_r, y_r = X_train, y_train
        return X_r, y_r, None

    if strategy == "rus":
        resampler = RandomUnderSampler(random_state=seed)
        X_r, y_r = resampler.fit_resample(X_train, y_train)
        return X_r, y_r, None

    # cost_sensitive returns (X_train unchanged, y_train unchanged, scale_pos_weight)
    if strategy == "cost_sensitive":
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw   = n_neg / max(n_pos, 1)
        return X_train, y_train, spw

    raise ValueError(f"Unknown strategy: {strategy}")


def train_model_with_strategy(X_train, y_train, strategy, seed=RANDOM_SEED):
    """Train XGBoost after applying resampling strategy."""
    X_r, y_r, extra = apply_resampling(X_train, y_train, strategy, seed)

    if strategy == "cost_sensitive":
        model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
            scale_pos_weight=extra,
            use_label_encoder=False, eval_metric="logloss",
            random_state=seed, verbosity=0,
        )
    else:
        model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=seed, verbosity=0,
        )
    model.fit(X_r, y_r)
    return model, X_r    # return resampled X for LIME background


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 EXPERIMENT LOOP
# ════════════════════════════════════════════════════════════════════════════

STRATEGIES = ["baseline", "smote", "adasyn", "rus", "cost_sensitive"]
STRATEGY_LABELS = {
    "baseline":       "Baseline (no resampling)",
    "smote":          "SMOTE",
    "adasyn":         "ADASYN",
    "rus":            "Random Under-Sampling",
    "cost_sensitive": "Cost-Sensitive XGBoost",
}


# def run_phase3_experiment(X, y, feature_names, dataset_name="dataset",
#                           default_rates=None, n_iterations=30, n_targets=50,
#                           seed=RANDOM_SEED):
#     """
#     For each (default_rate × strategy) combination:
#     - Run n_iterations training/explanation cycles
#     - Compute LIME CV, SHAP CV, LIME SRA, SHAP SRA, LIME VSI
#     Returns a long-form DataFrame.
#     """
#     if default_rates is None:
#         default_rates = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50]

#     print(f"\n{'='*65}")
#     print(f"Phase 3 Experiment — Dataset: {dataset_name}")
#     print(f"{'='*65}")

#     target_idx = sample_targets(X, y, n_targets=n_targets, seed=seed)
#     X_target   = X[target_idx]
#     y_target   = y[target_idx]

#     records = []

#     for strategy in STRATEGIES:
#         print(f"\n  Strategy: {STRATEGY_LABELS[strategy]}")

#         for dr in default_rates:
#             print(f"    DR={dr*100:.1f}%", end="", flush=True)

#             lime_imp_all = {i: [] for i in range(len(target_idx))}
#             shap_imp_all = {i: [] for i in range(len(target_idx))}
#             lime_fl_all  = {i: [] for i in range(len(target_idx))}   # feature lists for VSI

#             for b in range(n_iterations):
#                 print(".", end="", flush=True)

#                 X_tr, y_tr = build_training_set(X, y, target_idx, dr, seed=seed + b)

#                 # Skip if only one class present
#                 if len(np.unique(y_tr)) < 2:
#                     continue

#                 try:
#                     model, X_bg = train_model_with_strategy(X_tr, y_tr, strategy, seed=seed + b)
#                 except Exception as e:
#                     print(f"[err:{e}]", end="")
#                     continue

#                 for i, xi in enumerate(X_target):
#                     try:
#                         limp = get_lime_importances(model, X_bg, xi, feature_names, seed=seed + b)
#                         lime_imp_all[i].append(limp)
#                         lime_fl_all[i].append([f for f, v in limp.items() if v > 0])
#                     except Exception:
#                         pass
#                     try:
#                         simp = get_shap_importances(model, X_bg, xi, feature_names)
#                         shap_imp_all[i].append(simp)
#                     except Exception:
#                         pass

#             # Compute per-target stability metrics
#             lime_cv_list, shap_cv_list = [], []
#             lime_sra_list, shap_sra_list = [], []
#             lime_vsi_list = []

#             for i in range(len(target_idx)):
#                 if len(lime_imp_all[i]) < 2:
#                     continue

#                 B_actual = len(lime_imp_all[i])
#                 lime_mat = np.array([[lime_imp_all[i][b][f] for f in feature_names]
#                                      for b in range(B_actual)])
#                 shap_mat_data = shap_imp_all[i]
#                 if len(shap_mat_data) < 2:
#                     continue
#                 shap_mat = np.array([[shap_mat_data[b][f] for f in feature_names]
#                                      for b in range(len(shap_mat_data))])

#                 lime_cv_list.append(compute_cv(lime_mat))
#                 shap_cv_list.append(compute_cv(shap_mat))

#                 lime_rank = np.array([importance_to_ranking(lime_imp_all[i][b], feature_names)
#                                       for b in range(B_actual)])
#                 shap_rank = np.array([importance_to_ranking(shap_mat_data[b], feature_names)
#                                       for b in range(len(shap_mat_data))])
#                 lime_sra_list.append(np.mean(compute_sra_all_depths(lime_rank)))
#                 shap_sra_list.append(np.mean(compute_sra_all_depths(shap_rank)))
#                 lime_vsi_list.append(compute_vsi(lime_fl_all[i]))

#             records.append({
#                 "dataset":       dataset_name,
#                 "strategy":      strategy,
#                 "strategy_label": STRATEGY_LABELS[strategy],
#                 "default_rate":  dr,
#                 "lime_cv":       np.nanmean(lime_cv_list),
#                 "shap_cv":       np.nanmean(shap_cv_list),
#                 "lime_sra":      np.nanmean(lime_sra_list),
#                 "shap_sra":      np.nanmean(shap_sra_list),
#                 "lime_vsi":      np.nanmean(lime_vsi_list),
#             })

#     df = pd.DataFrame(records)
#     out_path = f"phase3_results_{dataset_name}.csv"
#     df.to_csv(out_path, index=False)
#     print(f"\n  Saved → {out_path}")
#     return df

def run_phase3_experiment(X, y, feature_names, dataset_name="dataset",
                          default_rates=None, n_iterations=30, n_targets=50,
                          seed=RANDOM_SEED):

    if default_rates is None:
        default_rates = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50]

    print(f"\n{'='*65}")
    print(f"Phase 3 Experiment — Dataset: {dataset_name}")
    print(f"{'='*65}")

    target_idx = sample_targets(X, y, n_targets=n_targets, seed=seed)
    X_target   = X[target_idx]

    records = []

    for strategy in STRATEGIES:
        print(f"\n  Strategy: {STRATEGY_LABELS[strategy]}")

        for dr in default_rates:
            print(f"    DR={dr*100:.1f}%", end="", flush=True)

            shap_imp_all = {i: [] for i in range(len(target_idx))}

            for b in range(n_iterations):
                print(".", end="", flush=True)

                X_tr, y_tr = build_training_set(X, y, target_idx, dr, seed=seed + b)

                if len(np.unique(y_tr)) < 2:
                    continue

                try:
                    model, X_bg = train_model_with_strategy(X_tr, y_tr, strategy, seed=seed + b)
                except Exception:
                    continue

                for i, xi in enumerate(X_target):
                    try:
                        simp = get_shap_importances(model, X_bg, xi, feature_names)
                        shap_imp_all[i].append(simp)
                    except Exception:
                        pass

            # =========================
            # SHAP METRICS ONLY
            # =========================
            shap_cv_list = []
            shap_sra_list = []

            for i in range(len(target_idx)):
                if len(shap_imp_all[i]) < 2:
                    continue

                shap_mat = np.array([
                    [shap_imp_all[i][b][f] for f in feature_names]
                    for b in range(len(shap_imp_all[i]))
                ])

                shap_cv_list.append(compute_cv(shap_mat))

                shap_rank = np.array([
                    importance_to_ranking(shap_imp_all[i][b], feature_names)
                    for b in range(len(shap_imp_all[i]))
                ])

                shap_sra_list.append(np.mean(compute_sra_all_depths(shap_rank)))

            records.append({
                "dataset": dataset_name,
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy],
                "default_rate": dr,
                "shap_cv": np.nanmean(shap_cv_list),
                "shap_sra": np.nanmean(shap_sra_list),
            })

    df = pd.DataFrame(records)
    out_path = f"phase3_results_{dataset_name}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n  Saved → {out_path}")
    return df
# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 PLOTTING — before vs after, per metric
# ════════════════════════════════════════════════════════════════════════════

COLORS = {
    "baseline":       "#888888",
    "smote":          "#1f77b4",
    "adasyn":         "#ff7f0e",
    "rus":            "#2ca02c",
    "cost_sensitive": "#d62728",
}
LINESTYLES = {
    "baseline":       "-",
    "smote":          "--",
    "adasyn":         "-.",
    "rus":            ":",
    "cost_sensitive": (0, (5, 2)),
}


def plot_phase3(df, dataset_name):
    # metrics = [
    #     ("lime_cv",  "LIME CV (↑ = less stable)"),
    #     ("shap_cv",  "SHAP CV (↑ = less stable)"),
    #     ("lime_sra", "LIME SRA (↑ = less stable)"),
    #     ("shap_sra", "SHAP SRA (↑ = less stable)"),
    #     ("lime_vsi", "LIME VSI (↑ = more stable)"),
    # ]
    metrics = [
    ("shap_cv",  "SHAP CV (↑ = less stable)"),
    ("shap_sra", "SHAP SRA (↑ = less stable)")
]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
    fig.suptitle(
        f"Phase 3: Effect of Resampling on XAI Stability — {dataset_name}\n"
        f"(x-axis: default rate %, from extreme imbalance 1% → balanced 50%)",
        fontsize=11, fontweight="bold"
    )

    for ax, (metric, ylabel) in zip(axes, metrics):
        for strategy in STRATEGIES:
            sub = df[df["strategy"] == strategy].sort_values("default_rate")
            x   = sub["default_rate"] * 100
            y   = sub[metric]
            ax.plot(x, y,
                    label=STRATEGY_LABELS[strategy],
                    color=COLORS[strategy],
                    linestyle=LINESTYLES[strategy],
                    marker="o", markersize=4, linewidth=1.8)

        ax.set_xlabel("Default rate (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(metric.upper())
        ax.invert_xaxis()   # match paper's x-axis direction
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(STRATEGIES),
               bbox_to_anchor=(0.5, -0.12), fontsize=9)
    plt.tight_layout()
    out_path = f"phase3_plot_{dataset_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved → {out_path}")


def plot_improvement_heatmap(df, dataset_name):
    """
    For each (strategy, default_rate), show % improvement vs baseline.
    Positive = more stable than baseline; negative = worse.
    Uses SHAP CV as the primary metric (lower = better).
    """
    # baseline = df[df["strategy"] == "baseline"][["default_rate", "lime_cv"]].set_index("default_rate")
    baseline = df[df["strategy"] == "baseline"][["default_rate", "shap_cv"]].set_index("default_rate")
    rows = []
    for strategy in STRATEGIES:
        if strategy == "baseline":
            continue
        sub = df[df["strategy"] == strategy].set_index("default_rate")
        for dr in sub.index:
            # base_val = baseline.loc[dr, "lime_cv"] if dr in baseline.index else np.nan
            # strat_val = sub.loc[dr, "lime_cv"]
            base_val = baseline.loc[dr, "shap_cv"]
            strat_val = sub.loc[dr, "shap_cv"]
            improvement = (base_val - strat_val) / (base_val + 1e-9) * 100   # % reduction in CV
            rows.append({"strategy": STRATEGY_LABELS[strategy],
                         "default_rate": f"{dr*100:.0f}%",
                         "improvement": improvement})

    heat_df = pd.DataFrame(rows).pivot(index="strategy", columns="default_rate", values="improvement")

    fig, ax = plt.subplots(figsize=(10, 4))
    import matplotlib.cm as cm
    im = ax.imshow(heat_df.values, cmap="RdYlGn", aspect="auto", vmin=-50, vmax=50)
    ax.set_xticks(range(len(heat_df.columns)))
    ax.set_xticklabels(heat_df.columns, rotation=45)
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index)
    ax.set_title(f"% Improvement in LIME CV vs Baseline — {dataset_name}\n(green = more stable, red = less stable)")
    for i in range(len(heat_df.index)):
        for j in range(len(heat_df.columns)):
            val = heat_df.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="% reduction in CV (positive = improvement)")
    plt.tight_layout()
    out_path = f"phase3_heatmap_{dataset_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Heatmap saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 extension: resampling vs XAI stability")
    parser.add_argument("--dataset",    choices=["taiwan", "german", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--targets",    type=int, default=50)
    parser.add_argument("--quick",      action="store_true")
    args = parser.parse_args()

    if args.quick:
        default_rates = [0.01, 0.05, 0.10, 0.50]
        n_iter = 5
        n_targets = 10
        print("⚡ Quick mode")
    else:
        default_rates = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        n_iter = args.iterations
        n_targets = args.targets

    datasets = []
    if args.dataset in ("taiwan", "both"):
        try:
            X, y, feat = load_taiwan()
            datasets.append((X, y, feat, "Taiwan"))
        except FileNotFoundError:
            print("⚠  taiwan_credit.csv not found")
    if args.dataset in ("german", "both"):
        try:
            X, y, feat = load_german()
            datasets.append((X, y, feat, "German"))
        except FileNotFoundError:
            print("⚠  SouthGermanCredit.asc not found")

    for X, y, feat, dname in datasets:
        df_p3 = run_phase3_experiment(X, y, feat,
                                      dataset_name=dname,
                                      default_rates=default_rates,
                                      n_iterations=n_iter,
                                      n_targets=n_targets)
        plot_phase3(df_p3, dname)
        plot_improvement_heatmap(df_p3, dname)

    print("\n✅ Phase 3 complete!")
