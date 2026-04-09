"""
Experiment C: Defaulter vs Non-Defaulter SHAP Stability Split
==============================================================
Research Question:
    Do defaulters (minority class, rejected applicants) receive
    systematically less stable SHAP explanations than non-defaulters?

Why it matters:
    The original paper averages stability across all 200 targets (100
    defaults + 100 non-defaults). ECOA and GDPR specifically protect
    people who are DENIED credit — i.e., the defaulter predictions.
    If instability is concentrated in defaulter explanations, this is
    a direct regulatory compliance risk that the original paper misses.

What we do:
    Re-run the Phase 3 experiment (same 5 strategies, same default rates)
    but track SHAP CV and SRA separately for:
        - Defaulter targets  (y=1, loan applicants predicted to default)
        - Non-defaulter targets (y=0, applicants predicted safe)

Output:
    - CSV with stability split by class
    - 4 plots: CV and SRA, each split by class, across all strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

import shap

# ── Re-use helpers from replication script ───────────────────────────────
from replicate_paper import (
    load_taiwan, load_german,
    sample_targets, build_training_set,
    get_shap_importances,
    compute_cv, compute_sra_all_depths, importance_to_ranking,
    RANDOM_SEED,
)

STRATEGIES = {
    "baseline":       "Baseline",
    "smote":          "SMOTE",
    "adasyn":         "ADASYN",
    "rus":            "RUS",
    "cost_sensitive": "Cost-Sensitive",
}
COLORS = {
    "baseline":       "#888888",
    "smote":          "#1f77b4",
    "adasyn":         "#ff7f0e",
    "rus":            "#2ca02c",
    "cost_sensitive": "#d62728",
}


# ════════════════════════════════════════════════════════════════════════════
# TRAINING WITH STRATEGY
# ════════════════════════════════════════════════════════════════════════════

def train_with_strategy(X_train, y_train, strategy, seed=RANDOM_SEED):
    if strategy == "smote":
        n_min = y_train.sum()
        k = min(5, max(1, n_min - 1))
        X_r, y_r = SMOTE(k_neighbors=k, random_state=seed).fit_resample(X_train, y_train)
    elif strategy == "adasyn":
        n_min = y_train.sum()
        k = min(5, max(1, n_min - 1))
        try:
            X_r, y_r = ADASYN(n_neighbors=k, random_state=seed).fit_resample(X_train, y_train)
        except Exception:
            X_r, y_r = X_train, y_train
    elif strategy == "rus":
        X_r, y_r = RandomUnderSampler(random_state=seed).fit_resample(X_train, y_train)
    else:
        X_r, y_r = X_train, y_train

    spw = 1.0
    if strategy == "cost_sensitive":
        n_neg, n_pos = (y_r == 0).sum(), (y_r == 1).sum()
        spw = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        use_label_encoder=False, eval_metric="logloss",
        random_state=seed, verbosity=0,
    )
    model.fit(X_r, y_r)
    return model, X_r


# ════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

def run_experiment_c(X, y, feature_names, dataset_name="Taiwan",
                     default_rates=None, n_iterations=30, n_targets=50,
                     seed=RANDOM_SEED):

    if default_rates is None:
        default_rates = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20,
                         0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print(f"\n{'='*60}")
    print(f"Experiment C — {dataset_name}: Defaulter vs Non-Defaulter Split")
    print(f"{'='*60}")

    # Sample fixed targets — keep track of which are defaulters
    target_idx = sample_targets(X, y, n_targets=n_targets, seed=seed)
    X_target = X[target_idx]
    y_target = y[target_idx]  # 0 = non-defaulter, 1 = defaulter

    default_mask    = y_target == 1
    nondefault_mask = y_target == 0

    print(f"  Targets: {default_mask.sum()} defaulters, {nondefault_mask.sum()} non-defaulters")

    records = []

    for strategy in STRATEGIES:
        print(f"\n  Strategy: {STRATEGIES[strategy]}")

        for dr in default_rates:
            print(f"    DR={dr*100:.1f}%", end="", flush=True)

            # Per-target importance lists
            shap_imp_all = {i: [] for i in range(len(target_idx))}

            for b in range(n_iterations):
                print(".", end="", flush=True)

                X_tr, y_tr = build_training_set(X, y, target_idx, dr, seed=seed + b)
                if len(np.unique(y_tr)) < 2:
                    continue

                try:
                    model, X_bg = train_with_strategy(X_tr, y_tr, strategy, seed=seed + b)
                except Exception as e:
                    continue

                for i, xi in enumerate(X_target):
                    try:
                        simp = get_shap_importances(model, X_bg, xi, feature_names)
                        shap_imp_all[i].append(simp)
                    except Exception:
                        pass

            # Compute stability split by class
            def stability_for_group(mask_indices):
                cv_list, sra_list = [], []
                for i in mask_indices:
                    if len(shap_imp_all[i]) < 2:
                        continue
                    B = len(shap_imp_all[i])
                    mat = np.array([[shap_imp_all[i][b][f] for f in feature_names]
                                    for b in range(B)])
                    cv_list.append(compute_cv(mat))
                    rank_mat = np.array([importance_to_ranking(shap_imp_all[i][b], feature_names)
                                         for b in range(B)])
                    sra_list.append(np.mean(compute_sra_all_depths(rank_mat)))
                return np.nanmean(cv_list), np.nanmean(sra_list)

            def_idx    = np.where(default_mask)[0]
            nondef_idx = np.where(nondefault_mask)[0]

            cv_def,  sra_def  = stability_for_group(def_idx)
            cv_non,  sra_non  = stability_for_group(nondef_idx)
            cv_all,  sra_all  = stability_for_group(list(range(len(target_idx))))

            records.append({
                "dataset":        dataset_name,
                "strategy":       strategy,
                "strategy_label": STRATEGIES[strategy],
                "default_rate":   dr,
                # Defaulters
                "shap_cv_defaulters":     cv_def,
                "shap_sra_defaulters":    sra_def,
                # Non-defaulters
                "shap_cv_nondefaulters":  cv_non,
                "shap_sra_nondefaulters": sra_non,
                # Overall (for cross-check)
                "shap_cv_all":    cv_all,
                "shap_sra_all":   sra_all,
                # Stability gap (defaulters worse = positive)
                "cv_gap":         cv_def  - cv_non,
                "sra_gap":        sra_def - sra_non,
            })
            print()

    df = pd.DataFrame(records)
    out = f"experiment_c_{dataset_name}.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved → {out}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_experiment_c(df, dataset_name):
    """
    4 plots:
      Row 1: SHAP CV for defaulters vs non-defaulters (one panel per strategy)
      Row 2: Stability GAP (defaulter CV - non-defaulter CV) across strategies
    """

    strategies = list(STRATEGIES.keys())
    n_strat    = len(strategies)
    x_all      = sorted(df["default_rate"].unique()) 

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Experiment C: Defaulter vs Non-Defaulter SHAP Stability\nDataset: {dataset_name}",
        fontsize=13, fontweight="bold"
    )

    # ── Plot 1: SHAP CV split, baseline only (clearest comparison) ────────
    ax = axes[0, 0]
    sub = df[df["strategy"] == "baseline"].sort_values("default_rate")
    x = sub["default_rate"] * 100
    ax.plot(x, sub["shap_cv_defaulters"],  "o-",  color="#d62728", linewidth=2,
            label="Defaulters (y=1)", markersize=5)
    ax.plot(x, sub["shap_cv_nondefaulters"], "s--", color="#1f77b4", linewidth=2,
            label="Non-Defaulters (y=0)", markersize=5)
    ax.fill_between(x, sub["shap_cv_defaulters"], sub["shap_cv_nondefaulters"],
                    alpha=0.12, color="#d62728", label="Stability gap")
    ax.set_title("SHAP CV by class — Baseline", fontsize=11)
    ax.set_xlabel("Default rate (%)")
    ax.set_ylabel("SHAP CV (higher = less stable)")
    ax.legend(fontsize=9)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: SHAP SRA split, baseline only ────────────────────────────
    ax = axes[0, 1]
    sub = df[df["strategy"] == "baseline"].sort_values("default_rate")
    x = sub["default_rate"] * 100
    ax.plot(x, sub["shap_sra_defaulters"],    "o-",  color="#d62728", linewidth=2,
            label="Defaulters (y=1)", markersize=5)
    ax.plot(x, sub["shap_sra_nondefaulters"], "s--", color="#1f77b4", linewidth=2,
            label="Non-Defaulters (y=0)", markersize=5)
    ax.fill_between(x, sub["shap_sra_defaulters"], sub["shap_sra_nondefaulters"],
                    alpha=0.12, color="#d62728")
    ax.set_title("SHAP SRA by class — Baseline", fontsize=11)
    ax.set_xlabel("Default rate (%)")
    ax.set_ylabel("SHAP SRA (higher = less stable)")
    ax.legend(fontsize=9)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: CV Gap across all strategies ─────────────────────────────
    ax = axes[1, 0]
    for strat in strategies:
        sub = df[df["strategy"] == strat].sort_values("default_rate")
        x = sub["default_rate"] * 100
        ax.plot(x, sub["cv_gap"],
                color=COLORS[strat],
                linestyle="-" if strat == "baseline" else "--",
                marker="o", markersize=4, linewidth=1.8,
                label=STRATEGIES[strat])
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("CV Stability Gap (defaulters - non-defaulters)\nPositive = defaulters more unstable", fontsize=10)
    ax.set_xlabel("Default rate (%)")
    ax.set_ylabel("CV gap")
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # ── Plot 4: SRA Gap across all strategies ────────────────────────────
    ax = axes[1, 1]
    for strat in strategies:
        sub = df[df["strategy"] == strat].sort_values("default_rate")
        x = sub["default_rate"] * 100
        ax.plot(x, sub["sra_gap"],
                color=COLORS[strat],
                linestyle="-" if strat == "baseline" else "--",
                marker="o", markersize=4, linewidth=1.8,
                label=STRATEGIES[strat])
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("SRA Stability Gap (defaulters - non-defaulters)\nPositive = defaulters less stable rankings", fontsize=10)
    ax.set_xlabel("Default rate (%)")
    ax.set_ylabel("SRA gap")
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"experiment_c_plot_{dataset_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved → {out}")


def print_summary_c(df, dataset_name):
    """Print key numbers for the report."""
    print(f"\n{'='*55}")
    print(f"EXPERIMENT C KEY NUMBERS — {dataset_name}")
    print(f"{'='*55}")
    sub = df[(df["strategy"] == "baseline") & (df["default_rate"] <= 0.05)]
    for _, row in sub.sort_values("default_rate").iterrows():
        dr = row["default_rate"]
        print(f"\nDefault rate {dr*100:.1f}%:")
        print(f"  Defaulter   CV={row['shap_cv_defaulters']:.4f}  SRA={row['shap_sra_defaulters']:.2f}")
        print(f"  Non-Default CV={row['shap_cv_nondefaulters']:.4f}  SRA={row['shap_sra_nondefaulters']:.2f}")
        print(f"  CV gap      = {row['cv_gap']:+.4f}  SRA gap = {row['sra_gap']:+.2f}")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["taiwan", "german", "both"], default="taiwan")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--targets",    type=int, default=50)
    parser.add_argument("--quick",      action="store_true")
    args = parser.parse_args()

    if args.quick:
        default_rates = [0.01, 0.05, 0.10, 0.25, 0.50]
        n_iter, n_targets = 5, 20
        print("⚡ Quick mode")
    else:
        default_rates = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20,
                         0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        n_iter    = args.iterations
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
        df = run_experiment_c(X, y, feat,
                              dataset_name=dname,
                              default_rates=default_rates,
                              n_iterations=n_iter,
                              n_targets=n_targets)
        plot_experiment_c(df, dname)
        print_summary_c(df, dname)

    print("\n✅ Experiment C complete!")
