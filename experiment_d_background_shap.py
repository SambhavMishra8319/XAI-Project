"""
Experiment D: SHAP Background Dataset Size and Composition Effect
=================================================================
Research Question:
    Can we restore SHAP stability at extreme imbalance by modifying
    the background dataset used by Tree SHAP — without changing the
    training procedure at all?

Why it matters:
    Finding 4 from our Phase 3 results showed that all strategies
    converge at 50% balance, confirming instability comes from sparse
    minority representation in the data SHAP uses.

    Tree SHAP computes interventional expectations over a background
    dataset (Eq. 5, Chen et al. 2024):
        f_S(x_S) = E[f(X) | do(X_S = x_S)]
    where X is sampled from the background dataset.

    The paper ALWAYS uses the full training set as background.
    If the training set is 1% defaulters, the background is also 1%
    defaulters — SHAP barely sees minority class samples.

    Our proposal: instead of resampling the training data, resample
    ONLY the background dataset passed to shap.TreeExplainer.
    This is a post-hoc fix that does not change model training at all.

Conditions tested (all at fixed default_rate=0.01, most extreme):
    1. Full training set as background (paper's approach) — baseline
    2. Background size: 50, 100, 200, 500, 1000, full (size ablation)
    3. Background composition:
       a. Random subset (same imbalance as training)
       b. Class-balanced subset (50/50 from training set)
       c. Minority-only background (only defaulters)
       d. Majority-only background (only non-defaulters)

Output:
    - CSV with SHAP CV and SRA per background condition
    - Plot showing stability vs background size and composition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import shap
from xgboost import XGBClassifier

from replicate_paper import (
    load_taiwan, load_german,
    sample_targets, build_training_set,
    compute_cv, compute_sra_all_depths, importance_to_ranking,
    RANDOM_SEED,
)


# ════════════════════════════════════════════════════════════════════════════
# SHAP WITH CUSTOM BACKGROUND
# ════════════════════════════════════════════════════════════════════════════

def get_shap_with_background(model, background_X, target_instance, feature_names):
    """
    Run Tree SHAP using a specific background dataset.
    Returns dict {feature: |shap_value|}.
    """
    explainer = shap.TreeExplainer(
        model,
        data=background_X,
        feature_perturbation="interventional"
    )
    shap_vals = explainer.shap_values(target_instance.reshape(1, -1))
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    abs_vals = np.abs(shap_vals[0])
    return dict(zip(feature_names, abs_vals))


def select_background(X_train, y_train, condition, size, seed=RANDOM_SEED):
    """
    Select background dataset according to condition.

    Conditions:
        'random'    — random subset of training data (same imbalance)
        'balanced'  — equal numbers of defaults and non-defaults
        'minority'  — only defaulters (minority class)
        'majority'  — only non-defaulters (majority class)
        'full'      — entire training set (paper's original approach)
    """
    rng = np.random.RandomState(seed)

    if condition == "full":
        return X_train, y_train

    idx_def    = np.where(y_train == 1)[0]
    idx_nondef = np.where(y_train == 0)[0]

    if condition == "random":
        n = min(size, len(X_train))
        sel = rng.choice(len(X_train), size=n, replace=False)
        return X_train[sel], y_train[sel]

    elif condition == "balanced":
        n_each = size // 2
        n_def  = min(n_each, len(idx_def))
        n_non  = min(n_each, len(idx_nondef))
        if n_def == 0:
            # No defaults available — fallback to random
            sel = rng.choice(len(X_train), size=min(size, len(X_train)), replace=False)
            return X_train[sel], y_train[sel]
        sel_def = rng.choice(idx_def,    size=n_def, replace=False)
        sel_non = rng.choice(idx_nondef, size=n_non, replace=False)
        sel = np.concatenate([sel_def, sel_non])
        return X_train[sel], y_train[sel]

    elif condition == "minority":
        n = min(size, len(idx_def))
        if n == 0:
            # No defaults — return full as fallback
            return X_train, y_train
        sel = rng.choice(idx_def, size=n, replace=len(idx_def) < n)
        return X_train[sel], y_train[sel]

    elif condition == "majority":
        n = min(size, len(idx_nondef))
        sel = rng.choice(idx_nondef, size=n, replace=False)
        return X_train[sel], y_train[sel]

    else:
        raise ValueError(f"Unknown condition: {condition}")


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT D — PART 1: Background SIZE ablation
# ════════════════════════════════════════════════════════════════════════════

def run_size_ablation(X, y, feature_names, dataset_name="Taiwan",
                      fixed_dr=0.01, n_iterations=30, n_targets=30,
                      seed=RANDOM_SEED):
    """
    Fix default_rate=fixed_dr (most extreme imbalance).
    Vary background size: 50, 100, 200, 500, 1000, full.
    Compare 'random' vs 'balanced' composition at each size.
    """
    print(f"\n{'='*60}")
    print(f"Experiment D (Part 1): Background Size Ablation — {dataset_name}")
    print(f"Fixed default rate: {fixed_dr*100:.1f}%")
    print(f"{'='*60}")

    target_idx = sample_targets(X, y, n_targets=n_targets, seed=seed)
    X_target   = X[target_idx]

    bg_sizes      = [50, 100, 200, 500, 1000, "full"]
    bg_conditions = ["random", "balanced"]

    records = []

    for condition in bg_conditions:
        print(f"\n  Condition: {condition}")
        for bg_size in bg_sizes:
            print(f"    bg_size={bg_size}", end="", flush=True)

            shap_imp_all = {i: [] for i in range(len(target_idx))}

            for b in range(n_iterations):
                print(".", end="", flush=True)

                X_tr, y_tr = build_training_set(X, y, target_idx, fixed_dr, seed=seed + b)
                if len(np.unique(y_tr)) < 2:
                    continue

                # Train model on FULL imbalanced training set (no resampling)
                model = XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=seed + b, verbosity=0,
                )
                model.fit(X_tr, y_tr)

                # Select background
                actual_size = len(X_tr) if bg_size == "full" else bg_size
                X_bg, y_bg = select_background(X_tr, y_tr, condition, actual_size, seed=seed + b)

                for i, xi in enumerate(X_target):
                    try:
                        simp = get_shap_with_background(model, X_bg, xi, feature_names)
                        shap_imp_all[i].append(simp)
                    except Exception:
                        pass

            # Compute stability
            cv_list, sra_list = [], []
            for i in range(len(target_idx)):
                if len(shap_imp_all[i]) < 2:
                    continue
                B = len(shap_imp_all[i])
                mat = np.array([[shap_imp_all[i][b][f] for f in feature_names]
                                for b in range(B)])
                cv_list.append(compute_cv(mat))
                rank_mat = np.array([importance_to_ranking(shap_imp_all[i][b], feature_names)
                                     for b in range(B)])
                sra_list.append(np.mean(compute_sra_all_depths(rank_mat)))

            n_def_in_bg = None
            if bg_size != "full":
                # Estimate typical minority count in background
                last_X_tr, last_y_tr = build_training_set(X, y, target_idx, fixed_dr, seed=seed)
                X_bg_est, y_bg_est = select_background(last_X_tr, last_y_tr, condition,
                                                        bg_size, seed=seed)
                n_def_in_bg = int(y_bg_est.sum())

            records.append({
                "dataset":      dataset_name,
                "condition":    condition,
                "bg_size":      bg_size,
                "n_defaults_in_bg": n_def_in_bg,
                "default_rate": fixed_dr,
                "shap_cv":      np.nanmean(cv_list),
                "shap_sra":     np.nanmean(sra_list),
            })
            print()

    df = pd.DataFrame(records)
    out = f"experiment_d_size_{dataset_name}.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved → {out}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT D — PART 2: Background COMPOSITION across default rates
# ════════════════════════════════════════════════════════════════════════════

def run_composition_ablation(X, y, feature_names, dataset_name="Taiwan",
                              default_rates=None, bg_size=200,
                              n_iterations=30, n_targets=30,
                              seed=RANDOM_SEED):
    """
    Compare background composition (full vs balanced vs minority vs majority)
    across all default rates. Fixed background size = bg_size.
    """
    if default_rates is None:
        default_rates = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50]

    print(f"\n{'='*60}")
    print(f"Experiment D (Part 2): Background Composition — {dataset_name}")
    print(f"Background size: {bg_size}")
    print(f"{'='*60}")

    target_idx = sample_targets(X, y, n_targets=n_targets, seed=seed)
    X_target   = X[target_idx]

    conditions = ["full", "random", "balanced", "minority", "majority"]
    condition_labels = {
        "full":     "Full training set (paper baseline)",
        "random":   f"Random subset (n={bg_size}, same imbalance)",
        "balanced": f"Balanced subset (n={bg_size}, 50/50)",
        "minority": f"Minority only (defaults, n≤{bg_size})",
        "majority": f"Majority only (non-defaults, n={bg_size})",
    }

    records = []

    for condition in conditions:
        print(f"\n  Background: {condition_labels[condition]}")
        for dr in default_rates:
            print(f"    DR={dr*100:.1f}%", end="", flush=True)

            shap_imp_all = {i: [] for i in range(len(target_idx))}

            for b in range(n_iterations):
                print(".", end="", flush=True)
                X_tr, y_tr = build_training_set(X, y, target_idx, dr, seed=seed + b)
                if len(np.unique(y_tr)) < 2:
                    continue

                model = XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=seed + b, verbosity=0,
                )
                model.fit(X_tr, y_tr)

                actual_size = len(X_tr) if condition == "full" else bg_size
                X_bg, y_bg = select_background(X_tr, y_tr, condition, actual_size, seed=seed + b)

                for i, xi in enumerate(X_target):
                    try:
                        simp = get_shap_with_background(model, X_bg, xi, feature_names)
                        shap_imp_all[i].append(simp)
                    except Exception:
                        pass

            cv_list, sra_list = [], []
            for i in range(len(target_idx)):
                if len(shap_imp_all[i]) < 2:
                    continue
                B = len(shap_imp_all[i])
                mat = np.array([[shap_imp_all[i][b][f] for f in feature_names]
                                for b in range(B)])
                cv_list.append(compute_cv(mat))
                rank_mat = np.array([importance_to_ranking(shap_imp_all[i][b], feature_names)
                                     for b in range(B)])
                sra_list.append(np.mean(compute_sra_all_depths(rank_mat)))

            records.append({
                "dataset":       dataset_name,
                "condition":     condition,
                "condition_label": condition_labels[condition],
                "default_rate":  dr,
                "shap_cv":       np.nanmean(cv_list),
                "shap_sra":      np.nanmean(sra_list),
            })
            print()

    df = pd.DataFrame(records)
    out = f"experiment_d_composition_{dataset_name}.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved → {out}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════

COND_COLORS = {
    "full":     "#888888",
    "random":   "#1f77b4",
    "balanced": "#2ca02c",
    "minority": "#d62728",
    "majority": "#ff7f0e",
}

def plot_size_ablation(df, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Experiment D (Part 1): Background Size Effect on SHAP Stability\n"
        f"Dataset: {dataset_name} | Fixed default rate: {df['default_rate'].iloc[0]*100:.1f}%",
        fontsize=11, fontweight="bold"
    )

    # Numeric x-axis: replace "full" with actual max size
    for ax, metric, ylabel in [
        (axes[0], "shap_cv",  "SHAP CV (higher = less stable)"),
        (axes[1], "shap_sra", "SHAP SRA (higher = less stable)")
    ]:
        for condition in ["random", "balanced"]:
            sub = df[df["condition"] == condition].copy()
            # Map "full" to a large number for plotting
            max_real = sub[sub["bg_size"] != "full"]["bg_size"].astype(int).max()
            sub["bg_size_num"] = sub["bg_size"].apply(
                lambda x: max_real * 1.3 if x == "full" else int(x))
            sub = sub.sort_values("bg_size_num")
            ax.plot(sub["bg_size_num"], sub[metric],
                    "o-" if condition == "random" else "s--",
                    label=f"{condition.capitalize()} subset",
                    color="#1f77b4" if condition == "random" else "#2ca02c",
                    linewidth=2, markersize=6)
        ax.set_xlabel("Background dataset size")
        ax.set_ylabel(ylabel)
        ax.set_title(metric.upper())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"experiment_d_size_plot_{dataset_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved → {out}")


def plot_composition_ablation(df, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Experiment D (Part 2): Background Composition Effect on SHAP Stability\n"
        f"Dataset: {dataset_name}",
        fontsize=11, fontweight="bold"
    )

    conditions = df["condition"].unique()

    for ax, metric, ylabel in [
        (axes[0], "shap_cv",  "SHAP CV (higher = less stable)"),
        (axes[1], "shap_sra", "SHAP SRA (higher = less stable)")
    ]:
        for cond in ["full", "random", "balanced", "minority", "majority"]:
            if cond not in conditions:
                continue
            sub = df[df["condition"] == cond].sort_values("default_rate")
            x = sub["default_rate"] * 100
            lbl = sub["condition_label"].iloc[0]
            ls = "-" if cond == "full" else "--"
            ax.plot(x, sub[metric],
                    color=COND_COLORS.get(cond, "black"),
                    linestyle=ls, marker="o", markersize=4,
                    linewidth=2 if cond == "full" else 1.5,
                    label=lbl)
        ax.set_xlabel("Default rate (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(metric.upper())
        ax.invert_xaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"experiment_d_composition_plot_{dataset_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved → {out}")


def print_summary_d(df_size, df_comp, dataset_name):
    print(f"\n{'='*55}")
    print(f"EXPERIMENT D KEY NUMBERS — {dataset_name}")
    print(f"{'='*55}")

    if df_size is not None:
        print("\nPart 1 — Size ablation at 1% default rate:")
        for _, row in df_size.iterrows():
            print(f"  {row['condition']:10s} bg_size={str(row['bg_size']):5s} → "
                  f"CV={row['shap_cv']:.4f}  SRA={row['shap_sra']:.2f}")

    if df_comp is not None:
        print("\nPart 2 — Composition at 1% default rate:")
        sub = df_comp[df_comp["default_rate"] == df_comp["default_rate"].min()]
        for _, row in sub.iterrows():
            print(f"  {row['condition']:10s} → CV={row['shap_cv']:.4f}  SRA={row['shap_sra']:.2f}")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["taiwan", "german", "both"], default="taiwan")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--targets",    type=int, default=30)
    parser.add_argument("--bg_size",    type=int, default=200)
    parser.add_argument("--quick",      action="store_true")
    parser.add_argument("--part",       choices=["1", "2", "both"], default="both",
                        help="Run Part 1 (size ablation), Part 2 (composition), or both")
    args = parser.parse_args()

    if args.quick:
        n_iter, n_targets = 5, 15
        default_rates = [0.01, 0.05, 0.10, 0.50]
        print("⚡ Quick mode")
    else:
        n_iter = args.iterations
        n_targets = args.targets
        default_rates = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20,
                         0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

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
        df_size, df_comp = None, None

        if args.part in ("1", "both"):
            df_size = run_size_ablation(
                X, y, feat, dataset_name=dname,
                fixed_dr=0.01, n_iterations=n_iter, n_targets=n_targets)
            plot_size_ablation(df_size, dname)

        if args.part in ("2", "both"):
            df_comp = run_composition_ablation(
                X, y, feat, dataset_name=dname,
                default_rates=default_rates,
                bg_size=args.bg_size,
                n_iterations=n_iter, n_targets=n_targets)
            plot_composition_ablation(df_comp, dname)

        print_summary_d(df_size, df_comp, dname)

    print("\n✅ Experiment D complete!")
