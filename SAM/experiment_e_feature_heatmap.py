"""
Experiment E: Feature-Level SHAP Stability Heatmap
===================================================
Research Question:
    Which specific features lose SHAP stability first as class
    imbalance increases? Are domain-critical features (like balance,
    payment history) disproportionately affected?

Why it matters:
    The original paper only reports AGGREGATE stability (averaged
    across all features). A practitioner using SHAP for credit
    decisions needs to know: "Is the SHAP value for THIS specific
    feature trustworthy at my dataset's imbalance level?"

    If high-stakes features (payment history, credit limit) become
    unstable at moderate imbalance (e.g., 10%) while low-stakes
    features remain stable, that changes the regulatory picture
    entirely.

What we produce:
    1. Heatmap: features (rows) x default rates (cols), colour = CV
       → shows exactly when each feature becomes unstable
    2. Ranking stability per feature: does each feature's rank
       position change as imbalance increases?
    3. "Critical threshold" per feature: the default rate below
       which CV exceeds 0.5 (unstable zone)

Run time: ~2-3 hours full, ~10 min quick mode
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
import shap

from replicate_paper import (
    load_taiwan, load_german,
    sample_targets, build_training_set,
    get_shap_importances,
    importance_to_ranking,
    RANDOM_SEED,
)

# CV threshold above which we consider a feature's SHAP "unstable"
INSTABILITY_THRESHOLD = 0.50


# ════════════════════════════════════════════════════════════════════════════
# CORE EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

def run_experiment_e(X, y, feature_names, dataset_name="Taiwan",
                     default_rates=None, n_iterations=30, n_targets=50,
                     seed=RANDOM_SEED):
    """
    For each (default_rate, feature), compute:
      - Mean CV of SHAP values across iterations  (instability measure)
      - Mean rank position of feature             (importance measure)
      - Rank variance                             (ranking instability)

    Returns a dict of DataFrames keyed by metric name.
    """
    if default_rates is None:
        default_rates = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20,
                         0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print(f"\n{'='*60}")
    print(f"Experiment E: Feature-Level Heatmap — {dataset_name}")
    print(f"Features: {len(feature_names)} | Targets: {n_targets} | Iterations: {n_iterations}")
    print(f"{'='*60}")

    target_idx = sample_targets(X, y, n_targets=n_targets, seed=seed)
    X_target   = X[target_idx]
    P          = len(feature_names)

    # Storage: for each (dr, feature) → list of shap values across iterations & targets
    # shape will be (n_iterations * n_targets,) per feature per dr
    results_cv   = {}   # {dr: array shape (P,)}
    results_rank = {}   # {dr: array shape (P,)}  mean rank position (1=most important)
    results_rankvar = {}  # {dr: array shape (P,)} rank variance

    for dr in default_rates:
        print(f"\n  Default rate: {dr*100:.1f}%", end="", flush=True)

        # Per-feature accumulators across all targets and iterations
        feat_shap_vals = {p: [] for p in range(P)}   # absolute shap values
        feat_ranks     = {p: [] for p in range(P)}   # rank positions

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

            for xi in X_target:
                try:
                    imp = get_shap_importances(model, X_tr, xi, feature_names)
                    vals = np.array([imp[f] for f in feature_names])
                    ranks = importance_to_ranking(imp, feature_names)
                    for p in range(P):
                        feat_shap_vals[p].append(vals[p])
                        feat_ranks[p].append(ranks[p])
                except Exception:
                    pass

        # Compute per-feature CV, mean rank, rank variance
        cv_arr      = np.zeros(P)
        rank_arr    = np.zeros(P)
        rankvar_arr = np.zeros(P)

        for p in range(P):
            vals = np.array(feat_shap_vals[p])
            nonzero = vals[vals > 0]
            if len(nonzero) >= 2 and nonzero.mean() > 0:
                cv_arr[p] = nonzero.std(ddof=1) / nonzero.mean()
            else:
                cv_arr[p] = np.nan

            r = np.array(feat_ranks[p], dtype=float)
            if len(r) >= 2:
                rank_arr[p]    = r.mean()
                rankvar_arr[p] = r.var()
            else:
                rank_arr[p]    = np.nan
                rankvar_arr[p] = np.nan

        results_cv[dr]      = cv_arr
        results_rank[dr]    = rank_arr
        results_rankvar[dr] = rankvar_arr

    # Build DataFrames
    cv_df      = pd.DataFrame(results_cv,      index=feature_names).T
    rank_df    = pd.DataFrame(results_rank,    index=feature_names).T
    rankvar_df = pd.DataFrame(results_rankvar, index=feature_names).T

    cv_df.index.name = "default_rate"
    rank_df.index.name = "default_rate"

    # Compute critical threshold per feature
    thresholds = {}
    for feat in feature_names:
        col = cv_df[feat].dropna()
        exceeded = col[col > INSTABILITY_THRESHOLD]
        if len(exceeded) > 0:
            thresholds[feat] = exceeded.index.max()   # highest DR where unstable
        else:
            thresholds[feat] = None    # always stable

    thresh_df = pd.Series(thresholds, name="critical_dr").to_frame()
    thresh_df.index.name = "feature"

    # Save
    cv_df.to_csv(f"experiment_e_cv_{dataset_name}.csv")
    rank_df.to_csv(f"experiment_e_rank_{dataset_name}.csv")
    thresh_df.to_csv(f"experiment_e_thresholds_{dataset_name}.csv")
    print(f"\n  Saved CSVs for {dataset_name}")

    return cv_df, rank_df, rankvar_df, thresh_df


# ════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_experiment_e(cv_df, rank_df, thresh_df, dataset_name, top_n=15):
    """
    3 plots:
      1. Heatmap of SHAP CV per feature per default rate
         (sorted by CV at most extreme imbalance — most unstable at top)
      2. Line chart: top-10 most unstable features CV over default rates
      3. Bar chart: critical threshold per feature
    """

    # Sort features by CV at most extreme imbalance (first row, ascending DR)
    most_extreme_dr = cv_df.index.min()
    feature_order   = cv_df.loc[most_extreme_dr].sort_values(ascending=False).index.tolist()
    top_features    = feature_order[:top_n]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Experiment E: Feature-Level SHAP Stability Heatmap — {dataset_name}\n"
        f"(Sorted by instability at {most_extreme_dr*100:.1f}% default rate)",
        fontsize=13, fontweight="bold"
    )

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)

    # ── Plot 1: Full heatmap ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    heat_data = cv_df[feature_order[:min(top_n, len(feature_order))]].T
    heat_data.columns = [f"{c*100:.0f}%" for c in heat_data.columns]

    cmap = plt.cm.RdYlGn_r   # red = unstable, green = stable
    im   = ax1.imshow(heat_data.values, cmap=cmap, aspect="auto",
                      vmin=0.3, vmax=1.0)

    ax1.set_xticks(range(len(heat_data.columns)))
    ax1.set_xticklabels(heat_data.columns, fontsize=9)
    ax1.set_yticks(range(len(heat_data.index)))
    ax1.set_yticklabels(heat_data.index, fontsize=9)
    ax1.set_xlabel("Default rate", fontsize=10)
    ax1.set_title(f"SHAP CV per feature (top {top_n} most unstable) — Red = less stable", fontsize=11)

    # Annotate cells with CV value
    for i in range(len(heat_data.index)):
        for j in range(len(heat_data.columns)):
            val = heat_data.values[i, j]
            if not np.isnan(val):
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=7, color="black" if 0.4 < val < 0.8 else "white")

    plt.colorbar(im, ax=ax1, label="SHAP CV (higher = less stable)", shrink=0.6)

    # ── Plot 2: Line chart top-5 vs bottom-5 features ───────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    top5    = feature_order[:5]
    bottom5 = feature_order[-5:]
    x = [dr * 100 for dr in cv_df.index]

    for feat in top5:
        ax2.plot(x, cv_df[feat].values, "o-", linewidth=1.5, markersize=3,
                 label=f"{feat} (unstable)", alpha=0.85)
    for feat in bottom5:
        ax2.plot(x, cv_df[feat].values, "s--", linewidth=1.2, markersize=3,
                 label=f"{feat} (stable)", alpha=0.6)

    ax2.axhline(INSTABILITY_THRESHOLD, color="red", linewidth=1, linestyle=":",
                label=f"Instability threshold ({INSTABILITY_THRESHOLD})")
    ax2.set_xlabel("Default rate (%)")
    ax2.set_ylabel("SHAP CV")
    ax2.set_title("Most vs least unstable features")
    ax2.invert_xaxis()
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Critical threshold bar chart ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    thresh_sorted = thresh_df["critical_dr"].dropna().sort_values(ascending=False)
    # Convert to percentage
    dr_vals = (thresh_sorted.values * 100).astype(float)
    colors  = ["#d62728" if v <= 5 else "#ff7f0e" if v <= 15 else "#2ca02c"
               for v in dr_vals]

    bars = ax3.barh(range(len(thresh_sorted)), dr_vals, color=colors, height=0.6)
    ax3.set_yticks(range(len(thresh_sorted)))
    ax3.set_yticklabels(thresh_sorted.index, fontsize=8)
    ax3.set_xlabel("Highest default rate (%) where CV > threshold\n(lower = unstable even at mild imbalance)")
    ax3.set_title(f"Feature instability threshold (CV > {INSTABILITY_THRESHOLD})")
    ax3.axvline(5,  color="red",    linestyle=":", linewidth=1, label="5% (mortgage rate)")
    ax3.axvline(10, color="orange", linestyle=":", linewidth=1, label="10%")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="x")

    plt.savefig(f"experiment_e_plot_{dataset_name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved → experiment_e_plot_{dataset_name}.png")


def print_summary_e(cv_df, thresh_df, dataset_name):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT E KEY NUMBERS — {dataset_name}")
    print(f"{'='*60}")

    most_extreme = cv_df.index.min()
    print(f"\nTop 5 most unstable features at {most_extreme*100:.1f}% default rate:")
    top5 = cv_df.loc[most_extreme].sort_values(ascending=False).head(5)
    for feat, cv in top5.items():
        crit = thresh_df.loc[feat, "critical_dr"] if feat in thresh_df.index else None
        crit_str = f"{crit*100:.1f}%" if crit is not None else "always stable"
        print(f"  {feat:<35s} CV={cv:.4f}  unstable until DR={crit_str}")

    print(f"\nTop 5 most stable features:")
    bot5 = cv_df.loc[most_extreme].sort_values(ascending=True).head(5)
    for feat, cv in bot5.items():
        print(f"  {feat:<35s} CV={cv:.4f}")

    # How many features are unstable at each DR
    print(f"\nFeatures exceeding CV>{INSTABILITY_THRESHOLD} threshold:")
    for dr in sorted(cv_df.index):
        n_unstable = (cv_df.loc[dr] > INSTABILITY_THRESHOLD).sum()
        print(f"  DR={dr*100:5.1f}%  →  {n_unstable}/{len(cv_df.columns)} features unstable")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["taiwan", "german", "both"], default="taiwan")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--targets",    type=int, default=50)
    parser.add_argument("--top_n",      type=int, default=15,
                        help="Number of features to show in heatmap")
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
            from replicate_paper import load_taiwan
            X, y, feat = load_taiwan()
            datasets.append((X, y, feat, "Taiwan"))
        except FileNotFoundError:
            print("⚠  taiwan_credit.csv not found")

    if args.dataset in ("german", "both"):
        try:
            from replicate_paper import load_german
            X, y, feat = load_german()
            datasets.append((X, y, feat, "German"))
        except FileNotFoundError:
            print("⚠  SouthGermanCredit.asc not found")

    for X, y, feat, dname in datasets:
        cv_df, rank_df, rankvar_df, thresh_df = run_experiment_e(
            X, y, feat,
            dataset_name=dname,
            default_rates=default_rates,
            n_iterations=n_iter,
            n_targets=n_targets,
        )
        plot_experiment_e(cv_df, rank_df, thresh_df, dname, top_n=args.top_n)
        print_summary_e(cv_df, thresh_df, dname)

    print("\n✅ Experiment E complete!")
