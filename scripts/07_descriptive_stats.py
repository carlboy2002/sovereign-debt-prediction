"""
07_descriptive_stats.py — Exploratory data analysis and visualizations.

Produces:
    - Summary statistics table
    - Crisis timeline chart
    - Divergence index distribution
    - Correlation heatmap
    - Map of crisis episodes

Output: outputs/figures/ and outputs/tables/
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PROCESSED, FIGURES, TABLES

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})


def summary_table(df):
    """Generate summary statistics by crisis/non-crisis."""
    feature_cols = [
        "gdp_growth", "inflation", "debt_to_gdp", "current_account_gdp",
        "reserves_months", "nightlight_growth", "divergence_index",
        "avg_tone", "avg_goldstein",
    ]
    existing = [c for c in feature_cols if c in df.columns]

    summary = df.groupby("crisis_target")[existing].agg(["mean", "std", "count"])
    summary.to_csv(TABLES / "summary_stats_by_crisis.csv")
    print("  Saved summary_stats_by_crisis.csv")

    # Print key differences
    print("\n  Mean comparison (crisis=0 vs crisis=1):")
    for col in existing:
        m0 = df.loc[df["crisis_target"] == 0, col].mean()
        m1 = df.loc[df["crisis_target"] == 1, col].mean()
        print(f"    {col:30s}: {m0:8.2f} vs {m1:8.2f}  (diff = {m1-m0:+.2f})")


def plot_crisis_timeline(df):
    """Timeline of crisis episodes by country."""
    crisis_obs = df[df["in_crisis"] == 1][["iso3c", "year"]].drop_duplicates()
    if crisis_obs.empty:
        print("  No crisis episodes to plot.")
        return

    countries = sorted(crisis_obs["iso3c"].unique())
    country_idx = {c: i for i, c in enumerate(countries)}

    fig, ax = plt.subplots(figsize=(14, max(6, len(countries) * 0.3)))
    for _, row in crisis_obs.iterrows():
        ax.barh(
            country_idx[row["iso3c"]], 1, left=row["year"],
            height=0.8, color="#d32f2f", alpha=0.8
        )

    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=8)
    ax.set_xlabel("Year")
    ax.set_title("Sovereign Debt Crisis Episodes by Country")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES / "crisis_timeline.png", bbox_inches="tight")
    plt.close()
    print("  Saved crisis_timeline.png")


def plot_divergence_distribution(df):
    """Distribution of divergence index: crisis vs non-crisis."""
    col = "divergence_index"
    if col not in df.columns:
        print(f"  Skipping divergence plot ({col} not found)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [(0, "#1976d2"), (1, "#d32f2f")]:
        subset = df[df["crisis_target"] == label][col].dropna()
        if len(subset) > 0:
            ax.hist(
                subset, bins=40, alpha=0.6, color=color, density=True,
                label=f"{'Crisis' if label else 'No crisis'} (n={len(subset)})"
            )

    ax.set_xlabel("Divergence Index (Nightlight Growth − Official GDP Growth)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Divergence Index by Crisis Status")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "divergence_distribution.png")
    plt.close()
    print("  Saved divergence_distribution.png")


def plot_correlation_heatmap(df):
    """Correlation heatmap of key features."""
    feature_cols = [
        "gdp_growth", "inflation", "debt_to_gdp", "current_account_gdp",
        "reserves_months", "nightlight_growth", "divergence_index",
        "avg_tone", "avg_goldstein", "crisis_target",
    ]
    existing = [c for c in feature_cols if c in df.columns]
    corr = df[existing].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        square=True, ax=ax, vmin=-1, vmax=1,
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "correlation_heatmap.png")
    plt.close()
    print("  Saved correlation_heatmap.png")


def plot_feature_over_time(df):
    """Average divergence index over time for crisis vs non-crisis countries."""
    col = "divergence_index"
    if col not in df.columns:
        return

    yearly = df.groupby(["year", "crisis_target"])[col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, name in [(0, "#1976d2", "No crisis"), (1, "#d32f2f", "Crisis")]:
        subset = yearly[yearly["crisis_target"] == label]
        ax.plot(subset["year"], subset[col], "o-", color=color, label=name, alpha=0.8)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Divergence Index")
    ax.set_title("Average Divergence Index Over Time")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "divergence_over_time.png")
    plt.close()
    print("  Saved divergence_over_time.png")


def main():
    print("Running descriptive statistics...")

    df = pd.read_csv(DATA_PROCESSED / "panel_features.csv")
    print(f"  Loaded: {df.shape}")

    # Use only train + val for EDA (never look at test set)
    df_eda = df[df["split"].isin(["train", "val"])].copy()
    print(f"  EDA sample (excl. test): {len(df_eda)} rows")

    summary_table(df_eda)
    plot_crisis_timeline(df_eda)
    plot_divergence_distribution(df_eda)
    plot_correlation_heatmap(df_eda)
    plot_feature_over_time(df_eda)

    print("\nDone.")


if __name__ == "__main__":
    main()
