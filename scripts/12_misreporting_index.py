"""
12_misreporting_index.py — Country-level statistical misreporting index.

Per TA feedback: produce an indicator of estimated degree of misreporting by country.

Approach:
    Misreporting index = (official GDP growth − satellite nightlight growth)
    A country consistently reporting higher growth than satellites suggest is
    flagged as a suspected over-reporter.

We use the VIIRS era only (2014-2023) to avoid the DMSP/VIIRS sensor mismatch.

Output: outputs/tables/misreporting_index.csv, outputs/figures/
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_PROCESSED, FIGURES, TABLES


def compute_raw_index(df):
    """
    Raw misreporting index = GDP growth - nightlight growth.
    Positive = government reports higher growth than satellites show.
    """
    if "divergence_index" not in df.columns:
        print("  ERROR: divergence_index not found. Run 05_clean_merge.py first.")
        return None

    # divergence_index = nightlight_growth - gdp_growth
    # misreport = -divergence = gdp_growth - nightlight_growth
    df["misreport_raw"] = -df["divergence_index"]

    return df


def compute_country_index(df):
    """
    Aggregate to country level with multiple robust metrics.
    """
    # Country-level average (kept for reference)
    country_avg = df.groupby("iso3c").agg(
        misreport_mean=("misreport_raw", "mean"),
        misreport_std=("misreport_raw", "std"),
        misreport_median=("misreport_raw", "median"),
        n_years=("misreport_raw", "count"),
        gdp_growth_mean=("gdp_growth", "mean"),
        nightlight_growth_mean=("nightlight_growth", "mean"),
    ).reset_index()

    # T-test: is mean significantly > 0?
    country_avg["t_stat"] = (
        country_avg["misreport_mean"]
        / (country_avg["misreport_std"] / np.sqrt(country_avg["n_years"]))
    )
    country_avg["significant"] = (country_avg["t_stat"].abs() > 1.96).astype(int)

    # ── Filter out saturated/tiny economies ─────────────────────────────────
    # Developed countries have saturated nightlights (NYC is already bright);
    # tiny island economies have huge percentage noise. Keep only countries
    # with moderate nightlight variation — these are where the signal is
    # actually meaningful.
    nl_std_by_country = (
        df.groupby("iso3c")["nightlight_growth"]
        .std()
        .reset_index(name="nl_growth_std")
    )
    valid_countries = nl_std_by_country[
        (nl_std_by_country["nl_growth_std"] > 5) &
        (nl_std_by_country["nl_growth_std"] < 30)
    ]
    df_filtered = df[df["iso3c"].isin(valid_countries["iso3c"])].copy()
    print(f"  Filtered to {df_filtered['iso3c'].nunique()} countries with "
          f"meaningful nightlight variation")

    # ── Cross-country Z-score ───────────────────────────────────────────────
    # How abnormal is a country-year relative to the GLOBAL distribution?
    global_mean = df_filtered["misreport_raw"].mean()
    global_std = df_filtered["misreport_raw"].std()
    df_filtered["misreport_zscore"] = (
        df_filtered["misreport_raw"] - global_mean
    ) / (global_std + 1e-8)

    # ── Anomaly flags per country-year ──────────────────────────────────────
    # A year is "suspicious" if z-score > 1.0 (strongly positive divergence)
    df_filtered["is_anomaly"] = (df_filtered["misreport_zscore"] > 1.0).astype(int)

    anomaly_counts = (
        df_filtered.groupby("iso3c")
        .agg(
            anomaly_count=("is_anomaly", "sum"),
            peak_anomaly_zscore=("misreport_zscore", "max"),
            years_observed=("year", "count"),
        )
        .reset_index()
    )
    anomaly_counts["anomaly_rate"] = (
        anomaly_counts["anomaly_count"] / anomaly_counts["years_observed"]
    )

    # Year of peak anomaly
    peak_year_idx = df_filtered.groupby("iso3c")["misreport_zscore"].idxmax()
    peak_year = df_filtered.loc[peak_year_idx.dropna(), ["iso3c", "year"]].rename(
        columns={"year": "peak_anomaly_year"}
    )

    # ── Merge everything ────────────────────────────────────────────────────
    country_avg = country_avg.merge(anomaly_counts, on="iso3c", how="inner")
    country_avg = country_avg.merge(peak_year, on="iso3c", how="left")

    # ── Rank by anomaly rate (% of years flagged), then by peak severity ────
    country_avg = country_avg.sort_values(
        ["anomaly_rate", "peak_anomaly_zscore"], ascending=[False, False]
    )
    country_avg["rank"] = range(1, len(country_avg) + 1)

    # Country-year level (for time-series plots)
    country_year = df_filtered.groupby(["iso3c", "year"]).agg(
        misreport_raw=("misreport_raw", "mean"),
        misreport_zscore=("misreport_zscore", "mean"),
    ).reset_index()

    return country_avg, country_year


def add_metadata(country_avg, df):
    """Add country names and regions."""
    meta_cols = ["iso3c", "country_name", "region_code", "income_level"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    if meta_cols:
        meta = df[meta_cols].drop_duplicates(subset=["iso3c"])
        country_avg = country_avg.merge(meta, on="iso3c", how="left")
    return country_avg


def plot_top_misreporters(country_avg, n=25):
    """Bar chart of top suspected misreporters by anomaly rate."""
    top = country_avg.head(n).copy()
    if "country_name" in top.columns:
        labels = top["country_name"].fillna(top["iso3c"])
    else:
        labels = top["iso3c"]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d32f2f" if r > 0.3 else "#f57c00" if r > 0.15 else "#1976d2"
              for r in top["anomaly_rate"]]
    ax.barh(range(len(top)), top["anomaly_rate"] * 100, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("% of Years with Suspicious Over-Reporting (z > 1.5)")
    ax.set_title(f"Top {n} Suspected Over-Reporters\n"
                 "(GDP Growth Significantly Exceeds Nightlight Growth)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES / "misreporting_top_countries.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved misreporting_top_countries.png")


def plot_misreporting_over_time(country_year, country_avg):
    """Time series of misreporting z-score for top 8 countries."""
    top_countries = country_avg.head(8)["iso3c"].tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    for iso3c in top_countries:
        subset = country_year[country_year["iso3c"] == iso3c].sort_values("year")
        if len(subset) > 0:
            ax.plot(subset["year"], subset["misreport_zscore"], "o-",
                    label=iso3c, markersize=5)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(1.5, color="red", linestyle=":", alpha=0.5, label="Anomaly threshold (z=1.5)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Misreporting Z-Score")
    ax.set_title("Misreporting Over Time — Top 8 Suspected Over-Reporters")
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "misreporting_time_series.png")
    plt.close()
    print("  Saved misreporting_time_series.png")


def main():
    print("Computing misreporting index...")

    # Load data (train + val only, don't touch test set)
    df = pd.read_csv(DATA_PROCESSED / "panel_features.csv")
    df = df[df["split"].isin(["train", "val"])].copy()

    # Compute raw index
    df = compute_raw_index(df)
    if df is None:
        return

    # Aggregate
    country_avg, country_year = compute_country_index(df)

    # Add country names/regions
    country_avg = add_metadata(country_avg, df)

    # Save
    country_avg.to_csv(TABLES / "misreporting_index.csv", index=False)
    country_year.to_csv(TABLES / "misreporting_index_by_year.csv", index=False)
    print(f"  Saved misreporting_index.csv ({len(country_avg)} countries)")

    # Print top 15
    print("\n  Top 15 suspected over-reporters (by % of suspicious years):")
    display_cols = ["rank", "iso3c", "anomaly_rate", "anomaly_count",
                    "peak_anomaly_zscore", "peak_anomaly_year", "years_observed"]
    if "country_name" in country_avg.columns:
        display_cols.insert(2, "country_name")
    print(country_avg[display_cols].head(15).to_string(index=False))

    # Plots
    plot_top_misreporters(country_avg)
    plot_misreporting_over_time(country_year, country_avg)

    # Summary stats
    n_flagged = (country_avg["anomaly_rate"] > 0.3).sum()
    print(f"\n  {n_flagged} countries flagged in >30% of observed years")


if __name__ == "__main__":
    main()
