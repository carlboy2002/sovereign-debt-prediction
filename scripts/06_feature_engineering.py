"""
06_feature_engineering.py — Construct all features for modeling.

Takes the merged panel and produces the final analysis-ready dataset with:
    - Lagged features (t-1, t-2, t-3)
    - Rolling statistics (volatility, trends)
    - Interaction terms
    - Train / validation / test split markers

Output: data/processed/panel_features.csv
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import (
    DATA_PROCESSED, LAGS, SEED,
    TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
)


def add_lags(df, cols, lags=LAGS):
    """Add lagged versions of specified columns."""
    df = df.sort_values(["iso3c", "year"])
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            df[lag_col] = df.groupby("iso3c")[col].shift(lag)
    return df


def add_rolling_stats(df, cols, window=3):
    """Add rolling mean and std (volatility) over past N years."""
    df = df.sort_values(["iso3c", "year"])
    for col in cols:
        if col not in df.columns:
            continue
        grouped = df.groupby("iso3c")[col]
        df[f"{col}_roll_mean_{window}y"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=2).mean()
        )
        df[f"{col}_roll_std_{window}y"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=2).std()
        )
    return df


def add_yoy_changes(df, cols):
    """Add year-over-year changes."""
    df = df.sort_values(["iso3c", "year"])
    for col in cols:
        if col not in df.columns:
            continue
        df[f"{col}_change"] = df.groupby("iso3c")[col].diff()
    return df


def add_split_labels(df):
    """Add train/val/test split labels."""
    conditions = [
        df["year"] <= TRAIN_END,
        df["year"].between(VAL_START, VAL_END),
        df["year"].between(TEST_START, TEST_END),
    ]
    choices = ["train", "val", "test"]
    df["split"] = np.select(conditions, choices, default="train")
    return df


def define_feature_blocks(df):
    """
    Identify which columns belong to which feature block.
    Returns a dict mapping block name to list of column names.
    Used downstream for nested model comparison.
    """
    all_cols = set(df.columns)

    # Macro features: anything from WDI
    macro_base = [
        "gdp_growth", "inflation", "debt_to_gdp", "current_account_gdp",
        "reserves_months", "external_debt_gni", "fiscal_balance_gdp",
        "exchange_rate",
    ]
    macro_features = [c for c in all_cols if any(c.startswith(m) for m in macro_base)]

    # Satellite features: nightlight-derived
    sat_base = ["nightlight_raw", "nightlight_growth", "divergence_index"]
    sat_features = [c for c in all_cols if any(c.startswith(s) for s in sat_base)]

    # Text features: GDELT-derived
    text_base = [
        "avg_tone", "total_events", "economic_events", "conflict_events",
        "protest_events", "avg_goldstein", "avg_mentions",
        "econ_event_share", "conflict_event_share", "tone_change",
    ]
    text_features = [c for c in all_cols if any(c.startswith(t) for t in text_base)]

    blocks = {
        "macro": sorted(macro_features),
        "satellite": sorted(sat_features),
        "text": sorted(text_features),
        "all": sorted(set(macro_features) | set(sat_features) | set(text_features)),
    }

    return blocks


def main():
    print("Engineering features...")

    # Load merged panel
    df = pd.read_csv(DATA_PROCESSED / "panel_merged.csv")
    print(f"  Input: {len(df)} rows, {df.shape[1]} columns")

    # ── Core features to lag ────────────────────────────────────────────────
    lag_cols = [
        # Macro
        "gdp_growth", "inflation", "debt_to_gdp", "current_account_gdp",
        "reserves_months", "external_debt_gni", "fiscal_balance_gdp",
        # Satellite
        "nightlight_raw", "nightlight_growth", "divergence_index",
        # GDELT
        "avg_tone", "economic_events", "conflict_events",
        "avg_goldstein", "econ_event_share", "tone_change",
    ]
    existing_lag_cols = [c for c in lag_cols if c in df.columns]

    # Add lags
    print(f"  Adding lags ({LAGS}) for {len(existing_lag_cols)} features...")
    df = add_lags(df, existing_lag_cols)

    # ── Rolling statistics ──────────────────────────────────────────────────
    vol_cols = ["gdp_growth", "nightlight_growth", "avg_tone", "exchange_rate"]
    existing_vol_cols = [c for c in vol_cols if c in df.columns]
    print(f"  Adding rolling stats for {len(existing_vol_cols)} features...")
    df = add_rolling_stats(df, existing_vol_cols, window=3)

    # ── Year-over-year changes ──────────────────────────────────────────────
    change_cols = ["debt_to_gdp", "reserves_months", "external_debt_gni"]
    existing_change_cols = [c for c in change_cols if c in df.columns]
    df = add_yoy_changes(df, existing_change_cols)

    # ── Interaction: divergence × debt level ────────────────────────────────
    if "divergence_index_lag1" in df.columns and "debt_to_gdp_lag1" in df.columns:
        df["divergence_x_debt_lag1"] = (
            df["divergence_index_lag1"] * df["debt_to_gdp_lag1"]
        )
        print("  Added interaction: divergence × debt_to_gdp")

    # ── Train/val/test split ────────────────────────────────────────────────
    df = add_split_labels(df)

    # ── Identify feature blocks ─────────────────────────────────────────────
    blocks = define_feature_blocks(df)
    print(f"\n  Feature blocks:")
    for name, cols in blocks.items():
        print(f"    {name}: {len(cols)} features")

    # Save feature block definitions
    import json
    block_path = DATA_PROCESSED / "feature_blocks.json"
    with open(block_path, "w") as f:
        json.dump(blocks, f, indent=2)
    print(f"  Saved feature blocks: {block_path}")

    # ── Report split sizes ──────────────────────────────────────────────────
    print(f"\n  Split sizes:")
    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name]
        n_pos = subset["crisis_target"].sum()
        n_tot = len(subset)
        print(f"    {split_name}: {n_tot} obs, {n_pos} positive ({n_pos/max(n_tot,1)*100:.1f}%)")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = DATA_PROCESSED / "panel_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Final shape: {df.shape}")


if __name__ == "__main__":
    main()
