"""
05_clean_merge.py — Clean individual datasets and merge into one panel.

Reads from: data/raw/*.csv
Output:     data/processed/panel_merged.csv
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import (
    DATA_RAW, DATA_PROCESSED, YEAR_START, YEAR_END,
    EXCLUDE_INCOME_GROUPS, REGION_GROUPS, HORIZON_YEARS,
)

try:
    from config import INCLUDE_INCOME_GROUPS
except ImportError:
    INCLUDE_INCOME_GROUPS = ["LIC", "LMC", "UMC"]

# ── GDELT country code to ISO3 mapping ──────────────────────────────────────
# GDELT uses FIPS 10-4 codes (2 letters). This maps common ones to ISO 3166 alpha-3.
# Not exhaustive — expand as needed.
FIPS_TO_ISO3 = {
    "AR": "ARG", "BR": "BRA", "CH": "CHN", "CO": "COL", "EC": "ECU",
    "EG": "EGY", "ET": "ETH", "GH": "GHA", "GM": "DEU", "GR": "GRC",
    "ID": "IDN", "IN": "IND", "IR": "IRN", "IZ": "IRQ", "JA": "JPN",
    "KE": "KEN", "KS": "KOR", "LE": "LBN", "LY": "LBY", "MX": "MEX",
    "MY": "MYS", "NI": "NGA", "PK": "PAK", "PE": "PER", "PH": "PHL",
    "PO": "POL", "RS": "RUS", "SA": "SAU", "SF": "ZAF", "SN": "SEN",
    "SP": "ESP", "TH": "THA", "TU": "TUR", "TX": "TKM", "UA": "UKR",
    "UP": "UKR", "VE": "VEN", "VM": "VNM", "ZA": "ZMB", "ZI": "ZWE",
    "CE": "LKA", "CG": "COG", "CF": "COG", "IV": "CIV", "MZ": "MOZ",
    "SU": "SDN", "UG": "UGA", "TZ": "TZA", "AO": "AGO", "CM": "CMR",
    "BC": "BWA", "WA": "NAM", "RW": "RWA", "BY": "BDI", "HA": "HTI",
    "DR": "DOM", "JM": "JAM", "BL": "BOL", "CI": "CHL", "PA": "PRY",
    "UY": "URY", "BG": "BGD", "BM": "MMR", "CB": "KHM", "LA": "LAO",
    "RP": "PHL", "TI": "TJK", "KG": "KGZ", "UZ": "UZB", "KZ": "KAZ",
    "TK": "TCA", "MO": "MAR", "TS": "TUN", "AG": "DZA", "LI": "LBR",
    "ML": "MLI", "SG": "SEN", "NG": "NER", "BF": "BFA", "CT": "CAF",
    "CD": "TCD", "DJ": "DJI", "ER": "ERI", "SO": "SOM", "MG": "MDG",
}


INCOME_GROUP_ALIASES = {
    "High income": "HIC",
    "Low income": "LIC",
    "Lower middle income": "LMC",
    "Upper middle income": "UMC",
    "Not classified": "INX",
}


def normalized_income_exclusions():
    """Return excluded income groups using WDI short codes."""
    return {
        INCOME_GROUP_ALIASES.get(group, group)
        for group in EXCLUDE_INCOME_GROUPS
    }


def load_wdi():
    """Load and filter WDI data."""
    path = DATA_RAW / "wdi_panel.csv"
    df = pd.read_csv(path)

    # Keep only real, classified low- and middle-income economies.
    if "income_level" in df.columns and INCLUDE_INCOME_GROUPS:
        include = set(INCLUDE_INCOME_GROUPS)
        before = len(df)
        df = df[df["income_level"].isin(include)]
        print(f"  Kept {len(df)} WDI rows from income groups: {sorted(include)}")
        print(f"  Dropped {before - len(df)} rows outside the project country scope")

    # Backward-compatible exclusion hook.
    if "income_level" in df.columns:
        excluded = normalized_income_exclusions()
        before = len(df)
        df = df[~df["income_level"].isin(excluded)]
        if before != len(df):
            print(f"  Excluded {before - len(df)} WDI rows from income groups: {sorted(excluded)}")

    # Filter year range
    df = df[df["year"].between(YEAR_START, YEAR_END)]

    print(f"  WDI: {df['iso3c'].nunique()} countries, {len(df)} rows")
    return df


def load_nightlights():
    """Load nightlights data."""
    path = DATA_RAW / "nightlights_panel.csv"
    df = pd.read_csv(path)
    df = df[df["year"].between(YEAR_START, YEAR_END)]
    print(f"  Nightlights: {df['iso3c'].nunique()} countries, {len(df)} rows")
    return df


def load_gdelt(df_wdi=None):
    """Load GDELT data. GDELT 2.0 uses ISO3 codes in Actor1CountryCode."""
    path = DATA_RAW / "gdelt_panel.csv"
    df = pd.read_csv(path)

    # GDELT 2.0 actually uses ISO3 codes (not FIPS as originally thought)
    df["iso3c"] = df["gdelt_country_code"]

    df = df[df["year"].between(YEAR_START, YEAR_END)]

    # Drop GDELT-specific columns
    gdelt_cols = [c for c in df.columns if c not in ["gdelt_country_code"]]
    df = df[gdelt_cols]

    print(f"  GDELT: {df['iso3c'].nunique()} countries, {len(df)} rows")
    return df


def load_crisis_labels():
    """Load crisis labels."""
    path = DATA_RAW / "crisis_labels.csv"
    df = pd.read_csv(path)
    print(f"  Crisis: {df['iso3c'].nunique()} countries with crisis episodes")
    return df


def build_prediction_target(panel, horizon=HORIZON_YEARS):
    """
    Build forward-looking target variable:
    crisis_next_N = 1 if a crisis onset occurs within the next N years.
    """
    # Get onset years
    onsets = panel[panel.get("crisis_onset", pd.Series(dtype=int)) == 1][
        ["iso3c", "year"]
    ].copy()
    onsets = onsets.rename(columns={"year": "onset_year"})

    panel["crisis_target"] = 0
    for iso3c in panel["iso3c"].unique():
        country_onsets = onsets[onsets["iso3c"] == iso3c]["onset_year"].values
        for onset_year in country_onsets:
            # Mark years where onset is within horizon
            mask = (
                (panel["iso3c"] == iso3c)
                & (panel["year"] >= onset_year - horizon)
                & (panel["year"] < onset_year)
            )
            panel.loc[mask, "crisis_target"] = 1

    return panel


def main():
    print("Cleaning and merging datasets...")

    # Load all
    wdi = load_wdi()
    nightlights = load_nightlights()
    gdelt = load_gdelt()
    crisis = load_crisis_labels()

    # Start with WDI as base (it has the most complete country coverage)
    panel = wdi.copy()

    # Merge nightlights
    nl_cols = ["iso3c", "year", "nightlight_raw", "nightlight_growth"]
    nl_cols = [c for c in nl_cols if c in nightlights.columns]
    panel = panel.merge(nightlights[nl_cols], on=["iso3c", "year"], how="left")

    # Merge GDELT
    gdelt_cols = ["iso3c", "year"] + [
        c for c in gdelt.columns
        if c not in ["iso3c", "year", "gdelt_country_code"]
    ]
    panel = panel.merge(gdelt[gdelt_cols], on=["iso3c", "year"], how="left")

    # Merge crisis labels
    crisis_cols = [
        "iso3c", "year", "in_crisis", "crisis_onset",
        "crisis_start", "crisis_end", "crisis_source",
    ]
    crisis_cols = [c for c in crisis_cols if c in crisis.columns]
    panel = panel.merge(crisis[crisis_cols], on=["iso3c", "year"], how="left")

    # Fill missing crisis flags with 0 (no crisis)
    panel["in_crisis"] = panel["in_crisis"].fillna(0).astype(int)
    panel["crisis_onset"] = panel["crisis_onset"].fillna(0).astype(int)

    # Build prediction target
    panel = build_prediction_target(panel)

    # Compute the DIVERGENCE INDEX (key variable)
    # = nightlight growth - official GDP growth
    if "nightlight_growth" in panel.columns and "gdp_growth" in panel.columns:
        panel["divergence_index"] = (
            panel["nightlight_growth"] - panel["gdp_growth"]
        )
    else:
        print("  WARNING: Cannot compute divergence index (missing columns)")

    # Sort
    panel = panel.sort_values(["iso3c", "year"]).reset_index(drop=True)

    # Report coverage
    total = len(panel)
    crisis_count = panel["crisis_target"].sum()
    print(f"\n  Merged panel:")
    print(f"    {panel['iso3c'].nunique()} countries")
    print(f"    {panel['year'].min()}-{panel['year'].max()}")
    print(f"    {total} total observations")
    print(f"    {crisis_count} positive labels (crisis_target=1)")
    print(f"    {crisis_count/total*100:.1f}% crisis rate")

    # Report missing data
    print(f"\n  Missing data (% of total):")
    for col in panel.columns:
        pct = panel[col].isna().mean() * 100
        if pct > 0:
            print(f"    {col}: {pct:.1f}%")

    # Save
    out_path = DATA_PROCESSED / "panel_merged.csv"
    panel.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
