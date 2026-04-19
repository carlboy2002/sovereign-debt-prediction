"""
01_fetch_wdi.py — Download macroeconomic indicators from World Bank WDI.
Uses the wbgapi package (official World Bank Python API).

Output: data/raw/wdi_panel.csv
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import wbgapi as wb
import pandas as pd
from config import DATA_RAW, WDI_INDICATORS, YEAR_START, YEAR_END

def fetch_wdi():
    print("Fetching World Bank WDI data...")

    frames = []
    for indicator_code, indicator_name in WDI_INDICATORS.items():
        print(f"  Downloading {indicator_name} ({indicator_code})...")
        try:
            df = wb.data.DataFrame(
                indicator_code,
                time=range(YEAR_START, YEAR_END + 1),
                columns="time",
            )
            # wb returns wide format with years as columns
            # Reset index: economy (country code) becomes a column
            df = df.reset_index()
            year_cols = [c for c in df.columns if c.startswith("YR")]
            df = df[["economy"] + year_cols]
            df = df.melt(
                id_vars=["economy"],
                value_vars=year_cols,
                var_name="year",
                value_name=indicator_name,
                )
            # Clean year column: "YR2000" -> 2000
            df["year"] = df["year"].str.replace("YR", "").astype(int)
            df = df.rename(columns={"economy": "iso3c"})
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: Could not download {indicator_code}: {e}")

    # Merge all indicators
    if not frames:
        print("ERROR: No data downloaded.")
        return

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["iso3c", "year"], how="outer")

    # Add country metadata (name, region, income group)
    print("  Adding country metadata...")
    meta = wb.economy.DataFrame().reset_index()
    meta = meta[["id", "name", "aggregate", "region", "incomeLevel"]].copy()
    meta = meta.rename(columns={
        "id": "iso3c",
        "name": "country_name",
        "region": "region_code",
        "incomeLevel": "income_level",
    })
    # Keep only non-aggregate economies (countries, not regions)
    meta = meta[meta["aggregate"] == False].drop(columns=["aggregate"])

    result = result.merge(meta, on="iso3c", how="left")

    # Save
    out_path = DATA_RAW / "wdi_panel.csv"
    result.to_csv(out_path, index=False)
    n_countries = result["iso3c"].nunique()
    n_years = result["year"].nunique()
    print(f"  Saved: {out_path}")
    print(f"  {n_countries} countries x {n_years} years = {len(result)} rows")
    print(f"  Columns: {list(result.columns)}")

if __name__ == "__main__":
    fetch_wdi()
