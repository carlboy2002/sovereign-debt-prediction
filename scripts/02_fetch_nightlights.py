"""
02_fetch_nightlights.py — Obtain country-level nighttime light data.

Nighttime lights serve as a satellite-derived proxy for true economic activity.
We need country-year level luminosity values from 2000-2023.

DATA SOURCES (choose one, instructions below):

    Option A (RECOMMENDED — easiest):
        Download the Harmonized DMSP-VIIRS country-level dataset from:
        https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGIVCD
        (Li et al., 2020 — "Harmonized Global Nighttime Light Dataset 1992-2020")
        Save the CSV to: data/raw/nightlights_harmonized.csv

    Option B (most complete, requires GIS processing):
        1. DMSP-OLS annual composites (1992-2013):
           https://eogdata.mines.edu/products/dmsp/
        2. VIIRS annual composites (2012-2023):
           https://eogdata.mines.edu/products/vnl/
        Save GeoTIFFs to: data/raw/nightlights_rasters/
        This script will aggregate to country level using Natural Earth boundaries.

    Option C (quick fallback — uses World Bank electricity data as proxy):
        No manual download needed. This script auto-downloads from WDI.
        Less accurate but gets you running immediately.

Output: data/raw/nightlights_panel.csv
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_RAW, YEAR_START, YEAR_END

# ── OPTION GEE (RECOMMENDED): Output from 02_fetch_nightlights_gee.py ───────


def aggregate_country_year(df):
    """Collapse multipart country features to one row per ISO3-year."""
    required = {"iso3c", "year", "nightlight_raw"}
    if not required.issubset(df.columns):
        return df

    before = len(df)
    agg_spec = {"nightlight_raw": "mean"}
    for col in ["nightlight_sum"]:
        if col in df.columns:
            agg_spec[col] = "sum"
    for col in ["country_name", "iso_code"]:
        if col in df.columns:
            agg_spec[col] = "first"

    df = df.groupby(["iso3c", "year"], as_index=False).agg(agg_spec)
    collapsed = before - len(df)
    if collapsed > 0:
        print(f"  Collapsed {collapsed} multipart country-year rows")
    return df


def process_gee_csv():
    """
    Process the CSV exported from Google Earth Engine.
    Maps FAO GAUL country codes to ISO3 using the country_name field.
    """
    path = DATA_RAW / "nightlights_by_country.csv"
    if not path.exists():
        return None

    print("  Processing GEE nightlights export...")
    df = pd.read_csv(path)

    # GEE uses FAO GAUL numeric codes. Map via country name to ISO3.
    # Load WDI country list as the authoritative source for names -> ISO3
    wdi_path = DATA_RAW / "wdi_panel.csv"
    if wdi_path.exists():
        wdi = pd.read_csv(wdi_path)
        name_to_iso = (
            wdi[["iso3c", "country_name"]]
            .drop_duplicates(subset=["country_name"])
            .set_index("country_name")["iso3c"]
            .to_dict()
        )
    else:
        print("  WARNING: wdi_panel.csv not found. Run 01_fetch_wdi.py first.")
        return None

    # Some GAUL names differ from WDI names — add manual overrides for emerging markets
    name_overrides = {
        "United States of America": "USA",
        "Russian Federation": "RUS",
        "Iran  (Islamic Republic of)": "IRN",
        "Iran (Islamic Republic of)": "IRN",
        "Syrian Arab Republic": "SYR",
        "Venezuela (Bolivarian Republic of)": "VEN",
        "Bolivia (Plurinational State of)": "BOL",
        "Lao People's Democratic Republic": "LAO",
        "Viet Nam": "VNM",
        "Republic of Korea": "KOR",
        "Democratic People's Republic of Korea": "PRK",
        "United Republic of Tanzania": "TZA",
        "Egypt": "EGY",
        "Libya": "LBY",
        "Somalia": "SOM",
        "Swaziland": "SWZ",
        "Cape Verde": "CPV",
        "Kyrgyzstan": "KGZ",
        "Yemen": "YEM",
        "Moldova, Republic of": "MDA",
        "Dem People's Rep of Korea": "PRK",
        "Kosovo": "XKX",
        "South Sudan": "SSD",
        "Micronesia (Federated States of)": "FSM",
        "Congo": "COG",
        "Democratic Republic of the Congo": "COD",
        "C\u00f4te d'Ivoire": "CIV",
        "Cote d'Ivoire": "CIV",
        "Czech Republic": "CZE",
        "Czechia": "CZE",
        "Slovakia": "SVK",
        "The former Yugoslav Republic of Macedonia": "MKD",
        "Republic of Moldova": "MDA",
        "Brunei Darussalam": "BRN",
        "Timor-Leste": "TLS",
        "Turkey": "TUR",
        "Gambia": "GMB",
        "The Gambia": "GMB",
        "United Kingdom": "GBR",
    }
    name_to_iso.update(name_overrides)

    df["iso3c"] = df["country_name"].map(name_to_iso)

    # Report unmapped
    unmapped = df[df["iso3c"].isna()]["country_name"].unique()
    if len(unmapped) > 0:
        print(f"  WARNING: {len(unmapped)} country names could not be mapped to ISO3:")
        for name in unmapped[:20]:
            print(f"    - {name}")

    # Drop unmapped rows
    df = df.dropna(subset=["iso3c"])

    # Rename for consistency
    df = df.rename(columns={"nl_mean": "nightlight_raw"})

    # Only use VIIRS era (2014+) to avoid DMSP/VIIRS unit mismatch
    df = df[df["year"].between(2014, YEAR_END)]
    df = aggregate_country_year(df)
    print(f"  Mapped {df['iso3c'].nunique()} countries to ISO3")
    return df


# ── OPTION A: Pre-aggregated harmonized dataset ─────────────────────────────

def process_harmonized_csv():
    """
    Process the Li et al. (2020) harmonized nightlights dataset.
    Expected format: columns include ISO3, Year, and a luminosity measure.
    Adjust column names below based on the actual file you download.
    """
    path = DATA_RAW / "nightlights_harmonized.csv"
    if not path.exists():
        return None

    print("  Processing harmonized nightlights dataset (Option A)...")
    df = pd.read_csv(path)

    # The Li et al. dataset may have different column names
    # Common variants — adjust as needed after inspecting the file:
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "iso" in cl or "country" in cl or "code" in cl:
            col_map[col] = "iso3c"
        elif "year" in cl:
            col_map[col] = "year"
        elif "light" in cl or "lumin" in cl or "ntl" in cl or "sol" in cl:
            col_map[col] = "nightlight_raw"

    df = df.rename(columns=col_map)

    if "iso3c" not in df.columns or "year" not in df.columns:
        print("  WARNING: Could not auto-detect columns. Please inspect the CSV")
        print(f"  Available columns: {list(df.columns)}")
        return None

    df = df[df["year"].between(YEAR_START, YEAR_END)]
    return df[["iso3c", "year", "nightlight_raw"]]


# ── OPTION B: Aggregate from rasters ────────────────────────────────────────

def aggregate_from_rasters():
    """
    Compute country-level mean radiance from GeoTIFF rasters.
    Requires: rasterio, geopandas, rasterstats
    """
    raster_dir = DATA_RAW / "nightlights_rasters"
    if not raster_dir.exists() or not list(raster_dir.glob("*.tif")):
        return None

    print("  Aggregating nightlights from rasters (Option B)...")

    try:
        import geopandas as gpd
        from rasterstats import zonal_stats
    except ImportError:
        print("  ERROR: Install geopandas and rasterstats: pip install geopandas rasterstats")
        return None

    # Download Natural Earth country boundaries (110m resolution is fine)
    world = gpd.read_file(
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    world = world[["ISO_A3", "NAME", "geometry"]].rename(
        columns={"ISO_A3": "iso3c", "NAME": "country_name"}
    )
    world = world[world["iso3c"] != "-99"]  # Remove Antarctica etc.

    records = []
    for tif_path in sorted(raster_dir.glob("*.tif")):
        # Extract year from filename (e.g., "VNL_v21_2020.tif" -> 2020)
        year = None
        for part in tif_path.stem.split("_"):
            if part.isdigit() and len(part) == 4:
                year = int(part)
                break

        if year is None or not (YEAR_START <= year <= YEAR_END):
            continue

        print(f"    Processing {tif_path.name} ({year})...")
        stats = zonal_stats(
            world, str(tif_path), stats=["mean", "sum"], nodata=0
        )
        for i, row in world.iterrows():
            records.append({
                "iso3c": row["iso3c"],
                "year": year,
                "nightlight_raw": stats[i]["mean"],
                "nightlight_sum": stats[i]["sum"],
            })

    if not records:
        return None

    return pd.DataFrame(records)


# ── OPTION C: World Bank electricity proxy ──────────────────────────────────

def fetch_electricity_proxy():
    """
    Use World Bank electricity consumption per capita as a proxy.
    Not ideal but gets the pipeline running immediately.
    """
    print("  Using electricity consumption proxy (Option C — fallback)...")
    try:
        import wbgapi as wb
    except ImportError:
        print("  ERROR: pip install wbgapi")
        return None

    # EG.USE.ELEC.KH.PC = Electric power consumption (kWh per capita)
    df = wb.data.DataFrame(
        "EG.USE.ELEC.KH.PC",
        time=range(YEAR_START, YEAR_END + 1),
        columns="time",
    ).reset_index()

    df = df.melt(id_vars=["economy"], var_name="year", value_name="nightlight_raw")
    df["year"] = df["year"].str.replace("YR", "").astype(int)
    df = df.rename(columns={"economy": "iso3c"})

    print("  WARNING: This is electricity consumption, not actual nightlights.")
    print("  Replace with satellite data (Option A or B) before final submission.")
    return df


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Fetching nighttime lights data...")

    # Try each option in order
    df = process_gee_csv()
    if df is None:
        df = process_harmonized_csv()
    if df is None:
        df = aggregate_from_rasters()
    if df is None:
        df = fetch_electricity_proxy()
    if df is None:
        print("ERROR: No nightlights data available. See docstring for options.")
        sys.exit(1)

    # Compute derived columns
    df = aggregate_country_year(df)
    df = df.sort_values(["iso3c", "year"])
    df["nightlight_growth"] = (
        df.groupby("iso3c")["nightlight_raw"]
        .pct_change() * 100
    )

    # Save
    out_path = DATA_RAW / "nightlights_panel.csv"
    df.to_csv(out_path, index=False)
    n = df["iso3c"].nunique()
    print(f"  Saved: {out_path}")
    print(f"  {n} countries, {df['year'].min()}-{df['year'].max()}")


if __name__ == "__main__":
    main()
