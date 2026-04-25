"""
02_fetch_nightlights_gee.py - compute country-level nightlights data with Google Earth Engine

This script runs all computations in the cloud, so the local machine does not
need to process large files.

Prerequisites:
    1. Register for GEE: https://earthengine.google.com -> Register
    2. pip install earthengine-api
    3. earthengine authenticate (browser authorization)

Output:
    - Google Drive will contain nightlights_by_country.csv.
    - Download it to data/raw/nightlights_by_country.csv.

Data sources:
    - Li et al. (2020) Harmonized NTL dataset, available in the GEE community catalog.
    - FAO GAUL country boundaries.
"""

import ee
import time

# Initialization
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from config import GCP_PROJECT_ID

    project_id = GCP_PROJECT_ID
except ImportError:
    project_id = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id-here")

ee.Initialize(project=project_id)

print("Google Earth Engine initialized successfully")

# Load data

# Country boundaries (FAO GAUL 2015).
countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
print(f"  Country boundaries loaded: {countries.size().getInfo()} countries/territories")

# Nightlights data from the DMSP and VIIRS datasets available in GEE.
# DMSP-OLS: 1992-2013
dmsp = ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS").select("stable_lights")
# VIIRS-DNB: 2014-present
viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").select("avg_rad")


# Core functions: compute country-level nightlights values for one year.


def compute_country_stats_dmsp(year):
    """Compute country-level DMSP nightlights statistics for a given year."""
    year = ee.Number(year)

    # DMSP has one to two satellites per year; use the mean of all images in the year.
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    annual = dmsp.filterDate(start, end).mean()

    # Compute the mean and sum for each country.
    def reduce_country(feature):
        stats = annual.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True),
            geometry=feature.geometry(),
            scale=1000,  # 1 km resolution.
            maxPixels=1e9,
            bestEffort=True,
        )
        return feature.set(
            {
                "year": year,
                "nl_mean": stats.get("stable_lights_mean"),
                "nl_sum": stats.get("stable_lights_sum"),
                "country_name": feature.get("ADM0_NAME"),
                "iso_code": feature.get("ADM0_CODE"),
            }
        )

    return countries.map(reduce_country)


def compute_country_stats_viirs(year):
    """Compute country-level VIIRS nightlights statistics for a given year."""
    year = ee.Number(year)

    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    annual = viirs.filterDate(start, end).mean()

    # Remove negative values, which are noise.
    annual = annual.max(0)

    def reduce_country(feature):
        stats = annual.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True),
            geometry=feature.geometry(),
            scale=1000,
            maxPixels=1e9,
            bestEffort=True,
        )
        return feature.set(
            {
                "year": year,
                "nl_mean": stats.get("avg_rad_mean"),
                "nl_sum": stats.get("avg_rad_sum"),
                "country_name": feature.get("ADM0_NAME"),
                "iso_code": feature.get("ADM0_CODE"),
            }
        )

    return countries.map(reduce_country)


# Main workflow


def main():
    print("\nStarting country-level nightlights computation...")
    print("All computations run on Google servers; please be patient.\n")

    # DMSP years: 2000-2013.
    dmsp_years = list(range(2000, 2014))
    # VIIRS years: 2014-2023.
    viirs_years = list(range(2014, 2024))

    all_features = []

    # Process DMSP data (2000-2013).
    print("=== DMSP (2000-2013) ===")
    for year in dmsp_years:
        print(f"  Processing {year}...", end=" ", flush=True)
        fc = compute_country_stats_dmsp(year)
        all_features.append(fc)
        print("OK")

    # Process VIIRS data (2014-2023).
    print("\n=== VIIRS (2014-2023) ===")
    for year in viirs_years:
        print(f"  Processing {year}...", end=" ", flush=True)
        fc = compute_country_stats_viirs(year)
        all_features.append(fc)
        print("OK")

    # Merge all years.
    print("\nMerging all years...")
    merged = ee.FeatureCollection(all_features).flatten()

    # Export to Google Drive.
    print("Exporting to Google Drive...")
    task = ee.batch.Export.table.toDrive(
        collection=merged,
        description="nightlights_by_country",
        fileNamePrefix="nightlights_by_country",
        fileFormat="CSV",
        selectors=["country_name", "iso_code", "year", "nl_mean", "nl_sum"],
    )
    task.start()
    print(f"  Export task submitted: {task.status()['state']}")

    # Wait for completion.
    print("\nWaiting for export to complete; this usually takes 10-30 minutes...")
    print("You can also check progress at https://code.earthengine.google.com/tasks\n")

    while task.active():
        status = task.status()
        print(f"  Status: {status['state']}...", flush=True)
        time.sleep(30)

    final_status = task.status()
    if final_status["state"] == "COMPLETED":
        print("\nExport complete.")
        print("  File location: Google Drive root directory -> nightlights_by_country.csv")
        print("  Download it to: data/raw/nightlights_by_country.csv")
    else:
        print(f"\nExport failed: {final_status}")
        print("  Common causes: timeout or insufficient memory")
        print("  Suggested fix: reduce the year range and export in batches")


# Alternative: batch exports if a single full export times out.


def export_single_year(year, sensor="viirs"):
    """Export one year separately for troubleshooting."""
    ee.Initialize()

    if sensor == "dmsp":
        fc = compute_country_stats_dmsp(year)
    else:
        fc = compute_country_stats_viirs(year)

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f"nightlights_{year}",
        fileNamePrefix=f"nightlights_{year}",
        fileFormat="CSV",
        selectors=["country_name", "iso_code", "year", "nl_mean", "nl_sum"],
    )
    task.start()
    print(f"  {year} export task submitted")
    return task


if __name__ == "__main__":
    main()
