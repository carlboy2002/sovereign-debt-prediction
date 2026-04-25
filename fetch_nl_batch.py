import os
import ee

# Read project ID from environment variable or config.py


def get_project_id():
    try:
        from config import GCP_PROJECT_ID

        project_id = GCP_PROJECT_ID
    except ImportError:
        project_id = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id-here")

    if project_id == "your-gcp-project-id-here":
        raise ValueError("Set GCP_PROJECT_ID in config.py or the environment before exporting.")
    return project_id


def export_year(year, countries, dmsp, viirs):
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)

    if year <= 2013:
        img = dmsp.filterDate(start, end).mean()
        band = "stable_lights"
    else:
        img = viirs.filterDate(start, end).mean().max(0)
        band = "avg_rad"

    def reduce(feature):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature.geometry(),
            scale=5000,  # 5 km resolution is much faster than 1 km and precise enough here.
            maxPixels=1e9,
            bestEffort=True,
        )
        return feature.set({
            "year": year,
            "nl_mean": stats.get(band),
            "country_name": feature.get("ADM0_NAME"),
            "iso_code": feature.get("ADM0_CODE"),
        })

    fc = countries.map(reduce)

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f"nl_{year}",
        fileNamePrefix=f"nl_{year}",
        fileFormat="CSV",
        selectors=["country_name", "iso_code", "year", "nl_mean"],
    )
    task.start()
    print(f"  {year} submitted")
    return task


def main():
    project_id = get_project_id()
    ee.Initialize(project=project_id)

    countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
    dmsp = ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS").select("stable_lights")
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").select("avg_rad")

    tasks = []
    for year in range(2000, 2024):
        tasks.append(export_year(year, countries, dmsp, viirs))

    print(f"\nSubmitted {len(tasks)} tasks")
    print("Check progress at https://code.earthengine.google.com/tasks")
    print("After all tasks complete, Google Drive should contain nl_2000.csv through nl_2023.csv")


if __name__ == "__main__":
    main()
