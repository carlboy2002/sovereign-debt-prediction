import os
import ee
import time

# Read project ID from environment variable or config.py
try:
    from config import GCP_PROJECT_ID
    project_id = GCP_PROJECT_ID
except ImportError:
    project_id = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id-here")

ee.Initialize(project=project_id)

countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
dmsp = ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS").select("stable_lights")
viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").select("avg_rad")

def export_year(year):
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
            scale=5000,  # 5km分辨率，比1km快很多，精度够用
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
    print(f"  {year} 已提交")
    return task

# 提交所有年份
tasks = []
for year in range(2000, 2024):
    tasks.append(export_year(year))

print(f"\n已提交 {len(tasks)} 个任务")
print("去 https://code.earthengine.google.com/tasks 查看进度")
print("全部完成后，Google Drive里会有 nl_2000.csv 到 nl_2023.csv")