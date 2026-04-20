"""
02_fetch_nightlights_gee.py — 用Google Earth Engine计算国家级夜间灯光数据

这个脚本在云端完成所有计算，本地不需要处理大文件。

前置步骤：
    1. 注册GEE: https://earthengine.google.com → Register
    2. pip install earthengine-api
    3. earthengine authenticate（浏览器授权）

输出：
    - Google Drive里会出现一个 nightlights_by_country.csv
    - 下载后放到 data/raw/nightlights_by_country.csv

数据源：
    - Li et al. (2020) Harmonized NTL dataset（GEE社区目录已收录）
    - FAO GAUL 国家边界
"""

import ee
import time

# ── 初始化 ──────────────────────────────────────────────────────────────────
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

print("Google Earth Engine 初始化成功")

# ── 加载数据 ─────────────────────────────────────────────────────────────────

# 国家边界（FAO GAUL 2015）
countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
print(f"  国家边界已加载: {countries.size().getInfo()} 个国家/地区")

# 夜间灯光 — 使用GEE内置的DMSP和VIIRS数据
# DMSP-OLS: 1992-2013
dmsp = ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS").select("stable_lights")
# VIIRS-DNB: 2014-至今
viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").select("avg_rad")


# ── 核心函数：计算一年的国家级灯光值 ────────────────────────────────────────


def compute_country_stats_dmsp(year):
    """计算DMSP卫星某一年的国家级灯光统计"""
    year = ee.Number(year)

    # DMSP每年有1-2颗卫星，取该年所有影像的均值
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    annual = dmsp.filterDate(start, end).mean()

    # 对每个国家计算均值和总和
    def reduce_country(feature):
        stats = annual.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), sharedInputs=True),
            geometry=feature.geometry(),
            scale=1000,  # 1km分辨率
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
    """计算VIIRS卫星某一年的国家级灯光统计"""
    year = ee.Number(year)

    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    annual = viirs.filterDate(start, end).mean()

    # 去掉负值（噪声）
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


# ── 主流程 ───────────────────────────────────────────────────────────────────


def main():
    print("\n开始计算国家级夜间灯光数据...")
    print("所有计算在Google服务器上运行，请耐心等待。\n")

    # DMSP年份: 2000-2013
    dmsp_years = list(range(2000, 2014))
    # VIIRS年份: 2014-2023
    viirs_years = list(range(2014, 2024))

    all_features = []

    # 处理DMSP（2000-2013）
    print("=== DMSP (2000-2013) ===")
    for year in dmsp_years:
        print(f"  处理 {year}...", end=" ", flush=True)
        fc = compute_country_stats_dmsp(year)
        all_features.append(fc)
        print("✓")

    # 处理VIIRS（2014-2023）
    print("\n=== VIIRS (2014-2023) ===")
    for year in viirs_years:
        print(f"  处理 {year}...", end=" ", flush=True)
        fc = compute_country_stats_viirs(year)
        all_features.append(fc)
        print("✓")

    # 合并所有年份
    print("\n合并所有年份...")
    merged = ee.FeatureCollection(all_features).flatten()

    # 导出到Google Drive
    print("导出到Google Drive...")
    task = ee.batch.Export.table.toDrive(
        collection=merged,
        description="nightlights_by_country",
        fileNamePrefix="nightlights_by_country",
        fileFormat="CSV",
        selectors=["country_name", "iso_code", "year", "nl_mean", "nl_sum"],
    )
    task.start()
    print(f"  导出任务已提交: {task.status()['state']}")

    # 等待完成
    print("\n等待导出完成（通常需要10-30分钟）...")
    print("你也可以去 https://code.earthengine.google.com/tasks 查看进度\n")

    while task.active():
        status = task.status()
        print(f"  状态: {status['state']}...", flush=True)
        time.sleep(30)

    final_status = task.status()
    if final_status["state"] == "COMPLETED":
        print("\n✓ 导出完成！")
        print("  文件位置: Google Drive 根目录 → nightlights_by_country.csv")
        print("  下载后放到: data/raw/nightlights_by_country.csv")
    else:
        print(f"\n✗ 导出失败: {final_status}")
        print("  常见原因: 超时或内存不足")
        print("  解决方案: 减少年份范围，分批导出")


# ── 备选方案：分批导出（如果一次性导出超时）──────────────────────────────────


def export_single_year(year, sensor="viirs"):
    """单独导出某一年的数据（用于排查问题）"""
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
    print(f"  {year} 导出任务已提交")
    return task


if __name__ == "__main__":
    main()
