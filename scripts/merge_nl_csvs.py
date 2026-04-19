"""
merge_nl_csvs.py — 合并24个分年份的夜间灯光CSV成一个文件

前置：
    - 24个 nl_YYYY.csv 文件已下载并放在 data/raw/ 下
输出：
    - data/raw/nightlights_by_country.csv（02_fetch_nightlights.py会读这个）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import DATA_RAW


def main():
    # 找到所有 nl_YYYY.csv 文件
    csv_files = sorted(DATA_RAW.glob("nl_*.csv"))

    if not csv_files:
        print(f"ERROR: 在 {DATA_RAW} 找不到 nl_*.csv 文件")
        print("请确保已从Google Drive下载所有nl_YYYY.csv并放到data/raw/")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 逐个读取并合并
    frames = []
    for path in csv_files:
        df = pd.read_csv(path)
        print(f"  {path.name}: {len(df)} 行")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # 移除空值行（海洋、没有数据的国家）
    merged = merged.dropna(subset=["nl_mean"])

    # 保存为02脚本能读的格式
    output = DATA_RAW / "nightlights_by_country.csv"
    merged.to_csv(output, index=False)

    print(f"\n合并完成:")
    print(f"  总行数: {len(merged)}")
    print(f"  国家数: {merged['country_name'].nunique()}")
    print(f"  年份范围: {merged['year'].min()} - {merged['year'].max()}")
    print(f"  保存到: {output}")


if __name__ == "__main__":
    main()