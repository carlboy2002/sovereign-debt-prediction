"""
merge_nl_csvs.py - merge 24 yearly nightlights CSV files into one file

Prerequisite:
    - 24 nl_YYYY.csv files have been downloaded and placed under data/raw/.
Output:
    - data/raw/nightlights_by_country.csv, which is read by 02_fetch_nightlights.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import DATA_RAW


def main():
    # Find all nl_YYYY.csv files.
    csv_files = sorted(DATA_RAW.glob("nl_*.csv"))

    if not csv_files:
        print(f"ERROR: No nl_*.csv files found in {DATA_RAW}")
        print("Please download all nl_YYYY.csv files from Google Drive and place them in data/raw/.")
        return

    print(f"Found {len(csv_files)} CSV files")

    # Read and merge files one by one.
    frames = []
    for path in csv_files:
        df = pd.read_csv(path)
        print(f"  {path.name}: {len(df)} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Remove rows with missing nightlights data, such as oceans or countries without data.
    merged = merged.dropna(subset=["nl_mean"])

    # Save in the format expected by 02_fetch_nightlights.py.
    output = DATA_RAW / "nightlights_by_country.csv"
    merged.to_csv(output, index=False)

    print("\nMerge complete:")
    print(f"  Total rows: {len(merged)}")
    print(f"  Countries: {merged['country_name'].nunique()}")
    print(f"  Year range: {merged['year'].min()} - {merged['year'].max()}")
    print(f"  Saved to: {output}")


if __name__ == "__main__":
    main()
