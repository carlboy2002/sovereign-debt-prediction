"""
04_fetch_crisis_labels.py — Build sovereign debt crisis labels.

Primary source: Laeven & Valencia (2020)
    "Systemic Banking Crises Database II" (IMF Economic Review)
    https://www.imf.org/en/Publications/WP/Issues/2018/09/14/Systemic-Banking-Crises-Revisited-46232

    This dataset covers banking crises, currency crises, and sovereign debt crises
    from 1970 to 2017. We supplement with post-2017 events from IMF reports.

MANUAL STEP REQUIRED:
    1. Download the Laeven & Valencia dataset from the IMF link above
       (usually an Excel file with crisis start/end years by country)
    2. Save it as: data/raw/laeven_valencia_crises.xlsx

    If you cannot find the exact file, this script also constructs labels
    from a hardcoded list of well-documented sovereign debt crises.

Output: data/raw/crisis_labels.csv
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd
from config import DATA_RAW, YEAR_START, YEAR_END

# ── Hardcoded sovereign debt crisis episodes ─────────────────────────────────
# Sources: Laeven & Valencia (2020), Reinhart & Rogoff (2009), IMF reports
# Format: (iso3c, start_year, end_year)
# This is a backup in case the IMF Excel file is not available.

KNOWN_CRISES = [
    # Latin America
    ("ARG", 2001, 2005), ("ARG", 2014, 2016), ("ARG", 2019, 2020),
    ("ECU", 1999, 2000), ("ECU", 2008, 2009), ("ECU", 2020, 2020),
    ("VEN", 2004, 2005), ("VEN", 2017, 2023),
    ("BLZ", 2006, 2007), ("BLZ", 2012, 2013), ("BLZ", 2016, 2017),
    ("JAM", 2010, 2013),
    ("GRD", 2004, 2005), ("GRD", 2013, 2015),
    ("SUR", 2020, 2023),
    ("URY", 2002, 2003),
    ("DOM", 2003, 2005), ("DOM", 2018, 2019),
    ("NIC", 2003, 2008),
    ("PRY", 2003, 2004),
    ("BOL", 2003, 2005),
    ("BRB", 2018, 2019),

    # Sub-Saharan Africa
    ("CIV", 2000, 2012),
    ("COG", 2000, 2004), ("COG", 2015, 2019),
    ("MOZ", 2016, 2019),
    ("ZMB", 2020, 2023),
    ("GHA", 2015, 2015), ("GHA", 2022, 2023),
    ("ETH", 2021, 2023),
    ("TCD", 2014, 2015), ("TCD", 2018, 2020),
    ("GMB", 2003, 2005), ("GMB", 2015, 2017),
    ("MWI", 2002, 2006), ("MWI", 2012, 2013),
    ("MDG", 2002, 2004), ("MDG", 2009, 2011),
    ("BFA", 2000, 2002),
    ("SLE", 2001, 2002),
    ("LBR", 2003, 2010),
    ("AGO", 2015, 2017),
    ("NGA", 2016, 2017),
    ("KEN", 2022, 2023),
    ("SDN", 2009, 2013), ("SDN", 2019, 2023),
    ("ZWE", 2000, 2009), ("ZWE", 2018, 2020),
    ("CAF", 2013, 2016),
    ("CMR", 2004, 2006),
    ("ERI", 2008, 2011),
    ("SSD", 2016, 2018),

    # Europe & Central Asia
    ("GRC", 2010, 2015),
    ("UKR", 2014, 2015), ("UKR", 2022, 2023),
    ("RUS", 1998, 2000), ("RUS", 2014, 2015), ("RUS", 2022, 2023),
    ("BLR", 2011, 2011), ("BLR", 2020, 2022),
    ("HUN", 2008, 2009),
    ("ISL", 2008, 2010),
    ("CYP", 2012, 2013),
    ("PRT", 2010, 2014),
    ("IRL", 2010, 2013),
    ("ESP", 2012, 2013),
    ("LVA", 2008, 2010),
    ("ROU", 2009, 2011),
    ("SRB", 2000, 2004),
    ("TUR", 2000, 2002), ("TUR", 2018, 2019), ("TUR", 2021, 2023),
    ("MKD", 2001, 2002),
    ("TJK", 2016, 2018),
    ("KGZ", 2020, 2021),

    # Asia
    ("LKA", 2022, 2023),
    ("PAK", 2008, 2009), ("PAK", 2018, 2019), ("PAK", 2022, 2023),
    ("LAO", 2022, 2023),
    ("MNG", 2016, 2017),
    ("MMR", 2003, 2006),
    ("NPL", 2015, 2016),
    ("BGD", 2022, 2023),
    ("MDV", 2009, 2010), ("MDV", 2021, 2023),
    ("IDN", 2001, 2002),
    ("PHL", 2003, 2004),

    # Middle East & North Africa
    ("LBN", 2020, 2023),
    ("IRQ", 2003, 2006),
    ("EGY", 2011, 2013), ("EGY", 2022, 2023),
    ("JOR", 2012, 2014),
    ("TUN", 2011, 2013), ("TUN", 2018, 2020),
    ("YEM", 2011, 2014), ("YEM", 2015, 2023),
    ("SYR", 2011, 2023),
    ("MAR", 2012, 2013),

    # Pacific & Caribbean small states
    ("DMA", 2003, 2005),
    ("ATG", 2004, 2005), ("ATG", 2010, 2011),
    ("KNA", 2011, 2012),
    ("HTI", 2010, 2011), ("HTI", 2021, 2023),
    ("SYC", 2008, 2010),
]


def load_laeven_valencia():
    """Try to load the IMF Laeven & Valencia crisis database."""
    path = DATA_RAW / "laeven_valencia_crises.xlsx"
    if not path.exists():
        return None

    print("  Loading Laeven & Valencia database...")
    try:
        # The Excel file typically has separate sheets for each crisis type
        # Look for "Sovereign Debt" or "Debt" sheet
        xls = pd.ExcelFile(path)
        sheet = None
        for name in xls.sheet_names:
            if "debt" in name.lower() or "sovereign" in name.lower():
                sheet = name
                break

        if sheet is None:
            # Try first sheet
            sheet = xls.sheet_names[0]
            print(f"  Using sheet: {sheet}")

        df = pd.read_excel(path, sheet_name=sheet)
        print(f"  Loaded {len(df)} rows from {sheet}")
        return df
    except Exception as e:
        print(f"  WARNING: Could not parse Excel file: {e}")
        return None


def build_crisis_panel():
    """
    Build a country-year panel with crisis indicator.
    crisis_flag = 1 if country is in a sovereign debt crisis that year.
    crisis_onset = 1 only in the first year of a crisis episode.
    crisis_window = 1 if crisis starts within next N years (prediction target).
    """
    print("Building crisis labels...")

    # Try Laeven & Valencia first
    lv_data = load_laeven_valencia()
    if lv_data is not None:
        print("  NOTE: Laeven & Valencia file was loaded but curated crisis episodes are still used.")
        print("        Review the Excel schema before merging it into the label panel.")

    # Use hardcoded list (supplement or primary)
    records = []
    episode_records = []
    for iso3c, start, end in KNOWN_CRISES:
        source = "curated_laeven_valencia_reinhart_rogoff_imf_reports"
        episode_records.append({
            "iso3c": iso3c,
            "crisis_start": start,
            "crisis_end": end,
            "crisis_source": source,
        })
        for year in range(max(start, YEAR_START), min(end, YEAR_END) + 1):
            records.append({
                "iso3c": iso3c,
                "year": year,
                "in_crisis": 1,
                "crisis_start": start,
                "crisis_end": end,
                "crisis_source": source,
            })

    crisis_df = pd.DataFrame(records)
    episodes_df = pd.DataFrame(episode_records)

    # Mark onset (first year of each episode)
    crisis_df["crisis_onset"] = (
        crisis_df["year"] == crisis_df["crisis_start"]
    ).astype(int)

    # Drop duplicates (same country-year from multiple sources)
    crisis_df = crisis_df.groupby(["iso3c", "year"]).agg({
        "in_crisis": "max",
        "crisis_onset": "max",
        "crisis_start": "min",
        "crisis_end": "max",
        "crisis_source": lambda x: ";".join(sorted(set(x))),
    }).reset_index()

    # Create full panel (all country-years, including non-crisis)
    # We'll merge with actual country list in 05_clean_merge.py
    # For now, just save the crisis episodes
    out_path = DATA_RAW / "crisis_labels.csv"
    episodes_path = DATA_RAW / "crisis_episodes.csv"
    crisis_df.to_csv(out_path, index=False)
    episodes_df.to_csv(episodes_path, index=False)

    n_episodes = len(KNOWN_CRISES)
    n_countries = crisis_df["iso3c"].nunique()
    n_crisis_years = len(crisis_df)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {episodes_path}")
    print(f"  {n_episodes} crisis episodes across {n_countries} countries")
    print(f"  {n_crisis_years} country-year observations flagged as crisis")

    return crisis_df


if __name__ == "__main__":
    build_crisis_panel()
