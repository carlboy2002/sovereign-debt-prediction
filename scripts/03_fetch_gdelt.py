"""
03_fetch_gdelt.py — Query GDELT Global Knowledge Graph via Google BigQuery.

GDELT tracks news events worldwide in 100+ languages. We extract:
    1. Monthly country-level average tone (sentiment)
    2. Event counts related to economic distress
    3. IMF/creditor mention frequency

SETUP:
    1. Go to https://console.cloud.google.com/bigquery
    2. Make sure your project has BigQuery API enabled
    3. Set GCP_PROJECT_ID in config.py to your project ID
    4. Run: gcloud auth application-default login
       (or set GOOGLE_APPLICATION_CREDENTIALS environment variable)

    Free tier: 1 TB/month of queries — more than enough for this project.

Output: data/raw/gdelt_panel.csv
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd
from config import DATA_RAW, YEAR_START, YEAR_END, GCP_PROJECT_ID


FIPS_TO_ISO3 = {
    "AA": "ABW", "AC": "ATG", "AE": "ARE", "AF": "AFG", "AG": "DZA",
    "AJ": "AZE", "AL": "ALB", "AM": "ARM", "AN": "AND", "AO": "AGO",
    "AR": "ARG", "AS": "AUS", "AU": "AUT", "BA": "BHR", "BB": "BRB",
    "BC": "BWA", "BD": "BMU", "BE": "BEL", "BF": "BHS", "BG": "BGD",
    "BH": "BLZ", "BK": "BIH", "BL": "BOL", "BM": "MMR", "BN": "BEN",
    "BO": "BLR", "BP": "SLB", "BR": "BRA", "BU": "BGR", "BY": "BDI",
    "CA": "CAN", "CB": "KHM", "CD": "TCD", "CE": "LKA", "CF": "COG",
    "CG": "COD", "CH": "CHN", "CI": "CHL", "CM": "CMR", "CN": "COM",
    "CO": "COL", "CS": "CRI", "CT": "CAF", "CU": "CUB", "CV": "CPV",
    "CY": "CYP", "DA": "DNK", "DJ": "DJI", "DO": "DMA", "DR": "DOM",
    "EC": "ECU", "EG": "EGY", "EI": "IRL", "EK": "GNQ", "EN": "EST",
    "ER": "ERI", "ES": "SLV", "ET": "ETH", "EZ": "CZE", "FI": "FIN",
    "FJ": "FJI", "FR": "FRA", "GA": "GMB", "GB": "GAB", "GG": "GEO",
    "GH": "GHA", "GJ": "GRD", "GM": "DEU", "GR": "GRC", "GT": "GTM",
    "GV": "GIN", "GY": "GUY", "HA": "HTI", "HO": "HND", "HR": "HRV",
    "HU": "HUN", "IC": "ISL", "ID": "IDN", "IN": "IND", "IR": "IRN",
    "IS": "ISR", "IT": "ITA", "IV": "CIV", "IZ": "IRQ", "JA": "JPN",
    "JM": "JAM", "JO": "JOR", "KE": "KEN", "KG": "KGZ", "KN": "PRK",
    "KS": "KOR", "KT": "KWT", "KZ": "KAZ", "LA": "LAO", "LE": "LBN",
    "LG": "LVA", "LI": "LBR", "LO": "SVK", "LS": "LIE", "LT": "LTU",
    "LU": "LUX", "LY": "LBY", "MA": "MDG", "MD": "MDA", "MG": "MNG",
    "MK": "MKD", "ML": "MLI", "MN": "MCO", "MO": "MAR", "MP": "MUS",
    "MR": "MRT", "MT": "MLT", "MU": "OMN", "MV": "MDV", "MX": "MEX",
    "MY": "MYS", "MZ": "MOZ", "NG": "NER", "NI": "NGA", "NL": "NLD",
    "NO": "NOR", "NP": "NPL", "NZ": "NZL", "PA": "PRY", "PE": "PER",
    "PK": "PAK", "PL": "POL", "PO": "PRT", "QA": "QAT", "RO": "ROU",
    "RP": "PHL", "RS": "RUS", "RW": "RWA", "SA": "SAU", "SE": "SYC",
    "SF": "ZAF", "SG": "SEN", "SI": "SVN", "SL": "SLE", "SN": "SGP",
    "SO": "SOM", "SP": "ESP", "SU": "SDN", "SW": "SWE", "SY": "SYR",
    "SZ": "CHE", "TD": "TTO", "TH": "THA", "TI": "TJK", "TN": "TON",
    "TO": "TGO", "TS": "TUN", "TU": "TUR", "TX": "TKM", "TZ": "TZA",
    "UG": "UGA", "UK": "GBR", "UP": "UKR", "US": "USA", "UV": "BFA",
    "UY": "URY", "UZ": "UZB", "VE": "VEN", "VM": "VNM", "WA": "NAM",
    "WI": "ESH", "YM": "YEM", "ZA": "ZMB", "ZI": "ZWE",
}


def normalize_country_code(code):
    """Normalize GDELT country codes to ISO3 when possible."""
    if pd.isna(code):
        return None
    code = str(code).strip().upper()
    if len(code) == 3:
        return code
    return FIPS_TO_ISO3.get(code)


def query_gdelt_events():
    """
    Query GDELT Event Database for country-level monthly indicators.
    Uses the GDELT 2.0 Event Database on BigQuery (free public dataset).
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=GCP_PROJECT_ID)

    print("  Querying GDELT events data (this may take 1-2 minutes)...")

    query = f"""
    SELECT
        Actor1CountryCode AS country_code,
        EXTRACT(YEAR FROM PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))) AS year,
        EXTRACT(MONTH FROM PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))) AS month,

        -- Average tone (negative = bad news)
        AVG(AvgTone) AS avg_tone,

        -- Total event count
        COUNT(*) AS total_events,

        -- Economic distress events (CAMEO root codes for economic activity)
        COUNTIF(EventRootCode IN ('03', '04', '10', '13', '14')) AS economic_events,

        -- Conflict/instability events
        COUNTIF(EventRootCode IN ('14', '15', '17', '18', '19', '20')) AS conflict_events,

        -- Protests and demands
        COUNTIF(EventRootCode = '14') AS protest_events,

        -- Average Goldstein scale (negative = conflictual)
        AVG(GoldsteinScale) AS avg_goldstein,

        -- Number of mentions (media attention)
        AVG(NumMentions) AS avg_mentions

    FROM `gdelt-bq.gdeltv2.events`
    WHERE
        EXTRACT(YEAR FROM PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)))
            BETWEEN {YEAR_START} AND {YEAR_END}
        AND Actor1CountryCode IS NOT NULL
        AND Actor1CountryCode != ''

    GROUP BY country_code, year, month
    ORDER BY country_code, year, month
    """

    df = client.query(query).to_dataframe()
    return df


def query_gdelt_gkg():
    """
    Query GDELT Global Knowledge Graph for theme-based signals.
    This captures IMF mentions, debt-related coverage, etc.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=GCP_PROJECT_ID)

    print("  Querying GDELT GKG for IMF/debt themes (this may take 2-3 minutes)...")

    # GKG themes related to debt/financial crisis
    query = f"""
    WITH exploded AS (
        SELECT
            SPLIT(location_raw, '#')[SAFE_OFFSET(2)] AS country_code,
            EXTRACT(YEAR FROM PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS year,
            EXTRACT(MONTH FROM PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS month,
            V2Themes,
            V2Tone,
            DocumentIdentifier
        FROM `gdelt-bq.gdeltv2.gkg`,
        UNNEST(SPLIT(IFNULL(V2Locations, ''), ';')) AS location_raw
        WHERE
            EXTRACT(YEAR FROM PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)))
                BETWEEN {YEAR_START} AND {YEAR_END}
            AND location_raw IS NOT NULL
            AND location_raw != ''
    )
    SELECT
        country_code,
        year,
        month,

        -- Count articles mentioning debt/financial crisis themes
        COUNTIF(
            REGEXP_CONTAINS(V2Themes, r'ECON_DEBT|ECON_BANKRUPTCY|IMF|WORLD_BANK|CREDIT_RATING')
        ) AS debt_mentions,

        -- Count articles mentioning IMF specifically
        COUNTIF(REGEXP_CONTAINS(V2Themes, r'\\bIMF\\b')) AS imf_mentions,

        -- Average document tone
        AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)) AS avg_doc_tone,

        COUNT(DISTINCT DocumentIdentifier) AS total_articles

    FROM exploded
    WHERE country_code IS NOT NULL AND country_code != ''
    GROUP BY country_code, year, month
    HAVING total_articles > 10
    ORDER BY country_code, year, month
    """

    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        print(f"  WARNING: GKG query failed ({e}). Using events data only.")
        return None


def aggregate_to_country_year(events_df, gkg_df=None):
    """
    Aggregate monthly data to country-year level.
    """
    print("  Aggregating to country-year level...")

    yearly = events_df.groupby(["country_code", "year"]).agg({
        "avg_tone": "mean",
        "total_events": "sum",
        "economic_events": "sum",
        "conflict_events": "sum",
        "protest_events": "sum",
        "avg_goldstein": "mean",
        "avg_mentions": "mean",
    }).reset_index()

    # Compute derived features
    yearly["econ_event_share"] = (
        yearly["economic_events"] / yearly["total_events"].clip(lower=1)
    )
    yearly["conflict_event_share"] = (
        yearly["conflict_events"] / yearly["total_events"].clip(lower=1)
    )

    # Tone trend (year-over-year change)
    yearly = yearly.sort_values(["country_code", "year"])
    yearly["tone_change"] = yearly.groupby("country_code")["avg_tone"].diff()

    if gkg_df is not None and not gkg_df.empty:
        gkg_df = gkg_df.copy()
        gkg_df["country_code"] = gkg_df["country_code"].apply(normalize_country_code)
        gkg_df = gkg_df.dropna(subset=["country_code"])

        gkg_yearly = gkg_df.groupby(["country_code", "year"]).agg({
            "debt_mentions": "sum",
            "imf_mentions": "sum",
            "avg_doc_tone": "mean",
            "total_articles": "sum",
        }).reset_index()
        gkg_yearly["debt_article_share"] = (
            gkg_yearly["debt_mentions"] / gkg_yearly["total_articles"].clip(lower=1)
        )
        gkg_yearly["imf_article_share"] = (
            gkg_yearly["imf_mentions"] / gkg_yearly["total_articles"].clip(lower=1)
        )
        gkg_yearly = gkg_yearly.sort_values(["country_code", "year"])
        gkg_yearly["doc_tone_change"] = (
            gkg_yearly.groupby("country_code")["avg_doc_tone"].diff()
        )

        yearly = yearly.merge(gkg_yearly, on=["country_code", "year"], how="left")

    yearly = yearly.rename(columns={"country_code": "gdelt_country_code"})

    return yearly


def main():
    print("Fetching GDELT data via BigQuery...")

    if GCP_PROJECT_ID == "your-gcp-project-id":
        print("ERROR: Set GCP_PROJECT_ID in config.py to your Google Cloud project ID.")
        print("       e.g., GCP_PROJECT_ID = 'my-project-123456'")
        sys.exit(1)

    # Query events
    events_df = query_gdelt_events()
    print(f"  Events: {len(events_df)} rows")

    # Query GKG (optional, may fail on free tier due to table size)
    gkg_df = query_gdelt_gkg()

    # Aggregate
    yearly = aggregate_to_country_year(events_df, gkg_df)

    # Save
    out_path = DATA_RAW / "gdelt_panel.csv"
    yearly.to_csv(out_path, index=False)
    n = yearly["gdelt_country_code"].nunique()
    print(f"  Saved: {out_path}")
    print(f"  {n} countries, {yearly['year'].min()}-{yearly['year'].max()}")


if __name__ == "__main__":
    main()
