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
    SELECT
        SPLIT(V2Locations, ';')[SAFE_OFFSET(0)] AS location_raw,
        EXTRACT(YEAR FROM PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS year,
        EXTRACT(MONTH FROM PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS month,

        -- Count articles mentioning debt/financial crisis themes
        COUNTIF(
            REGEXP_CONTAINS(V2Themes, r'ECON_DEBT|ECON_BANKRUPTCY|IMF|WORLD_BANK|CREDIT_RATING')
        ) AS debt_mentions,

        -- Count articles mentioning IMF specifically
        COUNTIF(REGEXP_CONTAINS(V2Themes, r'\\bIMF\\b')) AS imf_mentions,

        -- Average document tone
        AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)) AS avg_doc_tone,

        COUNT(*) AS total_articles

    FROM `gdelt-bq.gdeltv2.gkg`
    WHERE
        EXTRACT(YEAR FROM PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)))
            BETWEEN {YEAR_START} AND {YEAR_END}

    GROUP BY location_raw, year, month
    HAVING total_articles > 10
    ORDER BY year, month
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

    # GDELT uses FIPS country codes; we need to map to ISO3
    # For now, use the 2-letter codes and map later in merge step
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

    # Rename for clarity
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
