"""
config.example.py — Template for local configuration.

SETUP:
    1. Copy this file to config.py (it's gitignored):
       cp config.example.py config.py
    2. Fill in your own Google Cloud Project ID below
    3. Never commit config.py to git
"""
import os
from pathlib import Path

# ── PATHS ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

for d in [DATA_RAW, DATA_PROCESSED, FIGURES, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

# ── TIME SCOPE ───────────────────────────────────────────────────────────────
YEAR_START = 2000
YEAR_END = 2023

# Train / validation / hold-out test split
TRAIN_END = 2018
VAL_START = 2019
VAL_END = 2020
TEST_START = 2021
TEST_END = 2023

# Keep False until the final model-freeze run. The test set should be evaluated once.
EVALUATE_TEST_SET = False

# ── CRISIS PREDICTION HORIZON ───────────────────────────────────────────────
HORIZON_YEARS = 3

# ── FEATURE LAGS ─────────────────────────────────────────────────────────────
LAGS = [1, 2, 3]

# ── COUNTRY SCOPE ────────────────────────────────────────────────────────────
# WDI income-level codes. Keep only non-high-income, classified economies.
INCLUDE_INCOME_GROUPS = ["LIC", "LMC", "UMC"]
EXCLUDE_INCOME_GROUPS = ["HIC"]

# ── REGION MAPPING ───────────────────────────────────────────────────────────
REGION_GROUPS = {
    "Sub-Saharan Africa": "SSF",
    "Latin America & Caribbean": "LCN",
    "East Asia & Pacific": "EAS",
    "South Asia": "SAS",
    "Middle East & North Africa": "MEA",
    "Europe & Central Asia": "ECS",
}

# ── WORLD BANK INDICATORS ───────────────────────────────────────────────────
WDI_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG":    "inflation",
    "GC.DOD.TOTL.GD.ZS": "debt_to_gdp",
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",
    "FI.RES.TOTL.MO":    "reserves_months",
    "DT.DOD.DECT.GN.ZS": "external_debt_gni",
    "GC.BAL.CASH.GD.ZS": "fiscal_balance_gdp",
    "PA.NUS.FCRF":        "exchange_rate",
    "NY.GDP.MKTP.KD":     "gdp_constant",
    "SP.POP.TOTL":        "population",
}

# ── GDELT BIGQUERY ───────────────────────────────────────────────────────────
# ▼▼▼ REPLACE THIS WITH YOUR OWN GOOGLE CLOUD PROJECT ID ▼▼▼
# Get it from https://console.cloud.google.com/ after creating a project.
# You can also set the GCP_PROJECT_ID environment variable instead.
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id-here")

# ── XGBOOST HYPERPARAMETERS ─────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 10,
    "random_state": 42,
    "eval_metric": "auc",
}

# ── LASSO ────────────────────────────────────────────────────────────────────
LASSO_ALPHAS = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]

# ── RANDOM SEED ──────────────────────────────────────────────────────────────
SEED = 42
