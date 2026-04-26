# Beyond Official Statistics

## Predicting Sovereign Debt Crises with Macro Data, Satellite Nightlights, and News Text

**Columbia SIPA - DSPC7100 Applying Machine Learning - Spring 2026**

**Team:** Jinen Wang, Levi Liu, Wuhao Xia, Chenyi Jiang

---

## Project Summary

This project builds an early warning model for sovereign debt crises in low- and middle-income economies. The central idea is that official macroeconomic statistics are often delayed, revised, or strategically reported, so a stronger warning system should combine official indicators with external signals that are harder for governments to directly manipulate.

The project combines three data streams:

| Data stream | Role in the project |
|---|---|
| World Bank WDI macro indicators | Standard early warning baseline |
| NASA VIIRS nighttime lights | Proxy for observed economic activity |
| GDELT events and GKG news text | Political, conflict, debt, IMF, and sentiment signals |

The main model is a nested XGBoost classifier. LASSO logistic regression is used as an interpretable baseline, and additional scripts audit regional fairness, simulate a structural signaling model, and construct an exploratory misreporting index.

The current preferred validation-selected specification is **Macro + Satellite** because it has the strongest average precision on the official validation split and the strongest mean average precision across rolling validation windows. The **All Features** specification has the highest validation AUC and is best treated as a robustness model.

---

## Current Methodological Status

The code now enforces a strict prediction-time information rule:

**Only information available before the prediction year is allowed into the model.**

All model features are built from:

- Lagged variables: `t-1`, `t-2`, `t-3`
- Rolling statistics computed from prior years only
- Prior year-over-year changes
- Interactions constructed from lagged inputs

Current-year raw macro, satellite, or text variables are not included in the feature blocks used by the models. This is important because using contemporaneous WDI or GDELT values would create look-ahead bias in an early warning setting.

The code still contains an optional evaluation path for **2021-2023**, but this period is right-censored under the current three-year early-warning target. A 2021 observation would require knowing whether a crisis starts through 2024, a 2022 observation through 2025, and a 2023 observation through 2026. Because the current label file ends in 2023, this period should not be used as the main final test set.

In `config.py`, keep:

```python
EVALUATE_TEST_SET = False
```

Set it to `True` only for an explicitly labeled diagnostic run, not for the main reported results, unless crisis labels are extended far enough to make the full three-year horizon observable.

---

## Current Results

### Official Validation Split

Train period: years through 2018  
Validation period: 2019-2020  
Censored future period: 2021-2023, not used as the main test set

Validation set size: 258 country-years, 28 positive crisis targets.

| Specification | Candidate features | Validation AUC | Average precision | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Macro Only | 28 | 0.794 | 0.461 | 0.459 | 0.607 |
| Macro + Satellite | 40 | 0.816 | 0.519 | 0.388 | 0.679 |
| Macro + Text | 80 | 0.848 | 0.447 | 0.324 | 0.786 |
| All Features | 92 | 0.904 | 0.437 | 0.317 | 0.929 |

Interpretation:

- `Macro + Satellite` is the strongest validation model by average precision, which is the better headline metric for rare crisis prediction.
- `All Features` has the highest AUC but lower average precision, so it is useful as a robustness specification rather than the cleanest main result.
- Text features add ranking signal in some windows, but they are less stable than the satellite block.

### Rolling-Origin Validation

The rolling validation script evaluates multiple historical validation windows before the censored 2021-2023 future period:

| Specification | Windows | Mean AUC | AUC std. dev. | Mean average precision | AP std. dev. |
|---|---:|---:|---:|---:|---:|
| Macro + Satellite | 5 | 0.759 | 0.049 | 0.366 | 0.182 |
| Macro Only | 5 | 0.759 | 0.045 | 0.345 | 0.137 |
| All Features | 5 | 0.790 | 0.081 | 0.335 | 0.145 |
| Macro + Text | 5 | 0.767 | 0.059 | 0.292 | 0.111 |

Interpretation:

- The model has real predictive signal under stricter lag-only features.
- Satellite features provide the most stable improvement in average precision.
- The text block is useful but should be framed carefully because GDELT coverage varies heavily by country and year.

### LASSO Baseline

LASSO logistic regression is intentionally conservative:

- Validation AUC: 0.696
- Average precision: 0.202
- Selected features: 3 out of 91 usable features

Top nonzero features:

| Feature | Block | Direction |
|---|---|---|
| `reserves_months_lag1` | macro | lower reserves increase risk |
| `nightlight_raw_lag3` | satellite | selected as a slow-moving activity signal |
| `conflict_events_lag1` | text | conflict coverage increases risk |

This supports the interpretation that a linear sparse model captures only a small part of the signal, while XGBoost benefits from nonlinearities and interactions.

### Fairness Audit

The fairness audit evaluates validation performance by region using the same XGBoost training logic as the main model.

| Region | Observations | Crises | AUC | Average precision |
|---|---:|---:|---:|---:|
| Sub-Saharan Africa | 92 | 5 | 0.947 | 0.431 |
| Latin America & Caribbean | 44 | 4 | 0.900 | 0.458 |
| East Asia & Pacific | 44 | 2 | 0.857 | 0.200 |
| South Asia | 12 | 6 | 0.833 | 0.882 |
| Middle East & North Africa | 30 | 5 | 0.888 | 0.486 |
| Europe & Central Asia | 36 | 6 | 0.928 | 0.663 |

Regional AUC gap: 0.114.

The script also writes region-specific threshold recommendations to:

```text
outputs/tables/fairness_thresholds_by_region.csv
```

These thresholds target a similar false-positive rate across regions. They are a diagnostic, not the main model.

### Structural and Misreporting Outputs

The structural signaling model is illustrative rather than fully estimated. It checks whether the project narrative is directionally consistent with the empirical nightlight-divergence patterns.

Current structural validation:

| Quantity | Value |
|---|---:|
| Simulated debt-divergence correlation | -0.037 |
| Empirical debt-divergence correlation | -0.133 |
| Simulated divergence among defaulters | 0.161 |
| Simulated divergence among non-defaulters | 0.144 |

The misreporting index is exploratory. After filtering to countries with meaningful nightlight variation, the strongest current flags are Moldova and Bangladesh, each with one suspicious year out of six observed years. No country is flagged in more than 30 percent of observed years, so this output should be presented as a case-generation tool, not a validated ranking of manipulation.

---

## Repository Structure

```text
.
|-- auth.py                         # Earth Engine authentication helper
|-- config.example.py               # Config template; copy to config.py
|-- config.py                       # Local config, gitignored
|-- fetch_nl_batch.py               # Submit Google Earth Engine nightlight exports
|-- requirements.txt                # Python dependencies
|-- README.md                       # This file
|-- sovereign_debt_dashboard.html   # Static dashboard generated from outputs
|
|-- scripts/
|   |-- 01_fetch_wdi.py             # Download World Bank WDI indicators
|   |-- 02_fetch_nightlights.py     # Process merged VIIRS nightlight CSV
|   |-- 03_fetch_gdelt.py           # Query GDELT events and GKG through BigQuery
|   |-- 04_fetch_crisis_labels.py   # Build crisis episode labels
|   |-- 05_clean_merge.py           # Merge all raw data into one panel
|   |-- 06_feature_engineering.py   # Build lag-only model features
|   |-- 07_descriptive_stats.py     # EDA tables and figures
|   |-- 08_xgboost_model.py         # Nested XGBoost models and SHAP plots
|   |-- 09_lasso_model.py           # LASSO baseline and feature selection
|   |-- 10_fairness_audit.py        # Regional and income-group audit
|   |-- 11_structural_model.py      # Signaling-game simulation
|   |-- 12_misreporting_index.py    # Exploratory country-level index
|   |-- 13_rolling_validation.py    # Rolling-origin robustness checks
|   |-- 14_generate_dashboard.py    # Rebuild self-contained HTML dashboard
|   |-- 15_generate_presentation.py # Rebuild two-minute PPTX deck
|   |-- merge_nl_csvs.py            # Merge per-year GEE exports
|
|-- data/
|   |-- raw/                        # Raw downloaded/generated inputs
|   |-- processed/                  # Merged panel and feature matrices
|
|-- outputs/
|   |-- figures/                    # PNG plots
|   |-- tables/                     # CSV result tables
|
|-- presentation/
|   |-- Beyond_Official_Statistics_2min.pptx
|   |-- PROJECT_QA_GUIDE.md
```

Raw and processed data files are regenerated by the scripts and may be excluded from version control.

---

## Data Sources

| Source | Access path | Notes |
|---|---|---|
| World Bank WDI | `wbgapi` | Macro indicators, income groups, and regions |
| NASA VIIRS Black Marble | Google Earth Engine | Nighttime lights; VIIRS-only to avoid DMSP transition artifacts |
| GDELT 2.0 Events | Google BigQuery | Event tone, event type, Goldstein scale, media volume |
| GDELT 2.0 GKG | Google BigQuery | IMF/debt mentions, article tone, article volume |
| Crisis labels | Curated episode file generated by script 04 | Used to define forward-looking crisis targets |

---

## Setup

### 1. Install Dependencies

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

If geospatial packages fail locally, the core pipeline can still run because satellite aggregation is handled through Google Earth Engine.

### 2. Configure Google Cloud

BigQuery is required for GDELT, and Earth Engine is required for nightlights.

1. Create or select a Google Cloud project.
2. Enable the BigQuery API.
3. Enable the Earth Engine API.
4. Register for Earth Engine under the academic/noncommercial tier.
5. Authenticate locally:

```bash
gcloud auth login
gcloud auth application-default login
python auth.py
```

Make sure the active Google account has access to the GCP project in `config.py`.

### 3. Create Local Config

```bash
cp config.example.py config.py
```

Then edit:

```python
GCP_PROJECT_ID = "your-project-id"
EVALUATE_TEST_SET = False
```

`config.py` is local and should not be committed.

---

## Pipeline

Run scripts from the repository root.

### Phase 1: Data Acquisition

```bash
python scripts/01_fetch_wdi.py

python fetch_nl_batch.py
# Wait for Earth Engine tasks to finish.
# Download the exported nl_YYYY.csv files from Google Drive into data/raw/.
python scripts/merge_nl_csvs.py
python scripts/02_fetch_nightlights.py

python scripts/03_fetch_gdelt.py
python scripts/04_fetch_crisis_labels.py
```

### Phase 2: Processing and Feature Engineering

```bash
python scripts/05_clean_merge.py
python scripts/06_feature_engineering.py
```

The feature engineering step writes:

```text
data/processed/panel_features.csv
data/processed/feature_blocks.json
```

Feature blocks are checked so that only lagged or prior-year features enter the models.

### Phase 3: Modeling and Diagnostics

```bash
python scripts/07_descriptive_stats.py
python scripts/08_xgboost_model.py
python scripts/09_lasso_model.py
python scripts/10_fairness_audit.py
python scripts/11_structural_model.py
python scripts/12_misreporting_index.py
python scripts/13_rolling_validation.py
python scripts/14_generate_dashboard.py
python scripts/15_generate_presentation.py
```

All figures are written to `outputs/figures/`.  
All result tables are written to `outputs/tables/`.
Presentation materials are written to `presentation/`.

### Quick Code Check

```bash
python -m compileall -q scripts
```

---

## Holdout Protocol

The current project uses a three-year forward-looking target: a country-year is positive if a crisis onset occurs within the next three years. Since crisis labels currently run through 2023, the last fully observable validation years are 2019-2020.

For final reporting, use this protocol:

1. Treat 2019-2020 as the main out-of-time validation period.
2. Recommended main model: `Macro + Satellite`.
3. Recommended robustness model: `All Features`.
4. Use `scripts/13_rolling_validation.py` to show that results are not driven by a single split.
5. Keep `EVALUATE_TEST_SET = False` for the main reported results.

The 2021-2023 block can be evaluated only as a diagnostic future-period exercise. If you do so, label it clearly as censored and do not use it for model selection:

```bash
python scripts/08_xgboost_model.py
```

To make 2021-2023 a proper final test set under the current target, the crisis label file would need to be extended through 2026. A second option is to redesign the project around a one-year horizon, but that would likely produce fewer positive labels and a less stable annual crisis model.

---

## Important Limitations

### Censored Future Period

The 2021-2023 period is not a clean final test set under a three-year target because future crisis onsets after 2023 are not observable in the current label file. Any evaluation on this period should be treated as diagnostic and clearly marked as right-censored.

### VIIRS-Only Satellite History

The project does not currently harmonize DMSP and VIIRS nightlight sensors. DMSP and VIIRS are not directly comparable because of different calibration, dynamic range, and saturation behavior. To avoid artificial jumps around the 2013-2014 sensor transition, the model uses VIIRS-based nightlight features only. This reduces historical coverage but avoids contaminating trend features.

### Nightlight Saturation and Missingness

Nightlights can saturate in highly lit areas and can be missing or noisy for small countries. The misreporting index applies filters for meaningful variation, but the main model still inherits some satellite measurement error.

### GDELT Coverage Bias

GDELT media volume varies sharply across countries. Countries with more English-language or international coverage have richer text signals. The model includes article/event volume features, but GDELT should still be interpreted as media coverage, not ground truth.

### Crisis Label Uncertainty

The project uses curated crisis episodes to construct a forward-looking target. Crisis boundaries are inherently debatable, especially for multi-year restructurings or prolonged distress episodes. This introduces label noise.

### Fairness Sample Size

Regional fairness metrics are computed on small validation subsets. AUC by region is useful as a diagnostic, but regional threshold recommendations should not be treated as deployment-ready policy rules.

### Structural Model Scope

The structural signaling game supports the project narrative but is not fully estimated from the data. It should be framed as theoretical grounding and directional validation, not as a separate causal identification strategy.

---

## Approaches Considered but Not Used

| Approach | Why it was not used |
|---|---|
| Difference-in-differences using VIIRS rollout | VIIRS had near-global coverage from the start, so there is no staggered rollout variation to exploit. |
| DMSP plus VIIRS long satellite history | The sensor transition creates artificial jumps unless a full harmonization procedure is implemented. |
| Monthly GDELT panel | WDI variables and crisis labels are annual; monthly interpolation would create false precision. |
| LSTM or deep sequence models | The effective panel is too small and unbalanced for a credible deep sequence model. |
| Random forest as a main model | It did not offer a clear methodological advantage over XGBoost and was less central to the project narrative. |
| Elastic net baseline | LASSO is simpler and more interpretable for feature selection; elastic net adds another tuning dimension. |

---

## Suggested Next Improvements

1. If a true final test set is required, either extend crisis labels through 2026 or switch to a one-year horizon before freezing results.
2. Update the static dashboard if final reported figures change.
3. Add a short appendix explaining why average precision is prioritized over AUC.
4. Add a sensitivity table comparing `Macro + Satellite` and `All Features`.
5. If extending the project after the deadline, implement DMSP-VIIRS harmonization to recover a longer satellite history.

---

## Main Output Files

| File | Description |
|---|---|
| `outputs/tables/xgboost_nested_comparison.csv` | Main validation metrics by model specification |
| `outputs/tables/xgboost_rolling_validation.csv` | Window-level rolling validation results |
| `outputs/tables/xgboost_rolling_validation_summary.csv` | Rolling validation summary table |
| `outputs/tables/lasso_coefficients.csv` | LASSO feature selection output |
| `outputs/tables/fairness_by_region.csv` | Regional fairness audit |
| `outputs/tables/fairness_thresholds_by_region.csv` | Region-specific threshold diagnostics |
| `outputs/tables/misreporting_index.csv` | Exploratory country-level misreporting index |
| `outputs/figures/roc_comparison.png` | ROC comparison across nested XGBoost specs |
| `outputs/figures/shap_All Features.png` | SHAP feature importance for the all-features model |
| `outputs/figures/fairness_auc_by_region.png` | Regional AUC audit plot |
| `sovereign_debt_dashboard.html` | Self-contained dashboard rebuilt from latest outputs |
| `presentation/Beyond_Official_Statistics_2min.pptx` | Two-minute widescreen presentation deck |
| `presentation/PROJECT_QA_GUIDE.md` | English project brief and Q&A guide |

---

## Citation

```text
Wang, J., Liu, L., Xia, W., and Jiang, C. (2026).
Beyond Official Statistics: Predicting Sovereign Debt Crises with
Macro Data, Satellite Nightlights, and News Text.
Columbia SIPA, DSPC7100 Applying Machine Learning Final Project.
```

---

Last updated: April 25, 2026
