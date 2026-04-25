# Beyond Official Statistics - Project Brief and Q&A Guide

## 1. One-Sentence Summary

This project builds a multimodal early-warning model for sovereign debt crises in low- and middle-income economies by combining official macroeconomic indicators with satellite nightlights and global news text.

## 2. Two-Minute Presentation Script

**Problem.** Sovereign debt crises are hard to anticipate because traditional warning systems rely heavily on official statistics such as GDP growth, fiscal balances, reserves, and debt ratios. These data often arrive with lags, are revised, and may be strategically reported by governments under financial stress.

**Data.** We combine three country-year data streams from 2000 to 2023. First, World Bank WDI macro indicators provide the standard baseline. Second, VIIRS nighttime lights proxy for observed economic activity and allow us to measure divergence between reported GDP growth and satellite-observed growth. Third, GDELT events and GKG news data capture media tone, conflict and protest events, and IMF or debt-related article mentions.

**Approach.** The main model is XGBoost. We compare four nested specifications: Macro Only, Macro + Satellite, Macro + Text, and All Features. To avoid look-ahead bias, every model feature is lagged, a prior rolling statistic, or a prior change. We also use LASSO as an interpretable baseline, SHAP for feature interpretation, rolling-origin validation for robustness, a fairness audit across regions, and a country-level misreporting index.

**Findings.** The preferred main model is Macro + Satellite. On the 2019-2020 out-of-time validation set, it achieves average precision of 0.519, compared with 0.461 for Macro Only. Across five rolling validation windows, Macro + Satellite also has the highest mean average precision. The All Features model has the highest validation AUC at 0.904, but its average precision is lower, so we treat it as a robustness specification rather than the main result.

**Next steps.** The main next steps are to extend crisis labels through 2026 for a clean 2021-2023 final test, harmonize DMSP and VIIRS nightlights to recover a longer satellite history, and validate misreporting flags with qualitative evidence such as IMF Article IV reports.

## 3. Core Facts to Remember

| Item | Current Project Status |
|---|---|
| Unit of observation | Country-year |
| Country scope | Low-, lower-middle-, and upper-middle-income economies |
| Panel size | 3,096 country-years |
| Countries | 129 |
| Years | 2000-2023 |
| Prediction target | Crisis onset within the next three years |
| Main validation period | 2019-2020 |
| Censored future period | 2021-2023 |
| Preferred model | Macro + Satellite |
| Robustness model | All Features |
| Main validation metric | Average precision |
| Main model validation AP | 0.519 |
| Main model validation AUC | 0.816 |
| All Features validation AUC | 0.904 |
| Rolling validation best mean AP | Macro + Satellite, 0.366 |
| Fairness AUC gap | 0.114 across regions |
| Misreporting index output | `outputs/tables/misreporting_index.csv` |

## 4. Research Questions and Answers

### Q1. Do satellite trends and news sentiment contain predictive information beyond official macro data?

Yes, with the strongest evidence for satellite nightlights. Macro + Satellite improves validation average precision from 0.461 to 0.519 relative to Macro Only. It also has the best mean average precision across rolling validation windows.

Text features contain signal, especially in the official validation split, but they are less stable across rolling windows. This is likely because GDELT coverage varies heavily by country and year.

### Q2. Does satellite-official divergence proxy for misreporting and predict crisis risk?

Partially. The project implements a divergence-based misreporting index:

```text
misreporting = official GDP growth - nightlight growth
```

The model-level result is stronger at the block level than at the single-feature level: satellite features as a group improve precision, while the misreporting index is best framed as exploratory case generation rather than a validated manipulation ranking.

### Q3. Is model performance consistent across regions and income groups?

Mostly, but sample sizes are small. Regional AUCs range from 0.833 to 0.947 on the validation period, producing an AUC gap of 0.114. The audit also produces region-specific threshold diagnostics, but these are not deployment-ready policy rules.

## 5. Data Details

### World Bank WDI

Main indicators:

- GDP growth
- Inflation
- Debt-to-GDP
- Current account balance
- Reserves in months of imports
- External debt to GNI
- Fiscal balance
- Exchange rate

WDI also supplies region and income-group metadata. The project keeps LIC, LMC, and UMC economies and excludes high-income economies.

### VIIRS Nighttime Lights

Nightlights are used as an external proxy for observed economic activity. The project uses VIIRS-only features to avoid contaminating growth rates with the DMSP-VIIRS sensor transition.

Key features:

- Nightlight level
- Nightlight growth
- Nightlight rolling volatility
- Satellite-official divergence index

### GDELT Events and GKG

GDELT contributes two types of features:

- Event features: tone, economic events, conflict events, protest events, Goldstein scale, mentions.
- GKG article features: IMF mentions, debt mentions, article tone, article volume, article-share measures.

GDELT features are aggregated to country-year level to match WDI and crisis labels.

### Crisis Labels

Crisis labels are curated from well-documented sovereign debt crisis episodes, drawing on Laeven and Valencia, Reinhart and Rogoff, and IMF reports. The model target is forward-looking:

```text
crisis_target = 1 if a crisis onset occurs within the next three years
```

This is an early-warning target, not a contemporaneous crisis classifier.

## 6. Feature Engineering

The most important methodological choice is strict lag-only feature construction. This prevents look-ahead bias.

Allowed model features:

- `*_lag1`, `*_lag2`, `*_lag3`
- prior rolling means and standard deviations
- prior year-over-year changes
- interactions built from lagged inputs

Disallowed from feature blocks:

- current-year WDI values
- current-year GDELT values
- current-year nightlight values

This makes the prediction task more credible because all features would be available before or at the prediction point.

## 7. Model Design

### XGBoost

XGBoost is the primary model because it handles nonlinear relationships, interactions, missingness after imputation, and imbalanced classification better than a simple linear model.

Nested specifications:

1. Macro Only
2. Macro + Satellite
3. Macro + Text
4. All Features

### LASSO

LASSO logistic regression is used as an interpretable baseline. It selected only three nonzero features:

- `reserves_months_lag1`
- `nightlight_raw_lag3`
- `conflict_events_lag1`

Its validation AUC is 0.696 and average precision is 0.202, which supports the claim that the nonlinear XGBoost model captures more useful structure.

### SHAP

SHAP is used to interpret feature importance in the All Features robustness model. It supports interpretability but should not be interpreted as causal evidence.

## 8. Validation Protocol

### Main out-of-time validation

The main validation period is 2019-2020. It is out-of-time relative to training years through 2018 and is fully observable under the three-year target because the label file runs through 2023.

### Rolling-origin validation

The rolling validation script evaluates five pre-2021 windows:

- train through 2014, validate 2015-2016
- train through 2015, validate 2016-2017
- train through 2016, validate 2017-2018
- train through 2017, validate 2018-2019
- train through 2018, validate 2019-2020

This directly addresses the TA's concern about unintentional overfitting to one validation window.

### Why 2021-2023 is not the main test set

The project predicts crisis onset within the next three years. Since crisis labels currently end in 2023, the 2021-2023 period is right-censored:

- 2021 requires observing 2022-2024.
- 2022 requires observing 2023-2025.
- 2023 requires observing 2024-2026.

Therefore, 2021-2023 should not be used as the main official test set unless crisis labels are extended through 2026. It can only be used as a clearly labeled diagnostic future-period exercise.

## 9. Main Results

### Official validation split, 2019-2020

| Specification | AUC | Average Precision | Precision | Recall |
|---|---:|---:|---:|---:|
| Macro Only | 0.794 | 0.461 | 0.459 | 0.607 |
| Macro + Satellite | 0.816 | 0.519 | 0.388 | 0.679 |
| Macro + Text | 0.848 | 0.447 | 0.324 | 0.786 |
| All Features | 0.904 | 0.437 | 0.317 | 0.929 |

### Rolling validation summary

| Specification | Mean AUC | Mean AP |
|---|---:|---:|
| Macro + Satellite | 0.759 | 0.366 |
| Macro Only | 0.759 | 0.345 |
| All Features | 0.790 | 0.335 |
| Macro + Text | 0.767 | 0.292 |

### Preferred interpretation

The safest headline claim is:

> Satellite nightlight features improve rare-crisis precision beyond macroeconomic baselines, while text features are informative but less stable across time windows.

## 10. Misreporting Index

The TA specifically suggested an indicator of the estimated degree of misreporting by country. This is implemented in `scripts/12_misreporting_index.py`.

Definition:

```text
misreport_raw = GDP growth - nightlight growth
```

Country-level outputs:

- mean misreporting
- median misreporting
- standard deviation
- anomaly count
- anomaly rate
- peak anomaly z-score
- peak anomaly year

Current top countries by anomaly rate:

- Moldova
- Bangladesh

Each is flagged in one out of six observed years. No country is flagged in more than 30 percent of observed years, so the index should be presented cautiously.

## 11. Fairness Audit

The fairness audit checks whether the model performs similarly across regions and income groups.

Regional validation AUCs:

- Sub-Saharan Africa: 0.947
- Latin America & Caribbean: 0.900
- East Asia & Pacific: 0.857
- South Asia: 0.833
- Middle East & North Africa: 0.888
- Europe & Central Asia: 0.928

The regional AUC gap is 0.114. This is not large enough to dominate the project, but it is important enough to report. The threshold file `fairness_thresholds_by_region.csv` shows how thresholds could be adjusted to target similar false-positive rates across regions.

## 12. Structural Model

The structural model is a two-period signaling game:

1. A distressed government chooses whether to report accurately or misreport.
2. Creditors update beliefs and the country either services debt or defaults.

The model supports the project narrative: under distress, governments may have incentives to overreport economic strength, producing divergence between official statistics and externally observed activity.

Current validation:

- Simulated debt-divergence correlation: -0.037
- Empirical debt-divergence correlation: -0.133
- Simulated divergence among defaulters: 0.161
- Simulated divergence among non-defaulters: 0.144

This is theoretical grounding, not a causal identification strategy.

## 13. Deviations from the Proposal and Why They Are Reasonable

### Proposal: use DMSP + VIIRS nightlights from 2000-2023

Current project: VIIRS-only nightlight features.

Reason: DMSP and VIIRS are not directly comparable. Without harmonization, combining them creates artificial jumps around the sensor transition. The current approach sacrifices some coverage to avoid biased satellite growth features.

### Proposal: predict crisis within 12 months

Current project: predict crisis onset within three years.

Reason: The panel is annual, WDI indicators are annual, and sovereign debt crises often develop slowly. A three-year early-warning horizon is more stable and more policy-relevant for annual surveillance.

### Proposal: monthly GDELT panel

Current project: annual GDELT aggregation.

Reason: WDI and crisis labels are annual. Monthly interpolation would add false precision.

### Proposal: difference-in-differences using VIIRS rollout

Current project: not implemented.

Reason: VIIRS had near-global coverage from the beginning, so there is no meaningful staggered rollout for a credible DiD design. Forcing this design would be weaker than dropping it and being transparent.

## 14. Likely Questions and Strong Answers

### Why is Macro + Satellite the main model if All Features has higher AUC?

Because sovereign debt crises are rare, average precision is more informative than AUC. Macro + Satellite has the best validation AP and the best rolling-window mean AP. All Features ranks well by AUC but has lower AP, so we frame it as a robustness model.

### Why not use the 2021-2023 test set?

Because the target is crisis onset within the next three years, and labels only run through 2023. The 2021-2023 period is right-censored. We would need labels through 2026 to evaluate it cleanly.

### Did you address the TA feedback?

Yes. The project keeps a holdout-style out-of-time validation period and adds rolling-origin validation to test unintentional overfitting. It also implements a country-level misreporting index.

### Is the misreporting index proof that a government manipulated statistics?

No. It is a statistical anomaly indicator, not proof. It identifies cases where reported GDP growth is high relative to nightlight growth. It should be validated with qualitative evidence such as IMF Article IV reports, political economy context, or national accounts revisions.

### Why use XGBoost?

XGBoost handles nonlinearities, interactions, class imbalance, and mixed feature blocks well in small-to-medium tabular data. It is also appropriate for an Applied Machine Learning project because it can be compared cleanly across nested feature sets.

### Why use average precision?

The positive class is rare. Average precision focuses on whether high-risk predictions are actually enriched with crises. AUC measures ranking across all thresholds and can look strong even when precision is modest.

### Why aggregate GDELT annually?

The target and macro data are annual. Monthly GDELT would require interpolating annual macro data and crisis labels, creating false precision.

### Could GDELT reflect media bias?

Yes. GDELT is media coverage, not ground truth. The model includes volume-related features, and the report treats text as a noisy supplementary signal rather than the central finding.

### Why exclude high-income countries?

The project focuses on emerging and developing economies, where sovereign debt crises are more relevant, nightlight saturation is less severe, and macro reporting concerns are more central.

### Is this causal?

No. The main model is predictive. The structural model gives theoretical grounding, but the project does not claim causal identification.

### What would make the project stronger?

The strongest extensions are:

1. Extend crisis labels through 2026.
2. Harmonize DMSP and VIIRS nightlights.
3. Validate misreporting flags with IMF reports and country case studies.
4. Add calibrated decision thresholds for policy use.

## 15. Files to Know

| File | Purpose |
|---|---|
| `README.md` | Main project documentation |
| `sovereign_debt_dashboard.html` | Self-contained dashboard |
| `presentation/Beyond_Official_Statistics_2min.pptx` | Two-minute presentation deck |
| `scripts/06_feature_engineering.py` | Lag-only feature construction |
| `scripts/08_xgboost_model.py` | Main model comparison |
| `scripts/10_fairness_audit.py` | Regional and income-group audit |
| `scripts/12_misreporting_index.py` | Country-level misreporting index |
| `scripts/13_rolling_validation.py` | Rolling-origin robustness checks |
| `scripts/14_generate_dashboard.py` | Rebuilds dashboard from outputs |
| `scripts/15_generate_presentation.py` | Rebuilds PPTX from outputs |

## 16. Bottom Line

The project is strongest as an applied machine learning early-warning system. It does not prove manipulation causally, but it shows that external data, especially satellite nightlights, can improve rare-crisis prediction beyond official macroeconomic baselines while producing interpretable diagnostics for misreporting and fairness.
