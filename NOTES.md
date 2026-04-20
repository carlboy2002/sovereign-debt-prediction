# Project Notes

*Last updated: 2026-04-20*

---

## 1. Known Data Quality Issues

### Nighttime Lights (VIIRS)

**Sensor transition artifact (DMSP → VIIRS, 2013–2014).**
The DMSP and VIIRS satellite systems are not directly comparable due to differences in sensor design, dynamic range, and calibration. Naively stacking the two series introduces a systematic discontinuity around 2013–2014 that contaminates any trend or growth-rate feature spanning the transition.

- **Current workaround:** We use VIIRS data only (2014 onward), discarding the full DMSP period. This eliminates the artifact but reduces effective sample size by roughly 40%, weakening statistical power for models that need long pre-crisis windows.
- **Proper fix:** Intercalibrate using the cross-sensor harmonization approach in Li et al. (2020), which maps DMSP radiance values onto the VIIRS scale. Not yet implemented — requires downloading and aligning raster files at country-year level.

**Saturation bias in dense urban areas.**
VIIRS detects light intensity up to a sensor ceiling. High-income countries with heavily lit cities hit this ceiling frequently, making their nightlight growth measures uninformative (compressed near zero regardless of actual economic change). The misreporting index in `scripts/12_misreporting_index.py` partially filters these out, but the threshold is ad hoc and the issue also affects features used in the main model.

**GEE export latency and coverage gaps.**
The Google Earth Engine batch export (`fetch_nl_batch.py`) submits per-year tasks that can take hours and occasionally fail silently. Several small island states and landlocked countries with sparse VIIRS coverage return NaN for entire years, which propagates into `divergence_index` and all lagged/rolling features derived from it.

### GDELT / News Text

**FIPS 10-4 → ISO 3166 code mapping.**
GDELT uses FIPS 10-4 country codes; the rest of the pipeline uses ISO alpha-3. The mapping table in `scripts/03_fetch_gdelt.py` is manually maintained and known to be incomplete for about a dozen territories (Western Sahara, Kosovo, South Sudan, etc.). These countries receive NaN sentiment features and are effectively dropped from any model specification that includes text.

**GDELT event volume varies sharply by country.**
Large, English-language-prominent countries (e.g., Argentina, Turkey) have orders-of-magnitude more events per year than smaller or less covered countries (e.g., Togo, Bhutan). The aggregated `avg_tone` and `goldstein_avg` features are therefore much noisier for low-coverage countries, potentially introducing heteroskedastic error that the models do not explicitly account for.

**Sentiment scores capture media coverage, not ground truth.**
GDELT tone reflects how international media frames events, not the actual economic or political trajectory. During crises that received little international attention (some Sub-Saharan African episodes in the 2000s), the signal is weak or absent.

### Crisis Labels

**Label sparsity and class imbalance.**
Crisis onset years represent roughly 8% of country-year observations. The `scale_pos_weight` in XGBoost and `class_weight="balanced"` in LASSO compensate, but the model still sees very few true positives in the validation window (2019–2020), making AUC estimates on that window noisy.

**Label uncertainty at crisis boundaries.**
The Laeven & Valencia database defines crisis start years, but the hardcoded episodes in `scripts/04_fetch_crisis_labels.py` extend these with some manual judgement calls. Reasonable people differ on whether a given year is "onset" vs. "escalation." This introduces label noise particularly around multi-year crises.

### World Bank WDI

**Reporting lags and revisions.**
WDI data for recent years (2021–2023) is often preliminary and subject to revision. The test set spans exactly this window, so features derived from official statistics may be less reliable there than in the training period.

---

## 2. Approaches Tried but Abandoned

### Difference-in-Differences with VIIRS Rollout as Instrument

**Original plan:** Use the staggered expansion of VIIRS satellite coverage across countries as a natural experiment. Countries that gained VIIRS coverage in a given year would serve as a control group before coverage and treatment group after, allowing causal identification of the relationship between true economic activity and reported GDP.

**Why it was dropped:** VIIRS had near-global coverage from its first operational year (2012). There is no meaningful staggered rollout to exploit — essentially all emerging-market countries entered the VIIRS sample at the same time. The DiD design would have zero identifying variation. We switched to using commodity price shocks as a potential instrument, but this was also shelved due to time constraints.

### DMSP-Based Long History (Pre-2014 Nightlights)

**Original plan:** Use the full DMSP archive back to 2000 to give the model a longer pre-crisis runway, consistent with the WDI sample period.

**Why it was dropped:** The DMSP-VIIRS transition artifact (see Section 1) creates spurious growth spikes or drops that are correlated with the sensor switch, not with economic conditions. Any lag or rolling-window feature spanning 2013–2014 picks up sensor noise as signal. Rather than risk contaminating the full feature set, we restricted to VIIRS only. The ~40% sample reduction was judged preferable to biased features.

### Random Forest and Neural Network Specifications

**Original plan:** The proposal discussed testing a wider model class including random forests and a simple LSTM for sequential country-year data.

**Why it was dropped:** Given the small effective sample (roughly 1,500–2,000 country-year observations after VIIRS filtering), deep models overfit quickly and did not produce interpretable outputs. XGBoost with depth-4 trees already captures nonlinear interactions; adding random forests provided minimal AUC lift with much higher variance. The LSTM was dropped entirely because the panel is unbalanced (missing years for many countries) and sequence lengths are short.

### Elastic Net as LASSO Alternative

**Original plan:** When LASSO showed marginal convergence at several regularization values (noted in `scripts/09_lasso_model.py`), elastic net (L1 + L2 combined) was considered as a more stable alternative.

**Why it was dropped:** Elastic net adds a second hyperparameter (the L1/L2 mixing ratio) to cross-validate. With the small training set and 7-fold CV already used for alpha selection, adding this dimension would have required a grid search that is expensive to interpret cleanly. PCA preprocessing was also considered to address near-collinearity among lagged features, but this would destroy the direct interpretability of which original features survive — which is a key output of the LASSO analysis.

### Monthly Panel Modeling

**Original plan:** GDELT data is available monthly; building a monthly panel would increase effective observations and allow earlier detection of sentiment shifts.

**Why it was dropped:** WDI and crisis labels are annual; interpolating annual macro indicators to monthly frequency adds spurious precision. More importantly, the crisis label (onset within next 3 years) is inherently an annual concept tied to debt restructuring calendars. Aggregating GDELT monthly features to annual means was the cleaner choice, even at the cost of losing within-year signal.

---

## 3. Next Priority Items

### High Priority (before May 9 deadline)

1. **Implement Li et al. (2020) DMSP-VIIRS harmonization.**
   Extending the nightlights series back to 2000 would nearly double the training sample and allow the model to observe full pre-crisis trajectories for the large wave of crises in the early 2000s (Argentina 2001, Ecuador 1999 effects, etc.). This is the single change most likely to improve model calibration on the test set.

2. **Fix the FIPS → ISO code mapping gaps.**
   Audit which countries are silently dropped from the GDELT merge. For the dozen or so countries with missing mappings, manually add entries or impute national-level sentiment from regional aggregates. This will recover text features for a meaningful fraction of the crisis label set.

3. **Evaluate the model on the held-out test set (2021–2023).**
   The test-set evaluation block in `scripts/08_xgboost_model.py` is intentionally commented out. Uncomment and run once for the final reported numbers. Do not tune further after this.

4. **Recalibrate thresholds by region for the fairness audit.**
   The fairness gap of 0.141 AUC between regions is large enough to matter in a real deployment. Script 10 already computes regional AUCs; the next step is to select classification thresholds that equalize false-positive rates (or precision) across Sub-Saharan Africa, Latin America, and South Asia before reporting final results.

### Medium Priority

5. **Quantify GDELT volume bias.**
   Add a coverage-weight feature (log of annual event count per country) and test whether including it as a feature or using it to down-weight low-coverage country-years changes the SHAP rankings for sentiment variables. This would also support the claim that GDELT is informative even after accounting for differential coverage.

6. **Estimate structural model parameters from data.**
   The signaling-game simulation in `scripts/11_structural_model.py` uses illustrative parameter values. Using simulated method of moments (SMM) to fit the misreporting cost parameter (c) and the creditor prior to the observed divergence distribution would make the structural section more credible as empirical validation rather than illustrative narrative.

7. **Cross-validate the misreporting index against IMF Article IV consultations.**
   The ~12 countries flagged as systematic over-reporters in `scripts/12_misreporting_index.py` should be spot-checked against qualitative evidence (IMF consultation language, press freedom indices, political economy literature). This would either validate the index as a useful output or reveal that it is picking up measurement noise.

### Lower Priority / Post-Deadline

8. **Instrument with commodity price shocks.**
   For countries whose exports are dominated by a single commodity, global commodity price changes are plausibly exogenous to domestic reporting choices. This could serve as the causal instrument that the DiD approach was supposed to provide.

9. **Address VIIRS saturation bias in feature construction.**
   Flag country-years where radiance is near the VIIRS sensor ceiling and exclude or winsorize them before computing growth rates, rather than relying on the misreporting index filter as a downstream catch.

10. **Dashboard data refresh.**
    `sovereign_debt_dashboard.html` currently embeds static outputs. If the test-set evaluation or threshold recalibration changes figures, regenerate and re-embed the affected plots before final submission.
