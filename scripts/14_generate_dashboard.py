"""
14_generate_dashboard.py - Build a self-contained HTML dashboard from latest outputs.

The script reads current result tables and figures, then rewrites
sovereign_debt_dashboard.html so the dashboard stays synchronized with the
latest model run.
"""
import base64
import html
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import DATA_PROCESSED, FIGURES, ROOT, TABLES  # noqa: E402


OUT_PATH = ROOT / "sovereign_debt_dashboard.html"


def fmt(value, digits=3):
    """Format numeric values compactly."""
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def pct(value, digits=1):
    """Format a decimal as a percentage."""
    if pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def image_data_uri(path):
    """Return a base64 data URI for an image file."""
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def table_html(df, columns, formats=None):
    """Render a compact HTML table."""
    formats = formats or {}
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col, _ in columns:
            value = row[col]
            formatter = formats.get(col)
            text = formatter(value) if formatter else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def load_metrics():
    """Load result tables used in the dashboard."""
    xgb = pd.read_csv(TABLES / "xgboost_nested_comparison.csv")
    rolling = pd.read_csv(TABLES / "xgboost_rolling_validation_summary.csv")
    fairness = pd.read_csv(TABLES / "fairness_by_region.csv")
    lasso = pd.read_csv(TABLES / "lasso_coefficients.csv")
    misreport = pd.read_csv(TABLES / "misreporting_index.csv")
    structural = pd.read_csv(TABLES / "structural_model_validation.csv")
    panel = pd.read_csv(DATA_PROCESSED / "panel_features.csv")
    return xgb, rolling, fairness, lasso, misreport, structural, panel


def build_html():
    """Build the dashboard HTML string."""
    xgb, rolling, fairness, lasso, misreport, structural, panel = load_metrics()

    macro_sat = xgb[xgb["specification"] == "Macro + Satellite"].iloc[0]
    all_features = xgb[xgb["specification"] == "All Features"].iloc[0]
    macro_only = xgb[xgb["specification"] == "Macro Only"].iloc[0]
    rolling_best = rolling.sort_values("mean_avg_precision", ascending=False).iloc[0]
    fairness_gap = fairness["auc"].max() - fairness["auc"].min()
    lasso_selected = int((lasso["abs_coef"] > 0).sum())
    top_lasso = lasso[lasso["abs_coef"] > 0].copy()
    top_misreport = misreport.head(8).copy()
    structural_row = structural.iloc[0]

    split_summary = (
        panel.groupby("split")["crisis_target"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "observations", "sum": "positives", "mean": "positive_rate"})
    )

    xgb_table = table_html(
        xgb,
        [
            ("specification", "Specification"),
            ("auc", "AUC"),
            ("avg_precision", "Avg. Precision"),
            ("precision", "Precision"),
            ("recall", "Recall"),
        ],
        {
            "auc": fmt,
            "avg_precision": fmt,
            "precision": fmt,
            "recall": fmt,
        },
    )

    rolling_table = table_html(
        rolling,
        [
            ("specification", "Specification"),
            ("mean_auc", "Mean AUC"),
            ("std_auc", "AUC SD"),
            ("mean_avg_precision", "Mean AP"),
            ("std_avg_precision", "AP SD"),
        ],
        {
            "mean_auc": fmt,
            "std_auc": fmt,
            "mean_avg_precision": fmt,
            "std_avg_precision": fmt,
        },
    )

    fairness_table = table_html(
        fairness,
        [
            ("region", "Region"),
            ("n_obs", "Obs."),
            ("n_crises", "Crises"),
            ("auc", "AUC"),
            ("avg_precision", "AP"),
        ],
        {"auc": fmt, "avg_precision": fmt},
    )

    lasso_table = table_html(
        top_lasso,
        [
            ("feature", "Feature"),
            ("block", "Block"),
            ("coefficient", "Coefficient"),
        ],
        {"coefficient": fmt},
    )

    misreport_table = table_html(
        top_misreport,
        [
            ("rank", "Rank"),
            ("country_name", "Country"),
            ("iso3c", "ISO3"),
            ("anomaly_rate", "Anomaly Rate"),
            ("peak_anomaly_zscore", "Peak Z"),
        ],
        {"anomaly_rate": pct, "peak_anomaly_zscore": fmt},
    )

    split_table = table_html(
        split_summary,
        [
            ("split", "Split"),
            ("observations", "Observations"),
            ("positives", "Positive Targets"),
            ("positive_rate", "Positive Rate"),
        ],
        {"positive_rate": pct},
    )

    roc_img = image_data_uri(FIGURES / "roc_comparison.png")
    fairness_img = image_data_uri(FIGURES / "fairness_auc_by_region.png")
    misreport_img = image_data_uri(FIGURES / "misreporting_top_countries.png")
    structural_img = image_data_uri(FIGURES / "structural_model_predictions.png")
    shap_img = image_data_uri(FIGURES / "shap_All Features.png")

    today = date.today().isoformat()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Beyond Official Statistics Dashboard</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5a6578;
      --line: #d8dee8;
      --paper: #f7f8fb;
      --white: #ffffff;
      --navy: #19324d;
      --teal: #0f766e;
      --wine: #9f2842;
      --gold: #b7791f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background: var(--paper);
      line-height: 1.55;
    }}
    header {{
      background: var(--navy);
      color: var(--white);
      padding: 44px 8vw 34px;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: clamp(34px, 5vw, 58px);
      line-height: 1.04;
      letter-spacing: 0;
    }}
    header p {{
      max-width: 980px;
      margin: 10px 0 0;
      color: #dbe6f4;
      font-size: 18px;
    }}
    nav {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      padding: 14px 8vw;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    nav a {{ color: var(--navy); text-decoration: none; font-weight: 650; font-size: 14px; }}
    main {{ padding: 26px 8vw 54px; }}
    section {{ margin: 28px 0 44px; }}
    h2 {{ font-size: 28px; margin: 0 0 14px; }}
    h3 {{ font-size: 19px; margin: 0 0 10px; }}
    .lead {{ max-width: 1000px; font-size: 18px; color: #2f3b52; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--white);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      min-height: 130px;
    }}
    .metric {{
      font-size: 34px;
      font-weight: 760;
      margin: 5px 0;
    }}
    .metric.teal {{ color: var(--teal); }}
    .metric.wine {{ color: var(--wine); }}
    .metric.gold {{ color: var(--gold); }}
    .label {{ color: var(--muted); font-size: 14px; font-weight: 650; text-transform: uppercase; }}
    .note {{ color: var(--muted); font-size: 14px; }}
    .band {{
      background: var(--white);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      margin-top: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--white);
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    td:not(:first-child), th:not(:first-child) {{ text-align: right; }}
    .two-col {{
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
      gap: 22px;
      align-items: start;
    }}
    figure {{
      margin: 0;
      background: var(--white);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    figure img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    .callout {{
      border-left: 5px solid var(--teal);
      background: #eef8f6;
      padding: 14px 16px;
      border-radius: 6px;
      margin-top: 14px;
    }}
    footer {{
      padding: 26px 8vw;
      color: var(--muted);
      border-top: 1px solid var(--line);
      background: var(--white);
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      .two-col {{ grid-template-columns: 1fr; }}
      nav {{ position: static; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Beyond Official Statistics</h1>
    <p>Predicting sovereign debt crises with macro data, satellite nightlights, and global news text.</p>
    <p class="note">Dashboard refreshed from current output tables and figures on {today}.</p>
  </header>
  <nav>
    <a href="#overview">Overview</a>
    <a href="#data">Data</a>
    <a href="#models">Models</a>
    <a href="#validation">Validation</a>
    <a href="#fairness">Fairness</a>
    <a href="#misreporting">Misreporting</a>
    <a href="#limits">Limitations</a>
  </nav>
  <main>
    <section id="overview">
      <h2>Executive Takeaway</h2>
      <p class="lead">The project asks whether alternative data can improve sovereign-debt early warning systems when official statistics are delayed, revised, or strategically reported. The strongest main specification is <strong>Macro + Satellite</strong>: it has the best validation average precision and the best rolling-window average precision. The All Features model has the highest validation AUC and is best treated as a robustness specification.</p>
      <div class="grid">
        <div class="card">
          <div class="label">Panel</div>
          <div class="metric">{panel["iso3c"].nunique()}</div>
          <div class="note">LIC, LMC, and UMC countries, 2000-2023</div>
        </div>
        <div class="card">
          <div class="label">Main Model AP</div>
          <div class="metric teal">{fmt(macro_sat["avg_precision"])}</div>
          <div class="note">Macro + Satellite, validation 2019-2020</div>
        </div>
        <div class="card">
          <div class="label">Highest AUC</div>
          <div class="metric gold">{fmt(all_features["auc"])}</div>
          <div class="note">All Features, validation robustness model</div>
        </div>
        <div class="card">
          <div class="label">Fairness Gap</div>
          <div class="metric wine">{fmt(fairness_gap)}</div>
          <div class="note">Regional max-min AUC on validation period</div>
        </div>
      </div>
    </section>

    <section id="data">
      <h2>Data and Target</h2>
      <div class="two-col">
        <div class="band">
          <h3>Three data streams</h3>
          <p><strong>WDI macro indicators</strong> provide the official baseline: GDP growth, inflation, debt, current account, reserves, external debt, fiscal balance, and exchange rates.</p>
          <p><strong>VIIRS nighttime lights</strong> proxy for observed economic activity and support a satellite-official divergence index.</p>
          <p><strong>GDELT events and GKG</strong> provide media tone, event intensity, conflict/protest indicators, and IMF/debt article mentions.</p>
          <div class="callout">All model features are lagged or computed from prior years only. Current-year variables are excluded to avoid look-ahead bias.</div>
        </div>
        <div class="band">
          <h3>Split summary</h3>
          {split_table}
          <p class="note">The 2021-2023 block is right-censored under the three-year target and is not used as the main final test set.</p>
        </div>
      </div>
    </section>

    <section id="models">
      <h2>Nested XGBoost Results</h2>
      <div class="two-col">
        <div class="band">
          {xgb_table}
          <p class="note">Average precision is emphasized because debt crises are rare. AUC remains useful for ranking performance but can look strong even when precision is modest.</p>
        </div>
        <figure>
          <img src="{roc_img}" alt="ROC comparison">
          <figcaption>ROC curves for nested XGBoost specifications generated from the latest run.</figcaption>
        </figure>
      </div>
    </section>

    <section id="validation">
      <h2>Robustness and Interpretation</h2>
      <div class="two-col">
        <div class="band">
          <h3>Rolling-origin validation</h3>
          {rolling_table}
          <p class="note">Rolling windows test whether results depend on one favorable split. Macro + Satellite has the highest mean average precision across five pre-2021 windows.</p>
        </div>
        <figure>
          <img src="{shap_img}" alt="SHAP feature importance">
          <figcaption>SHAP feature importance for the All Features robustness model.</figcaption>
        </figure>
      </div>
      <div class="band">
        <h3>LASSO baseline</h3>
        <p>LASSO selected {lasso_selected} nonzero features and achieved validation AUC 0.696 with average precision 0.202. This supports using it as an interpretable baseline rather than the primary model.</p>
        {lasso_table}
      </div>
    </section>

    <section id="fairness">
      <h2>Fairness Audit</h2>
      <div class="two-col">
        <div class="band">
          {fairness_table}
          <p class="note">Region-specific threshold diagnostics are saved in outputs/tables/fairness_thresholds_by_region.csv.</p>
        </div>
        <figure>
          <img src="{fairness_img}" alt="Fairness by region">
          <figcaption>Validation AUC by region. Sample sizes are small, so these are diagnostics rather than deployment rules.</figcaption>
        </figure>
      </div>
    </section>

    <section id="misreporting">
      <h2>Misreporting Index and Structural Grounding</h2>
      <div class="two-col">
        <div class="band">
          <h3>Country-level misreporting indicator</h3>
          <p>The index compares official GDP growth with satellite nightlight growth. A positive anomaly means reported growth exceeds what satellite growth would suggest.</p>
          {misreport_table}
          <p class="note">No country is flagged in more than 30 percent of observed years, so the index is best used for case generation.</p>
        </div>
        <figure>
          <img src="{misreport_img}" alt="Misreporting top countries">
          <figcaption>Top suspected over-reporters by share of suspicious years.</figcaption>
        </figure>
      </div>
      <div class="two-col" style="margin-top: 22px;">
        <div class="band">
          <h3>Structural model check</h3>
          <p>Simulated debt-divergence correlation: <strong>{fmt(structural_row["sim_debt_divergence_corr"])}</strong></p>
          <p>Empirical debt-divergence correlation: <strong>{fmt(structural_row["emp_debt_divergence_corr"])}</strong></p>
          <p>Simulated divergence is higher among defaulters ({fmt(structural_row["sim_default_divergence"])}) than non-defaulters ({fmt(structural_row["sim_nondefault_divergence"])}).</p>
        </div>
        <figure>
          <img src="{structural_img}" alt="Structural model predictions">
          <figcaption>The structural model is theoretical grounding, not causal identification.</figcaption>
        </figure>
      </div>
    </section>

    <section id="limits">
      <h2>Limitations and Next Steps</h2>
      <div class="grid">
        <div class="card">
          <h3>Right-censoring</h3>
          <p>The current labels end in 2023, so 2021-2023 cannot be a clean final test period under a three-year horizon.</p>
        </div>
        <div class="card">
          <h3>Satellite history</h3>
          <p>The project uses VIIRS-only nightlights to avoid DMSP-VIIRS sensor discontinuities.</p>
        </div>
        <div class="card">
          <h3>GDELT coverage</h3>
          <p>News volume varies by country, so text signals should be interpreted as media coverage rather than ground truth.</p>
        </div>
        <div class="card">
          <h3>Next steps</h3>
          <p>Extend crisis labels, update the dashboard after final figures, and consider DMSP-VIIRS harmonization for a longer satellite history.</p>
        </div>
      </div>
    </section>
  </main>
  <footer>
    Beyond Official Statistics - Columbia SIPA DSPC7100 Applying Machine Learning Final Project.
  </footer>
</body>
</html>
"""


def main():
    OUT_PATH.write_text(build_html(), encoding="utf-8")
    print(f"Saved dashboard: {OUT_PATH}")


if __name__ == "__main__":
    main()
