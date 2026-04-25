"""
13_rolling_validation.py - Rolling-origin validation for XGBoost specifications.

This script checks whether validation performance is stable across multiple
pre-test time windows. It never evaluates the locked 2021-2023 test period.

Output:
    outputs/tables/xgboost_rolling_validation.csv
    outputs/tables/xgboost_rolling_validation_summary.csv
"""
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import DATA_PROCESSED, SEED, TABLES, TEST_START, XGB_PARAMS  # noqa: E402


WINDOWS = [
    {"train_end": 2014, "val_start": 2015, "val_end": 2016},
    {"train_end": 2015, "val_start": 2016, "val_end": 2017},
    {"train_end": 2016, "val_start": 2017, "val_end": 2018},
    {"train_end": 2017, "val_start": 2018, "val_end": 2019},
    {"train_end": 2018, "val_start": 2019, "val_end": 2020},
]


def load_inputs():
    """Load engineered panel and feature block definitions."""
    df = pd.read_csv(DATA_PROCESSED / "panel_features.csv")
    with open(DATA_PROCESSED / "feature_blocks.json", encoding="utf-8") as f:
        blocks = json.load(f)
    return df, blocks


def dedupe(features):
    """Preserve feature order while removing duplicates."""
    return list(dict.fromkeys(features))


def make_specs(blocks):
    """Build nested model specifications."""
    return {
        "Macro Only": blocks["macro"],
        "Macro + Satellite": dedupe(blocks["macro"] + blocks["satellite"]),
        "Macro + Text": dedupe(blocks["macro"] + blocks["text"]),
        "All Features": blocks["all"],
    }


def prepare_matrices(train, val, features):
    """Fit imputation on train only, then transform train and validation."""
    existing = [f for f in features if f in train.columns]
    existing = [f for f in existing if train[f].notna().any()]
    if not existing:
        raise ValueError("No usable features remain after dropping all-missing columns.")

    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(
        imputer.fit_transform(train[existing]),
        columns=existing,
        index=train.index,
    )
    X_val = pd.DataFrame(
        imputer.transform(val[existing]),
        columns=existing,
        index=val.index,
    )
    y_train = train["crisis_target"].to_numpy()
    y_val = val["crisis_target"].to_numpy()
    return X_train, y_train, X_val, y_val


def train_xgb(X_train, y_train, X_val, y_val):
    """Train XGBoost with validation early stopping."""
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = n_neg / n_pos
    params["early_stopping_rounds"] = 30

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate(model, X_val, y_val):
    """Compute threshold-free ranking metrics."""
    y_prob = model.predict_proba(X_val)[:, 1]
    return {
        "auc": roc_auc_score(y_val, y_prob),
        "avg_precision": average_precision_score(y_val, y_prob),
    }


def summarize(results_df):
    """Summarize metric stability across rolling windows."""
    summary = (
        results_df.groupby("specification", as_index=False)
        .agg(
            n_windows=("window", "count"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            min_auc=("auc", "min"),
            mean_avg_precision=("avg_precision", "mean"),
            std_avg_precision=("avg_precision", "std"),
            min_avg_precision=("avg_precision", "min"),
        )
        .sort_values("mean_avg_precision", ascending=False)
    )
    return summary


def main():
    print("Running rolling-origin validation...")
    df, blocks = load_inputs()
    df = df[df["year"] < TEST_START].copy()
    specs = make_specs(blocks)

    rows = []
    for window in WINDOWS:
        train = df[df["year"] <= window["train_end"]].copy()
        val = df[df["year"].between(window["val_start"], window["val_end"])].copy()
        window_name = (
            f"train<= {window['train_end']}, "
            f"val {window['val_start']}-{window['val_end']}"
        )

        print(f"\n  Window: {window_name}")
        if train["crisis_target"].nunique() < 2 or val["crisis_target"].nunique() < 2:
            print("    Skipping: train or validation window has only one class.")
            continue

        for spec_name, features in specs.items():
            X_train, y_train, X_val, y_val = prepare_matrices(train, val, features)
            model = train_xgb(X_train, y_train, X_val, y_val)
            metrics = evaluate(model, X_val, y_val)

            row = {
                "window": window_name,
                "train_end": window["train_end"],
                "val_start": window["val_start"],
                "val_end": window["val_end"],
                "specification": spec_name,
                "n_features": X_train.shape[1],
                "n_train": len(train),
                "n_train_positive": int(y_train.sum()),
                "n_val": len(val),
                "n_val_positive": int(y_val.sum()),
                **metrics,
            }
            rows.append(row)
            print(
                f"    {spec_name}: AUC={metrics['auc']:.3f}, "
                f"AP={metrics['avg_precision']:.3f}"
            )

    results_df = pd.DataFrame(rows)
    out_path = TABLES / "xgboost_rolling_validation.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved {out_path}")

    summary = summarize(results_df)
    summary_path = TABLES / "xgboost_rolling_validation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved {summary_path}")

    print("\nRolling validation summary:")
    display_cols = [
        "specification",
        "n_windows",
        "mean_auc",
        "std_auc",
        "mean_avg_precision",
        "std_avg_precision",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
