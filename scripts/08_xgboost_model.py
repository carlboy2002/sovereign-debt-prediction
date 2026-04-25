"""
08_xgboost_model.py — XGBoost prediction with nested model comparison and SHAP.

Runs four specifications:
    (1) Macro features only
    (2) Macro + Satellite
    (3) Macro + Text
    (4) All features

Reports AUC, precision, recall for each.
Produces SHAP feature importance plots.

Output: outputs/figures/, outputs/tables/
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, confusion_matrix,
)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
from config import DATA_PROCESSED, FIGURES, TABLES, XGB_PARAMS, SEED

try:
    from config import EVALUATE_TEST_SET
except ImportError:
    EVALUATE_TEST_SET = False

np.random.seed(SEED)


def load_data():
    """Load feature panel and split into train/val/test."""
    df = pd.read_csv(DATA_PROCESSED / "panel_features.csv")

    with open(DATA_PROCESSED / "feature_blocks.json") as f:
        blocks = json.load(f)

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)} (LOCKED)")
    return train, val, test, blocks


def prepare_xy(df, features):
    """Extract X, y arrays with imputation."""
    existing = [f for f in features if f in df.columns]
    existing = [f for f in existing if df[f].notna().any()]
    X = df[existing].copy()

    # Impute missing with median
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=existing, index=X.index)

    y = df["crisis_target"].values
    return X, y, imputer


def train_xgb(X_train, y_train, X_val, y_val):
    """Train XGBoost with early stopping on validation set."""
    # Adjust scale_pos_weight for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = n_neg / n_pos
    params["early_stopping_rounds"] = 30

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate(model, X, y, spec_name):
    """Compute evaluation metrics."""
    y_prob = model.predict_proba(X)[:, 1]

    results = {"specification": spec_name}
    results["auc"] = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else np.nan
    results["avg_precision"] = average_precision_score(y, y_prob)
    results["n_obs"] = len(y)
    results["n_positive"] = int(y.sum())

    # Optimal threshold (Youden's J)
    if len(np.unique(y)) > 1:
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        results["best_threshold"] = thresholds[best_idx]
        y_pred = (y_prob >= thresholds[best_idx]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        results["precision"] = tp / max(tp + fp, 1)
        results["recall"] = tp / max(tp + fn, 1)
        results["f1"] = 2 * results["precision"] * results["recall"] / max(
            results["precision"] + results["recall"], 1e-8
        )

    return results


def plot_roc_comparison(models_results, X_val_dict, y_val):
    """Plot ROC curves for all specifications on same axes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1976d2", "#388e3c", "#f57c00", "#d32f2f"]

    for (spec_name, model), color in zip(models_results.items(), colors):
        X = X_val_dict[spec_name]
        y_prob = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc = roc_auc_score(y_val, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{spec_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Nested Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_comparison.png")
    plt.close()
    print("  Saved roc_comparison.png")


def plot_shap(model, X, spec_name):
    """SHAP summary plot for a given model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, plot_type="bar", max_display=20, show=False, rng=rng
    )
    plt.title(f"SHAP Feature Importance - {spec_name}")
    plt.tight_layout()
    plt.savefig(FIGURES / f"shap_{spec_name.replace(' + ', '_')}.png", bbox_inches="tight")
    plt.close()

    # Also save beeswarm
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, max_display=15, show=False, rng=np.random.default_rng(SEED)
    )
    plt.title(f"SHAP Summary - {spec_name}")
    plt.tight_layout()
    plt.savefig(FIGURES / f"shap_beeswarm_{spec_name.replace(' + ', '_')}.png", bbox_inches="tight")
    plt.close()

    print(f"  Saved SHAP plots for {spec_name}")


def main():
    print("Training XGBoost models...")

    train, val, test, blocks = load_data()

    # Define four nested specifications
    specs = {
        "Macro Only": blocks["macro"],
        "Macro + Satellite": blocks["macro"] + blocks["satellite"],
        "Macro + Text": blocks["macro"] + blocks["text"],
        "All Features": blocks["all"],
    }

    all_results = []
    models = {}
    imputers = {}
    feature_columns = {}
    X_val_dict = {}

    for spec_name, features in specs.items():
        print(f"\n  === {spec_name} ({len(features)} candidate features) ===")

        # Prepare data
        X_train, y_train, imp = prepare_xy(train, features)
        existing = list(X_train.columns)
        if len(existing) != len(features):
            print(f"    Usable features after dropping all-missing train columns: {len(existing)}")
        X_v = val[existing].copy()
        X_v = pd.DataFrame(
            imp.transform(X_v), columns=existing, index=X_v.index
        )
        y_v = val["crisis_target"].values

        # Train
        model = train_xgb(X_train, y_train, X_v, y_v)

        # Evaluate on validation set
        results = evaluate(model, X_v, y_v, spec_name)
        all_results.append(results)
        models[spec_name] = model
        imputers[spec_name] = imp
        feature_columns[spec_name] = existing
        X_val_dict[spec_name] = X_v

        print(f"    AUC: {results.get('auc', 'N/A'):.3f}")
        print(f"    Avg Precision: {results.get('avg_precision', 'N/A'):.3f}")
        print(f"    Precision: {results.get('precision', 'N/A'):.3f}")
        print(f"    Recall: {results.get('recall', 'N/A'):.3f}")

    # Save results table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLES / "xgboost_nested_comparison.csv", index=False)
    print(f"\n  Saved xgboost_nested_comparison.csv")

    # Plot ROC comparison
    plot_roc_comparison(models, X_val_dict, val["crisis_target"].values)

    # SHAP for best model (All Features)
    best_name = "All Features"
    plot_shap(models[best_name], X_val_dict[best_name], best_name)

    if EVALUATE_TEST_SET:
        print("\n  === FINAL TEST SET EVALUATION ===")
        test_results_all = []
        for spec_name, model in models.items():
            imp = imputers[spec_name]
            existing = feature_columns[spec_name]
            X_test = test[existing].copy()
            X_test = pd.DataFrame(
                imp.transform(X_test), columns=existing, index=X_test.index
            )
            y_test = test["crisis_target"].values
            test_results = evaluate(model, X_test, y_test, f"{spec_name} (TEST)")
            test_results_all.append(test_results)
            print(
                f"  {spec_name}: AUC={test_results['auc']:.3f}, "
                f"AP={test_results['avg_precision']:.3f}"
            )
        pd.DataFrame(test_results_all).to_csv(
            TABLES / "xgboost_test_results.csv", index=False
        )
        print("  Saved xgboost_test_results.csv")
    else:
        print("\nDone. Test set evaluation is locked. Set EVALUATE_TEST_SET=True for the final run.")


if __name__ == "__main__":
    main()
