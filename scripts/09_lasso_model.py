"""
09_lasso_model.py — LASSO logistic regression baseline + feature selection.

LASSO serves two purposes:
    1. Interpretable baseline to compare against XGBoost
    2. Feature selection — which signals survive L1 regularization?

Output: outputs/tables/, outputs/figures/
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from config import DATA_PROCESSED, FIGURES, TABLES, LASSO_ALPHAS, SEED


def main():
    print("Training LASSO logistic regression...")

    # Load data
    df = pd.read_csv(DATA_PROCESSED / "panel_features.csv")
    with open(DATA_PROCESSED / "feature_blocks.json") as f:
        blocks = json.load(f)

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    features = blocks["all"]
    existing = [f for f in features if f in df.columns]

    # Prepare data
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = train[existing].copy()
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=existing)
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=existing)
    y_train = train["crisis_target"].values

    X_val = val[existing].copy()
    X_val = pd.DataFrame(imputer.transform(X_val), columns=existing)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=existing)
    y_val = val["crisis_target"].values

    # ── LASSO with cross-validation ─────────────────────────────────────────
    print("  Running LASSO CV...")
    lasso_cv = LogisticRegressionCV(
        Cs=[1/a for a in LASSO_ALPHAS],  # sklearn uses C = 1/alpha
        penalty="l1",
        solver="saga",
        cv=5,
        scoring="roc_auc",
        class_weight="balanced",
        random_state=SEED,
        max_iter=5000,
    )
    lasso_cv.fit(X_train_scaled, y_train)

    best_C = lasso_cv.C_[0]
    print(f"  Best C (1/alpha): {best_C:.4f}")

    # Evaluate
    y_prob = lasso_cv.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else np.nan
    ap = average_precision_score(y_val, y_prob)
    print(f"  Validation AUC: {auc:.3f}")
    print(f"  Avg Precision: {ap:.3f}")

    # ── Feature selection: which coefficients survived? ─────────────────────
    coefs = pd.DataFrame({
        "feature": existing,
        "coefficient": lasso_cv.coef_[0],
        "abs_coef": np.abs(lasso_cv.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    n_selected = (coefs["abs_coef"] > 0).sum()
    n_total = len(coefs)
    print(f"\n  LASSO selected {n_selected}/{n_total} features")

    # Classify features by block
    def get_block(feat):
        if any(feat.startswith(m) for m in ["gdp", "inflation", "debt", "current", "reserves", "external", "fiscal", "exchange"]):
            return "macro"
        elif any(feat.startswith(s) for s in ["nightlight", "divergence"]):
            return "satellite"
        else:
            return "text"

    coefs["block"] = coefs["feature"].apply(get_block)

    # Save
    coefs.to_csv(TABLES / "lasso_coefficients.csv", index=False)
    print(f"  Saved lasso_coefficients.csv")

    # Print top features
    print(f"\n  Top 15 features (by |coefficient|):")
    for _, row in coefs.head(15).iterrows():
        direction = "+" if row["coefficient"] > 0 else "−"
        print(f"    [{row['block']:9s}] {direction} {row['feature']:40s} {row['abs_coef']:.4f}")

    # ── Plot: LASSO coefficient path ────────────────────────────────────────
    print("\n  Computing regularization path...")
    coef_paths = []
    for alpha in LASSO_ALPHAS:
        lr = LogisticRegression(
            C=1/alpha, penalty="l1", solver="saga",
            class_weight="balanced", random_state=SEED, max_iter=5000,
        )
        lr.fit(X_train_scaled, y_train)
        coef_paths.append(lr.coef_[0])

    coef_paths = np.array(coef_paths)  # (n_alphas, n_features)

    # Plot top 10 features
    top_idx = coefs.head(10).index.tolist()
    top_features = coefs.head(10)["feature"].tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (idx, name) in enumerate(zip(top_idx, top_features)):
        ax.plot(LASSO_ALPHAS, coef_paths[:, idx], "o-", label=name, markersize=3)

    ax.set_xscale("log")
    ax.set_xlabel("Alpha (regularization strength)")
    ax.set_ylabel("Coefficient value")
    ax.set_title("LASSO Regularization Path — Top 10 Features")
    ax.legend(fontsize=8, loc="best")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "lasso_regularization_path.png")
    plt.close()
    print("  Saved lasso_regularization_path.png")

    # ── Plot: surviving features by block ───────────────────────────────────
    surviving = coefs[coefs["abs_coef"] > 0]
    block_counts = surviving["block"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    block_counts.plot(kind="bar", color=["#1976d2", "#388e3c", "#f57c00"], ax=ax)
    ax.set_ylabel("Number of surviving features")
    ax.set_title("Features Surviving LASSO by Data Source")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES / "lasso_surviving_by_block.png")
    plt.close()
    print("  Saved lasso_surviving_by_block.png")


if __name__ == "__main__":
    main()
