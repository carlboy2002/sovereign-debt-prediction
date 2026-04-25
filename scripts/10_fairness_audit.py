"""
10_fairness_audit.py — Evaluate model fairness across regions and income groups.

Tests whether the early warning model systematically underperforms for
certain regions, which could cause harm if used for IMF surveillance.

Output: outputs/tables/, outputs/figures/
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, roc_curve
from sklearn.impute import SimpleImputer
import xgboost as xgb
from config import DATA_PROCESSED, FIGURES, TABLES, XGB_PARAMS, SEED, REGION_GROUPS


def metrics_at_threshold(y_true, y_prob, threshold):
    """Compute classification metrics at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fp / max(fp + tn, 1),
        "tpr": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def choose_threshold_for_target_fpr(y_true, y_prob, target_fpr):
    """Choose the threshold whose validation FPR is closest to target_fpr."""
    candidate_thresholds = np.r_[np.inf, np.sort(np.unique(y_prob))[::-1], -np.inf]
    metrics = [
        metrics_at_threshold(y_true, y_prob, threshold)
        for threshold in candidate_thresholds
    ]
    return min(
        metrics,
        key=lambda row: (abs(row["fpr"] - target_fpr), -row["tpr"], -row["threshold"]),
    )


def main():
    print("Running fairness audit...")

    # Load data
    df = pd.read_csv(DATA_PROCESSED / "panel_features.csv")
    with open(DATA_PROCESSED / "feature_blocks.json") as f:
        blocks = json.load(f)

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    features = blocks["all"]
    existing = [f for f in features if f in df.columns]
    existing = [f for f in existing if train[f].notna().any()]

    # Prepare train and validation matrices
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(train[existing]), columns=existing
    )
    y_train = train["crisis_target"].values
    X_val = pd.DataFrame(
        imputer.transform(val[existing]), columns=existing
    )
    y_val = val["crisis_target"].values

    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = n_neg / n_pos
    params["early_stopping_rounds"] = 30

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_prob = model.predict_proba(X_val)[:, 1]
    val_with_preds = val.copy()
    val_with_preds["y_prob"] = y_prob

    # ── Evaluate by region ──────────────────────────────────────────────────
    print("\n  Performance by region:")
    region_results = []

    if "region_code" in val.columns:
        group_col = "region_code"
    else:
        print("  WARNING: No region column found. Skipping regional audit.")
        return

    for region_name, region_code in REGION_GROUPS.items():
        mask = val_with_preds[group_col] == region_code
        subset = val_with_preds[mask]

        if len(subset) < 10 or subset["crisis_target"].nunique() < 2:
            print(f"    {region_name}: insufficient data ({len(subset)} obs)")
            continue

        y_true = subset["crisis_target"].values
        y_pred = subset["y_prob"].values
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        n_pos = y_true.sum()

        region_results.append({
            "region": region_name,
            "n_obs": len(subset),
            "n_crises": int(n_pos),
            "crisis_rate": n_pos / len(subset),
            "auc": auc,
            "avg_precision": ap,
        })
        print(f"    {region_name:35s}: AUC={auc:.3f}, AP={ap:.3f}, n={len(subset)}, crises={n_pos}")

    if not region_results:
        print("  No regional results to save.")
        return

    region_df = pd.DataFrame(region_results)
    region_df.to_csv(TABLES / "fairness_by_region.csv", index=False)
    print(f"\n  Saved fairness_by_region.csv")

    # ── Region-specific thresholds targeting the overall false-positive rate ─
    threshold_rows = []
    if len(np.unique(y_val)) > 1:
        fpr, tpr, thresholds = roc_curve(y_val, y_prob)
        best_idx = np.argmax(tpr - fpr)
        overall_threshold = thresholds[best_idx]
        overall_metrics = metrics_at_threshold(y_val, y_prob, overall_threshold)
        target_fpr = overall_metrics["fpr"]
        print(f"\n  Overall validation threshold: {overall_threshold:.4f}")
        print(f"  Target FPR for regional recalibration: {target_fpr:.3f}")

        for region_name, region_code in REGION_GROUPS.items():
            subset = val_with_preds[val_with_preds[group_col] == region_code]
            if len(subset) < 10 or subset["crisis_target"].nunique() < 2:
                continue

            y_true = subset["crisis_target"].values
            y_region_prob = subset["y_prob"].values
            global_threshold_metrics = metrics_at_threshold(
                y_true, y_region_prob, overall_threshold
            )
            regional_threshold_metrics = choose_threshold_for_target_fpr(
                y_true, y_region_prob, target_fpr
            )
            threshold_rows.append({
                "region": region_name,
                "n_obs": len(subset),
                "n_crises": int(y_true.sum()),
                "overall_threshold": overall_threshold,
                "region_threshold": regional_threshold_metrics["threshold"],
                "global_threshold_fpr": global_threshold_metrics["fpr"],
                "global_threshold_recall": global_threshold_metrics["recall"],
                "region_threshold_fpr": regional_threshold_metrics["fpr"],
                "region_threshold_recall": regional_threshold_metrics["recall"],
                "region_threshold_precision": regional_threshold_metrics["precision"],
            })

        if threshold_rows:
            pd.DataFrame(threshold_rows).to_csv(
                TABLES / "fairness_thresholds_by_region.csv", index=False
            )
            print("  Saved fairness_thresholds_by_region.csv")

    # ── Evaluate by income group ────────────────────────────────────────────
    if "income_level" in val.columns:
        print("\n  Performance by income group:")
        income_results = []
        for income in val_with_preds["income_level"].dropna().unique():
            mask = val_with_preds["income_level"] == income
            subset = val_with_preds[mask]
            if len(subset) < 10 or subset["crisis_target"].nunique() < 2:
                continue

            y_true = subset["crisis_target"].values
            y_pred = subset["y_prob"].values
            auc = roc_auc_score(y_true, y_pred)

            income_results.append({
                "income_group": income,
                "n_obs": len(subset),
                "auc": auc,
            })
            print(f"    {income:30s}: AUC={auc:.3f}, n={len(subset)}")

        if income_results:
            pd.DataFrame(income_results).to_csv(
                TABLES / "fairness_by_income.csv", index=False
            )

    # ── Plot: AUC by region ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    region_df_sorted = region_df.sort_values("auc", ascending=True)
    colors = ["#d32f2f" if a < 0.6 else "#f57c00" if a < 0.7 else "#388e3c"
              for a in region_df_sorted["auc"]]
    ax.barh(region_df_sorted["region"], region_df_sorted["auc"], color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.set_xlabel("AUC")
    ax.set_title("Model Performance by Region (Fairness Audit)")
    ax.set_xlim(0, 1)

    # Annotate
    overall_auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5
    ax.axvline(overall_auc, color="#1976d2", linestyle="-", alpha=0.7, label=f"Overall ({overall_auc:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fairness_auc_by_region.png")
    plt.close()
    print("  Saved fairness_auc_by_region.png")

    # ── Fairness gap ────────────────────────────────────────────────────────
    max_auc = region_df["auc"].max()
    min_auc = region_df["auc"].min()
    gap = max_auc - min_auc
    print(f"\n  Fairness gap (max - min AUC): {gap:.3f}")
    if gap > 0.15:
        print("  WARNING: Significant performance gap detected.")
        print("  Consider recalibration or region-specific thresholds.")


if __name__ == "__main__":
    main()
