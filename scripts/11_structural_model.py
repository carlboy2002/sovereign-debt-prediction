"""
11_structural_model.py — Two-period government signaling model.

Models the government's decision to misreport economic statistics when
facing debt distress. Derives testable predictions and compares them
with empirical patterns observed in the data.

The model:
    Period 1: Government observes true state θ ∈ {good, bad}.
              Chooses reported GDP = θ_hat (can misreport upward).
              Misreporting has a cost c (reputation, future credibility).
    Period 2: Creditors observe θ_hat, set interest rate r(θ_hat).
              Government either services debt at rate r, or defaults.

Equilibrium prediction:
    - Governments in "bad" state misreport if cost of default > cost of misreporting
    - As θ worsens, divergence (θ_hat - θ) increases
    - This maps to our empirical divergence index

Output: outputs/figures/, outputs/tables/
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_PROCESSED, FIGURES, TABLES, SEED

np.random.seed(SEED)


# ── MODEL PARAMETERS ────────────────────────────────────────────────────────

def run_signaling_model(
    n_countries=500,
    n_periods=20,
    default_cost=0.15,          # Fixed cost of default (% of GDP)
    misreport_cost_base=0.02,  # Base cost of misreporting per unit
    debt_level_mean=0.7,       # Mean debt/GDP
    debt_level_std=0.25,        # Std dev of debt/GDP
):
    """
    Simulate the government signaling game.

    Each country-period:
        1. Draw true growth θ ~ N(2, 3)
        2. Draw debt level d ~ N(0.5, 0.2)
        3. Government chooses report θ_hat to maximize:
           U = -r(θ_hat) * d - c * (θ_hat - θ)^2   if service
           U = -default_cost                          if default
        4. Creditors set r(θ_hat) = r_base - β * θ_hat
    """
    records = []

    for country in range(n_countries):
        debt = np.clip(np.random.normal(debt_level_mean, debt_level_std), 0.1, 1.5)

        for t in range(n_periods):
            # True economic state
            theta_true = np.random.normal(2.0, 3.0)

            # Debt evolves
            debt = debt + np.random.normal(0.02, 0.05)
            debt = np.clip(debt, 0.1, 2.0)

            # Creditor pricing function: r = 0.08 - 0.005 * θ_hat
            r_base = 0.15
            r_sensitivity = 0.01

            # Government's optimization: choose θ_hat
            # For each candidate report, compute utility of servicing vs default
            best_report = theta_true
            best_utility = -np.inf

            for theta_hat in np.linspace(theta_true, theta_true + 15, 100):
                # Interest rate given report
                r = max(r_base - r_sensitivity * theta_hat, 0.01)

                # Cost of servicing
                service_cost = r * debt

                # Cost of misreporting
                misreport_cost = misreport_cost_base * (theta_hat - theta_true) ** 2

                # Utility of servicing
                u_service = -service_cost - misreport_cost

                # Utility of default
                u_default = -default_cost

                # Government chooses to service if u_service > u_default
                if u_service > u_default and u_service > best_utility:
                    best_utility = u_service
                    best_report = theta_hat

            # Did the government default?
            r_actual = max(r_base - r_sensitivity * best_report, 0.01)
            u_service = -r_actual * debt - misreport_cost_base * (best_report - theta_true) ** 2
            u_default = -default_cost
            defaulted = u_service < u_default

            records.append({
                "country": country,
                "period": t,
                "theta_true": theta_true,
                "theta_reported": best_report,
                "divergence": best_report - theta_true,
                "debt_level": debt,
                "interest_rate": r_actual,
                "defaulted": int(defaulted),
            })

    return pd.DataFrame(records)


def validate_predictions(sim_df, empirical_df):
    """
    Compare structural model predictions with empirical data.

    Prediction 1: Higher debt → more misreporting (larger divergence)
    Prediction 2: Divergence increases before default
    Prediction 3: Countries that default have higher average divergence
    """
    results = {}

    # Prediction 1: Debt-divergence correlation
    corr = sim_df[["debt_level", "divergence"]].corr().iloc[0, 1]
    results["sim_debt_divergence_corr"] = corr
    print(f"  Prediction 1 (debt → divergence):")
    print(f"    Simulated correlation: {corr:.3f}")

    if "divergence_index" in empirical_df.columns and "debt_to_gdp" in empirical_df.columns:
        emp_corr = empirical_df[["debt_to_gdp", "divergence_index"]].dropna().corr().iloc[0, 1]
        results["emp_debt_divergence_corr"] = emp_corr
        print(f"    Empirical correlation:  {emp_corr:.3f}")

    # Prediction 2: Divergence before default
    defaulters = sim_df[sim_df["defaulted"] == 1]["country"].unique()
    if len(defaulters) > 0:
        default_div = sim_df[sim_df["country"].isin(defaulters)]["divergence"].mean()
        nondefault_div = sim_df[~sim_df["country"].isin(defaulters)]["divergence"].mean()
        results["sim_default_divergence"] = default_div
        results["sim_nondefault_divergence"] = nondefault_div
        print(f"\n  Prediction 3 (defaulters vs non-defaulters):")
        print(f"    Defaulters avg divergence:     {default_div:.3f}")
        print(f"    Non-defaulters avg divergence:  {nondefault_div:.3f}")

    return results


def plot_model_results(sim_df):
    """Visualize structural model predictions."""
    # Plot 1: Divergence vs debt level
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sample = sim_df.sample(min(2000, len(sim_df)), random_state=SEED)
    colors = ["#1976d2" if d == 0 else "#d32f2f" for d in sample["defaulted"]]
    ax.scatter(sample["debt_level"], sample["divergence"], alpha=0.3, s=10, c=colors)
    ax.set_xlabel("Debt / GDP")
    ax.set_ylabel("Divergence (Reported − True Growth)")
    ax.set_title("Model: Debt Level vs. Misreporting")

    # Plot 2: Divergence distribution by default status
    ax = axes[1]
    for label, color, name in [(0, "#1976d2", "No default"), (1, "#d32f2f", "Default")]:
        subset = sim_df[sim_df["defaulted"] == label]["divergence"]
        ax.hist(subset, bins=40, alpha=0.6, color=color, density=True, label=name)
    ax.set_xlabel("Divergence (Reported − True)")
    ax.set_ylabel("Density")
    ax.set_title("Model: Divergence Distribution by Default Status")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES / "structural_model_predictions.png")
    plt.close()
    print("  Saved structural_model_predictions.png")


def main():
    print("Running structural signaling model...")

    # Simulate
    sim_df = run_signaling_model()
    n_defaults = sim_df["defaulted"].sum()
    print(f"  Simulated {len(sim_df)} country-periods, {n_defaults} defaults")

    # Load empirical data for comparison
    try:
        emp_df = pd.read_csv(DATA_PROCESSED / "panel_features.csv")
        emp_df = emp_df[emp_df["split"].isin(["train", "val"])]
    except FileNotFoundError:
        emp_df = pd.DataFrame()

    # Validate predictions
    results = validate_predictions(sim_df, emp_df)

    # Save simulation results
    sim_df.to_csv(TABLES / "structural_model_simulation.csv", index=False)

    # Plot
    plot_model_results(sim_df)

    # Save validation results
    pd.DataFrame([results]).to_csv(TABLES / "structural_model_validation.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
