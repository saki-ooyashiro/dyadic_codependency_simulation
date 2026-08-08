import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def simulate_dyads(
    n=3000,
    seed=42,
    beta_anxiety=0.55,
    beta_control=0.70,
    noise_sd=0.90,
):
    """
    Three-layer directed-dyad simulation.

    Distal layer:
        differentiation_X, differentiation_Y

    Proximal layer:
        D_X, Aut_X, D_Y, Aut_Y

    Attribution layer:
        own_anxiety_coping_X
        partner_control_Y_to_X

    The simulation is illustrative, not an empirical estimate.
    """

    rng = np.random.default_rng(seed)

    differentiation_X = rng.normal(0, 1, n)
    differentiation_Y = rng.normal(0, 1, n)

    relational_context = rng.normal(0, 0.65, n)

    D_X = (
        0.45 * relational_context
        - 0.35 * differentiation_X
        + rng.normal(0, 0.85, n)
    )

    D_Y = (
        0.45 * relational_context
        - 0.35 * differentiation_Y
        + rng.normal(0, 0.85, n)
    )

    own_anxiety_coping_X = (
        0.75 * (-differentiation_X)
        + 0.45 * D_X
        + rng.normal(0, 0.75, n)
    )

    own_anxiety_coping_Y = (
        0.75 * (-differentiation_Y)
        + 0.45 * D_Y
        + rng.normal(0, 0.75, n)
    )

    control_Y_to_X = (
        0.65 * (-differentiation_Y)
        + 0.30 * D_Y
        + rng.normal(0, 0.85, n)
    )

    control_X_to_Y = (
        0.65 * (-differentiation_X)
        + 0.30 * D_X
        + rng.normal(0, 0.85, n)
    )

    Aut_X = (
        1.25 * differentiation_X
        - beta_anxiety * own_anxiety_coping_X
        - beta_control * control_Y_to_X
        + rng.normal(0, noise_sd, n)
    )

    Aut_Y = (
        1.25 * differentiation_Y
        - beta_anxiety * own_anxiety_coping_Y
        - beta_control * control_X_to_Y
        + rng.normal(0, noise_sd, n)
    )

    own_effect_X = np.abs(beta_anxiety * own_anxiety_coping_X)
    partner_effect_X = np.abs(beta_control * control_Y_to_X)

    low_autonomy_X = Aut_X < 0

    X_pattern = np.full(n, "Other", dtype=object)
    X_pattern[
        low_autonomy_X & (own_effect_X > partner_effect_X)
    ] = "X: codependent edge"
    X_pattern[
        low_autonomy_X & (partner_effect_X > own_effect_X)
    ] = "X: subjugated edge"

    df = pd.DataFrame(
        {
            "differentiation_X": differentiation_X,
            "differentiation_Y": differentiation_Y,
            "D_X": D_X,
            "Aut_X": Aut_X,
            "D_Y": D_Y,
            "Aut_Y": Aut_Y,
            "own_anxiety_coping_X": own_anxiety_coping_X,
            "own_anxiety_coping_Y": own_anxiety_coping_Y,
            "control_Y_to_X": control_Y_to_X,
            "control_X_to_Y": control_X_to_Y,
            "X_pattern": X_pattern,
        }
    )

    return df


def linear_regression(y, X):
    """Simple OLS without external statistics packages."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_design = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]

    residuals = y - X_design @ beta
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot

    return beta, r2


def make_graphs(df, output_dir="simulation_output"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.scatter(
        df["D_X"],
        df["Aut_X"],
        s=10,
        alpha=0.25,
    )
    plt.axhline(0)
    plt.axvline(0)
    plt.xlabel("Dependence D_X")
    plt.ylabel("Autonomy / boundary retention Aut_X")
    plt.title("Simulated proximal state space for X")
    plt.tight_layout()
    plt.savefig(output_dir / "01_proximal_state_space_X.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter(
        df["own_anxiety_coping_X"],
        -df["Aut_X"],
        s=10,
        alpha=0.22,
        label="Own anxiety-regulation coping",
    )
    plt.scatter(
        df["control_Y_to_X"],
        -df["Aut_X"],
        s=10,
        alpha=0.22,
        label="Partner control Y→X",
    )
    plt.xlabel("Explanatory variable")
    plt.ylabel("Autonomy loss (-Aut_X)")
    plt.title("Two competing explanations of X's autonomy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "02_competing_explanations.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter(
        df["differentiation_X"],
        df["Aut_X"],
        s=10,
        alpha=0.25,
    )
    plt.axhline(0)
    plt.axvline(0)
    plt.xlabel("Distal differentiation_X")
    plt.ylabel("Proximal autonomy Aut_X")
    plt.title("Distal differentiation and proximal autonomy")
    plt.tight_layout()
    plt.savefig(output_dir / "03_distal_to_proximal.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter(
        df["control_Y_to_X"],
        df["control_X_to_Y"],
        s=10,
        alpha=0.25,
    )
    plt.axhline(0)
    plt.axvline(0)
    plt.xlabel("Control Y→X")
    plt.ylabel("Control X→Y")
    plt.title("Directed control asymmetry")
    plt.tight_layout()
    plt.savefig(output_dir / "04_directed_control_asymmetry.png", dpi=180)
    plt.close()


def main():
    df = simulate_dyads()

    corr = df[["D_X", "Aut_X", "D_Y", "Aut_Y"]].corr()

    y = -df["Aut_X"].to_numpy()
    X = df[["own_anxiety_coping_X", "control_Y_to_X"]].to_numpy()

    beta, r2 = linear_regression(y, X)

    print("=== Simulation summary ===")
    print(f"Dyads: {len(df)}")
    print(df["X_pattern"].value_counts())
    print()

    print("=== Correlations among proximal states ===")
    print(corr.round(2))
    print()

    print("=== Linear model: X autonomy loss ===")
    print(f"Intercept: {beta[0]:.3f}")
    print(f"Own coping: {beta[1]:.3f}")
    print(f"Partner control Y→X: {beta[2]:.3f}")
    print(f"R²: {r2:.3f}")
    print()

    make_graphs(df)

    df.to_csv("simulation_output/simulated_dyads.csv", index=False)

    print("=== Output ===")
    print("simulation_output/01_proximal_state_space_X.png")
    print("simulation_output/02_competing_explanations.png")
    print("simulation_output/03_distal_to_proximal.png")
    print("simulation_output/04_directed_control_asymmetry.png")
    print("simulation_output/simulated_dyads.csv")


if __name__ == "__main__":
    main()