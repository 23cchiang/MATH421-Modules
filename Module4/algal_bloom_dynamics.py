"""
MATH-421 Module 4: Environmental Impacts of Industrial Agriculture

Generates numerical solutions and plots for eta < 0 and eta > 0.
Requirements: numpy, scipy, matplotlib.
"""

"""
Algal Bloom Dynamics - MATH 421 Module 4

Model:
    da/dt = eta*a - gamma*a**2
    dC/dt = q_C - delta*a - epsilon*C

Creates time-series and phase-plane plots for two positive eta values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# ------------------------------------------------------------
# Model parameters
# ------------------------------------------------------------
GAMMA = 0.01
Q_C = 0.8
DELTA = 0.001
EPSILON = 0.2

# Initial conditions
A0 = 0.5
C0 = 4.1

# Simulation time (days)
T_START = 0.0
T_END = 60.0
NUM_TIME_POINTS = 1200
TIME_POINTS = np.linspace(T_START, T_END, NUM_TIME_POINTS)

# Output folder
OUTPUT_DIR = (
    Path(__file__).resolve().parent / "algal_bloom_outputs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def model(t, state, eta):
    """Calculate the derivatives da/dt and dC/dt."""
    a, c = state

    da_dt = eta * a - GAMMA * a**2
    dC_dt = Q_C - DELTA * a - EPSILON * c

    return [da_dt, dC_dt]


def get_equilibria(eta):
    """Calculate the algae-free and nonzero algebraic equilibria."""
    algae_free = (0.0, Q_C / EPSILON)

    if GAMMA == 0:
        nonzero = None
    else:
        a_star = eta / GAMMA
        c_star = (Q_C - DELTA * a_star) / EPSILON
        nonzero = (a_star, c_star)

    return algae_free, nonzero


def solve_case(label, eta):
    """Solve the system and save its plots."""
    print(f"\n--- {label}: eta = {eta} ---")

    # Solve the differential equation system
    solution = solve_ivp(
        fun=lambda t, state: model(t, state, eta),
        t_span=(T_START, T_END),
        y0=[A0, C0],
        t_eval=TIME_POINTS,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    # Check the solver
    if not solution.success:
        print(f"Solver failed: {solution.message}")
        return

    # Extract solution arrays BEFORE plotting
    t = solution.t
    a = solution.y[0]
    c = solution.y[1]

    # Calculate equilibria
    algae_free, nonzero = get_equilibria(eta)

    print(
        f"Algae-free equilibrium: "
        f"a*={algae_free[0]:.6f}, C*={algae_free[1]:.6f}"
    )

    if nonzero is not None:
        print(
            f"Nonzero algebraic equilibrium: "
            f"a*={nonzero[0]:.6f}, C*={nonzero[1]:.6f}"
        )

        if nonzero[0] > 0 and nonzero[1] > 0:
            print("Nonzero equilibrium has positive coordinates.")
        else:
            print(
                "Nonzero equilibrium is not biologically feasible "
                "because at least one coordinate is nonpositive."
            )

    print(
        f"Final state at t={t[-1]:.2f}: "
        f"a={a[-1]:.6f}, C={c[-1]:.6f}"
    )

    # --------------------------------------------------------
    # Time-series plot
    # --------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    line1, = ax1.plot(
        t, a, label="Algae density a(t)"
    )

    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Algae density a (µg/L)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    line2, = ax2.plot(
        t, c, "--", label="Dissolved oxygen C(t)"
    )

    ax2.set_ylabel("Dissolved oxygen C (mg/L)")

    # Both lines in one legend
    ax1.legend(
        [line1, line2],
        [line1.get_label(), line2.get_label()],
        loc="best",
    )

    ax1.set_title(
        f"Algal Bloom Dynamics: {label} (eta={eta})"
    )

    fig.tight_layout()

    time_series_path = OUTPUT_DIR / f"{label}_time_series.png"
    fig.savefig(time_series_path, dpi=300)
    plt.close(fig)

    print(f"Time-series plot saved to: {time_series_path}")

    # --------------------------------------------------------
    # Phase-plane plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(a, c, label="Numerical trajectory")

    ax.scatter(
        [A0], [C0],
        marker="o",
        label="Initial condition",
    )

    ax.scatter(
        [algae_free[0]], [algae_free[1]],
        marker="x",
        s=80,
        label="Algae-free equilibrium",
    )

    if nonzero is not None:
        ax.scatter(
            [nonzero[0]], [nonzero[1]],
            marker="*",
            s=120,
            label="Nonzero algebraic equilibrium",
        )

    ax.set_xlabel("Algae density a (µg/L)")
    ax.set_ylabel("Dissolved oxygen C (mg/L)")
    ax.set_title(f"Phase Plane: {label} (eta={eta})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    phase_path = OUTPUT_DIR / f"{label}_phase_plot.png"
    fig.savefig(phase_path, dpi=300)
    plt.close(fig)

    print(f"Phase-plane plot saved to: {phase_path}")

def main():
    print("Algal bloom model")
    print(
        f"Parameters: gamma={GAMMA}, q_C={Q_C}, "
        f"delta={DELTA}, epsilon={EPSILON}"
    )
    print(f"Initial conditions: a0={A0}, C0={C0}")
    print(f"Plots will be saved in: {OUTPUT_DIR}")

    # Case 1: Negative growth parameter
    solve_case("eta_negative", eta=-0.08)

    # Case 2: Positive growth parameter
    solve_case("eta_positive", eta=0.08)


if __name__ == "__main__":
    main()
