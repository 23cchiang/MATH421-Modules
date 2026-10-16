import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Original model parameters
eta = 7.5
sigma = 0.2
gamma = 0.01
delta = sigma * gamma /(20 * eta)

# Internal competition coefficients
c_x = 0.5 / 100000
c_y = 0.2 / 100000

# Original equilibrium
X_eq_original = sigma / delta
Y_eq_original = eta / gamma

# Initial conditions from the provided MATLAB example
X0 = 0.9 * X_eq_original
Y0 = 1.25 * Y_eq_original
U0 = [X0, Y0]

# Original Lotka-Volterra model
def original_model(t, U):
    X, Y = U

    dX = eta * X - gamma * X * Y
    dY = delta * X * Y -sigma * Y

    return [dX, dY]

# New Model with interal competition
def competition_model(t, U):
    X, Y = U

    dX = eta * X - gamma * X * Y - c_x * X**2
    dY = delta * X * Y -sigma * Y - c_y * Y**2

    return [dX, dY]

# Solve both models
t_eval = np.linspace(0, 20, 2001)

original = solve_ivp(
    original_model, (0, 20), U0, t_eval=t_eval,
    rtol=1e-9, atol=1e-10
)

competition = solve_ivp(
    competition_model, (0, 20), U0, t_eval=t_eval,
    rtol=1e-9, atol=1e-10
)
if not original.success or not competition.success:
    raise RuntimeError("A numerical solution failed.")

# Figure 1: Population vs time
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

axes[0].plot(original.t, original.y[0], label="Original")
axes[0].plot(competition.t, competition.y[0],
             label="Internal competition")
axes[0].set_ylabel("Prey population")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(original.t, original.y[1], label="Original")
axes[1].plot(competition.t, competition.y[1],
             label="Internal competition")
axes[1].set_xlabel("Time (years)")
axes[1].set_ylabel("Predator population")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("population_vs_time.png", dpi=200)
plt.show()

# Figure 2: Phase plane
plt.figure(figsize=(8,6))
plt.plot(original.y[0], original.y[1], label="Original")
plt.plot(competition.y[0], competition.y[1],
         label="Internal competition")
plt.xlabel("Prey population")
plt.ylabel("Predator population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("phase_plane.png", dpi=200)
plt.show()

# Print the original equilibrium and the new equilibrium
# Solve the two nonzero equilibrium equations:
# c_x X + gamma Y = eta
# delta X - c_y Y = sigma

A = np.array([
    [c_x, gamma],
    [delta, -c_y]
], dtype=float)

b = np.array([eta, sigma], dtype=float)

X_eq_new, Y_eq_new = np.linalg.solve(A, b)

print("Original equilibrium")
print(f"Prey: {X_eq_original:.4f}")
print(f"Predators: {Y_eq_original:.4f}")

print("\nInternal-competition equilibrium")
print(f"Prey: {X_eq_new:.4f}")
print(f"Predators: {Y_eq_new:.4f}")

print("\nFinal simulated populations at t = 20:")
print(f"Original Prey: {original.y[0, -1]:.4f}")
print(f"Original predators: {original.y[1, -1]:.4f}")
print(f"Competition prety: {competition.y[0, -1]:.4f}")
print(f"Competition predators: {competition.y[1, -1]:.4f}")
