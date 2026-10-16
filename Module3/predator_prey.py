import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate import solve_ivp

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
Y_eq_original = eta / delta

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

