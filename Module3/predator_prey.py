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
