
import numpy as np
from sklearn.datasets import load_iris

# ==========================================
# MATH-421 - SHEET 2 - QUESTION 2(c)
# Three-Neuron Iris Classification
# ==========================================

# 1. Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

species = iris.target_names
features = iris.feature_names

# 2. Calculate the three class mean vectors
means = []

for k in range(3):
    Xk = X[y == k]
    means.append(np.mean(Xk, axis=0))

means = np.array(means)

print("CLASS MEAN VECTORS")
for k in range(3):
    print(f"\n{species[k]}:")
    print(means[k])

# 3. Calculate within-class scatter matrix
SW = np.zeros((4, 4))

for k in range(3):
    Xk = X[y == k]
    centered = Xk - means[k]

    SW += centered.T @ centered

print("\nWITHIN-CLASS SCATTER MATRIX:")
print(SW)

# 4. Calculate pooled covariance matrix
Sigma = SW / 147

print("\nPOOLED COVARIANCE MATRIX:")
print(Sigma)

# 5. Calculate neuron weight vectors
# Each row of W represents one neuron's weights
W = np.linalg.solve(Sigma, means.T).T

print("\nNEURON WEIGHT VECTORS:")
for k in range(3):
    print(f"\n{species[k]}:")
    print(W[k])

# 6. Calculate neuron biases
biases = np.zeros(3)

for k in range(3):
    biases[k] = -0.5 * means[k].T @ W[k]

print("\nNEURON BIASES:")
for k in range(3):
    print(f"{species[k]}: {biases[k]:.6f}")

# 7. Display complete neuron equations
print("\nNEURAL NETWORK EQUATIONS:")

for k in range(3):
    equation = " + ".join(
        f"({W[k, j]:.4f})x{j+1}"
        for j in range(4)
    )
    print(f"\nNeuron {k+1} ({species[k]}):")
    print(f"g{k+1}(x) = {equation} + ({biases[k]:.4f})")

print("\nTRAINING COMPLETE!")
