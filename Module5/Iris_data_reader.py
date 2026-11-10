
import numpy as np
from sklearn.datasets import load_iris

### Used pip install scikit-learn numpy

# ==========================================
# PART 1 - QUESTION 3: BINARY IRIS CLASSIFIER
# ==========================================

# 1. Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Select only setosa (0) and versicolor (1)
X1 = X[y == 0]
X2 = X[y == 1]

# 2. Compute class mean vectors
mu1 = np.mean(X1, axis=0)
mu2 = np.mean(X2, axis=0)

print("Class 1 Mean (Setosa):")
print(mu1)

print("\nClass 2 Mean (Versicolor):")
print(mu2)

# 3. Compute within-class scatter matrix
S1 = (X1 - mu1).T @ (X1 - mu1)
S2 = (X2 - mu2).T @ (X2 - mu2)

SW = S1 + S2

print("\nWithin-Class Scatter Matrix:")
print(SW)

# 4. Compute optimal weight vector
# First calculate the direction specified in the assignment.
w_original = np.linalg.solve(SW, mu1 - mu2)

# Reverse the direction to match the assignment's
# threshold convention:
# Setosa = -1 when score < 0
# Versicolor = +1 when score >= 0

w = -w_original

# Normalize the weight vector to unit length
w = w / np.linalg.norm(w)

print("\nOptimal Weight Vector:")
print(w)

# 5. Compute projected class means
mu_tilde1 = w.T @ mu1
mu_tilde2 = w.T @ mu2

print("\nProjected Mean - Setosa:", mu_tilde1)
print("Projected Mean - Versicolor:", mu_tilde2)

# 6. Calculate optimal bias
b = -(mu_tilde1 + mu_tilde2) / 2

print("\nOptimal Bias:")
print(b)

# 7. Predict the classes of all 100 flowers
X_test = X[:100]
y_test = y[:100]

scores = X_test @ w + b

predictions = np.where(scores < 0, -1, 1)

# Convert actual labels to -1 and +1
actual = np.where(y_test == 0, -1, 1)

# 8. Calculate classification accuracy
correct = np.sum(predictions == actual)
total = len(actual)

accuracy = correct / total * 100

print("\nClassification Results:")
print("Correct Predictions:", correct)
print("Total Flowers:", total)
print("Incorrect Predictions:", total - correct)
print("Accuracy:", accuracy, "%")

# 9. Identify the feature with the largest absolute weight
features = iris.feature_names

largest_weight_index = np.argmax(np.abs(w))

print("\nFeature with Largest Absolute Weight:")
print(features[largest_weight_index])
print("Weight:", w[largest_weight_index])

# 10. Display all weights
print("\nAll Feature Weights:")
for feature, weight in zip(features, w):
    print(f"{feature}: {weight:.6f}")
