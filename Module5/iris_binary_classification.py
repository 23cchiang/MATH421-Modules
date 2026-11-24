import numpy as np
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Select only Iris setosa (class 0) and Iris versicolor (class 1)
X_setosa = X[y == 0]
X_versicolor = X[y == 1]

# Number of observations in each class
N1 = len(X_setosa)
N2 = len(X_versicolor)

# Calculate class mean vectors
mu1 = np.mean(X_setosa, axis=0)
mu2 = np.mean(X_versicolor, axis=0)

# Display results
print("CLASS MEANS")
print("------------------")
print("Setosa mean vector:")
print(mu1)

print("\nVersicolor mean vector:")
print(mu2)

print("\nNumber of observations:")
print(f"Setosa: {N1}")
print(f"Versicolor: {N2}")

print("\nFeature dimnensions:", mu1.shape)
print("Feature units: centimeters (cm)")

# Calculate within-class scatter matrix
S1 = np.zeros((4, 4))
S2 = np.zeros((4, 4))

for x in X_setosa:
    diff = x - mu1
    S1 += np.outer(diff, diff)

for x in X_versicolor:
    diff = x - mu2
    S2 += np.outer(diff, diff)

SW = S1 + S2

# Calculate between-class scatter matrix
mean_diff = mu1 - mu2
SB = np.outer(mean_diff, mean_diff)

# Display results
print("\nWITHIN-CLASS SCATTER MATRIX (SW)")
print(SW)

print("\nBETWEEN-CLASS SCATTER MATRIX (SB)")
print(SB)

print("\nMatrix dimensions:")
print("SW:", SW.shape)
print("SB:", SB.shape)

print("\nPhysical units: cm^2")
print("Rank of SB:", np.linalg.matrix_rank(SB))

# Calculate optimal Fisher discriminant weight vector
mean_diff = mu1 - mu2

w_star = np.linalg.solve(SW, mean_diff)

# Normalize the vector to unit length
w_star = w_star / np.linalg.norm(w_star)

# Display results
print("\nOPTIMAL FISHER DISCRIMINANT")
print("--------------------------------")
print("Unnormalized weight vector:")
print(np.linalg.solve(SW, mean_diff))

print("\nNormalized weight vector:")
print(w_star)

print("\nVector magnitude:")
print(np.linalg.norm(w_star))

# Calculate projected class means
mu1_projected = np.dot(w_star, mu1)
mu2_projected = np.dot(w_star, mu2)

# Calculate optimal neuron bias
b_star = -(mu1_projected + mu2_projected) / 2

# Reverse the Fisher vector to match the class label convention
w_neuron = -w_star
b_neuron = -b_star

# Display results
print("\nPROJECTED CLASS MEANS")
print("--------------------------------")
print("Setosa projected mean:", mu1_projected)
print("Versicolor projected mean:", mu2_projected)

print("\nOPTIMAL NEURON BIAS")
print("--------------------------------")
print("Bias:", b_star)
print("Decision boundary:", -b_star)
# Combine the two classes into one dataset
X_binary = np.vstack((X_setosa, X_versicolor))

# Assign true labels
# Setosa = -1, Versicolor = +1
true_labels = np.array([-1] * 50 + [1] * 50)

# Calculate neuron outputs
scores = X_binary @ w_neuron + b_neuron

# Classify observations
predicted_labels = np.where(scores < 0, -1, 1)

# Calculate accuracy
correct = np.sum(predicted_labels == true_labels)
accuracy = correct / len(true_labels) * 100

print("\nCLASSIFICATION RESULTS")
print("--------------------------------")
print("Correct classifications:", correct)
print("Incorrect classifications:", len(true_labels) - correct)
print("Accuracy:", accuracy, "%")

# Identify the feature with the largest absolute weight
feature_names = [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width"
]

largest_weight_index = np.argmax(np.abs(w_star))

print("\nMOST INFLUENTIAL FEATURE")
print("--------------------------------")
print("Feature:", feature_names[largest_weight_index])
print("Weight:", w_star[largest_weight_index])
