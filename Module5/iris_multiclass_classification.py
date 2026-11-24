
import numpy as np
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Separate the three species
X_setosa = X[y == 0]
X_versicolor = X[y == 1]
X_virginica = X[y == 2]

# Store the classes for easier iteration
classes = [X_setosa, X_versicolor, X_virginica]

class_names = ["Setosa", "Versicolor", "Virginica"]

# Calculate class means
mu1 = np.mean(X_setosa, axis=0)
mu2 = np.mean(X_versicolor, axis=0)
mu3 = np.mean(X_virginica, axis=0)

# Calculate overall mean
mu = np.mean(X, axis=0)

# Display results
print("CLASS MEANS")
print("--------------------------------")

for name, mean in zip(class_names, [mu1, mu2, mu3]):
    print(f"{name}: {mean}")

print("\nOVERALL MEAN")
print("--------------------------------")
print(mu)

print("\nDATASET DIMENSIONS")
print("--------------------------------")
print("Full dataset:", X.shape)
print("Setosa:", X_setosa.shape)
print("Versicolor:", X_versicolor.shape)
print("Virginica:", X_virginica.shape)

# Calculate within-class scatter matrix
SW = np.zeros((4, 4))

for class_data, class_mean in zip(classes, [mu1, mu2, mu3]):
    for x in class_data:
        difference = x - class_mean
        SW += np.outer(difference, difference)

# Calculate between-class scatter matrix
SB = np.zeros((4, 4))

for class_data, class_mean in zip(classes, [mu1, mu2, mu3]):
    difference = class_mean - mu
    SB += len(class_data) * np.outer(difference, difference)

# Calculate pooled covariance matrix
Sigma = SW / 147

# Display results
print("\nWITHIN-CLASS SCATTER MATRIX (SW)")
print("--------------------------------")
print(SW)

print("\nBETWEEN-CLASS SCATTER MATRIX (SB)")
print("--------------------------------")
print(SB)

print("\nPOOLED COVARIANCE MATRIX (Sigma)")
print("--------------------------------")
print(Sigma)

print("\nMATRIX DIMENSIONS")
print("--------------------------------")
print("SW:", SW.shape)
print("SB:", SB.shape)
print("Sigma:", Sigma.shape)

print("\nRANK OF BETWEEN-CLASS SCATTER")
print("--------------------------------")
print(np.linalg.matrix_rank(SB))

# Calculate inverse pooled covariance using a linear solver
# First, select one observation to demonstrate
x_example = X[0]

print("\nMAHALANOBIS VS EUCLIDEAN DISTANCE")
print("--------------------------------")

for name, class_mean in zip(class_names, [mu1, mu2, mu3]):
    difference = x_example - class_mean

    # Squared Euclidean distance
    euclidean = difference @ difference

    # Squared Mahalanobis distance
    mahalanobis = difference @ np.linalg.solve(Sigma, difference)

    print(f"\n{name}:")
    print("Squared Euclidean distance:", euclidean)
    print("Squared Mahalanobis distance:", mahalanobis)

# Calculate weights and biases for each neuron

class_means = [mu1, mu2, mu3]

weights = []
biases = []

for class_mean in class_means:

    # Calculate neuron weight vector
    w = np.linalg.solve(Sigma, class_mean)

    # Calculate neuron bias
    b = -0.5 * (class_mean @ w)

    weights.append(w)
    biases.append(b)

# Display neuron parameters
print("\nNEURAL NETWORK WEIGHTS AND BIASES")
print("--------------------------------")

for i, name in enumerate(class_names):
    print(f"\nNeuron {i + 1}: {name}")
    print("Weights:", weights[i])
    print("Bias:", biases[i])

# Calculate scores for all 150 observations
# Each column represents one neuron's output
scores = X @ np.array(weights).T + np.array(biases)

# Predict the class with the highest score
predictions = np.argmax(scores, axis=1)

# Create the 3x3 confusion matrix
confusion_matrix = np.zeros((3, 3), dtype=int)

for actual, predicted in zip(y, predictions):
    confusion_matrix[actual, predicted] += 1

# Calculate classification accuracy
correct = np.sum(predictions == y)
incorrect = len(y) - correct
accuracy = correct / len(y) * 100

# Display confusion matrix
print("\nCONFUSION MATRIX")
print("--------------------------------")
print("Rows = Actual Class")
print("Columns = Predicted Class")
print("             Setosa  Versicolor  Virginica")

for i, name in enumerate(class_names):
    print(f"{name:12}", confusion_matrix[i])

# Display accuracy
print("\nCLASSIFICATION RESULTS")
print("--------------------------------")
print("Total observations:", len(y))
print("Correct classifications:", correct)
print("Misclassifications:", incorrect)
print("Accuracy:", accuracy, "%")

# Calculate individual sample covariance matrices
covariances = []

for name, class_data in zip(class_names, classes):
    covariance = np.cov(class_data, rowvar=False)
    covariances.append(covariance)

    print(f"\n{name.upper()} COVARIANCE MATRIX")
    print("--------------------------------")
    print(covariance)

# Compare the diagonal entries (feature variances)
feature_names = [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width"
]

print("\nVARIANCE COMPARISON")
print("--------------------------------")

for i, feature in enumerate(feature_names):
    print(f"\n{feature}:")
    for name, covariance in zip(class_names, covariances):
        print(f"{name}: {covariance[i, i]:.6f}")
# TASK 5: CLASSIFICATION USING PETAL FEATURES ONLY

# Keep only petal length and petal width
X_petals = X[:, 2:4]

# Separate the petal measurements by species
petal_classes = [
    X_petals[y == 0],
    X_petals[y == 1],
    X_petals[y == 2]
]

# Calculate class means and overall mean
petal_means = [
    np.mean(class_data, axis=0)
    for class_data in petal_classes
]

petal_overall_mean = np.mean(X_petals, axis=0)

# Calculate within-class scatter matrix
SW_petals = np.zeros((2, 2))

for class_data, class_mean in zip(petal_classes, petal_means):
    for observation in class_data:
        difference = observation - class_mean
        SW_petals += np.outer(difference, difference)

# Calculate pooled covariance matrix
Sigma_petals = SW_petals / 147

# Calculate neuron weights and biases
petal_weights = []
petal_biases = []

for class_mean in petal_means:
    w = np.linalg.solve(Sigma_petals, class_mean)
    b = -0.5 * (class_mean @ w)

    petal_weights.append(w)
    petal_biases.append(b)

# Calculate classification scores
petal_scores = (
    X_petals @ np.array(petal_weights).T
    + np.array(petal_biases)
)

# Predict the class with the highest score
petal_predictions = np.argmax(petal_scores, axis=1)

# Build confusion matrix
petal_confusion_matrix = np.zeros((3, 3), dtype=int)

for actual, predicted in zip(y, petal_predictions):
    petal_confusion_matrix[actual, predicted] += 1

# Calculate accuracy
petal_correct = np.sum(petal_predictions == y)
petal_incorrect = len(y) - petal_correct
petal_accuracy = petal_correct / len(y) * 100

# Display results
print("\nPETAL-ONLY CLASSIFICATION")
print("--------------------------------")
print("Features: Petal length and petal width")
print("\nPooled covariance matrix:")
print(Sigma_petals)

print("\nConfusion matrix:")
print("Rows = Actual Class")
print("Columns = Predicted Class")
for name, row in zip(class_names, petal_confusion_matrix):
    print(f"{name:<12} {row}")

print("\nClassification results:")
print(f"Total observations: {len(y)}")
print(f"Correct classifications: {petal_correct}")
print(f"Misclassifications: {petal_incorrect}")
print(f"Accuracy: {petal_accuracy:.2f}%")