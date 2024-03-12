import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_excel("training_mathbert 2.xlsx")

# Select two feature vectors for distance calculation
feature_1 = 'embed_0'
feature_2 = 'embed_1'

# Extract the values of the selected features
X = dataset[[feature_1, feature_2]].values

# Divide the dataset into train and test sets
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Calculate Minkowski distance for different values of r
r_values = range(1, 11)
distances = []

for r in r_values:
    distance_r = np.linalg.norm(X_train[:, 0] - X_train[:, 1], ord=r)
    distances.append(distance_r)

# Plot the distance versus r
plt.figure(figsize=(10, 6))
plt.plot(r_values, distances, marker='o', linestyle='-')
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.ylabel('Distance')
plt.grid(True)
plt.show()
