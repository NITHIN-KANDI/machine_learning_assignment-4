import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from an Excel file and return a DataFrame."""
    return pd.read_excel(file_path)

def calculate_distances(X_train):
    """Calculate Minkowski distance for different values of r."""
    r_values = range(1, 11)
    distances = []
    for r in r_values:
        distance_r = np.linalg.norm(X_train[:, 0] - X_train[:, 1], ord=r)
        distances.append(distance_r)
    return r_values, distances

def plot_distance_vs_r(r_values, distances):
    """Plot the distance versus r."""
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, distances, marker='o', linestyle='-')
    plt.title('Minkowski Distance vs. r')
    plt.xlabel('r')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.show()

def main():
    # Load the dataset
    dataset = load_data("training_mathbert 2.xlsx")

    # Select two feature vectors for distance calculation
    feature_1 = 'embed_0'
    feature_2 = 'embed_1'

    # Extract the values of the selected features
    X = dataset[[feature_1, feature_2]].values

    # Divide the dataset into train and test sets
    X_train, _ = train_test_split(X, test_size=0.3, random_state=42)

    # Calculate distances for different values of r
    r_values, distances = calculate_distances(X_train)

    # Plot the distance versus r
    plot_distance_vs_r(r_values, distances)

if __name__ == "__main__":
    main()
