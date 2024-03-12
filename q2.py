import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from an Excel file and return a DataFrame."""
    return pd.read_excel(file_path)

def plot_histogram(data, feature):
    """Plot a histogram of the selected feature."""
    feature_values = data[feature]
    plt.figure(figsize=(10, 6))
    plt.hist(feature_values, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {feature}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def calculate_statistics(feature_values, feature_name):
    """Calculate the mean and variance of the feature."""
    feature_mean = np.mean(feature_values)
    feature_variance = np.var(feature_values)
    print(f"Mean of {feature_name}: {feature_mean}")
    print(f"Variance of {feature_name}: {feature_variance}")

def main():
    # Load the dataset
    dataset = load_data("training_mathbert 2.xlsx")

    # Select the feature (column) for which you want to generate the histogram
    feature = 'embed_0'

    # Plot the histogram
    plot_histogram(dataset, feature)

    # Calculate the mean and variance of the feature
    calculate_statistics(dataset[feature], feature)

if __name__ == "__main__":
    main()
