import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_excel("training_mathbert 2.xlsx")

# Select the feature (column) for which you want to generate the histogram
feature = 'embed_0'

# Extract the values of the selected feature
feature_values = dataset[feature]

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(feature_values, bins=20, color='skyblue', edgecolor='black')
plt.title(f'Histogram of {feature}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate the mean and variance of the feature
feature_mean = np.mean(feature_values)
feature_variance = np.var(feature_values)

print(f"Mean of {feature}: {feature_mean}")
print(f"Variance of {feature}: {feature_variance}")
