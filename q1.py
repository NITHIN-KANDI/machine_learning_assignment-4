import pandas as pd
import numpy as np

dataset = pd.read_excel("training_mathbert 2.xlsx")


mathbert_columns = [col for col in dataset.columns if col.startswith("embed_")]
mathbert_data = dataset[mathbert_columns]

mathbert_data['output'] = dataset['output']

valid_output_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
mathbert_data = mathbert_data[mathbert_data['output'].isin(valid_output_values)]

grouped_data = mathbert_data.groupby('output')

class_centroids = grouped_data.mean()

class_spreads = grouped_data.std()

# Select two classes for interclass distance calculation
class_1 = class_centroids.iloc[0]  # For example, choose the first class
class_2 = class_centroids.iloc[1]  # For example, choose the second class

# Calculate the distance between mean vectors between classes
interclass_distance = np.linalg.norm(class_1 - class_2)

# Output the results
print("Class Centroids:")
print(class_centroids)
print("\nClass Spreads:")
print(class_spreads)
print("\nInterclass Distance between Class 1 and Class 2:", interclass_distance)
