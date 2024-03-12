import pandas as pd
import numpy as np

def load_data(file_path):
    """Load data from an Excel file and return a DataFrame."""
    return pd.read_excel(file_path)

def process_data(data):
    """Process the dataset to extract MathBERT embeddings and valid output values."""
    mathbert_columns = [col for col in data.columns if col.startswith("embed_")]
    mathbert_data = data[mathbert_columns]
    valid_output_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    mathbert_data['output'] = data['output']
    mathbert_data = mathbert_data[mathbert_data['output'].isin(valid_output_values)]
    return mathbert_data

def calculate_statistics(mathbert_data):
    """Calculate class centroids and spreads."""
    grouped_data = mathbert_data.groupby('output')
    class_centroids = grouped_data.mean()
    class_spreads = grouped_data.std()
    return class_centroids, class_spreads

def calculate_interclass_distance(class_centroids):
    """Calculate the interclass distance between two selected classes."""
    class_1 = class_centroids.iloc[0]  # Choose the first class
    class_2 = class_centroids.iloc[1]  # Choose the second class
    interclass_distance = np.linalg.norm(class_1 - class_2)
    return interclass_distance

def main():
    # Load data
    dataset = load_data("training_mathbert 2.xlsx")

    # Process data
    mathbert_data = process_data(dataset)

    # Calculate statistics
    class_centroids, class_spreads = calculate_statistics(mathbert_data)

    # Calculate interclass distance
    interclass_distance = calculate_interclass_distance(class_centroids)

    # Output the results
    print("Class Centroids:")
    print(class_centroids)
    print("\nClass Spreads:")
    print(class_spreads)
    print("\nInterclass Distance between Class 1 and Class 2:", interclass_distance)

if __name__ == "__main__":
    main()
