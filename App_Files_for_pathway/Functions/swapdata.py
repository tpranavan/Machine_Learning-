import pandas as pd

# Load the dataset from the CSV file
dataset = pd.read_csv('../Career_dataset.csv')  # Replace 'your_dataset.csv' with your actual dataset file path

# Shuffle the rows randomly
shuffled_dataset = dataset.sample(frac=1, random_state=42)  # frac=1 means all rows, random_state for reproducibility

# Save the shuffled dataset to a new CSV file
shuffled_dataset.to_csv('shuffled_dataset.csv', index=False)  # Replace 'shuffled_dataset.csv' with your desired output file name
