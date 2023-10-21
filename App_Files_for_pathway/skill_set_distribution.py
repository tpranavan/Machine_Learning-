import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
df = pd.read_csv('skill_set.csv')

# Create a bar plot to visualize the data distribution without ordering
plt.figure(figsize=(12, 6))
barplot = sns.barplot(x=df['Skills'].value_counts().values, y=df['Skills'].value_counts().index, palette='viridis', order=df['Skills'].unique())

# Reduce the font size for y-axis labels
barplot.set_yticklabels(barplot.get_yticklabels(), size=3)
plt.title('Skills Distribution in Datasets')
plt.xlabel('Count')
plt.ylabel('Skills')
plt.show()
