import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the font scale for Seaborn (adjust the value to your preference)
sns.set(font_scale=0.7)  # You can adjust the font scale as needed

# Load the dataset
df = pd.read_csv('Career_dataset.csv')

# Check the first few rows of the dataset
print(df.head())

# Explore the distribution of Y variables (Topup_Course and choice)
sns.countplot(data=df, x='Topup_Course')
plt.title('Distribution of Topup_Course')
plt.show()

sns.countplot(data=df, x='choice')
plt.title('Distribution of choice')
plt.show()

# Explore relationships between numerical X variables and Y variables with regression analysis
for col in df.columns[1:16]:
    plt.figure(figsize=(10, 6))

    # Relationship with 'Topup_Course' using regression
    sns.boxplot(data=df, x='Topup_Course', y=col)
    plt.title(f'Relationship between {col} and Topup_Course')
    plt.xticks(rotation=45)

    plt.show()

    plt.figure(figsize=(10, 6))

    # Relationship with 'choice' using regression
    sns.boxplot(data=df, x='choice', y=col)
    plt.title(f'Relationship between {col} and choice')
    plt.xticks(rotation=45)

    plt.show()
