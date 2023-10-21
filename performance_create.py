import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump

# Load the dataset
df = pd.read_csv('student_performance.csv')

# Convert categorical variables into numerical using one-hot encoding
df_encoded = pd.get_dummies(df.drop('performance_score', axis=1))

# Split the dataset into training and testing sets
X = df_encoded.values
y = df['performance_score'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
dump(model, 'performance_model.joblib')
print("Model saved!")