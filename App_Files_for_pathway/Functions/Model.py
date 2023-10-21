import joblib
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Step 1: Load the dataset
data = pd.read_csv('sdata.csv')

# Step 2: Preprocess the data
# Encode categorical variables
label_encoders = {}
categorical_columns = ['Interested_Area', 'Topup_Course', 'choice']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Step 3: Define features (X) and targets (y)
X = data.drop(['Topup_Course', 'choice'], axis=1)
y = data[['Topup_Course', 'choice']]

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Step 5: Train and evaluate different machine learning algorithms
models = [
    ('RandomForest', RandomForestClassifier(random_state=42, n_jobs=-1)),
    ('LogisticRegression', MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=10000, n_jobs=-1))),
    ('SVM', MultiOutputClassifier(SVC(random_state=42))),
    ('DecisionTree', DecisionTreeClassifier(random_state=42))
]

best_model = None
best_average_accuracy = 0

def train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy_topup_course = accuracy_score(y_test['Topup_Course'], y_pred[:, 0])
    accuracy_choice = accuracy_score(y_test['choice'], y_pred[:, 1])
    average_accuracy = (accuracy_topup_course + accuracy_choice) / 2

    return name, model, average_accuracy

# Use all CPU resources for parallel processing
with parallel_backend('loky', n_jobs=-1):
    results = Parallel()(delayed(train_and_evaluate)(name, model, X_train, y_train, X_test, y_test) for name, model in models)

for name, model, average_accuracy in results:
    if average_accuracy > best_average_accuracy:
        best_average_accuracy = average_accuracy
        best_model = model

# Step 6: Select the best-performing model based on average accuracy
print(f'Best Model: {best_model} with Average Accuracy: {round(best_average_accuracy, 2)}')

# Step 7: Save the best model and label encoders to a file
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Step 8: Evaluate the best model on the test data and print accuracy for 'Topup_Course' and 'choice'
y_pred = best_model.predict(X_test)

accuracy_topup_course = accuracy_score(y_test['Topup_Course'], y_pred[:, 0])
accuracy_choice = accuracy_score(y_test['choice'], y_pred[:, 1])

print(f'Accuracy for Topup_Course: {round(accuracy_topup_course, 2)}')
print(f'Accuracy for choice: {round(accuracy_choice, 2)}')
