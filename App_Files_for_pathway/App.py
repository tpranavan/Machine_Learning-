import joblib
import pandas as pd

# Load the saved model and label encoders
best_model = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Input data (replace with your input)
input_data = {
    'Interested_Area': ['Cyber'],
    'programming': [8],
    'Web': [7],
    'Mobile': [6],
    'Software': [7],
    'Linux': [3],
    'UML': [4],
    'UI': [3],
    'Network': [4],
    'Security': [5],
    'Cloud': [6],
    'BigData': [5],
    'ML': [3],
    'Mathematics': [4],
    'Statistics': [3],
    'Support': [4]
# Analytics,7,2,4,3,2,6,1,7,10,9,9,7,6,7,1,Data,Data Engineer
#topup_courses = ['CSE', 'CSN', 'Cyber', 'Data', 'ISE', 'IT', 'IM', 'SE', 'CS']
#interested_areas = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9']
}

# Create a DataFrame from the input data
input_df = pd.DataFrame(input_data)

# Encode categorical variables using label encoders
for column in input_df.columns:
    if column in label_encoders:
        le = label_encoders[column]
        input_df[column] = le.transform(input_df[column])

# Make predictions
predictions = best_model.predict(input_df)

# Decode the 'Topup_Course' and 'choice' predictions using label encoders
inverse_label_encoders = {}
for column in ['Topup_Course', 'choice']:
    le = label_encoders[column]
    inverse_label_encoders[column] = le.inverse_transform(predictions[:, 0 if column == 'Topup_Course' else 1])

# Print the predictions
for column, values in inverse_label_encoders.items():
    print(f'{column}: {values[0]}')
