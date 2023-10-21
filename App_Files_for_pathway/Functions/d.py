import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Define the number of samples
num_samples = 20



# topup_courses_mapping = {
#     'Systems': 'CSE',
#     'Networking': 'CSN',
#     'Cyber': 'CyberSecurity',
#     'Analytics': 'Data',
#     'Business': 'ISE',
#     'IT': 'Information',
#     'Multimedia': 'IM',
#     'Programming': 'SE',
#     'Computational Theory': 'CS'
# }

# Define the possible values for Interested_Area and their corresponding Topup_Course values
interested_areas = ['Systems', 'Networking', 'Cyber', 'Analytics', 'Business', 'IT', 'Multimedia', 'Programming', 'Computational Theory']
topup_courses = ['CSE', 'CSN', 'CyberSecurity', 'Data', 'ISE', 'Information', 'IM', 'SE', 'CS']
topup_to_y_mapping = {
    'CSE': 'C1',
    'CSN': 'C2',
    'CyberSecurity': 'C3',
    'Data': 'C4',
    'ISE': 'C5',
    'Information': 'C6',
    'IM': 'C7',
    'SE': 'C8',
    'CS': 'C9'
}

# Generate data for the dataset
data = []
for _ in range(num_samples):
    interested_area = np.random.choice(interested_areas)
    topup_course = np.random.choice(topup_courses)
    y_value = topup_to_y_mapping[topup_course]

    # Define the relationships between X and Y variables
    programming = np.random.randint(0, 11)
    web = np.random.randint(0, 11)
    mobile = np.random.randint(0, 11)
    software = np.random.randint(0, 11)
    linux = np.random.randint(0, 11)
    uml = np.random.randint(0, 11)
    ui = np.random.randint(0, 11)
    network = np.random.randint(0, 11)
    security = np.random.randint(0, 11)
    cloud = np.random.randint(0, 11)
    bigdata = np.random.randint(0, 11)
    ml = np.random.randint(0, 11)
    mathematics = np.random.randint(0, 11)
    statistics = np.random.randint(0, 11)
    support = np.random.randint(0, 11)
    data.append([interested_area, programming, web, mobile, software, linux, uml, ui, network, security, cloud, bigdata, ml, mathematics, statistics, support, topup_course,y_value])

# Create a DataFrame
columns = ['Interested_Area', 'programming', 'Web', 'Mobile', 'Software', 'Linux', 'UML', 'UI', 'Network', 'Security', 'Cloud', 'BigData', 'ML', 'Mathematics', 'Statistics', 'Support', 'Topup_Course','choice']
df = pd.DataFrame(data, columns=columns)
df_sorted = df.sort_values(by='choice')
# Save the dataset to a CSV file
df.to_csv('sdata2.csv', index=False)
# Save the sorted dataset to a new CSV file

print("Dataset sorted and saved to sorted_synthetic_dataset.csv")