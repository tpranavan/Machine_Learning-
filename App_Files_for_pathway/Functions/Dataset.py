import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Define the number of samples
num_samples = 100

# Define the possible values for Interested_Area and their corresponding Topup_Course values
interested_areas = ['Systems', 'Networking', 'Cyber', 'Analytics', 'Business', 'IT', 'Multimedia', 'Programming', 'Computational Theory']
#topup_courses = ['CSE', 'CSN', 'Cyber', 'Data', 'ISE', 'IT', 'IM', 'SE', 'CS']

topup_courses_mapping = {
    'Systems': 'CSE',
    'Networking': 'CSN',
    'Cyber': 'CyberSecurity',
    'Analytics': 'Data',
    'Business': 'ISE',
    'IT': 'Information',
    'Multimedia': 'IM',
    'Programming': 'SE',
    'Computational Theory': 'CS'
}

# Define the mapping of Topup_Course to Y values
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
   # topup_course = np.random.choice(topup_courses)
    topup_course = topup_courses_mapping[interested_area]
    y_value = topup_to_y_mapping[topup_course]
 #   Define the relationships between X and Y variables
    if topup_course in ['SE', 'IT', 'Data', 'CS']:
        programming = np.random.randint(6, 11)
    else:
        programming = np.random.randint(0, 5)

    if topup_course in ['SE', 'IT','IM','CS']:
        web = np.random.randint(5, 11)
    else:
        web = np.random.randint(0, 5)

    if topup_course in ['SE', 'IT', 'CS']:
        mobile = np.random.randint(6, 11)
    else:
        mobile = np.random.randint(0, 5)

    if topup_course in ['SE', 'IT', 'CS']:
        software = np.random.randint(6, 11)
    else:
        software = np.random.randint(0, 5)

    if topup_course in ['CSE', 'CSN', 'Cyber','CS']:
        linux = np.random.randint(6, 11)
    else:
        linux = np.random.randint(0, 5)

    if topup_course in ['Data', 'ISE', 'IM','CS','IT']:
        uml = np.random.randint(6, 11)
    else:
        uml = np.random.randint(0, 5)

    if topup_course in ['SE', 'IT','IM']:
        ui = np.random.randint(6, 11)
    else:
        ui = np.random.randint(0, 5)

    if topup_course in ['CSE','CSN','Cyber', 'ISE', 'IT']:
        network = np.random.randint(6, 11)
    else:
        network = np.random.randint(0, 5)

    if topup_course in ['CSN','Cyber', 'ISE', 'IT']:
        security = np.random.randint(6, 11)
    else:
        security = np.random.randint(0, 5)

    if topup_course in ['CSN', 'Cyber', 'SE', 'IT','CS','ISE']:
        cloud = np.random.randint(6, 11)
    else:
        cloud = np.random.randint(0, 5)

    if topup_course in ['Data', 'ISE','CS']:
        bigdata = np.random.randint(5, 11)
    else:
        bigdata = np.random.randint(0, 5)

    if topup_course in ['Data', 'ISE', 'IT', 'SE','CS']:
        ml = np.random.randint(5, 11)
    else:
        ml = np.random.randint(0, 5)

    if topup_course in ['Data', 'SE', 'IT', 'CSE','CS']:
        mathematics = np.random.randint(6, 11)
    else:
        mathematics = np.random.randint(0, 5)

    if topup_course in ['Data', 'SE', 'IT', 'CSE','CS']:
        statistics = np.random.randint(6, 11)
    else:
        statistics = np.random.randint(0, 5)

    if topup_course in ['IT','ISE','CSE']:
        support = np.random.randint(5, 11)
    else:
        support = np.random.randint(0, 5)

    data.append([interested_area, programming, web, mobile, software, linux, uml, ui, network, security, cloud, bigdata, ml, mathematics, statistics, support, topup_course,y_value])

# Create a DataFrame
columns = ['Interested_Area', 'programming', 'Web', 'Mobile', 'Software', 'Linux', 'UML', 'UI', 'Network', 'Security', 'Cloud', 'BigData', 'ML', 'Mathematics', 'Statistics', 'Support', 'Topup_Course','choice']
df = pd.DataFrame(data, columns=columns)
df_sorted = df.sort_values(by='choice')
# Save the dataset to a CSV file
df.to_csv('sdata3.csv', index=False)
# Save the sorted dataset to a new CSV file

print("Dataset sorted and saved to sorted_synthetic_dataset4.csv")