import random

import pandas as pd

# Define separate pools for each C1-C9 value
CSE = [
    "Computer System Design Engineer",
    "Aeronautics Engineer",
    "Industrial Engineer",
    "Computer Hardware Engineer",
    "Network Engineer",
    "Network Security Specialist",
    "Computer Support Specialist",
    "Systems Security Analyst"
]

CSN = [
    "Firewall Administrator",
    "Endpoint Engineer",
    "Network Administrator",
    "Network Engineer",
    "Network Designer",
    "Database Engineer",
    "Infrastructure Engineer",
    "System Designer",
    "Systems Administrator",
    "Embedded Systems Designer",
    "Embedded Software Designer",
    "Information Systems Auditor",
    "Information Security Manager",
    "Database Administrator",
    "Communication Engineer",
    "Cloud Engineer / Architect",
    "Network Architect",
    "Network Programmer"
]
CyberSecurity = [
    "Application security engineer",
    "Firewall Administrator",
    "Endpoint Engineer",
    "Risk Specialist",
    "Security Analyst",
    "Security Engineer",
    "Security Architect",
    "Architect-Security",
    "Forensics Investigator",
    "Network Security Engineer",
    "Information Assurance Engineer",
    "IT Auditor",
    "Network Administrator",
    "Network Engineer",
    "Security Administrator",
    "Penetration and Vulnerability Tester",
    "Secure Software Developer",
    "Incident Responder"
]

Data = [
    "Database Admin",
    "DataBase Engineer ",
    "IT Consultant ",
    "Data Engineer ",
    "Data Analyst ",
    "AI Engineer ",
    "Business Intelligence Engineer",
    "ML Engineer",
    "Data scientist ",
    "Business Analyst",
    "Data Consultant"
]

ISE = [
    "Information Security Analyst ",
    "Business Analyst",
    "Systems Analysts",
    "Business Consultant",
    "IS Auditor",
    "IT Consultant",
    "IT Technical Support Officer",
    "Data Analyst",
    "Database Administrator",
    "System Developer"
]

Information = [
    "QA Engineer",
    "SEO",
    "Web Designer",
    "Data Analyst ",
    "Business Analyst",
    "Operations Engineer",
    "Cloud Engineer",
    "Infrastructure Engineer",
    "Support Engineer",
    "TechOps Engineer"
    "Devops Engineer",
    "UX/UI Engineer",
    "Software Engineer",
    "Network Engineer",
    "System Engineer"
    "UI Engineer",
]

IM = [
    "Game Developer",
    "3d Modeler",
    "3D Visualizer",
    "Game Concept designer",
    "UI Engineer",
    "Interior Designers",
    "Multimedia Developer",
    "Multimedia Programmer"
]

SE = [
    "Software Engineer",
    "Application Engineer",
    "ERP Consultant",
    "Game Developer",
    "UI/UX Engineer",
    "Technical Consultant",
    "Software Architect",
    "Product Specialist",
]

CS = [
    "Network Engineer",
    "Software Architect",
    "Software Engineer",
    "ML Engineer",
    "Data scientist ",
    "Web Designer",
    "Network Engineer",
    "Technical Consultant"
]
# Load the dataset from synthetic_dataset.csv
df = pd.read_csv('sdata3.csv')

def replace_values(column_name):
    if column_name == 'C1':
        return random.choice(CSE)
    elif column_name == 'C2':
        return random.choice(CSN)
    elif column_name == 'C3':
        return random.choice(CyberSecurity)
    elif column_name == 'C4':
        return random.choice(Data)
    elif column_name == 'C5':
        return random.choice(ISE)
    elif column_name == 'C6':
        return random.choice(Information)
    elif column_name == 'C7':
        return random.choice(IM)
    elif column_name == 'C8':
        return random.choice(SE)
    elif column_name == 'C9':
        return random.choice(CS)
    else:
        # If it's not C1-C9, return the original value
        return column_name

# Create a function to parallelize the replacement
def parallel_replace(df):
    df['choice'] = df['choice'].apply(lambda x: replace_values(x))

# Call the parallel_replace function
parallel_replace(df)

# Save the updated dataset to the same CSV file, overwriting the original data
df.to_csv('sdata3.csv', index=False)

print("Values in the 'C1' to 'C9' of 'choice' column replaced in synthetic_dataset.csv")
