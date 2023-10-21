import random
import pandas as pd

# List of all keywords
all_keywords = [
    "Operation", "Database", "Framework", "API", "Agile", "Containerization", "Kubernetes",
   "Authentication",
    "Authorization", "Firewall", "Penetration testing", "VPN", "Intrusion detection", "AWS",
    "IoT", "Blockchain", "DevSecOps",
    "RESTful", "Microservices", "Continuous integration", "Data analytics", "Data science",
     "Natural language processing", "Neural networks", "Supervised learning",
    "Unsupervised learning", "Reinforcement learning", "Data visualization", "Linear algebra",
    "Calculus", "Probability", "Regression analysis", "Data mining", "Cloud Computing computing", "Serverless",
    "Hadoop", "MapReduce", "Spark", "Data warehousing",

    "Algorithms", "Data Structures", "Operating Systems", "Computer Architecture",
    "Artificial Intelligence", "Computer Graphics", "Databases Management Systems", "Cybersecurity",
    "Distributed Systems", "Web Technologies", "Human-Computer Interaction", "Programming Paradigms",
    "Computer Vision", "Natural Language Processing", "Cloud Services", "Internet of Things", "Big Data",
    "Virtualization", "Software Testing", "Parallel Computing", "High-Performance Computing", "Data Mining",
    "Mobile App Development", "Computer Networks", "Wireless Technologies", "Mobile Security", "Embedded Systems",
    "Compiler Design", "Game Development", "Human-Centered Design", "Agile Methodology", "Cloud Security",
    "Quantum Computing", "Network Protocols", "Real-Time Systems", "Scalability", "Cloud Deployment Models",
    "Machine Vision", "Cryptography", "Computer Ethics", "IT Project Management", "Virtual Reality",
    "Augmented Reality", "Cloud Service Models", "Web Security", "Edge Computing", "System Analysis",
    "IT Governance", "DevOps",

    "OOP", "web development", "Mobile Computing", "Software development", "Linux", "UML Modeling", "UI", "Networking", "Information Security",
    "Cloud Computing", "Database", "Machine Learning", "Mathematics", "Statistics", "Support"
]

# Define keywords with higher occurrence rates and their respective weights
high_occurrence_keywords = [
    ("OOPs", 25),
    ("web development", 26),
    ("Mobile Computing", 29),
    ("Software development", 28),
    ("Linux", 26),
    ("UML Modeling", 21),
    ("UI", 27),
    ("Network", 22),
    ("Information Security", 21),
    ("Cloud Computing", 26),
    ("Database", 26),
    ("Machine Learning", 23),
    ("Mathematics", 29),
    ("Statistics", 24),
    ("Support", 28)
]

# Calculate the total number of occurrences for high-occurrence keywords
total_high_occurrence_count = sum(weight for _, weight in high_occurrence_keywords)

# Calculate the number of occurrences for other keywords
total_other_count = 700 - total_high_occurrence_count

# Generate random weights for the other keywords within a specified range
min_weight = 1  # Minimum weight for other keywords
max_weight = 6  # Maximum weight for other keywords

# Calculate the remaining count of other keywords
remaining_other_count = total_other_count

other_keywords_weights = []

# Create a list of keywords that are not in high_occurrence_keywords
other_keywords = [kw for kw in all_keywords if kw not in [kw for kw, _ in high_occurrence_keywords]]

# Distribute the weights randomly among non-high-occurrence keywords
for keyword in other_keywords:
    # Generate a random weight within the specified range
    weight = random.randint(min_weight, max_weight)
    other_keywords_weights.append((keyword, weight))
    remaining_other_count -= weight

# Combine the high-occurrence and other keyword weights
combined_keyword_weights = high_occurrence_keywords + other_keywords_weights

# Shuffle the combined weights
random.shuffle(combined_keyword_weights)

# Distribute any remaining count randomly to other keywords
while remaining_other_count > 0:
    keyword, weight = random.choice(other_keywords_weights)
    increase_by = min(random.randint(1, remaining_other_count), max_weight - weight)
    weight += increase_by
    remaining_other_count -= increase_by

# Shuffle the combined weights again to ensure randomness
random.shuffle(combined_keyword_weights)

# Generate the data based on the weighted keywords
data = [keyword for keyword, weight in combined_keyword_weights for _ in range(weight)]

# Shuffle the data to make it random
random.shuffle(data)

# Create a DataFrame from the generated data
df = pd.DataFrame({"Skills": data})

# Save the data to a CSV file
df.to_csv('skill_set.csv', index=False)

