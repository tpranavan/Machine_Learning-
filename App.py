import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import load
import joblib
from flask import Flask, render_template, request, jsonify
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer

import pickle
import numpy as np

app = Flask(__name__)


# member 01 pathway
# Load the saved model and label encoders
best_model = joblib.load('App_Files_for_pathway/best_model.pkl')
label_encoders = joblib.load('App_Files_for_pathway/label_encoders.pkl')
df2 = pd.read_csv('App_Files_for_pathway/Career_dataset.csv')
@app.route('/pathway', methods=['GET', 'POST'])
def pathway():
    if request.method == "POST":
        # Retrieve input data from the form
        interested_area = request.form["interested_area"]
        programming = int(request.form["programming"])
        web = int(request.form["web"])
        mobile = int(request.form["mobile"])
        software = int(request.form["software"])
        linux = int(request.form["linux"])
        uml = int(request.form["uml"])
        ui = int(request.form["ui"])
        network = int(request.form["network"])
        security = int(request.form["security"])
        cloud = int(request.form["cloud"])
        big_data = int(request.form["big_data"])
        ml = int(request.form["ml"])
        mathematics = int(request.form["mathematics"])
        statistics = int(request.form["statistics"])
        support = int(request.form["support"])

        # Create a dictionary from the input data
        input_data = {
            'Interested_Area': [interested_area],
            'programming': [programming],
            'Web': [web],
            'Mobile': [mobile],
            'Software': [software],
            'Linux': [linux],
            'UML': [uml],
            'UI': [ui],
            'Network': [network],
            'Security': [security],
            'Cloud': [cloud],
            'BigData': [big_data],
            'ML': [ml],
            'Mathematics': [mathematics],
            'Statistics': [statistics],
            'Support': [support]
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

        # Decode the predictions using label encoders
        inverse_label_encoders = {}
        for column in ['Topup_Course', 'choice']:
            le = label_encoders[column]
            inverse_label_encoders[column] = le.inverse_transform(predictions[:, 0 if column == 'Topup_Course' else 1])

        topup_prediction = inverse_label_encoders['Topup_Course'][0]

        # Filter the DataFrame based on the topup_prediction value
        filtered_df = df2[df2['Topup_Course'] == topup_prediction]

        # Get unique values in the 'choice' column for the filtered DataFrame
        unique_choices = filtered_df['choice'].unique()

        # Print the unique choices for the filtered DataFrame
        print(f"Topup Course: {topup_prediction}")
        print("Unique Choices:")
        for choice in unique_choices:
            print(choice)
        print("\n")

        # Pass the predictions to the template
        return render_template("pathway.html",
                               topup_prediction=inverse_label_encoders['Topup_Course'][0],
                               career_choice_prediction=inverse_label_encoders['choice'][0],

        unique_choices = unique_choices
                               )

    return render_template("pathway.html", topup_prediction="", career_choice_prediction="",unique_choices='')


# member 02 student_performance
loaded_model = load('performance_model.joblib')
df = pd.read_csv('student_performance.csv')

# Define a route for the home page
@app.route('/student_performance', methods=['GET', 'POST'])
def student_performance():
    if request.method == 'POST':
        # Get user inputs from the form
        user_inputs = {
            'personality_Introverted': int(request.form.get('personality_Introverted', 0)),
            'personality_Extroverted': int(request.form.get('personality_Extroverted', 0)),
            'personality_Ambivert': int(request.form.get('personality_Ambivert', 0)),
            'studying_hours': int(request.form.get('studying_hours', 0)),
            'studying_time_Afternoon': int(request.form.get('studying_time_Afternoon', 0)),
            'studying_time_Evening': int(request.form.get('studying_time_Evening', 0)),
            'studying_time_Morning': int(request.form.get('studying_time_Morning', 0)),
            'prior_knowledge_High': int(request.form.get('prior_knowledge_High', 0)),
            'prior_knowledge_Medium': int(request.form.get('prior_knowledge_Medium', 0)),
            'prior_knowledge_Low': int(request.form.get('prior_knowledge_Low', 0)),
            'learning_style_Visual_Learner': int(request.form.get('learning_style_Visual_Learner', 0)),
            'learning_style_Auditory_Learner': int(request.form.get('learning_style_Auditory_Learner', 0)),
            'learning_style_Kinesthetic_Learner': int(request.form.get('learning_style_Kinesthetic_Learner', 0)),
            'motivation_Moderate': int(request.form.get('motivation_Moderate', 0)),
            'motivation_High': int(request.form.get('motivation_High', 0)),
            'motivation_Low': int(request.form.get('motivation_Low', 0)),
            'interest_High': int(request.form.get('interest_High', 0)),
            'interest_Moderate': int(request.form.get('interest_Moderate', 0)),
            'interest_Low': int(request.form.get('interest_Low', 0)),
            'study_environment_Quiet_and_Organized': int(request.form.get('study_environment_Quiet_and_Organized', 0)),
            'study_environment_Noisy_and_Disorganized': int(request.form.get('study_environment_Noisy_and_Disorganized', 0)),
            'study_techniques_Effective_Study_Techniques': int(request.form.get('study_techniques_Effective_Study_Techniques', 0)),
            'study_techniques_Time_Management': int(request.form.get('study_techniques_Time_Management', 0)),
            'study_techniques_Active_Learning_Methods': int(request.form.get('study_techniques_Active_Learning_Methods', 0)),
            'teacher_quality_Good': int(request.form.get('teacher_quality_Good', 0)),
            'teacher_quality_Average': int(request.form.get('teacher_quality_Average', 0)),
            'teacher_quality_Excellent': int(request.form.get('teacher_quality_Excellent', 0)),
            'personal_circumstances_Supportive_Family': int(request.form.get('personal_circumstances_Supportive_Family', 0)),
            'personal_circumstances_Financial_Difficulties': int(request.form.get('personal_circumstances_Financial_Difficulties', 0)),
            'peer_influence_Low': int(request.form.get('peer_influence_Low', 0)),
            'peer_influence_High': int(request.form.get('peer_influence_High', 0)),
            'peer_influence_Medium': int(request.form.get('peer_influence_Medium', 0)),
            'test_anxiety_Moderate': int(request.form.get('test_anxiety_Moderate', 0)),
            'test_anxiety_High': int(request.form.get('test_anxiety_High', 0)),
            'test_anxiety_Low': int(request.form.get('test_anxiety_Low', 0)),
            'resources_access_Adequate': int(request.form.get('resources_access_Adequate', 0)),
            'resources_access_Limited': int(request.form.get('resources_access_Limited', 0)),
        }

        # Create a DataFrame from user inputs
        user_inputs_df = pd.DataFrame(user_inputs, index=[0])

        # Reindex the user inputs DataFrame to match the training dataset's column order
        df_encoded = pd.get_dummies(df.drop('performance_score', axis=1))
        user_inputs_df = user_inputs_df.reindex(columns=df_encoded.columns, fill_value=0)

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(user_inputs_df.values)[0]

        return render_template('student_performance.html', prediction=prediction)

    return render_template('student_performance.html', prediction=None)



# member 03 ChatBot
lemmatizer = WordNetLemmatizer()
# Load chatbot model and data
# model = load_model('chatbot.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open("Uni_Data.json").read())

# Functions for chatbot
def clean_up_sentence(sentence):
    # Tokenize the pattern - split words into an array
    sentence_words = nltk.word_tokenize(sentence)

    # Stem each word - create short form for words
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))

def predict_class(sentence, model):
    # Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    error = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents, text):
    tag = predict_class(text )[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/chatbot')
def home():
    return render_template('chatbot.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.form['user_message']
    response = get_response(intents, user_message)
    return jsonify({'bot_response': response})



@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")




if __name__ == "__main__":
    app.run(debug=True)
