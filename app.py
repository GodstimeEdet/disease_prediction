import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
import ast

app = Flask(__name__)

# Load the model
with open('a.sav', 'rb') as f:
    model = pickle.load(f)

# List of symptoms
symptoms_list = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", 
    "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", 
    "fatigue", "weight_gain", "anxiety", "cold_hands_and_feet", "mood_swings", 
    "weight_loss", "restlessness", "lethargy", "patches_in_throat"
]

# Dictionary mapping disease numbers to names
disease_map = {
    0: "Paroymsal Positional Vertigo", 1: "AIDS", 2: "Acne", 3: "Alcoholic hepatitis", 
    4: "Allergy", 5: "Arthritis", 6: "Bronchial Asthma", 7: "Cervical spondylosis", 
    8: "Chicken pox", 9: "Chronic cholestasis", 10: "Common Cold", 11: "Dengue", 
    12: "Diabetes", 13: "Dimorphic hemmorhoids(piles)", 14: "Drug Reaction", 
    15: "Fungal infection", 16: "GERD", 17: "Gastroenteritis", 18: "Heart attack", 
    19: "Hepatitis B", 20: "Hepatitis C", 21: "Hepatitis D", 22: "Hepatitis E", 
    23: "Hypertension", 24: "Hyperthyroidism", 25: "Hypoglycemia", 
    26: "Hypothyroidism", 27: "Impetigo", 28: "Jaundice", 29: "Malaria", 
    30: "Migraine", 31: "Osteoarthristis", 32: "Paralysis (brain hemorrhage)", 
    33: "Peptic ulcer disease", 34: "Pneumonia", 35: "Psoriasis", 
    36: "Tuberculosis", 37: "Typhoid", 38: "Urinary tract infection", 
    39: "Varicose veins", 40: "hepatitis A"
}

# Load additional datasets
medications_df = pd.read_csv('medications.csv')
precautions_df = pd.read_csv('precautions_df.csv')
diets_df = pd.read_csv('diets.csv')
description_df = pd.read_csv('description.csv')

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = [0] * len(symptoms_list)
    for i, symptom in enumerate(symptoms_list):
        if request.form.get(symptom) == '1':
            symptoms[i] = 1
    
    input_data = np.array([symptoms])
    prediction = model.predict(input_data)
    predicted_disease_num = prediction[0]
    predicted_disease = disease_map.get(predicted_disease_num, "Unknown")
    
    # Fetching the medication, Precautions and the diets from Datasets
    medication_str = medications_df[medications_df['Disease'] == predicted_disease]['Medication'].values
    precautions = precautions_df[precautions_df['Disease'] == predicted_disease].values.flatten()[2:] #flatten the values for that paricular Disease in a into a list.
    diet_str = diets_df[diets_df['Disease'] == predicted_disease]['Diet'].values
    description = description_df[description_df['Disease'] == predicted_disease]['Description'].values[0]

    
    # Convert string representations of lists to actual lists from the datasets
    medication = ast.literal_eval(medication_str[0]) if len(medication_str) > 0 else []
    diet = ast.literal_eval(diet_str[0]) if len(diet_str) > 0 else []
    
    return render_template(
        'result.html', 
        disease=predicted_disease, 
        medication=medication, 
        precautions=precautions, 
        diet=diet,
        description=description
    )

if __name__ == '__main__':
    app.run(debug=True)
