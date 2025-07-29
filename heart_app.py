import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier

# Title
st.title("Heart Disease Prediction App")

# Load dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv("heart disease Dataset.csv")
    dataset.rename(columns={
        'age': 'age',
        'sex': 'gender',
        'cp': 'chest_pain',
        'trestbps': 'blood_pressure',
        'chol': 'cholesterol',
        'fbs': 'fasting_sugar',
        'restecg': 'ecg_result',
        'thalach': 'heart_rate',
        'exang': 'exercise_angina',
        'oldpeak': 'st_depression',
        'slope': 'st_slope',
        'ca': 'vessels_count',
        'thal': 'thalassemia_type',
        'condition': 'heart_disease'
    }, inplace=True)

    # Remove outliers
    Q1 = dataset['cholesterol'].quantile(0.25)
    Q3 = dataset['cholesterol'].quantile(0.75)
    IQR = Q3 - Q1
    dataset = dataset[
        (dataset['cholesterol'] >= Q1 - 1.5 * IQR) &
        (dataset['cholesterol'] <= Q3 + 1.5 * IQR)
    ]
    return dataset

dataset = load_data()

# Show data checkbox
if st.checkbox("Show Dataset (Preview)"):
    st.dataframe(dataset.head(10))

# Features & Labels
x = dataset.drop("heart_disease", axis=1)
y = dataset["heart_disease"]

# Scaling
scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RidgeClassifier(alpha=0.01, solver='lsqr', tol=0.01)
model.fit(x_train, y_train)

# User input section
st.subheader("Enter Patient Data:")
user_input = {}

# Input form
for col in x.columns:
    if col == "gender":
        st.markdown(":gray[Gender: 1 = Male, 0 = Female]")
    elif col == "fasting_sugar":
        st.markdown(":gray[Fasting Sugar > 120 mg/dl: 1 = Yes, 0 = No]")
    elif col == "exercise_angina":
        st.markdown(":gray[Exercise-induced Angina: 1 = Yes, 0 = No]")
    elif col == "chest_pain":
        st.markdown(":gray[Chest Pain Type: 0 = Typical Angina, 1 = Atypical, 2 = Non-anginal, 3 = Asymptomatic]")
    elif col == "ecg_result":
        st.markdown(":gray[Resting ECG: 0 = Normal, 1 = ST-T Abnormality, 2 = Left Ventricular Hypertrophy]")
    elif col == "st_slope":
        st.markdown(":gray[ST Slope: 0 = Upsloping, 1 = Flat, 2 = Downsloping]")
    elif col == "vessels_count":
        st.markdown(":gray[Number of Major Vessels Colored by Fluoroscopy: 0–3]")
    elif col == "thalassemia_type":
        st.markdown(":gray[Thalassemia: 1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect]")

    # Input field
    if col in ['gender', 'fasting_sugar', 'exercise_angina', 'chest_pain', 'ecg_result', 'st_slope', 'vessels_count', 'thalassemia_type']:
        user_input[col] = st.number_input(f"{col.replace('_', ' ').title()}:", min_value=0, max_value=3, step=1)
    else:
        user_input[col] = st.number_input(f"{col.replace('_', ' ').title()}:", format="%.2f")

# Prediction
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("⚠️ Patient is likely to have heart disease.")
    else:
        st.success("✅ Patient is not likely to have heart disease.")
