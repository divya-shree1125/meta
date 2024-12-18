# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# import pickle

# # Load dataset
# file_path = "gym_members_exercise_tracking_synthetic_data.csv"
# data = pd.read_csv(file_path)

# # Drop rows with missing target values
# data = data.dropna(subset=["Calories_Burned"])

# # Features and target
# X = data.drop(columns=["Calories_Burned", "Workout_Type", "Max_BPM", "Gender"])
# y = data["Calories_Burned"]

# # Preprocessing pipeline
# pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="mean")),
#     ("scaler", StandardScaler())
# ])

# X_processed = pipeline.fit_transform(X)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# # Train Random Forest Regressor
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save model and pipeline
# with open("model.pkl", "wb") as model_file:
#     pickle.dump(model, model_file)

# with open("pipeline.pkl", "wb") as pipeline_file:
#     pickle.dump(pipeline, pipeline_file)

# print("Model and pipeline saved successfully!")

import streamlit as st
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import pickle

# Initialize Firebase
cred = credentials.Certificate("tracker-ba5ed-1e9bbc078e3f.json")  # Replace with your Firebase key
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the trained model and pipeline
with open("model.pkl", "rb") as model_file, open("pipeline.pkl", "rb") as pipeline_file:
    model = pickle.load(model_file)
    pipeline = pickle.load(pipeline_file)

# Streamlit App
st.title("Fitness Tracker System")
st.header("Predict Your Calories Burned")

# User Input Form
with st.form("input_form"):
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.1)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, step=0.01)
    resting_bpm = st.number_input("Resting BPM", min_value=40.0, max_value=120.0, step=0.1)
    avg_bpm = st.number_input("Average BPM", min_value=50.0, max_value=200.0, step=0.1)
    session_duration = st.number_input("Session Duration (hours)", min_value=0.1, max_value=5.0, step=0.1)
    fat_percentage = st.number_input("Fat Percentage (%)", min_value=5.0, max_value=50.0, step=0.1)
    water_intake = st.number_input("Water Intake (liters)", min_value=0.5, max_value=5.0, step=0.1)
    workout_frequency = st.slider("Workout Frequency (days/week)", min_value=1, max_value=7, step=1)
    experience_level = st.slider("Experience Level (1=Beginner, 3=Expert)", min_value=1, max_value=3, step=1)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input for prediction
    user_input = np.array([[age, weight, height, resting_bpm, avg_bpm, session_duration,
                            fat_percentage, water_intake, workout_frequency, experience_level]])
    input_processed = pipeline.transform(user_input)

    # Predict calories burned
    prediction = model.predict(input_processed)[0]
    st.success(f"Estimated Calories Burned: {prediction:.2f} kcal")

    # Save user data to Firebase
    user_data = {
        "Age": age,
        "Weight (kg)": weight,
        "Height (m)": height,
        "Resting BPM": resting_bpm,
        "Average BPM": avg_bpm,
        "Session Duration (hours)": session_duration,
        "Fat Percentage (%)": fat_percentage,
        "Water Intake (liters)": water_intake,
        "Workout Frequency (days/week)": workout_frequency,
        "Experience Level": experience_level,
        "Predicted Calories Burned": prediction
    }
    db.collection("fitness_tracker").add(user_data)
    st.info("Your data has been saved!")
