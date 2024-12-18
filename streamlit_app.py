import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
file_path = "gym_members_exercise_tracking_synthetic_data.csv"
data = pd.read_csv(file_path)

# Drop rows with missing target values
data = data.dropna(subset=["Calories_Burned"])

# Features and target
X = data.drop(columns=["Calories_Burned", "Workout_Type", "Max_BPM", "Gender"])
y = data["Calories_Burned"]

# Preprocessing pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

X_processed = pipeline.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and pipeline
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("pipeline.pkl", "wb") as pipeline_file:
    pickle.dump(pipeline, pipeline_file)

print("Model and pipeline saved successfully!")


