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

# import streamlit as st
# import numpy as np
# import firebase_admin
# from firebase_admin import credentials, firestore
# import pickle

# # Initialize Firebase
# cred = credentials.Certificate("tracker-ba5ed-1e9bbc078e3f.json")  # Replace with your Firebase key
# # firebase_admin.initialize_app(cred)
# db = firestore.client()

# # Load the trained model and pipeline
# with open("model.pkl", "rb") as model_file, open("pipeline.pkl", "rb") as pipeline_file:
#     model = pickle.load(model_file)
#     pipeline = pickle.load(pipeline_file)

# # Streamlit App
# st.title("Fitness Tracker System")
# st.header("Predict Your Calories Burned")

# # User Input Form
# with st.form("input_form"):
#     age = st.number_input("Age", min_value=10, max_value=100, step=1)
#     weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.1)
#     height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, step=0.01)
#     resting_bpm = st.number_input("Resting BPM", min_value=40.0, max_value=120.0, step=0.1)
#     avg_bpm = st.number_input("Average BPM", min_value=50.0, max_value=200.0, step=0.1)
#     session_duration = st.number_input("Session Duration (hours)", min_value=0.1, max_value=5.0, step=0.1)
#     fat_percentage = st.number_input("Fat Percentage (%)", min_value=5.0, max_value=50.0, step=0.1)
#     water_intake = st.number_input("Water Intake (liters)", min_value=0.5, max_value=5.0, step=0.1)
#     workout_frequency = st.slider("Workout Frequency (days/week)", min_value=1, max_value=7, step=1)
#     experience_level = st.slider("Experience Level (1=Beginner, 3=Expert)", min_value=1, max_value=3, step=1)
    
#     submitted = st.form_submit_button("Predict")

# if submitted:
#     # Prepare input for prediction
#     user_input = np.array([[age, weight, height, resting_bpm, avg_bpm, session_duration,
#                             fat_percentage, water_intake, workout_frequency, experience_level]])
#     input_processed = pipeline.transform(user_input)

#     # Predict calories burned
#     prediction = model.predict(input_processed)[0]
#     st.success(f"Estimated Calories Burned: {prediction:.2f} kcal")

#     # Save user data to Firebase
#     user_data = {
#         "Age": age,
#         "Weight (kg)": weight,
#         "Height (m)": height,
#         "Resting BPM": resting_bpm,
#         "Average BPM": avg_bpm,
#         "Session Duration (hours)": session_duration,
#         "Fat Percentage (%)": fat_percentage,
#         "Water Intake (liters)": water_intake,
#         "Workout Frequency (days/week)": workout_frequency,
#         "Experience Level": experience_level,
#         "Predicted Calories Burned": prediction
#     }
#     db.collection("fitness_tracker").add(user_data)
#     st.info("Your data has been saved!")
# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error

# # Load and preprocess data
# @st.cache
# def load_data(file_path):
#     data = pd.read_csv("gym_members_exercise_tracking_synthetic_data.csv")
#     data['Max_BPM'] = pd.to_numeric(data['Max_BPM'], errors='coerce')  # Convert to numeric
#     data.dropna(inplace=True)  # Drop rows with missing values
#     return data

# data = load_data('gym_members_exercise_tracking_synthetic_data.csv')

# # Streamlit UI
# st.title("Fitness Tracker System")
# st.sidebar.header("Navigation")
# page = st.sidebar.selectbox("Select a Page", ["Data Overview", "Visualizations", "Calorie Prediction"])

# if page == "Data Overview":
#     st.header("Dataset Overview")
#     st.write(data.head())
#     st.write("Data Dimensions:", data.shape)

#     if st.checkbox("Show Summary Statistics"):
#         st.write(data.describe())

# elif page == "Visualizations":
#     st.header("Data Visualizations")
#     st.subheader("line chart:calories burned over sessions")
#     st.line_chart(data[['Calories_burned',"Session_Duration(hours)"]])
#     st.subheader("line_chart:avg  hros")
#     st.line_chart(data[["Avg_BPM","Resting_BPM"]])

  

# elif page == "Calorie Prediction":
#     st.header("Predict Calories Burned")

#     # Model training
#     X = data[['Age', 'Weight (kg)', 'Height (m)', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)', 
#               'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']]
#     y = data['Calories_Burned']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     error = mean_absolute_error(y_test, predictions)
    
#     st.write("Model trained with a Mean Absolute Error of:", round(error, 2))
    
#     # User input form
#     st.subheader("Enter Fitness Details")
#     user_input = {}
#     for col in X.columns:
#         user_input[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))
    
#     if st.button("Predict"):
#         input_df = pd.DataFrame([user_input])
#         prediction = model.predict(input_df)[0]
#         st.write(f"Predicted Calories Burned: {round(prediction, 2)}")

# st.sidebar.write("Developed by ChatGPT")
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import auth
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Initialize Firebase Admin SDK
cred = credentials.Certificate("majore-da630-64e56a73f37a.json")  # Path to your Firebase service account key
# firebase_admin.initialize_app(cred)
# firebaseConfig = {
#     apiKey: "AIzaSyAhOLmdSCoZ2ATcpswSqeFjQuSjiLg4SKM",
#     authDomain: "majore-da630.firebaseapp.com",
#     databaseURL: "https://majore-da630-default-rtdb.firebaseio.com",
#     projectId: "majore-da630",
#     storageBucket: "majore-da630.firebasestorage.app",
#     messagingSenderId: "300801355721",
#     appId: "1:300801355721:web:d3a7551c8c5ffc1b0f4218",
#     measurementId: "G-5N9K57XPLQ"}
# // For Firebase JS SDK v7.20.0 and later, measurementId is optional
# firebaseConfig = {
#     apiKey: "AIzaSyAhOLmdSCoZ2ATcpswSqeFjQuSjiLg4SKM",
#     authDomain: "majore-da630.firebaseapp.com",
#     databaseURL: "https://majore-da630-default-rtdb.firebaseio.com",
#     projectId: "majore-da630",
#     storageBucket: "majore-da630.firebasestorage.app",
#     messagingSenderId: "300801355721",
#     appId: "1:300801355721:web:d3a7551c8c5ffc1b0f4218",
#     measurementId: "G-5N9K57XPLQ"
# }

# firebase = Firebase(firebaseConfig)
# auth = firebase.auth()
db = firestore.client()

# Streamlit UI
st.title("Fitness Tracker with Firebase")
st.sidebar.header("Welcome!")
menu = st.sidebar.radio("Menu", ["Sign Up", "Log In", "Dashboard", "Predict My Calories"])

# Sign Up Page
if menu == "Sign Up":
    st.subheader("Create Your Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        # try:
        #     user = auth.create_user(email=email,password=password)
        # # Save user details to Firestore
        db.collection("users").document(email).set({"password": password})
        st.success("Account created! Please log in.")
        st.balloons()
        # except Exception as e:
        #     st.error(f"error: {e}")
# Log In Page
elif menu == "Log In":
    st.subheader("Log In to Your Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Log In"):
        # try:
        #     user = auth.get_user_by_email(email)
        #     st.success("login successfully")
        # except:
        #     st.warning("login failed")    
        # Check credentials
        doc = db.collection("users").document(email).get()
        if doc.exists and doc.to_dict()["password"] == password:
            st.success(f"Welcome back, {email}!")
            st.session_state["user"] = email
        else:
            st.error("Invalid credentials. Please try again.")

# Dashboard Page
elif menu == "Dashboard":
    if "user" not in st.session_state:
        st.warning("Please log in to access your dashboard.")
    else:
        user = st.session_state["user"]
        st.header(f"Welcome to Your Dashboard, {user}!")

        # Fetch user data
        user_docs = db.collection("fitness_data").where("user", "==", user).stream()
        user_data = [doc.to_dict() for doc in user_docs]
        
        if user_data:
            df = pd.DataFrame(user_data)
            st.write("Your Fitness Records:", df)
        else:
            st.write("No fitness records found. Add your first record below!")

        # Add new fitness record
        st.subheader("Add New Fitness Record")
        date = st.date_input("Date")
        age = st.number_input("Age", min_value=15, max_value=80, value=30)
        weight = st.number_input("Weight (kg)", min_value=40.0, max_value=200.0, value=70.0)
        height = st.number_input("Height (m)", min_value=1.2, max_value=2.5, value=1.7)
        session_duration = st.slider("Session Duration (hours)", min_value=0.5, max_value=3.0, value=1.0)
        total_steps = st.number_input("Total Steps", min_value=0, max_value=50000, value=5000)

        if st.button("Save and Predict"):
            # Dummy model for calorie prediction (replace with your model)
            features = [[age, weight, height, session_duration, total_steps]]
            model = RandomForestRegressor()
            model.fit([[30, 70, 1.7, 1.0, 5000]], [400])  # Train with dummy data
            calories_burned = model.predict(features)[0]

            # Save record to Firestore
            db.collection("fitness_data").add({
                "user": user,
                "date": str(date),
                "age": age,
                "weight": weight,
                "height": height,
                "session_duration": session_duration,
                "total_steps": total_steps,
                "calories_burned": calories_burned
            })

            st.success(f"Record saved! Estimated Calories Burned: {round(calories_burned, 2)}")

        # Display calorie trend
        if user_data:
            st.subheader("Calories Burned Trend")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            st.line_chart(df.set_index("date")["calories_burned"])

# Predict My Calories Page
elif menu == "Predict My Calories":
    st.header("Calorie Burn Prediction")

    if "user" not in st.session_state:
        st.warning("Please log in to use this feature.")
    else:
        user = st.session_state["user"]

        # Fetch all users' fitness data for model training
        docs = db.collection("fitness_data").stream()
        data = [doc.to_dict() for doc in docs]

        if data:
            df = pd.DataFrame(data)

            # Prepare features and target
            X = df[["age", "weight", "height", "session_duration", "total_steps"]]
            y = df["calories_burned"]

            # Train Random Forest Model
            model = RandomForestRegressor()
            model.fit(X, y)

            # User input for prediction
            st.subheader("Enter Your Details for Prediction")
            age = st.number_input("Age", min_value=15, max_value=80, value=30)
            weight = st.number_input("Weight (kg)", min_value=40.0, max_value=200.0, value=70.0)
            height = st.number_input("Height (m)", min_value=1.2, max_value=2.5, value=1.7)
            session_duration = st.slider("Session Duration (hours)", min_value=0.5, max_value=3.0, value=1.0)
            total_steps = st.number_input("Total Steps", min_value=0, max_value=50000, value=5000)

            if st.button("Predict"):
                prediction = model.predict([[age, weight, height, session_duration, total_steps]])[0]
                st.success(f"You are estimated to burn *{round(prediction, 2)} calories* in your session!")
        else:
            st.warning("Not enough data for prediction. Add more records in the dashboard.")