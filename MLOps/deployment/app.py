import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Download and load the trained model
try:
    model_repo_id = "ashikr/tourist_package_prediction_model"
    model_filename = "best_tourist_package_prediction_model_v1.joblib"
    model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Tourist Package Purchase Prediction")
st.write("Enter customer details to predict if they will take the tourist package.")

# User inputs for all features used in training
age = st.number_input("Age", 18, 100, 30)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration = st.number_input("Duration of Pitch", 0, 100, 10)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
num_person = st.number_input("Number of Person Visiting", 1, 10, 2)
num_followups = st.number_input("Number of Followups", 0, 10, 3)
product = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
star_rating = st.selectbox("Preferred Property Star", [3, 4, 5])
marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
num_trips = st.number_input("Number of Trips", 0, 20, 3)
passport = st.selectbox("Passport", [0, 1])
satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.selectbox("Own Car", [0, 1])
num_children = st.number_input("Number of Children Visiting", 0, 10, 1)
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
income = st.number_input("Monthly Income", 0, 100000, 20000)

# Map categorical values to numeric (matching prep.py LabelEncoding results where necessary)
# Note: OneHotEncoder in the pipeline handles string columns like 'Gender'
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': 1 if type_of_contact == "Self Enquiry" else 0,
    'CityTier': city_tier,
    'DurationOfPitch': duration,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': num_person,
    'NumberOfFollowups': num_followups,
    'ProductPitched': product,
    'PreferredPropertyStar': star_rating,
    'MaritalStatus': marital,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'PitchSatisfactionScore': satisfaction,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children,
    'Designation': designation,
    'MonthlyIncome': income
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    status = "Will Take Package" if prediction > 0.5 else "Will Not Take Package"
    st.subheader(f"Result: {status}")
    st.write(f"Confidence Score: {prediction:.2f}")

# Write requirements.txt
with open('MLOps/deployment/requirements.txt', 'w') as f:
    f.write('pandas==2.2.2\nhuggingface_hub==0.32.6\nstreamlit==1.43.2\njoblib==1.5.1\nscikit-learn==1.6.0\nxgboost==2.1.4\nmlflow==3.0.1')
