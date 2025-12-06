%%writefile tourism_project/deployment/app.py
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

#https://huggingface.co/divshiva1988/Tourism_package_acceptance_predictor_model/resolve/main/best_Tourism_package_acceptance_predictor_model_v1.joblib
# Download and load the trained model
repo_id="divshiva1988/Tourism_package_acceptance_predictor_model"
repo_type="space"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")
    
model_path = hf_hub_download(repo_id="divshiva1988/Tourism_package_acceptance_predictor_model", filename="best_Tourism_package_acceptance_predictor_model_v1.joblib")
model = joblib.load(model_path)

model=joblib.load("https://huggingface.co/divshiva1988/Tourism_package_acceptance_predictor_model/resolve/main/best_Tourism_package_acceptance_predictor_model_v1.joblib")

# Streamlit UI
st.title(" Tourism package acceptance Prediction")
st.write("""
This application predicts the  acceptance of a Tourism package.
""")

# User input
Designation = st.selectbox("Designation", ["AVP", "Executive", "Manager", "SENSenior Manager", "VP"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("MaritalStatus", ["Divorced", "Married", "Single", "UNmarried"])
ProductPitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "King","Standard","Super Deluxe"])
PreferredPropertyStar = st.selectbox("PreferredPropertyStar", [3,4,5])
Passport = st.selectbox("Passport", [0,1])
PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", [1,2,3,4,5])
OwnCar = st.selectbox("OwnCar", [0,1])
NumberOfChildrenVisiting = st.selectbox("NumberOfChildrenVisiting", [0,1,2,3])
Occupation = st.selectbox("Occupation", ["Free Lancer","Large Business", "Salaried", "Small Business"])
TypeofContact = st.selectbox("TypeofContact", ["Company Invited", "Self Enquiry"])

Age = st.number_input("Age of person", min_value=18.0, max_value=61.0, value=16.0, step=1)
CityTier = st.number_input("CityTier", min_value=1, max_value=3, value=1, step=1)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5, max_value=127, value=5, step=5)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=6, value=1)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=5, value=5, step=1)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1, max_value=22, value=1)
MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000,max_value=99000, value=1000,step=1000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Designation': Designation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Occupation': Occupation,
    'TypeofContact': TypeofContact,
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'MonthlyIncome': MonthlyIncome
    
}])

# Predict button
if st.button("Predict Tourism package acceptance"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Tourism package has strong chances of getting : **${prediction:,.2f} **")
