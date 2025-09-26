# %%
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("Glaucoma Prediction App")

# Load the trained model (update filename if needed)
try:
    with open('glaucoma_model.pkl', 'rb') as file:   # 👈 replace with your model file
        glaucoma_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'glaucoma_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Define the label encoder for decoding predictions
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Glaucoma', 'No Glaucoma'])  # 👈 adjust if your model has different outputs

# Sidebar for user inputs
st.sidebar.header("Enter Patient Details")

# Example input features (👈 update these based on your dataset features)
age = st.sidebar.slider("Age", min_value=18, max_value=90, value=45)
iop = st.sidebar.slider("Intraocular Pressure (mmHg)", min_value=10, max_value=40, value=20)
cup_disc_ratio = st.sidebar.slider("Cup-to-Disc Ratio", min_value=0.1, max_value=1.0, value=0.5, step=0.01)
corneal_thickness = st.sidebar.slider("Central Corneal Thickness (microns)", min_value=450, max_value=650, value=540)
visual_field_loss = st.sidebar.slider("Visual Field Loss (%)", min_value=0, max_value=100, value=10)

# Categorical features
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
family_history = st.sidebar.selectbox("Family History of Glaucoma", options=["Yes", "No"])

# Function to preprocess input data
def preprocess_input(age, iop, cup_disc_ratio, corneal_thickness, visual_field_loss, gender, family_history):
    data = {
        'Age': age,
        'IOP': iop,
        'Cup_Disc_Ratio': cup_disc_ratio,
        'Corneal_Thickness': corneal_thickness,
        'Visual_Field_Loss': visual_field_loss,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'Family_History_Yes': 1 if family_history == 'Yes' else 0,
        'Family_History_No': 1 if family_history == 'No' else 0,
    }
    df = pd.DataFrame([data])
    return df

# Button to make prediction
if st.sidebar.button("Predict"):
    input_df = preprocess_input(age, iop, cup_disc_ratio, corneal_thickness, visual_field_loss, gender, family_history)

    try:
        prediction = glaucoma_model.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Display result
        st.subheader("Prediction Result")
        st.write(f"The predicted outcome is: **{predicted_label}**")
        if predicted_label == "No Glaucoma":
            st.success("No signs of glaucoma detected.")
        else:
            st.warning("The patient may have glaucoma. Further medical examination is recommended.")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Display instructions
st.write("""
### Instructions
1. Use the sidebar to enter the patient's clinical details.
2. Adjust the sliders for numerical features like Age, IOP, Cup-to-Disc Ratio, etc.
3. Select appropriate options for Gender and Family History.
4. Click the 'Predict' button to see the glaucoma prediction.
""")



