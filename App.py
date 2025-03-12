import pickle
import numpy as np
import streamlit as st

# Load trained model & scaler
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details to predict survival.")

# User Input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, step=1)
fare = st.number_input("Fare", min_value=0.0, step=0.5)

# Embarked Selection
embarked = st.radio("Port of Embarkation", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
embarked_c = 1 if embarked == "Cherbourg (C)" else 0
embarked_q = 1 if embarked == "Queenstown (Q)" else 0
embarked_s = 1 if embarked == "Southampton (S)" else 0

# Convert categorical inputs to numerical values
sex = 1 if sex == "Male" else 0

# Make Prediction
if st.button("Predict"):
    try:
        # Scale numerical features (age, fare)
        scaled_features = scaler.transform([[age, fare]])  # Keeps it as 2D array with shape (1,2)

        # Ensure correct shape before concatenation
        features = np.hstack((scaled_features[0], [pclass, sex, embarked_c, embarked_q, embarked_s]))

        # Get prediction
        prediction = model.predict([features])[0]

        # Display prediction with colored band
        if prediction == 1:
            st.markdown(
                "<div style='background-color:#4CAF50; color:white; padding:10px; border-radius:5px; text-align:center;'>"
                "🎉 **Survived!**</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color:#ff4c4c; color:white; padding:10px; border-radius:5px; text-align:center;'>"
                "😞 **Did Not Survive**</div>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Error: {e}")