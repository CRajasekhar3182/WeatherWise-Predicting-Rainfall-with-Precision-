import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
try:
    model = joblib.load("rain_xgb.pkl")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model = None  # Ensure model is None if loading fails

# Title of the app
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 40px;
            color: #4CAF50;
            font-weight: bold;
        }
        .header {
            text-align: center;
            font-size: 30px;
            color: #FF5722;
        }
        .input-section {
            margin: 20px 0;
            padding: 10px;
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .footer {
            text-align: center;
            font-size: 12px;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Rain Prediction App</div>', unsafe_allow_html=True)

# Display model information
if model is not None:
    st.markdown("**Model Used:** XGBoost Classifier for predicting rain based on weather conditions.")
else:
    st.markdown("**Model Status:** Not loaded.")

# Input fields for user data
st.markdown('<div class="header">Input Weather Conditions</div>', unsafe_allow_html=True)

# Create a section for inputs
st.markdown('<div class="input-section">', unsafe_allow_html=True)

# Use number input for numerical fields
conditions = st.number_input("Conditions (Numerical Value)", min_value=0, value=5)
dew_temp = st.number_input("Dew Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0)

# Number inputs for binary fields, where 0 represents False and 1 represents True
fog = st.number_input("Fog (0 or 1)", min_value=0, max_value=1, value=0)
balls_of_ice = st.number_input("Balls of Ice (0 or 1)", min_value=0, max_value=1, value=0)
snow = st.number_input("Snow (0 or 1)", min_value=0, max_value=1, value=0)
tornado = st.number_input("Tornado (0 or 1)", min_value=0, max_value=1, value=0)

# Input for humidity, atmospheric pressure, temperature, and wind direction
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
atm_pressure = st.number_input("Atmospheric Pressure (hPa)", min_value=900, max_value=1100, value=1013)
temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
wind_direction = st.number_input("Wind Direction (°)", min_value=0, max_value=360, value=0)

# Close input section div
st.markdown('</div>', unsafe_allow_html=True)

# Button to make prediction
if st.button("Predict"):
    if model is not None:
        # Prepare input data as a 2D array
        input_data = np.array([[conditions, dew_temp, fog, balls_of_ice, humidity, atm_pressure, snow, temp, tornado, wind_direction]])
        
        # Make prediction
        try:
            prediction = model.predict(input_data)

            # Interpret prediction
            if prediction[0] == 1:
                st.success("Rain is predicted to come!")
            else:
                st.success("No rain is predicted.")

            # Log the prediction in session state
            if 'predictions' not in st.session_state:
                st.session_state['predictions'] = []
            st.session_state['predictions'].append({"Conditions": conditions, "Dew Temp": dew_temp, 
                                                    "Fog": fog, "Balls of Ice": balls_of_ice, 
                                                    "Humidity": humidity, "Pressure": atm_pressure, 
                                                    "Snow": snow, "Temp": temp, "Tornado": tornado, 
                                                    "Wind Direction": wind_direction, "Prediction": prediction[0]})

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Model not loaded. Cannot make predictions.")

# Display previous predictions
if 'predictions' in st.session_state and st.session_state['predictions']:
    st.markdown("### Previous Predictions:")
    predictions_df = pd.DataFrame(st.session_state['predictions'])
    st.dataframe(predictions_df)

    # Visualize predictions using a pie chart
    st.markdown("### Prediction Summary:")
    prediction_counts = predictions_df['Prediction'].value_counts()

    # Create pie chart only if there are predictions
    if not prediction_counts.empty:
        plt.figure(figsize=(6, 6))
        labels = ["No Rain", "Rain"]
        sizes = [prediction_counts.get(0, 0), prediction_counts.get(1, 0)]  # Get counts for both labels

        # Custom colors for the pie chart
        colors = ['#FF6347', '#4682B4']  # Tomato and Steel Blue
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title("Prediction Distribution (Rain vs No Rain)")
        st.pyplot(plt)
    else:
        st.warning("No predictions available to summarize.")

# Footer
st.markdown('<div class="footer">Developed by Your Name</div>', unsafe_allow_html=True)
