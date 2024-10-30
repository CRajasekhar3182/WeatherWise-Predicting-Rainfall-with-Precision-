import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load the trained model
try:
    model = joblib.load("rain_xgb.pkl")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model = None  # Ensure model is None if loading fails

# Title of the app
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
        }
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
            padding: 20px;
            border: 1px solid #4CAF50;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .footer {
            text-align: center;
            font-size: 12px;
            color: gray;
        }
        .reset-button {
            background-color: #FF5722;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .reset-button:hover {
            background-color: #FF7043;
        }
        .input-label {
            font-weight: bold;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌧️ Rain Prediction App 🌧️</div>', unsafe_allow_html=True)

# Display model information
if model is not None:
    st.markdown("**Model Used:** XGBoost Classifier for predicting rain based on weather conditions.")
else:
    st.markdown("**Model Status:** Not loaded.")

# Input fields for user data
st.markdown('<div class="header">Input Weather Conditions</div>', unsafe_allow_html=True)

st.image("rain.jpg.webp", caption="Rain Prediction Visual", use_column_width=True)

# Create a section for inputs
st.markdown('<div class="input-section">', unsafe_allow_html=True)

# Input fields with better options
conditions = st.slider("Conditions (Numerical Value)", min_value=0, max_value=100, value=5, step=1)
dew_temp = st.slider("Dew Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0)

# Select boxes for binary fields
fog = st.selectbox("Fog (0 or 1)", options=[0, 1])
balls_of_ice = st.selectbox("Balls of Ice (0 or 1)", options=[0, 1])
snow = st.selectbox("Snow (0 or 1)", options=[0, 1])
tornado = st.selectbox("Tornado (0 or 1)", options=[0, 1])

# Input for humidity, atmospheric pressure, temperature, and wind direction
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50)
atm_pressure = st.slider("Atmospheric Pressure (hPa)", min_value=900, max_value=1100, value=1013)
temp = st.slider("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
wind_direction = st.slider("Wind Direction (°)", min_value=0, max_value=360, value=0)

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
                st.success("🌧️ Rain is predicted to come! 🌧️")
            else:
                st.success("☀️ No rain is predicted. ☀️")

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

    # Visualize predictions using Plotly
    st.markdown("### Prediction Summary:")
    prediction_counts = predictions_df['Prediction'].value_counts()

    # Create a Plotly bar chart
    if not prediction_counts.empty:
        fig = go.Figure(data=[
            go.Bar(x=["No Rain", "Rain"], 
                   y=[prediction_counts.get(0, 0), prediction_counts.get(1, 0)],
                   marker_color=['#007bff', '#28a745'],
                   text=[f"{prediction_counts.get(0, 0)}", f"{prediction_counts.get(1, 0)}"],
                   textposition='auto')
        ])

        fig.update_layout(
            title="🌧️ Prediction Distribution 🌧️",
            xaxis_title="Predictions",
            yaxis_title="Counts",
            template="plotly_white",
            showlegend=False
        )

        # Show the plotly chart
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No predictions available to summarize.")

# Reset input fields
if st.button("Reset Inputs"):
    # Clear predictions
    if 'predictions' in st.session_state:
        del st.session_state['predictions']

    # Reset all input fields to their default values
    conditions = 5
    dew_temp = 20.0
    fog = 0
    balls_of_ice = 0
    snow = 0
    tornado = 0
    humidity = 50
    atm_pressure = 1013
    temp = 25.0
    wind_direction = 0

# Footer
st.markdown('<div class="footer">Developed by Your Name</div>', unsafe_allow_html=True)
