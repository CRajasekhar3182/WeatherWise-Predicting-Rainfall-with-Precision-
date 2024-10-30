import streamlit as st
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load("rain_xgb.pkl")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model = None  # Ensure model is None if loading fails

# Title of the app
st.title("Rain Prediction App")
st.write("App Loaded")  # Debugging line to check if app is running

# Input fields for user data
st.header("Input Weather Conditions")

# Define mappings for categorical features
conditions_mapping = { 
    "Smoke": 0, "Mist": 1, "Clear": 2, "Widespread Dust": 3, "Fog": 4, 
    "Scattered Clouds": 5, "Partly Cloudy": 6, "Shallow Fog": 7, 
    "Mostly Cloudy": 8, "Light Rain": 9, "Partial Fog": 10, "Patches of Fog": 11, 
    "Thunderstorms and Rain": 12, "Heavy Fog": 13, "Light Drizzle": 14, 
    "Rain": 15, "Unknown": 16, "Blowing Sand": 17, "Overcast": 18, 
    "Thunderstorm": 19, "Light Thunderstorms and Rain": 20, "Drizzle": 21, 
    "Light Fog": 22, "Light Thunderstorm": 23, "Heavy Rain": 24, 
    "Heavy Thunderstorms and Rain": 25, "Thunderstorms with Hail": 26, 
    "Squalls": 27, "Light Sandstorm": 28, "Light Rain Showers": 29, 
    "Volcanic Ash": 30, "Light Haze": 31, "Sandstorm": 32, "Funnel Cloud": 33, 
    "Rain Showers": 34, "Heavy Thunderstorms with Hail": 35, 
    "Light Hail Showers": 36, "Light Freezing Rain": 37
}
wind_direction_mapping = {
    "North": 0, "West": 1, "WNW": 2, "East": 3, "NW": 4, "WSW": 5, "ESE": 6, 
    "ENE": 7, "SE": 8, "SW": 9, "NNW": 10, "NE": 11, "SSE": 12, "NNE": 13, 
    "SSW": 14, "South": 15, "Variable": 16
}

# Dropdown selection for 'conditions' and map to numeric
conditions = st.selectbox("Conditions", options=list(conditions_mapping.keys()))
conditions_encoded = conditions_mapping[conditions]

# Input for 'dew_temp'
dew_temp = st.number_input("Dew Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0)

# Drop-downs for binary values
fog = st.selectbox("Fog", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
balls_of_ice = st.selectbox("Balls of Ice", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
snow = st.selectbox("Snow", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
tornado = st.selectbox("Tornado", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Number input for 'humidity', 'atm_pressure', and 'temp'
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
atm_pressure = st.number_input("Atmospheric Pressure (hPa)", min_value=900, max_value=1100, value=1013)
temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)

# Dropdown for 'wind_direction' and map to numeric
wind_direction = st.selectbox("Wind Direction", options=list(wind_direction_mapping.keys()))
wind_direction_encoded = wind_direction_mapping[wind_direction]

# Button to make prediction
if st.button("Predict"):
    if model is not None:
        # Prepare input data as a 2D array
        input_data = np.array([[conditions_encoded, dew_temp, fog, balls_of_ice, humidity, 
                                atm_pressure, snow, temp, tornado, wind_direction_encoded]])

        # Make prediction
        try:
            prediction = model.predict(input_data)

            # Interpret prediction
            if prediction[0] == 1:
                st.success("Rain is predicted to come!")
            else:
                st.success("No rain is predicted.")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Model not loaded. Cannot make predictions.")