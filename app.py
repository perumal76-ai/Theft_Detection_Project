import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import time
 
# 1. Load the AI Models and Scaler
# Use @st.cache_resource so they only load once
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model('combined_model.keras')
    iso_forest = joblib.load('iso_forest.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, iso_forest, scaler

model, iso_forest, scaler = load_models()

# 2. ThingSpeak Configuration (Replace with your actual keys)
CHANNEL_ID = "3256606"
READ_API_KEY = "YC5N2SQUWR1IYIR0"

def fetch_thingspeak_data():
    """Fetches the latest entry from ThingSpeak"""
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"
    try:
        response = requests.get(url).json()
        feed = response['feeds'][0]
        # Map your fields: Voltage, Current, Power, Energy, Frequency, Power_Factor
        data = [
            float(feed['field1']), float(feed['field2']), float(feed['field3']),
            float(feed['field4']), float(feed['field5']), float(feed['field6'])
        ]
        return np.array(data).reshape(1, -1), feed['created_at']
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

# 3. Streamlit UI Layout
st.title("⚡ ELCOT Electricity Theft Detection")
st.sidebar.header("System Settings")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 15)

# Container for real-time updates
placeholder = st.empty()

while True:
    raw_data, timestamp = fetch_thingspeak_data()
    
    if raw_data is not None:
        # Scale the data for the AI
        scaled_data = scaler.transform(raw_data)
        
        # --- MODEL 1: CNN-LSTM Theft Prediction ---
        # Reshape to (batch, steps, features) for the LSTM
        sequence_data = np.repeat(scaled_data[:, np.newaxis, :], 10, axis=1)
        theft_prob = model.predict(sequence_data)[0][0]
        
        # --- MODEL 2: Isolation Forest Outlier Detection ---
        iso_pred = iso_forest.predict(scaled_data) # -1 = Outlier, 1 = Normal

        with placeholder.container():
            # Display Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Voltage", f"{raw_data[0][0]} V")
            m2.metric("Current", f"{raw_data[0][1]} A")
            m3.metric("Power", f"{raw_data[0][2]} W")

            # Status Alerts
            if theft_prob > 0.5:
                st.error(f"🚨 ALERT: Theft Detected! (Confidence: {theft_prob:.2%})")
            elif iso_pred == -1:
                st.warning("⚠️ Unknown Device Detected! Is this a new device?")
                if st.button("Yes, Register as Normal"):
                    st.info("Recording device signature... Model will update on next training cycle.")
            else:
                st.success("✅ System Status: Normal")
            
            st.caption(f"Last updated: {timestamp}")

    time.sleep(refresh_rate) # Wait before next refresh
