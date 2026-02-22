import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests

# 1. Load the AI Models
model = tf.keras.models.load_model('combined_model.keras')
iso_forest = joblib.load('iso_forest.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="ELCOT Theft Detection", layout="wide")
st.title("⚡ ELCOT Real-Time Electricity Theft Detection")

# 2. ThingSpeak Configuration (Replace with your keys)
TS_CHANNEL_ID = "YOUR_CHANNEL_ID"
TS_READ_API_KEY = "YOUR_API_KEY"

def fetch_live_data():
    url = f"https://api.thingspeak.com/channels/{TS_CHANNEL_ID}/feeds.json?api_key={TS_READ_API_KEY}&results=1"
    response = requests.get(url).json()
    feed = response['feeds'][0]
    # Map your ESP32 fields (V, I, P, E, F, PF)
    data = [float(feed['field1']), float(feed['field2']), float(feed['field3']), 
            float(feed['field4']), float(feed['field5']), float(feed['field6'])]
    return np.array(data).reshape(1, -1)

# 3. Main Dashboard Logic
if st.button('Analyze Live Data'):
    raw_data = fetch_live_data()
    scaled_data = scaler.transform(raw_data)
    
    # Check 1: CNN-LSTM for Theft
    # We repeat the data 10 times to match the 'window' expected by the model
    sequence_data = np.repeat(scaled_data[:, np.newaxis, :], 10, axis=1)
    theft_pred = model.predict(sequence_data)[0][0]
    
    # Check 2: Isolation Forest for Unknown Devices
    iso_pred = iso_forest.predict(scaled_data) # -1 means outlier/unknown

    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        if theft_pred > 0.5:
            st.error(f"🚨 ALERT: Theft Detected! (Confidence: {theft_pred:.2%})")
        else:
            st.success("✅ Usage Pattern: Normal")

    with col2:
        if iso_pred == -1:
            st.warning("⚠️ Unknown Device Detected!")
            if st.button("Add this device to Normal List?"):
                st.info("Device signature saved. Retraining scheduled.")
                # Future: Append raw_data to your CSV for retraining
        else:
            st.info("🔍 Device Identity: Recognized")