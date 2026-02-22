import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import plotly.express as px
import time
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="Smart Electricity Monitor", layout="wide")

# Custom CSS for a clean UI
st.markdown("""
    <style>
    .stMetric { background-color: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; }
    .status-card { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Secret Keys and Assets
# Set these in the "Secrets" tab of Streamlit Cloud
try:
    CHANNEL_ID = st.secrets["3256606"]
    READ_API_KEY = st.secrets["YC5N2SQUWR1IYIR0"]
except:
    st.error("Secrets missing! Add TS_CHANNEL_ID and TS_READ_API_KEY to Streamlit Secrets.")
    st.stop()

@st.cache_resource
def load_ai_assets():
    model = tf.keras.models.load_model('combined_model.keras')
    iso_forest = joblib.load('iso_forest.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, iso_forest, scaler

model, iso_forest, scaler = load_ai_assets()

# 3. Data Fetching Logic
def fetch_data(results=50):
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={results}"
    try:
        response = requests.get(url).json()
        df = pd.DataFrame(response['feeds'])
        
        # Mapping hardware fields to labels
        rename_map = {
            'field1': 'Voltage', 'field2': 'Current', 'field3': 'Power',
            'field4': 'Energy', 'field5': 'Frequency', 'field6': 'Power_Factor'
        }
        df.rename(columns=rename_map, inplace=True)
        df['Time'] = pd.to_datetime(df['created_at'])
        
        cols = ['Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power_Factor']
        df[cols] = df[cols].apply(pd.to_numeric)
        return df[['Time'] + cols]
    except Exception:
        return None

# --- Main Dashboard UI ---
st.title("🛡️ Advanced Electricity Theft Detection System")
st.write(f"Monitoring Status: Active | Channel ID: {CHANNEL_ID}")

df = fetch_data()

if df is not None and not df.empty:
    latest = df.iloc[-1]
    
    # 4. Live KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Voltage", f"{latest['Voltage']} V")
    c2.metric("Current", f"{latest['Current']} A")
    c3.metric("Power", f"{latest['Power']} W")
    c4.metric("Frequency", f"{latest['Frequency']} Hz")

    st.divider()

    # 5. AI Security Analysis & History
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🤖 AI Security Scan")
        
        # Scale and segment data for prediction
        current_vals = latest[['Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power_Factor']].values.reshape(1, -1)
        scaled_features = scaler.transform(current_vals)
        
        # CNN-LSTM Logic (using 10-step window)
        sequence = np.repeat(scaled_features[:, np.newaxis, :], 10, axis=1)
        theft_prob = model.predict(sequence)[0][0]
        
        # Isolation Forest Logic (Detects unknown devices)
        iso_status = iso_forest.predict(scaled_features)[0]

        if theft_prob > 0.5:
            st.error(f"🚨 ALERT: Theft Detected ({theft_prob:.1%})")
            st.warning("Immediate inspection required.")
        elif iso_status == -1:
            st.warning("⚠️ Unknown Device Pattern Detected")
            if st.button("Mark as Safe"):
                st.success("Pattern Whitelisted.")
        else:
            st.success("✅ System Secure: Normal Pattern")
            
        st.info(f"Last sync: {datetime.now().strftime('%H:%M:%S')}")

    with col_right:
        st.subheader("📈 Recent Consumption Trend")
        fig = px.line(df, x='Time', y='Power', template='plotly_white', title="Power Usage (Watts)")
        fig.update_traces(line_color='#0ea5e9', fill='tozeroy')
        st.plotly_chart(fig, use_container_width=True)

# 6. Automated Refresh
time.sleep(16)
st.rerun()