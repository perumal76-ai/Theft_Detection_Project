import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
import joblib
import requests
from datetime import datetime

# Page Config
st.set_page_config(page_title="Electricity Theft Detection", layout="wide")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #d1d5db; }
    .status-card { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('combined_model.keras')
    iso_forest = joblib.load('iso_forest.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, iso_forest, scaler

model, iso_forest, scaler = load_assets()

# --- ThingSpeak Data Fetching ---
def fetch_data(results=50):
    url = f"https://api.thingspeak.com/channels/3256606/feeds.json?api_key=YC5N2SQUWR1IYIR0&results={results}"
    data = requests.get(url).json()
    df = pd.DataFrame(data['feeds'])
    # Rename fields to professional labels
    df.columns = ['Time', 'Entry_ID', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power_Factor']
    df[['Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power_Factor']] = df[['Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power_Factor']].apply(pd.to_numeric)
    df['Time'] = pd.to_datetime(df['Time'])
    return df

# --- Main UI ---
st.title("🛡️ ELCOT: Advanced Electricity Theft Detection")

# Refresh Button in Sidebar
if st.sidebar.button('🔄 Refresh System'):
    st.rerun()

# 1. LIVE METRICS ROW
df = fetch_data()
latest = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Live Voltage", f"{latest['Voltage']}V", delta="Stable")
c2.metric("Current Load", f"{latest['Current']}A")
c3.metric("Active Power", f"{latest['Power']}W")
c4.metric("Frequency", f"{latest['Frequency']}Hz")

st.divider()

# 2. ANALYSIS ROW (Models vs Charts)
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("🤖 AI Security Analysis")
    # Prepare data for AI
    current_features = np.array([[latest['Voltage'], latest['Current'], latest['Power'], 
                                latest['Energy'], latest['Frequency'], latest['Power_Factor']]])
    scaled_data = scaler.transform(current_features)
    
    # Predict
    sequence_data = np.repeat(scaled_data[:, np.newaxis, :], 10, axis=1)
    theft_prob = model.predict(sequence_data)[0][0]
    iso_pred = iso_forest.predict(scaled_data)[0]

    # Professional Status Cards
    if theft_prob > 0.5:
        st.error("🚨 ALERT: CRIMINAL THEFT DETECTED")
        st.progress(float(theft_prob))
    elif iso_pred == -1:
        st.warning("⚠️ UNKNOWN DEVICE DETECTED")
        if st.button("Register New Device as Safe"):
            st.success("Updating Device Database...")
    else:
        st.success("✅ SYSTEM STATUS: SECURE")
    
    st.info(f"Last Scan: {datetime.now().strftime('%H:%M:%S')}")

with col_right:
    st.subheader("📈 Consumption History (Recent)")
    fig = px.line(df, x='Time', y='Power', title='Power Load (Watts)', 
                  template='plotly_white', line_shape='spline')
    fig.update_traces(line_color='#1f77b4')
    st.plotly_chart(fig, use_container_width=True)

# 3. RAW DATA LOG
with st.expander("📄 View System Log History"):
    st.dataframe(df.sort_values(by='Time', ascending=False), use_container_width=True)