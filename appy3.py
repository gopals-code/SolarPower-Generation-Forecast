import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Solar Power Forecasting App",
    page_icon="⚡",
    layout="wide"
)

# --- Load Model and Scaler ---
# Using st.cache_resource ensures these large objects are loaded only once.
@st.cache_resource
def load_model_and_scaler():
    """
    Loads the pre-trained RandomForest model and the StandardScaler from disk.
    Update the file paths to match the location on your system.
    """
    try:
        # Note: Ensure you are loading the RandomForest model saved from the notebook.
        model = joblib.load("pkl/solar_Power_eneration_Forecasting_model.pkl")
        scaler = joblib.load("pkl/scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please check the file paths.")
        return None, None

model, scaler = load_model_and_scaler()

# --- Load Data for Visualization ---
# Using st.cache_data caches the data after the first load for faster performance.
@st.cache_data
def load_data():
    """
    Loads the merged dataset for creating visualizations.
    Update the file path to match the location on your system.
    """
    try:
        data = pd.read_csv('data/Plant1_Merged_Dataset.csv', parse_dates=['DATE_TIME'])
        data.set_index('DATE_TIME', inplace=True)
        return data
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the file path.")
        return pd.DataFrame()

final_data = load_data()


# --- Sidebar for User Inputs and Information ---
st.sidebar.title("Controls & Information")
st.sidebar.header("User Inputs")

# User input fields are placed in the sidebar.
date_input = st.sidebar.text_input("Enter Date and Time (YYYY-MM-DD HH:MM)", "2020-06-15 14:00")
IRRADIATION = st.sidebar.slider("IRRADIATION (W/m²)", 0.0, 1.25, 0.8, 0.01, help="Intensity of solar radiation")
MODULE_TEMPERATURE = st.sidebar.slider("MODULE TEMPERATURE (°C)", 15.0, 70.0, 45.0, 0.5)
AMBIENT_TEMPERATURE = st.sidebar.slider("AMBIENT TEMPERATURE (°C)", 15.0, 50.0, 30.0, 0.5)

# --- Model Information Displayed in Sidebar ---
st.sidebar.title("Model Information")
st.sidebar.info("This app uses a RandomForest model to forecast DC power based on environmental and time-based features.")
st.sidebar.metric("Model R² Score", "0.996")  # Value from your notebook

# --- Main Page Layout ---
st.title("☀️ Solar DC Power Forecasting App")
st.markdown("Use the controls in the sidebar to get a real-time power prediction.")
st.markdown("---")

# Main prediction logic is executed only if the model and scaler are loaded successfully.
if model and scaler:
    # A single button to trigger the prediction.
    if st.button("Predict DC Power"):
        try:
            # Parse date and time to create time-based features
            input_time = datetime.strptime(date_input, "%Y-%m-%d %H:%M")
            hour = input_time.hour
            day = input_time.day
            month = input_time.month
            day_of_week = input_time.weekday()

            # Create a DataFrame from user inputs in the correct order
            features_list = [
                'IRRADIATION', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE',
                'HOUR', 'DAY', 'MONTH', 'DAY_OF_WEEK'
            ]
            input_df = pd.DataFrame([[
                IRRADIATION, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE,
                hour, day, month, day_of_week
            ]], columns=features_list)

            # Scale the input features using the loaded scaler
            input_scaled = scaler.transform(input_df)

            # Make a prediction using the loaded model
            predicted_dc_power = model.predict(input_scaled)

            # Display the prediction result prominently
            st.subheader("Prediction Result")
            st.metric("Predicted DC Power", f"{predicted_dc_power[0]:.2f} kW")
            st.success("Prediction successful!")

        except ValueError:
            st.error("Invalid Date Format. Please use YYYY-MM-DD HH:MM.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- Additional Sections for Data Exploration and Model Insights ---
if not final_data.empty:
    with st.expander("📊 Explore the Training Data"):
        st.subheader("Historical DC Power Generation")
        st.line_chart(final_data['DC_POWER'])

        st.subheader("Distribution of Key Features")
        feature_to_plot = st.selectbox("Select a feature", ['IRRADIATION', 'MODULE_TEMPERATURE', 'DC_POWER'])
        fig, ax = plt.subplots()
        sns.histplot(final_data[feature_to_plot].dropna(), kde=True, bins=50, ax=ax, color='orange')
        ax.set_title(f"Distribution of {feature_to_plot}")
        st.pyplot(fig)

if model and hasattr(model, 'feature_importances_'):
    with st.expander("💡 Model Insights"):
        st.subheader("How Features Impact the Prediction")
        
        features_list = [
            'IRRADIATION', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE',
            'HOUR', 'DAY', 'MONTH', 'DAY_OF_WEEK'
        ]
        
        importance_df = pd.DataFrame({
            'Feature': features_list,
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'))
        st.markdown("`IRRADIATION` is clearly the most important factor, followed by temperatures and the time of day (`HOUR`).")
        



