import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load and preprocess the data
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')
    data.set_index('date', inplace=True)
    return data

# File uploader
uploaded_file = st.file_uploader("Upload Malaria Excel File", type=["xlsx"])

# Only run the app if a file is uploaded
if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Title and instructions
    st.title("Malaria Data Analysis and Prediction")
    st.write("This app allows you to analyze malaria data and predict future trends using ARIMA.")

    # Region selection
    region = st.selectbox("Select Region:", ['All Regions'] + list(data['region'].unique()))

    # County selection
    counties = sorted(data[data['region'] == region]['county'].unique()) if region != 'All Regions' else sorted(data['county'].unique())
    county = st.selectbox("Select County:", counties)

    # Analysis type
    analysis_type = st.selectbox("Select Analysis Type:", ["Total Cases", "Severe Cases", "Deaths", "Mosquito Density"])

    # ARIMA inputs
    weeks_to_predict = st.number_input("Weeks to Predict:", min_value=1, max_value=12, value=4)
    p = st.number_input("ARIMA p:", min_value=0, max_value=5, value=1)
    d = st.number_input("ARIMA d:", min_value=0, max_value=5, value=1)
    q = st.number_input("ARIMA q:", min_value=0, max_value=5, value=1)

    # Filter data
    filtered_data = data[data['region'] == region] if region != 'All Regions' else data
    filtered_data = filtered_data[filtered_data['county'] == county]

    # Update analysis
    if st.button("Update Analysis"):
        if analysis_type == "Total Cases":
            data_series = filtered_data['total_cases']
        elif analysis_type == "Severe Cases":
            data_series = filtered_data['severe_cases']
        elif analysis_type == "Deaths":
            data_series = filtered_data['deaths']
        else:
            data_series = filtered_data['mosquito_density']

        st.subheader(f"Data Summary for {analysis_type}")
        st.write(data_series.describe())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(filtered_data.index, data_series, label=analysis_type)
        ax.set_title(f"{analysis_type} Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel(analysis_type)
        ax.legend()
        st.pyplot(fig)

    # Run prediction
    if st.button("Run Prediction"):
        data_series = filtered_data['total_cases']

        model = ARIMA(data_series, order=(p, d, q))
        model_fit = model.fit()

        forecast_index = [filtered_data.index[-1] + pd.DateOffset(weeks=i) for i in range(1, weeks_to_predict + 1)]
        forecast = model_fit.forecast(steps=weeks_to_predict)

        st.subheader(f"Prediction for {weeks_to_predict} weeks:")
        for i, pred in enumerate(forecast):
            st.write(f"Week {i+1}: {pred:.2f} predicted total cases")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(filtered_data.index, data_series, label="Historical Data")
        ax.plot(forecast_index, forecast, label="Predicted Data", linestyle='--')
        ax.set_title(f"Total Cases Prediction for {weeks_to_predict} Weeks")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Cases")
        ax.legend()
        st.pyplot(fig)

else:
    st.warning("Please upload a malaria Excel file to continue.")
