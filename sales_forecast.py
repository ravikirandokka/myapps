import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# App Title
st.title("Sales Forecasting Dashboard")
st.write("Upload your sales data to forecast future trends!")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file with 'ds' (date) and 'y' (sales) columns", type=["csv"])

if uploaded_file:
    # Load Data
    data = pd.read_csv(uploaded_file)

    # Validate Data
    if 'ds' not in data.columns or 'y' not in data.columns:
        st.error("The dataset must contain 'ds' (date) and 'y' (sales) columns!")
    else:
        # Display Data
        st.write("Uploaded Dataset:")
        st.dataframe(data.head())

        # Visualize Data
        st.subheader("Sales Over Time")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data['ds'], data['y'], label='Sales', color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)

        # Forecasting with Prophet
        st.subheader("Forecast Sales")
        forecast_period = st.slider("Select forecast period (days)", min_value=7, max_value=365, value=30)

        # Prepare Data for Prophet
        model = Prophet()
        model.fit(data)

        # Make Future Dataframe
        future = model.make_future_dataframe(periods=forecast_period)
        forecast = model.predict(future)

        # Display Forecast
        st.write("Forecasted Data:")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Plot Forecast
        st.subheader("Forecasted Sales")
        fig2 = model.plot(forecast)
        st.pyplot(fig2)

        # Components
        st.subheader("Forecast Components")
        fig3 = model.plot_components(forecast)
        st.pyplot(fig3)
else:
    st.info("Please upload a CSV file to start.")

# Footer
st.caption("Built with ❤️ using Streamlit and Prophet")
