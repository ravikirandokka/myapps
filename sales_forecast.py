import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# App Title
st.title("Enhanced Sales Forecasting Dashboard")
st.write("Upload your sales data to forecast future trends using multiple models!")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file with 'ds' (date) and 'y' (sales) columns", type=["csv"])

if uploaded_file:
    # Load Data
    data = pd.read_csv(uploaded_file)
    data['ds'] = pd.to_datetime(data['ds'])

    # Validate Data
    if 'ds' not in data.columns or 'y' not in data.columns:
        st.error("The dataset must contain 'ds' (date) and 'y' (sales) columns!")
    else:
        st.write("Uploaded Dataset:")
        st.dataframe(data.head())

        # Plot Original Data
        st.subheader("Sales Over Time")
        plt.figure(figsize=(10, 5))
        plt.plot(data['ds'], data['y'], label='Sales', color='blue')
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        st.pyplot(plt)

        # User Input for Forecasting
        forecast_period = st.slider("Select forecast period (days)", min_value=7, max_value=365, value=30)

        # Prophet Model
        st.subheader("Prophet Forecast")
        prophet_model = Prophet()
        prophet_model.fit(data)
        future = prophet_model.make_future_dataframe(periods=forecast_period)
        forecast_prophet = prophet_model.predict(future)
        st.write("Prophet Forecast:")
        st.dataframe(forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Prophet Plot
        st.pyplot(prophet_model.plot(forecast_prophet))

        # ARIMA Model
        st.subheader("ARIMA Forecast")
        arima_order = (st.slider("p", 0, 5, 2), st.slider("d", 0, 2, 1), st.slider("q", 0, 5, 2))
        train, test = data['y'][:-forecast_period], data['y'][-forecast_period:]
        arima_model = ARIMA(train, order=arima_order).fit()
        forecast_arima = arima_model.forecast(steps=forecast_period)
        st.write("ARIMA Forecast:")
        arima_forecast_df = pd.DataFrame({"ds": data['ds'][-forecast_period:], "yhat": forecast_arima})
        st.dataframe(arima_forecast_df)
        plt.figure(figsize=(10, 5))
        plt.plot(data['ds'], data['y'], label="Original Data")
        plt.plot(arima_forecast_df['ds'], arima_forecast_df['yhat'], label="ARIMA Forecast", color='orange')
        plt.legend()
        st.pyplot(plt)

        # LSTM Model
        st.subheader("LSTM Forecast")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['y'].values.reshape(-1, 1))

        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)

        seq_length = 10
        X, y = create_sequences(scaled_data, seq_length)
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        # LSTM Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

        # LSTM Forecast
        predictions = []
        current_input = X_test[0]
        for _ in range(forecast_period):
            pred = model.predict(current_input.reshape(1, seq_length, 1))
            predictions.append(pred[0, 0])
            current_input = np.append(current_input[1:], pred).reshape(-1, 1)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        lstm_forecast_df = pd.DataFrame({"ds": pd.date_range(start=data['ds'].iloc[-1], periods=forecast_period + 1)[1:], "yhat": predictions.flatten()})
        st.write("LSTM Forecast:")
        st.dataframe(lstm_forecast_df)

        plt.figure(figsize=(10, 5))
        plt.plot(data['ds'], data['y'], label="Original Data")
        plt.plot(lstm_forecast_df['ds'], lstm_forecast_df['yhat'], label="LSTM Forecast", color='green')
        plt.legend()
        st.pyplot(plt)

        # Model Comparison
        st.subheader("Model Comparison")
        st.write("Comparing Prophet, ARIMA, and LSTM forecasts.")
        plt.figure(figsize=(10, 5))
        plt.plot(data['ds'], data['y'], label="Original Data", color='blue')
        plt.plot(forecast_prophet['ds'][-forecast_period:], forecast_prophet['yhat'][-forecast_period:], label="Prophet", color='red')
        plt.plot(arima_forecast_df['ds'], arima_forecast_df['yhat'], label="ARIMA", color='orange')
        plt.plot(lstm_forecast_df['ds'], lstm_forecast_df['yhat'], label="LSTM", color='green')
        plt.legend()
        st.pyplot(plt)


st.caption("Built with ❤️ using Streamlit and Prophet, ARIMA,LSTM models")
