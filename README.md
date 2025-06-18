# 📈 Stock Predictor App

This is a Streamlit-based web app that predicts future stock prices using deep learning models like LSTM, GRU, and BiLSTM. It also simulates investment strategies and gives users a sense of how their portfolio would perform using different forecasting approaches.

The goal was to build something practical , not just another ML model. I wanted an app that helps visualize predictions, compares model performance, and lets you experiment with real-world investment profiles.

# Features

- Forecast stock prices for **3 months, 6 months, or 1 year**
- Compare models (LSTM vs GRU vs BiLSTM) using:
  - RMSE, MAE, R², and Profit/Loss
- Hyperparameter tuning using **KerasTuner**
- Visualize:
  - Predicted vs Historical prices
  - Portfolio growth over time
- Simulate strategies for:
  -  Conservative investors
  -  Aggressive investors
- Deployed on **Azure App Service**
- CI/CD with **GitHub Actions** (auto-deploy on push)

# Tech Stack

- **Frontend**: Streamlit + Plotly
- **Backend**: TensorFlow (LSTM, GRU, BiLSTM), scikit-learn
- **Data**: yFinance API (live historical stock data)
- **Deployment**: Azure App Service (Python 3.11)
- **CI/CD**: GitHub Actions + Azure Publish Profile

# How It Works

1. You enter a stock symbol (like `AAPL` or `GOOGL`)
2. Select a forecast horizon and your investor profile
3. The app:
   - Downloads historical data
   - Trains deep learning models
   - Forecasts future prices
   - Compares performance
   - Simulates investment outcome
     

