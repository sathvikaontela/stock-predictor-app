import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
import keras_tuner as kt
import time

st.set_page_config(page_title="Pro Stock Predictor", layout="wide")

# === Styles & Navbar ===
st.markdown("""
<style>
body { background-color: #f0f2f6; }
.navbar { background-color: #001f3f; padding: 15px; border-radius: 0px 0px 10px 10px; margin-bottom: 20px; }
.navbar a { color: #ffffff; margin-right: 20px; text-decoration: none; font-weight: bold; }
.navbar a:hover { color: #ffdd57; }
.hero { text-align: center; padding: 40px; }
.hero h1 { font-size: 3rem; color: #001f3f; }
.card { background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }
.main-title { color: #001f3f; text-align: center; margin-top: 20px; }
</style>
<div class="navbar">
    <a href="#predict">Predict</a>
    <a href="#strategy">Strategy Simulator</a>
    <a href="#about">About</a>
</div>
<div class="hero">
    <h1>📈 Professional Stock Predictor</h1>
    <p>Compare models, forecast prices, and simulate trading strategies — all in a sleek, interactive dashboard.</p>
</div>
""", unsafe_allow_html=True)

def set_random_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# === Data Loading ===
def get_data(symbol, start, end, retries=3, delay=5):
    for _ in range(retries):
        try:
            df = yf.download(symbol, start=start, end=end, timeout=10)
            if not df.empty:
                return df
        except:
            time.sleep(delay)
    return None

def prep_data(arr, n):
    x, y = [], []
    for i in range(len(arr) - n):
        x.append(arr[i:i+n, 0]); y.append(arr[i+n, 0])
    return np.array(x), np.array(y)

# === Build Models ===
def build_model(type_, shape):
    model = Sequential()
    if type_=="LSTM":
        model.add(LSTM(64, return_sequences=True, input_shape=shape)); model.add(Dropout(0.2))
        model.add(LSTM(64))
    elif type_=="GRU":
        model.add(GRU(64, return_sequences=True, input_shape=shape)); model.add(Dropout(0.2))
        model.add(GRU(64))
    else:  # BiLSTM
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=shape)); model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2)); model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def tuner_build(hp, type_, shape):
    model = Sequential()
    units = hp.Int('units', 32, 128, step=32)
    dropout = hp.Float('dropout', 0.1, 0.5, step=0.1)
    if type_=="LSTM":
        model.add(LSTM(units, return_sequences=True, input_shape=shape)); model.add(Dropout(dropout))
        model.add(LSTM(units))
    elif type_=="GRU":
        model.add(GRU(units, return_sequences=True, input_shape=shape)); model.add(Dropout(dropout))
        model.add(GRU(units))
    else:
        model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=shape))
        model.add(Dropout(dropout)); model.add(Bidirectional(LSTM(units)))
    model.add(Dropout(dropout)); model.add(Dense(1))
    lr = hp.Choice('lr', [1e-3, 1e-4, 5e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    return model

# === Buttons & Input ===
symbol = st.text_input("Stock Symbol", "AAPL").upper()
years = st.selectbox("Historical Data (years)", [1,2,3,4,5], index=2)
period = st.selectbox("Forecast Period", ['3 Months','6 Months','1 Year'])
tuning = st.checkbox("Enable Hyperparameter Tuning")
investor = st.radio("Investor Profile", ["Conservative","Aggressive"])

days_map = {'3 Months':90,'6 Months':180,'1 Year':365}
days = days_map[period]

if st.button("Run Full Forecast & Strategy"):
    set_random_seeds()
    end = pd.Timestamp.today(); start = end - pd.DateOffset(years=years)
    df = get_data(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        st.error("No data found for symbol.")
    else:
        close = df['Close'].values.reshape(-1,1)
        scaler = MinMaxScaler(); scaled = scaler.fit_transform(close)
        n = 60
        x, y = prep_data(scaled, n)
        x = x.reshape((x.shape[0], x.shape[1], 1))
        results = []; model_store = {}

        for m in ["LSTM","GRU","BiLSTM"]:
            st.info(f"Training {m}...")
            if tuning:
                tuner = kt.RandomSearch(lambda hp: tuner_build(hp,m,(n,1)),
                                        objective='loss', max_trials=3, executions_per_trial=1,
                                        directory='tunedir', project_name=f"tune_{m}")
                tuner.search(x,y,epochs=10,batch_size=32,verbose=0)
                hp = tuner.get_best_hyperparameters(1)[0]
                model = tuner.hypermodel.build(hp)
            else:
                model = build_model(m,(n,1))
            model.fit(x,y,epochs=50,batch_size=32,verbose=0)
            pred = model.predict(x)
            inv_pred = scaler.inverse_transform(pred); inv_true = scaler.inverse_transform(y.reshape(-1,1))
            rmse = np.sqrt(mean_squared_error(inv_true,inv_pred))
            # Forecast
            inp = scaled[-n:]; fc = []
            for _ in range(days):
                p = model.predict(inp.reshape(1,n,1), verbose=0)[0][0]
                fc.append(p); inp = np.append(inp[1:], [[p]], axis=0)
            fc_inv = scaler.inverse_transform(np.array(fc).reshape(-1,1)).flatten()
            pred_price = fc_inv[-1]; curr = close[-1][0]; profit = pred_price - curr
            results.append({"Model":m,"RMSE":round(rmse,2),"Profit/Loss":round(profit,2),"Forecast Price":round(pred_price,2)})
            model_store[m]=model

        df_res = pd.DataFrame(results).set_index("Model")
        st.markdown("### 🔍 Model Results")
        st.dataframe(df_res)

        best = df_res['RMSE'].idxmin()
        st.success(f"🏆 Best Model: {best}")

        # Forecast & Strategy with best
        st.markdown("### 📈 Interactive Forecast Plot")
        model = model_store[best]
        inp = scaled[-n:]; fcs=[]
        for _ in range(days):
            p = model.predict(inp.reshape(1,n,1),verbose=0)[0][0]
            fcs.append(p); inp = np.append(inp[1:], [[p]], axis=0)
        fcs_inv = scaler.inverse_transform(np.array(fcs).reshape(-1,1)).flatten()
        dates = pd.date_range(df.index[-1]+pd.Timedelta(1,'days'), periods=days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=close.flatten(), name="Historical"))
        fig.add_trace(go.Scatter(x=dates, y=fcs_inv, name="Forecast", line=dict(dash='dash')))
        fig.update_layout(title="Close Price & Forecast", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)

        # Simulate strategy
        cash, shares = 10000,0; pf=[]
        for i in range(days-1):
            t, tn = fcs_inv[i], fcs_inv[i+1]
            if tn>t:
                if investor=="Aggressive" and cash>0:
                    shares+=cash/t; cash=0
                if investor=="Conservative" and cash>0:
                    invest=cash*0.25; shares+=invest/t; cash-=invest
            elif tn<t and shares>0:
                cash+=shares*t; shares=0
            pf.append(cash + shares*t)
        pf.append(cash + shares*fcs_inv[-1])
        fig2 = go.Figure(go.Scatter(x=dates, y=pf, mode='lines+markers'))
        fig2.update_layout(title="Portfolio Value Over Time", xaxis_title="Date", yaxis_title="Portfolio ($)")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"**Investor Profile**: {investor}  ")
        st.markdown(f"**Start Value**: $10,000 → **End Value**: ${pf[-1]:,.2f}  ")
        st.markdown(f"**Total Return**: ${pf[-1] - 10000:,.2f}")
