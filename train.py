import numpy as np
import pandas as pd

from fetch_data import load_timeseries
from metrics import mape, smape

from models.arima import ARIMAModel
from models.prophet_model import ProphetModel
from models.nbeats import NBeatsWrapper

from darts import TimeSeries

# =========================
# Load FAOSTAT data
# =========================

df = load_timeseries(
    "data/raw/faostat.csv",
    item_name="Apples",
    area_name="Russian Federation"
)

print(df.head(), "\nYears:", len(df))

y = df["Value"].values.astype(float)

split = int(len(y) * 0.8)
train, test = y[:split], y[split:]

# =========================
# ARIMA BASELINE
# =========================

arima = ARIMAModel()
arima.fit(train)
pred_arima = arima.predict(len(test))

# =========================
# PROPHET
# =========================

prophet_df = pd.DataFrame({
    "ds": pd.to_datetime(df["Year"], format="%Y"),
    "y": df["Value"]
})

train_prophet = prophet_df.iloc[:split]

prophet = ProphetModel()
prophet.fit(train_prophet)

pred_prophet = prophet.predict(len(test))

# =========================
# N-BEATS
# =========================

series = TimeSeries.from_values(y)
train_series = series[:split]

nbeats = NBeatsWrapper()
nbeats.fit(train_series)

pred_nbeats = nbeats.predict(len(test)).values().flatten()

# =========================
# RESULTS
# =========================

results = {
    "ARIMA": (
        mape(test, pred_arima),
        smape(test, pred_arima)
    ),
    "Prophet": (
        mape(test, pred_prophet),
        smape(test, pred_prophet)
    ),
    "N-BEATS": (
        mape(test, pred_nbeats),
        smape(test, pred_nbeats)
    )
}

print("\nMODEL COMPARISON\n")

for model, (m, s) in results.items():
    print(f"{model:10s} | MAPE: {m:.2f}% | SMAPE: {s:.2f}%")

# =========================
# FUTURE FORECAST
# =========================

horizon = 5   # years ahead

print("\nFUTURE FORECAST (next 5 years)\n")

# --- ARIMA ---
arima_full = ARIMAModel()
arima_full.fit(y)
future_arima = arima_full.predict(horizon)

# --- PROPHET ---
prophet_full = ProphetModel()
prophet_full.fit(prophet_df)
future_prophet = prophet_full.predict(horizon)

# --- N-BEATS ---
series_full = TimeSeries.from_values(y)
nbeats_full = NBeatsWrapper()
nbeats_full.fit(series_full)
future_nbeats = nbeats_full.predict(horizon).values().flatten()

years_future = list(range(int(df["Year"].iloc[-1]) + 1,
                          int(df["Year"].iloc[-1]) + 1 + horizon))

forecast_df = pd.DataFrame({
    "Year": years_future,
    "ARIMA": future_arima,
    "Prophet": future_prophet,
    "N-BEATS": future_nbeats
})

print(forecast_df)
