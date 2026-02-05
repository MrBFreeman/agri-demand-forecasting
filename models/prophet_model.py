from prophet import Prophet
import pandas as pd

class ProphetModel:
    def fit(self, df):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        self.model.fit(df)

    def predict(self, periods):
        future = self.model.make_future_dataframe(periods=periods, freq="Y")
        forecast = self.model.predict(future)
        return forecast["yhat"].tail(periods).values

