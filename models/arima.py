from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel:
    def fit(self, y):
        self.model = ARIMA(y, order=(1,1,1)).fit()

    def predict(self, steps):
        return self.model.forecast(steps)

