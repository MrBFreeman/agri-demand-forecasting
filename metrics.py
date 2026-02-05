import numpy as np

def mape(y, yhat):
    y = np.where(y == 0, 1e-8, y)
    return np.mean(np.abs((y - yhat) / y)) * 100

def smape(y, yhat):
    return 100 * np.mean(
        2 * np.abs(yhat - y) / (np.abs(y) + np.abs(yhat) + 1e-8)
    )

