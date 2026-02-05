from darts import TimeSeries
from darts.models import NBEATSModel

class NBeatsWrapper:
    def fit(self, series):
        self.model = NBEATSModel(
            input_chunk_length=10,
            output_chunk_length=3,
            n_epochs=300,
            random_state=42
        )
        self.model.fit(series)

    def predict(self, steps):
        return self.model.predict(steps)

