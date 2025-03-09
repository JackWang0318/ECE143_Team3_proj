import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.neighbors import KNeighborsRegressor

# This is a template file. Please add your model as a class.
# Your Model Class
class YourModel:
    def __init__(self, model_name="Yourmodel"):
        self.model_name = model_name
        self.model = None

    def train(self, X_train, y_train):
        """Train the model"""
        raise NotImplementedError("Each model must implement its own train method.")

    def predict(self, X_test):
        """Make predictions"""
        raise NotImplementedError("Each model must implement its own predict method.")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return {"Model": self.model_name, "RMSE": rmse, "MAE": mae}


# NN model, contributed by Ching-Hao Wang, pushed by Hongjie Wang
class NN_model:
    def __init__(self, model_name="NN", input_shape=None):
        self.model_name = model_name
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(input_shape,)),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
        )

    def predict(self, X_test):
        return self.model.predict(X_test).flatten()

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return {"Model": self.model_name, "RMSE": rmse, "MAE": mae}
    

class KNN_model:
    def __init__(self, model_name="KNN", n_neighbors=5):
        self.model_name = model_name
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)

    def train(self, X_train, y_train):
        """Train the KNN model"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using the KNN model"""
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance using RMSE and MAE"""
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return {"Model": self.model_name, "RMSE": rmse, "MAE": mae}