import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

#This is a tempele file. Please add your model as a class. 

# Your Model Class
class YourModel:
    def __init__(self, model_name="Yourmodel"):
        self.model_name = model_name
        self.model = None  # Placeholder for actual model

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
