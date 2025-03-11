import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

# NN model, contributed by Ching-Hao Wang, pushed by Hongjie Wang
class NN_model:
    def __init__(self, model_name="NN", input_shape=None):
        self.model_name = model_name
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        model = Sequential([
            Dense(64, activation="relu", input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ])
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
        return {"Predictions": predictions, "RMSE": rmse, "MAE": mae}


class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Linear Regression model trained.")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return {"Predictions": predictions, "RMSE": rmse, "MAE": mae}


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
        return {"Predictions": predictions, "RMSE": rmse, "MAE": mae}


class RandomForest_model:
    def __init__(self, model_name="RandomForest", n_estimators=100, max_depth=None):
        self.model_name = model_name
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )

    def train(self, X_train, y_train):
        """Train the Random Forest model"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using the Random Forest model"""
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance using RMSE and MAE"""
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return {"Predictions": predictions, "RMSE": rmse, "MAE": mae}


class SVM_model:
    def __init__(self, model_name="SVM", kernel="rbf", C=1.0, epsilon=0.1):
        self.model_name = model_name
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)

    def train(self, X_train, y_train):
        """Train the SVM model"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using the SVM model"""
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance using RMSE and MAE"""
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return {"Predictions": predictions, "RMSE": rmse, "MAE": mae}


class XGBoost_model:
    def __init__(self, model_name="XGBoost", n_estimators=100, learning_rate=0.1, max_depth=3):
        self.model_name = model_name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )

    def train(self, X_train, y_train):
        """Train the XGBoost model"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using the XGBoost model"""
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance using RMSE and MAE"""
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return {"Predictions": predictions, "RMSE": rmse, "MAE": mae}
