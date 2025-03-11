import os
import pandas as pd
import numpy as np
from preprocessing import preprocess_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from evaluation import load_data


def tune_knn(data_dir, preprocessing_options):
    # Load the data
    X_train, X_test, y_train, y_test = load_data(data_dir, **preprocessing_options)
    
    # Define parameter grid for tuning
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsRegressor()
    
    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found:", grid_search.best_params_)
    
    # Use the best model to make predictions on the test set
    best_knn = grid_search.best_estimator_
    predictions = best_knn.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print("Test RMSE:", rmse)
    print("Test MAE:", mae)
    
    return best_knn

if __name__ == "__main__":
    data_dir = "data/"
    # Preprocessing options as defined in your pipeline
    preprocessing_options = {
        "apply_pca": True,
        "n_components": 10,
        "add_poly": True,
        "poly_degree": 2,
        "select_features": True,
        "k": 5
    }
    
    best_model = tune_knn(data_dir, preprocessing_options)
