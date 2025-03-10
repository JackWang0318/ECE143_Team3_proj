import os
import pandas as pd
from model import LinearRegressionModel, NN_model # Import all model classes when more are added
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    # Load preprocessed data
    df = pd.read_csv(os.path.join(data_dir, 'processed_data.csv'))  # Assuming a single preprocessed file
    
    # Define features and target
    target_column = 'normalized_used_price'  # Modify if the target column is different
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def evaluate_models(models, data_dir, results_dir):
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_dir) 
    
    results = []
    predictions_data = {"Actual": y_test.values.flatten()}

    for model_name, model_class in models.items():
        print(f"Training and evaluating {model_name}...")
        
        # Initializing model
        if model_name == "NN Model":
            model = model_class("NN", X_train.shape[1])
        elif model_name == "Linear Regression":
            model = model_class()
        
        # model specific preprocessing
        #X_train_processed, y_train_processed = model.process_data(X_train, t_train)
        #X_test_processed, y_test_processed = model.process_data(X_test, t_test)

        # Training model
        model.train(X_train, y_train)
        
        # Evaluating model
        eval_results = model.evaluate(X_test, y_test)
        
        # Storing RMSE
        results.append({"Model": model_name, "RMSE": eval_results["RMSE"], "MAE": eval_results["MAE"]})
        
        # Storing predictions
        predictions_data[model_name] = eval_results["Predictions"]
    
    
    metrics_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(predictions_data)

    metrics_df.to_csv(os.path.join(results_dir, 'evaluation_metrics.csv'), index=False)
    predictions_df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)

if __name__ == "__main__":
    models = {
        "Linear Regression": LinearRegressionModel,
        "NN Model": NN_model
    }
    
    data_dir = "data/"
    results_dir = "results/"
    
    evaluate_models(models, data_dir, results_dir)