import os
import pandas as pd
from preprocessing import preprocess_data
from model import LinearRegressionModel, NN_model, KNN_model, RandomForest_model, SVM_model, XGBoost_model
from sklearn.model_selection import train_test_split

def load_data(data_dir, apply_pca=True, n_components=10,
              add_poly=True, poly_degree=2,
              select_features=True, k=5):
    """Load and preprocess data"""
    file_path = os.path.join(data_dir, 'used_device_data.csv')
    df = preprocess_data(file_path, apply_pca=apply_pca, n_components=n_components,
                         add_poly=add_poly, poly_degree=poly_degree,
                         select_features=select_features, k=k)

    # Check if preprocessing returned a valid DataFrame
    if df is None or df.empty:
        raise ValueError("Preprocessing failed. Dataset is empty!")

    # Define features and target
    target_column = 'normalized_used_price'  
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def evaluate_models(models, data_dir, results_dir, **preprocessing_options):
    """Train and evaluate multiple models"""
    # Load data with preprocessing options
    X_train, X_test, y_train, y_test = load_data(data_dir, **preprocessing_options)

    results = []
    predictions_data = {"Actual": y_test.values.flatten()}

    for model_name, model_class in models.items():
        print(f"Training and evaluating {model_name}...")

        # Initialize model
        if model_name == "NN Model":
            model = model_class("NN", X_train.shape[1])
        elif model_name == "Linear Regression":
            model = model_class()
        elif model_name == "KNN Model":
            model = model_class(n_neighbors=9,
                                weights='distance',
                                metric='euclidean')
        elif model_name == "Random Forest":
            model = model_class(n_estimators=200, max_depth=10)
        elif model_name == "SVM Model":
            model = model_class(kernel="rbf", C=1.5, epsilon=0.05)
        elif model_name == "XGBoost Model":
            model = model_class(n_estimators=150, learning_rate=0.05, max_depth=8)

        # Train model
        model.train(X_train, y_train)
        
        # Evaluate model
        eval_results = model.evaluate(X_test, y_test)

        # Store evaluation metrics and predictions
        results.append({
            "Model": model_name,
            "RMSE": eval_results["RMSE"],
            "MAE": eval_results["MAE"]
        })
        predictions_data[model_name] = eval_results["Predictions"]

    # Save metrics and predictions to CSV files
    os.makedirs(results_dir, exist_ok=True)  # Ensure results directory exists
    metrics_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(predictions_data)

    metrics_df.to_csv(os.path.join(results_dir, 'evaluation_metrics.csv'), index=False)
    predictions_df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
    print("Evaluation completed! Results saved to", results_dir)


if __name__ == "__main__":
    models = {
        "Linear Regression": LinearRegressionModel,
        "NN Model": NN_model,
        "KNN Model": KNN_model,
        "Random Forest": RandomForest_model,
        "SVM Model": SVM_model,
        "XGBoost Model": XGBoost_model
    }

    data_dir = "data/"
    results_dir = "results/"

    # Define preprocessing options
    preprocessing_options = {
        "apply_pca": True,
        "n_components": 10,
        "add_poly": True,
        "poly_degree": 2,
        "select_features": True,
        "k": 5
    }

    evaluate_models(models, data_dir, results_dir, **preprocessing_options)
