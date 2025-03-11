import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression


def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)


def handle_missing_values(df):
    """Fill missing values using appropriate strategies."""
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric NaNs with median
    return df


def encode_categorical_features(df):
    """Convert categorical features into numeric representations."""
    df["4g"] = df["4g"].map({"yes": 1, "no": 0})
    df["5g"] = df["5g"].map({"yes": 1, "no": 0})

    # One-hot encode 'device_brand' and 'os'
    df = pd.get_dummies(df, columns=["device_brand", "os"], drop_first=True, dtype=int)
    return df


def scale_features(df):
    """Scale numerical features using Min-Max Scaling."""
    scaler = MinMaxScaler()
    
    # Dynamically detect numeric columns
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    if not numeric_columns:
        print("No numeric columns found for scaling. Skipping MinMaxScaler.")
        return df  # Skip scaling if no numeric columns exist
    
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    print(f"Scaled {len(numeric_columns)} numeric columns.")
    
    return df


def add_polynomial_features(X, degree=2):
    """Add polynomial features to the dataset."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    print(f"Added Polynomial Features: New shape {X_poly.shape}")
    return pd.DataFrame(X_poly)


def apply_pca_function(X, n_components=10):
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    print(f"Applied PCA: Reduced to {X_pca.shape[1]} components")
    return pd.DataFrame(X_pca)


def select_best_features(X, y, k=10):
    """Select top k best features using f_regression."""
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    
    print(f"Selected Top {k} Features: New shape {X_new.shape}")
    return pd.DataFrame(X_new)


def preprocess_data(file_path, apply_pca=False, n_components=10, 
                    add_poly=False, poly_degree=2, 
                    select_features=False, k=10):
    """Complete preprocessing pipeline with optional PCA, Polynomial, Feature Selection"""
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = encode_categorical_features(df)

    # Separate features and target
    target_column = "normalized_used_price"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("Before preprocessing, dataset shape:", X.shape)

    # Scale features first
    X = scale_features(X)

    # Feature Engineering
    if add_poly:
        X = add_polynomial_features(X, poly_degree)

    if apply_pca:
        X = apply_pca_function(X, n_components=n_components)

    if select_features:
        X = select_best_features(X, y, k)

    print("Final dataset shape:", X.shape)

    # Save preprocessed data
    processed_df = pd.concat([X, y], axis=1)
    processed_df.to_csv("data/processed_data.csv", index=False)
    print("Preprocessing Completed. Data saved as processed_data.csv")
    return processed_df


if __name__ == "__main__":
    # Set preprocessing options here
    preprocess_data(
        file_path="data/used_device_data.csv",
        apply_pca=True,       # Apply PCA or not
        n_components=10,      # Number of PCA components
        add_poly=True,        # Apply Polynomial Features or not
        poly_degree=2,        # Polynomial degree
        select_features=True, # Select top k features or not
        k=5                   # Number of top features to select
    )
