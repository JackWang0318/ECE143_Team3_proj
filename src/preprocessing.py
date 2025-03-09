import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)


def handle_missing_values(df):
    """Fill missing values using appropriate strategies."""
    df.fillna(
        df.median(numeric_only=True), inplace=True
    )  # Fill numeric NaNs with median
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
    numeric_columns = [
        "screen_size",
        "rear_camera_mp",
        "front_camera_mp",
        "internal_memory",
        "ram",
        "battery",
        "weight",
        "release_year",
        "days_used",
        "normalized_new_price",
    ]
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


def preprocess_data(file_path):
    """Complete preprocessing pipeline."""
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = encode_categorical_features(df)
    df = scale_features(df)
    return df


if __name__ == "__main__":
    dataset_path = "../data/your_dataset.csv"
    df = preprocess_data(dataset_path)
    df.to_csv("../data/processed_data.csv", index=False)
    print("Preprocessing Completed. Data saved as processed_data.csv")
