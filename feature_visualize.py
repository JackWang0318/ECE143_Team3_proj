import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def feat_vis(df):
    """
    Visualize the distribution of features in the dataset.
    Parameters:
    df : pd.DataFrame       The dataset.
    Returns:
    None
    """
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert len(df) > 0, "DataFrame must not be empty"
    plt.figure(figsize=(10, 10), dpi=100)
    plt.subplot(3, 3, 1)
    sns.histplot(data=df, x="internal_memory")
    plt.xlabel(df.columns[7])
    plt.subplot(3, 3, 2)
    sns.histplot(data=df, x="os")
    plt.xlabel(df.columns[1])
    plt.subplot(3, 3, 3)
    sns.histplot(data=df, x="screen_size")
    plt.xlabel(df.columns[2])
    plt.subplot(3, 3, 4)
    sns.histplot(data=df, x="4g")
    plt.xlabel(df.columns[3])
    plt.subplot(3, 3, 5)
    sns.histplot(data=df, x="5g")
    plt.xlabel(df.columns[4])
    plt.subplot(3, 3, 6)
    sns.histplot(data=df, x="ram")
    plt.xlabel(df.columns[8])
    plt.subplot(3, 3, 7)
    sns.histplot(data=df, x="battery")
    plt.xlabel(df.columns[9])
    plt.subplot(3, 3, 8)
    sns.histplot(data=df, x="normalized_used_price", kde=True)
    plt.xlabel(df.columns[13])
    plt.subplot(3, 3, 9)
    sns.histplot(data=df, x="front_camera_mp")
    plt.xlabel(df.columns[6])
    plt.tight_layout()
    plt.show()
