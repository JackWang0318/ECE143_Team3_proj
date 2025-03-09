import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def cor_heatmap(df):
    correlation = df.corr().round(2)
    sns.heatmap(correlation, annot=True, cmap="flare")
    plt.figure(figsize=(14, 7))


def plot_hist(data, col):
    plt.figure(figsize=(20, 15))
    sns.histplot(y=col, hue=col, palette="Set2", data=data)


def plot_numeric_feat(df):
    num_feature = [feature for feature in df.columns if df[feature].dtype != "O"]

    fig = plt.figure(figsize=(10, 10))

    for i in range(len(num_feature)):
        plt.subplot(4, 3, i + 1)
        sns.boxplot(data=df, x=df[num_feature[i]])

    plt.tight_layout()
    plt.show()


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
