import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif


def get_df_info(df: pd.DataFrame) -> pd.DataFrame:

    """
    Prints and returns detailed information about a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing descriptive statistics of the input DataFrame.
    """

    print("\n\033[1mShape of DataFrame:\033[0m", df.shape)
    print("\n\033[1mColumns in DataFrame:\033[0m", df.columns.to_list())
    print("\n\033[1mData types of columns:\033[0m\n", df.dtypes)

    print("\n\033[1mInformation about DataFrame:\033[0m")
    df.info()

    print("\n\033[1mNumber of unique values in each column:\033[0m")
    for col in df.columns:
        print(f"\033[1m{col}\033[0m: {df[col].nunique()}")

    print("\n\033[1mNull values in columns:\033[0m")
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    if len(null_columns) > 0:
        for col, count in null_columns.items():
            print(f"\033[1m{col}\033[0m: {count}")
    else:
        print("There are no null values in the DataFrame.")

    print("\n\033[1mNumber of duplicate rows:\033[0m", df.duplicated().sum())

    print("\n\033[1mDescriptive statistics of DataFrame:\033[0m\n")
    descriptive_stats = df.describe().transpose()
    print(descriptive_stats)

    return descriptive_stats


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")