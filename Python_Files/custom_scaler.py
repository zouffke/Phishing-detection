import pandas as pd


def scale(df):
    for col in df.select_dtypes('bool'):
        df[col] = df[col].astype(int)

    for col in df.select_dtypes('object'):
        df[col], _ = df[col].factorize()

    scales = pd.read_csv('../models/scale.csv', header=0, index_col=0)

    for index in scales.index:
        df[index] = (df[index] - scales.loc[index]['Min']) / scales.loc[index]['Range']

    return df
