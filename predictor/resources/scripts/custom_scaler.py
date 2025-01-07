def scale(df, scales):
    for col in df.select_dtypes('bool'):
        df[col] = df[col].astype(int)

    for col in df.select_dtypes('object'):
        df[col], _ = df[col].factorize()

    for index in scales.index:
        df[index] = (df[index] - scales.loc[index]['Min']) / scales.loc[index]['Range']

    return df
